"""Minimal OpenAI-compatible chat completion client.

Replaces aztool's ChatCompletionClient with a thin wrapper over
``openai.AsyncOpenAI`` that exposes the same ``create(messages,
extra_create_args)`` interface the Fara agents expect, returning a
``CreateResult`` whose ``.content`` is the raw API message object
(``.content.content`` is the text).
"""

import json
from typing import Any, Dict, List

from openai import AsyncOpenAI

from .messages import CreateResult, LLMMessage, RequestUsage, message_to_openai_format


def _text_value(value: Any) -> str:
    """Flatten the text-like content variants used by compatible servers."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_text_value(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return _text_value(value[key])
        return ""
    for attribute in ("text", "content", "value"):
        nested = getattr(value, attribute, None)
        if nested is not None and nested is not value:
            return _text_value(nested)
    return ""


def extract_message_text(message: Any) -> str:
    """Get generated text from an OpenAI or OpenAI-compatible message.

    Some reasoning servers put the entire generated response in a non-standard
    reasoning field and leave ``content`` empty.  Native function calls are
    normalized back into the textual format expected by Fara's trained parser.
    """
    if isinstance(message, str):
        return message

    values: list[str] = []
    standard_content = _text_value(getattr(message, "content", None)).strip()

    extras = getattr(message, "model_extra", None) or {}
    for field_name in ("reasoning_content", "reasoning", "analysis"):
        value = getattr(message, field_name, None)
        if value is None and isinstance(extras, dict):
            value = extras.get(field_name)
        text = _text_value(value).strip()
        if text and text not in values:
            values.append(text)

    if standard_content and standard_content not in values:
        values.append(standard_content)

    tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        if function is None and isinstance(tool_call, dict):
            function = tool_call.get("function")
        name = (
            function.get("name")
            if isinstance(function, dict)
            else getattr(function, "name", None)
        )
        arguments = (
            function.get("arguments")
            if isinstance(function, dict)
            else getattr(function, "arguments", None)
        )
        if not name:
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                # Keep malformed arguments visible to the normal parser, which
                # will produce a useful error and trigger its bounded retry.
                tool_text = (
                    f'<tool_call>{{"name": {json.dumps(name)}, '
                    f'"arguments": {arguments}}}</tool_call>'
                )
                values.append(tool_text)
                continue
        tool_text = json.dumps(
            {"name": name, "arguments": arguments or {}}, ensure_ascii=False
        )
        values.append(f"<tool_call>{tool_text}</tool_call>")

    return "\n".join(values)


class ChatCompletionClient:
    """Chat completion client backed by an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def create(
        self,
        messages: List[LLMMessage],
        extra_create_args: Dict[str, Any] | None = None,
    ) -> CreateResult:
        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": [message_to_openai_format(m) for m in messages],
        }
        if extra_create_args:
            request_params.update(extra_create_args)

        response = await self._client.chat.completions.create(**request_params)
        usage = RequestUsage()
        if response.usage:
            usage = RequestUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
        return CreateResult(
            content=response.choices[0].message,
            usage=usage,
            finish_reason=response.choices[0].finish_reason,
        )
