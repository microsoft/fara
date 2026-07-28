"""Minimal OpenAI-compatible chat completion client.

Replaces aztool's ChatCompletionClient with a thin wrapper over
``openai.AsyncOpenAI`` that exposes the same ``create(messages,
extra_create_args)`` interface the Fara agents expect, returning a
``CreateResult`` whose ``.content`` is the raw API message object
(``.content.content`` is the text).
"""

from typing import Any, Dict, List

from openai import AsyncOpenAI

from .messages import CreateResult, LLMMessage, RequestUsage, message_to_openai_format


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

    async def close(self) -> None:
        """Close the underlying HTTP client.

        ``AsyncOpenAI`` owns an async httpx connection pool.  Leaving it for
        garbage collection can make it try to close sockets after
        ``asyncio.run`` has already closed the event loop.
        """
        await self._client.close()
