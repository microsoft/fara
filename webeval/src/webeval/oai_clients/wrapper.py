"""Chat completion client wrappers around the OpenAI / Azure OpenAI / Azure ML SDKs.

Auth: Azure clients accept ``azure_ad_token_provider`` as a callable or
the string ``"DEFAULT"`` (resolves to
``azure.identity.get_bearer_token_provider(DefaultAzureCredential(), ...)``).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import httpx
import tiktoken
from openai import AsyncAzureOpenAI, AsyncOpenAI

from .messages import (
    CreateResult,
    ImageObj,
    LLMMessage,
    RequestUsage,
    Tool,
    ToolSchema,
    message_to_openai_format,
)

logger = logging.getLogger(__name__)


_COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"


def _default_azure_token_provider(scope: str = _COGNITIVE_SCOPE) -> Callable[[], str]:
    """Build a bearer-token provider with sensible Azure credential priority.

    Order: AzureCliCredential → ManagedIdentityCredential →
    DefaultAzureCredential. We try the CLI first because shared workstations
    often have a system-assigned managed identity that does NOT carry the
    `Cognitive Services OpenAI User` role on the deployments we hit, while
    the user's `az login` (or `az login --identity` against a different
    user-assigned identity) does — so the CLI credential is the one we
    actually want when both are available. Override with the
    ``AZURE_TOKEN_CREDENTIALS`` env var (azure-identity 1.16+) if you need
    a different ordering.
    """
    from azure.identity import (
        AzureCliCredential,
        ChainedTokenCredential,
        DefaultAzureCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )

    cred = ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential(),
        DefaultAzureCredential(exclude_cli_credential=True, exclude_managed_identity_credential=True),
    )
    return get_bearer_token_provider(cred, scope)


def _resolve_token_provider(
    value: Union[None, str, Callable[[], str]],
    scope: str = _COGNITIVE_SCOPE,
) -> Optional[Callable[[], str]]:
    """Coerce an ``azure_ad_token_provider`` config value into a callable.

    Accepts ``None`` (default → DefaultAzureCredential), ``"DEFAULT"``
    (same), any other string (raises), or a callable (returned as-is).
    """
    if value is None or (isinstance(value, str) and value.upper() == "DEFAULT"):
        return _default_azure_token_provider(scope)
    if callable(value):
        return value
    if isinstance(value, str):
        raise ValueError(
            f"azure_ad_token_provider={value!r} is not a recognised value. "
            "Use the string 'DEFAULT' or pass a callable returning a bearer token."
        )
    return value


@dataclass
class ModelCapabilities:
    json_output: bool = True
    function_calling: bool = True
    vision: bool = False


def _image_token_cost(width: int = 1920, height: int = 1080) -> int:
    """Approximate image token cost (OpenAI high-detail vision pricing)."""
    w, h = float(width), float(height)
    if max(w, h) > 2048:
        scale = 2048 / max(w, h)
        w, h = w * scale, h * scale
    if min(w, h) > 768:
        scale = 768 / min(w, h)
        w, h = w * scale, h * scale
    tiles = math.ceil(w / 512) * math.ceil(h / 512)
    return tiles * 170 + 85


_VLLM_ONLY_SAMPLING_KEYS = frozenset(
    {
        "top_k",
        "min_p",
        "repetition_penalty",
        "length_penalty",
        "early_stopping",
        "use_beam_search",
        "best_of",
        "guided_choice",
        "guided_regex",
        "guided_json",
        "guided_grammar",
        "skip_special_tokens",
        "spaces_between_special_tokens",
    }
)


def _route_vllm_extras_to_body(kwargs: Dict[str, Any]) -> None:
    """Move vLLM-only sampling params into ``extra_body`` so the OpenAI SDK accepts them."""
    extras = {k: kwargs.pop(k) for k in list(kwargs) if k in _VLLM_ONLY_SAMPLING_KEYS}
    if extras:
        body = kwargs.setdefault("extra_body", {})
        body.update(extras)


class ChatCompletionClient:
    """Base interface for chat completion clients."""

    def __init__(self, max_tokens: Optional[int] = None):
        self.metadata: Dict[str, Any] = {}
        self._last_usage = RequestUsage()
        self._total_usage = RequestUsage()
        self._max_tokens = max_tokens or 115000
        self._encoding = None

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] = (),
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
    ) -> CreateResult:
        raise NotImplementedError

    async def close(self) -> None:
        pass

    def supports_json(self) -> bool:
        return True

    def _get_encoding(self):
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    def count_tokens(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] = (),
    ) -> int:
        encoding = self._get_encoding()
        total = 0
        for msg in messages:
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, str):
                total += len(encoding.encode(content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        total += len(encoding.encode(item))
                    elif isinstance(item, ImageObj):
                        total += _image_token_cost(item.image.width, item.image.height)
                    elif isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type in ("image_url", "image"):
                            total += _image_token_cost()
                        elif item_type == "text":
                            total += len(encoding.encode(item.get("text", "")))
                        else:
                            total += len(encoding.encode(str(item)))
        for tool in tools:
            total += len(encoding.encode(str(tool)))
        return total

    def remaining_tokens(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] = (),
    ) -> int:
        return self._max_tokens - self.count_tokens(messages, tools)

    def actual_usage(self) -> RequestUsage:
        return self._last_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    @property
    def endpoint(self) -> str:
        if "azure_endpoint" in self.metadata and "azure_deployment" in self.metadata:
            return f"{self.metadata['azure_endpoint']}{self.metadata['azure_deployment']}"
        if "model" in self.metadata:
            return self.metadata["model"]
        return "unknown_endpoint"

    @property
    def description(self) -> str:
        if "azure_endpoint" in self.metadata and "azure_deployment" in self.metadata:
            return f"{self.metadata['azure_endpoint']}{self.metadata['azure_deployment']}"
        if "model" in self.metadata:
            return self.metadata["model"]
        return "unknown_client"


def _to_oai_messages(messages: Sequence[LLMMessage]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg:
            out.append(msg)
        else:
            out.append(message_to_openai_format(msg))
    return out


def _record_openai_usage(response, client: ChatCompletionClient) -> RequestUsage:
    if not response.usage:
        return RequestUsage()
    reasoning = 0
    details = getattr(response.usage, "completion_tokens_details", None)
    if details is not None:
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
    usage = RequestUsage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        reasoning_tokens=reasoning,
    )
    client._last_usage = usage
    client._total_usage += usage
    return usage


class OpenAIClientWrapper(ChatCompletionClient):
    """Wrapper around the OpenAI Python SDK (``openai.AsyncOpenAI``)."""

    def __init__(self, **kwargs):
        max_tokens = kwargs.pop("max_tokens", None)
        super().__init__(max_tokens=max_tokens)
        self._create_args = kwargs.copy()
        self.model = kwargs.pop("model", "gpt-4")
        self.metadata = {"model": self.model, "provider": "openai"}
        kwargs.pop("model_capabilities", None)
        kwargs.pop("max_retries", None)
        self.client = AsyncOpenAI(**kwargs)

    async def close(self) -> None:
        await self.client.close()

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] = (),
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
    ) -> CreateResult:
        kwargs: Dict[str, Any] = {"model": self.model, "messages": _to_oai_messages(messages)}
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = list(tools)
        kwargs.update(extra_create_args)
        _route_vllm_extras_to_body(kwargs)

        response = await self.client.chat.completions.create(**kwargs)
        usage = _record_openai_usage(response, self)
        msg = response.choices[0].message
        return CreateResult(
            content=msg.content if msg.content is not None else "",
            usage=usage,
            finish_reason=response.choices[0].finish_reason,
            message=msg,
            tool_calls=list(msg.tool_calls) if getattr(msg, "tool_calls", None) else None,
        )


class OrcaRouterClientWrapper(OpenAIClientWrapper):
    """OpenAI-compatible client for the OrcaRouter gateway.

    OrcaRouter exposes many upstream models behind a single OpenAI-compatible
    endpoint (``https://api.orcarouter.ai/v1``) using ``vendor/model`` names,
    plus adaptive routing via ``orcarouter/auto``. When no ``base_url`` or
    ``api_key`` is supplied, it defaults to the public endpoint and the
    ``ORCAROUTER_API_KEY`` environment variable.
    """

    DEFAULT_BASE_URL = "https://api.orcarouter.ai/v1"

    def __init__(self, **kwargs):
        kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
        kwargs.setdefault("api_key", os.environ.get("ORCAROUTER_API_KEY"))
        super().__init__(**kwargs)
        self.metadata = {"model": self.model, "provider": "orcarouter"}


class AzureOpenAIClientWrapper(ChatCompletionClient):
    """Wrapper around Azure OpenAI Chat Completions."""

    def __init__(self, **kwargs):
        max_tokens = kwargs.pop("max_tokens", None)
        super().__init__(max_tokens=max_tokens)
        self._create_args = kwargs.copy()
        self.model = kwargs.pop("azure_deployment", kwargs.pop("model", "gpt-4"))
        self.metadata = {
            "azure_endpoint": kwargs.get("azure_endpoint", ""),
            "azure_deployment": self.model,
            "provider": "azure",
        }
        kwargs.pop("model_capabilities", None)
        kwargs.pop("max_retries", None)

        if "api_key" not in kwargs:
            kwargs["azure_ad_token_provider"] = _resolve_token_provider(
                kwargs.pop("azure_ad_token_provider", None)
            )
        else:
            kwargs.pop("azure_ad_token_provider", None)

        self._azure_kwargs = kwargs.copy()
        self.client = AsyncAzureOpenAI(**kwargs)

    async def close(self) -> None:
        await self.client.close()

    def refresh_credentials(self) -> None:
        if "api_key" in self._azure_kwargs:
            return
        self._azure_kwargs["azure_ad_token_provider"] = _resolve_token_provider(None)
        self.client = AsyncAzureOpenAI(**self._azure_kwargs)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] = (),
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
    ) -> CreateResult:
        kwargs: Dict[str, Any] = {"model": self.model, "messages": _to_oai_messages(messages)}
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = list(tools)
        kwargs.update(extra_create_args)
        _route_vllm_extras_to_body(kwargs)

        response = await self.client.chat.completions.create(**kwargs)
        usage = _record_openai_usage(response, self)
        msg = response.choices[0].message
        return CreateResult(
            content=msg.content if msg.content is not None else "",
            usage=usage,
            finish_reason=response.choices[0].finish_reason,
            message=msg,
            tool_calls=list(msg.tool_calls) if getattr(msg, "tool_calls", None) else None,
        )


class AzureOpenAIResponsesWrapper(ChatCompletionClient):
    """Wrapper around the Azure OpenAI Responses API (codex / o3-pro / gpt-5)."""

    def __init__(self, **kwargs):
        max_tokens = kwargs.pop("max_tokens", None)
        super().__init__(max_tokens=max_tokens)
        self._create_args = kwargs.copy()
        self.model = kwargs.pop("azure_deployment", kwargs.pop("model", "gpt-4"))
        self.metadata = {
            "azure_endpoint": kwargs.get("azure_endpoint", ""),
            "azure_deployment": self.model,
            "provider": "azure_responses",
        }
        kwargs.pop("model_capabilities", None)
        kwargs.pop("max_retries", None)

        if kwargs.get("api_version", "") < "2025-03-01":
            kwargs["api_version"] = "2025-03-01-preview"

        if "api_key" not in kwargs:
            kwargs["azure_ad_token_provider"] = _resolve_token_provider(
                kwargs.pop("azure_ad_token_provider", None)
            )
        else:
            kwargs.pop("azure_ad_token_provider", None)

        self._azure_kwargs = kwargs.copy()
        self.client = AsyncAzureOpenAI(**kwargs)

    async def close(self) -> None:
        await self.client.close()

    def refresh_credentials(self) -> None:
        if "api_key" in self._azure_kwargs:
            return
        self._azure_kwargs["azure_ad_token_provider"] = _resolve_token_provider(None)
        self.client = AsyncAzureOpenAI(**self._azure_kwargs)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] = (),
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
    ) -> CreateResult:
        kwargs: Dict[str, Any] = {"model": self.model, "input": _to_oai_messages(messages)}
        if json_output:
            kwargs["text"] = {"format": {"type": "json_object"}}
        if tools:
            kwargs["tools"] = list(tools)
        kwargs.update(extra_create_args)

        if "reasoning_effort" in kwargs:
            effort = kwargs.pop("reasoning_effort")
            reasoning = kwargs.get("reasoning", {})
            reasoning["effort"] = effort
            kwargs["reasoning"] = reasoning

        response = await self.client.responses.create(**kwargs)

        reasoning = 0
        if response.usage and getattr(response.usage, "output_tokens_details", None):
            reasoning = response.usage.output_tokens_details.reasoning_tokens or 0
        usage = RequestUsage(
            prompt_tokens=response.usage.input_tokens if response.usage else 0,
            completion_tokens=response.usage.output_tokens if response.usage else 0,
            reasoning_tokens=reasoning,
        )
        self._last_usage = usage
        self._total_usage += usage

        text = getattr(response, "output_text", None)
        if not text:
            chunks: list[str] = []
            for item in getattr(response, "output", []) or []:
                content = getattr(item, "content", None)
                if not content:
                    continue
                for block in content:
                    block_text = getattr(block, "text", None)
                    if block_text:
                        chunks.append(block_text)
            text = "".join(chunks)
        return CreateResult(
            content=text,
            usage=usage,
            finish_reason=response.status,
            message=response,
        )


class AzureMLClientWrapper(ChatCompletionClient):
    """Wrapper for Azure ML managed-online endpoints with a custom ``/score`` API."""

    def __init__(self, **kwargs):
        max_tokens = kwargs.pop("max_tokens", None)
        super().__init__(max_tokens=max_tokens)
        self.model = kwargs.pop("model", "unknown")
        self._score_url = kwargs.pop("score_url")
        self._capabilities = ModelCapabilities(**kwargs.pop("model_capabilities", {}))
        self._max_completion_tokens = kwargs.pop("max_completion_tokens", 4096)

        self._token_scope = kwargs.pop("azure_ad_token_scope", "https://ml.azure.com")
        self._token_provider = _resolve_token_provider(
            kwargs.pop("azure_ad_token_provider", None), self._token_scope
        )

        self.metadata = {
            "model": self.model,
            "provider": "azure_ml",
            "score_url": self._score_url,
        }

    def refresh_credentials(self) -> None:
        self._token_provider = _resolve_token_provider(None, self._token_scope)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] = (),
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
    ) -> CreateResult:
        parts: list[str] = []
        image_b64: Optional[str] = None
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = msg.source
                content = msg.content

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, ImageObj):
                        image_b64 = item.to_base64()
                    elif isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(item["text"])
                        elif item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                image_b64 = url.split(",", 1)[-1]
            elif isinstance(content, str) and content:
                prefix = f"[{role}] " if role == "system" else ""
                parts.append(prefix + content)

        data: Dict[str, Any] = {
            "user_prompt": "\n".join(parts),
            "max_new_tokens": self._max_completion_tokens,
        }
        if image_b64:
            data["image_input"] = image_b64

        token = self._token_provider() if self._token_provider else ""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

        async with httpx.AsyncClient(timeout=300) as http:
            resp = await http.post(self._score_url, headers=headers, content=json.dumps(data))
            if resp.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"{resp.status_code} for {self._score_url}: {resp.text}",
                    request=resp.request,
                    response=resp,
                )
            result = resp.json()

        if isinstance(result, dict):
            raw = result.get("output", result.get("result", ""))
            prompt_tokens = result.get("initial_input_tokens", 0)
            completion_tokens = result.get("output_tokens", 0)
        else:
            raw = result
            prompt_tokens = 0
            completion_tokens = 0

        if isinstance(raw, list):
            text = "".join(str(item) for item in raw)
        elif isinstance(raw, str):
            text = raw
        else:
            text = json.dumps(raw) if raw else ""

        usage = RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self._last_usage = usage
        self._total_usage += usage

        message = type("Message", (), {"content": text, "tool_calls": None})()
        return CreateResult(content=message, usage=usage, finish_reason="stop")


class ClientWrapper:
    """Namespace for the two ChatCompletionClient construction helpers.

    The wrappers in this module already carry the metadata callers
    historically read from a separate adapter, so this class just
    forwards to ``create_client_from_config`` / ``create_client_from_file``.
    """

    @staticmethod
    def from_config(config: Dict[str, Any]) -> ChatCompletionClient:
        from .create_utils import create_client_from_config

        return create_client_from_config(config)

    @staticmethod
    def from_file(path: str) -> ChatCompletionClient:
        from .create_utils import create_client_from_file

        return create_client_from_file(path)
