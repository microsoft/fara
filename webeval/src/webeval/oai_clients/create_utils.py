"""Factory helpers for building :class:`ChatCompletionClient` instances from JSON config."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from .wrapper import (
    AzureMLClientWrapper,
    AzureOpenAIClientWrapper,
    AzureOpenAIResponsesWrapper,
    ChatCompletionClient,
    OpenAIClientWrapper,
    OrcaRouterClientWrapper,
)

ENVIRON_KEY_CHAT_COMPLETION_PROVIDER = "CHAT_COMPLETION_PROVIDER"
ENVIRON_KEY_CHAT_COMPLETION_KWARGS_JSON = "CHAT_COMPLETION_KWARGS_JSON"

# Backwards-compatible aliases for fara-private code that still uses the
# legacy short names.
_KWARGS_JSON_KEY = ENVIRON_KEY_CHAT_COMPLETION_KWARGS_JSON
_PROVIDER_KEY = ENVIRON_KEY_CHAT_COMPLETION_PROVIDER


def create_completion_client_from_env(
    config: Dict,
    logger: Optional[logging.Logger] = None,
    use_responses_api: bool = False,
) -> ChatCompletionClient:
    """Construct a client from a config dict.

    The dict must contain ``CHAT_COMPLETION_PROVIDER`` ("openai",
    "orcarouter", "azure", "trapi", "azure_ml", or "graceful_retry") and
    ``CHAT_COMPLETION_KWARGS_JSON`` (the kwargs forwarded to the
    underlying SDK client constructor).
    """

    env_dict = copy.deepcopy(config)

    raw_kwargs = env_dict.get(_KWARGS_JSON_KEY, {})
    if isinstance(raw_kwargs, str):
        raw_kwargs = json.loads(raw_kwargs)
    _kwargs = dict(raw_kwargs)

    _provider = env_dict.get(_PROVIDER_KEY, "openai").lower().strip()

    if _provider == "openai":
        _kwargs.pop("proxies", None)
        return OpenAIClientWrapper(**_kwargs)
    if _provider == "orcarouter":
        _kwargs.pop("proxies", None)
        return OrcaRouterClientWrapper(**_kwargs)
    if _provider in ("azure", "trapi"):
        model = _kwargs.get("model", _kwargs.get("azure_deployment", ""))
        if "codex" in model or "o3-pro" in model or use_responses_api:
            return AzureOpenAIResponsesWrapper(**_kwargs)
        return AzureOpenAIClientWrapper(**_kwargs)
    if _provider == "azure_ml":
        return AzureMLClientWrapper(**_kwargs)
    if _provider == "graceful_retry":
        from .graceful_client import GracefulRetryClient, ResponsesGracefulRetryClient

        log = logger or logging.getLogger(__name__)
        clients = []
        eval_model = _kwargs.get("eval_model", "*")
        sub_responses = use_responses_api or _kwargs.get("use_responses_client", False)

        for cfg in _kwargs.get("client_configs", []):
            model_name = cfg.get(_KWARGS_JSON_KEY, {}).get("model", "")
            if eval_model == "*" or GracefulRetryClient._should_include_model(model_name, eval_model):
                clients.append(create_client_from_config(cfg, use_responses_api=sub_responses))

        for path in _kwargs.get("config_paths", []):
            with open(path) as f:
                cfg = json.load(f)
            model_name = cfg.get(_KWARGS_JSON_KEY, {}).get("model", "")
            if eval_model == "*" or GracefulRetryClient._should_include_model(model_name, eval_model):
                clients.append(create_client_from_config(cfg, use_responses_api=sub_responses))

        if _kwargs.get("config_dir"):
            config_path = Path(_kwargs["config_dir"]).resolve()
            if config_path.is_dir():
                for file_path in config_path.iterdir():
                    if file_path.suffix == ".json":
                        with open(file_path) as f:
                            cfg = json.load(f)
                        model_name = cfg.get(_KWARGS_JSON_KEY, {}).get("model", "")
                        if eval_model == "*" or GracefulRetryClient._should_include_model(model_name, eval_model):
                            clients.append(create_client_from_config(cfg, use_responses_api=sub_responses))
            elif config_path.is_file():
                with open(config_path) as f:
                    clients.append(create_client_from_config(json.load(f), use_responses_api=sub_responses))

        if not clients:
            raise ValueError(
                f"No clients for graceful_retry. Check config or eval_model: {eval_model}"
            )

        client_class = (
            ResponsesGracefulRetryClient
            if _kwargs.get("use_responses_client")
            else GracefulRetryClient
        )
        return client_class(
            clients=clients,
            logger=log,
            max_retries=_kwargs.get("max_retries", 8),
            max_tokens=_kwargs.get("max_tokens", 115000),
            timeout=_kwargs.get("timeout"),
            support_json=_kwargs.get("support_json", True),
        )
    raise ValueError(f"Unknown provider {_provider!r}")


def create_client_from_config(
    config: Dict,
    logger: Optional[logging.Logger] = None,
    use_responses_api: bool = False,
) -> ChatCompletionClient:
    """Alias for :func:`create_completion_client_from_env` taking a dict."""
    return create_completion_client_from_env(config, logger=logger, use_responses_api=use_responses_api)


def create_client_from_file(
    path: str,
    logger: Optional[logging.Logger] = None,
    use_responses_api: bool = False,
) -> ChatCompletionClient:
    with open(path, "r") as f:
        config = json.load(f)
    return create_completion_client_from_env(
        config, logger=logger, use_responses_api=use_responses_api
    )
