"""Smoke + behavioural tests for ``webeval.oai_clients``.

  * Importing the package never pulls ``autogen_core`` / ``autogen_ext``
    into ``sys.modules`` (regression guard against an autogen reintroduction).
  * ``GracefulRetryClient.from_path`` filters configs by model name.
  * ``message_to_openai_format`` produces dicts the OpenAI SDK accepts.
  * ``CreateResult.content`` is the response text string.
  * Full ``client.create()`` round trip works against a stubbed
    ``AsyncOpenAI`` (covers usage tracking and tool-call extraction).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from PIL import Image as PILImage


def test_oai_clients_does_not_pull_autogen():
    # Force a clean import.
    for mod in list(sys.modules):
        if mod.startswith("webeval.oai_clients") or mod.startswith("autogen"):
            del sys.modules[mod]

    import webeval.oai_clients  # noqa: F401

    leaked = [m for m in sys.modules if m.startswith("autogen")]
    assert not leaked, f"oai_clients pulled in autogen modules: {leaked}"


def test_message_to_openai_format_text_only():
    from webeval.oai_clients import SystemMessage, UserMessage, message_to_openai_format

    assert message_to_openai_format(SystemMessage(content="hi")) == {
        "role": "system",
        "content": "hi",
    }
    assert message_to_openai_format(UserMessage(content="ping")) == {
        "role": "user",
        "content": "ping",
    }


def test_message_to_openai_format_multimodal():
    from webeval.oai_clients import ImageObj, UserMessage, message_to_openai_format

    pil = PILImage.new("RGB", (4, 4), (10, 20, 30))
    msg = UserMessage(content=["look", ImageObj.from_pil(pil)])
    out = message_to_openai_format(msg)
    assert out["role"] == "user"
    parts = out["content"]
    assert len(parts) == 2
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_request_usage_addition():
    from webeval.oai_clients import RequestUsage

    a = RequestUsage(prompt_tokens=10, completion_tokens=2)
    b = RequestUsage(prompt_tokens=5, completion_tokens=3)
    s = a + b
    assert s.prompt_tokens == 15
    assert s.completion_tokens == 5
    assert s.num_calls == 2


def test_graceful_retry_from_path_filters_by_eval_model(tmp_path):
    from webeval.oai_clients import GracefulRetryClient

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    # Two configs: one gpt-4o, one o4-mini.
    (cfg_dir / "a.json").write_text(
        json.dumps(
            {
                "CHAT_COMPLETION_PROVIDER": "openai",
                "CHAT_COMPLETION_KWARGS_JSON": {
                    "model": "gpt-4o",
                    "api_key": "sk-test",
                },
            }
        )
    )
    (cfg_dir / "b.json").write_text(
        json.dumps(
            {
                "CHAT_COMPLETION_PROVIDER": "openai",
                "CHAT_COMPLETION_KWARGS_JSON": {
                    "model": "o4-mini",
                    "api_key": "sk-test",
                },
            }
        )
    )

    import logging

    logger = logging.getLogger(__name__)

    g4o = GracefulRetryClient.from_path(cfg_dir, logger=logger, eval_model="gpt-4o")
    assert len(g4o._clients) == 1
    assert "gpt-4o" in g4o._clients[0].description

    o4 = GracefulRetryClient.from_path(cfg_dir, logger=logger, eval_model="o4-mini")
    assert len(o4._clients) == 1
    assert "o4-mini" in o4._clients[0].description

    star = GracefulRetryClient.from_path(cfg_dir, logger=logger, eval_model="*")
    assert len(star._clients) == 2

    with pytest.raises(ValueError):
        GracefulRetryClient.from_path(cfg_dir, logger=logger, eval_model="nope")


def test_should_include_model_exact_match():
    from webeval.oai_clients import GracefulRetryClient

    assert GracefulRetryClient._should_include_model("gpt-4o", "gpt-4o")
    assert GracefulRetryClient._should_include_model("gpt-4o", "*")
    assert GracefulRetryClient._should_include_model("o4-mini", ["gpt-4o", "o4-mini"])
    assert not GracefulRetryClient._should_include_model("gpt-4o", "o4-mini")
    assert not GracefulRetryClient._should_include_model("gpt-4o-mini", "gpt-4o")


class _AlwaysFailsClient:
    """Fake ChatCompletionClient whose create() always raises ``exc``.

    Exposes the minimal surface GracefulRetryClient.create() touches:
    ``endpoint``/``description``/``count_tokens``/``create`` (+ an optional
    ``refresh_credentials`` no-op). Records how many times create() ran so the
    test can assert the loop is bounded.
    """

    def __init__(self, exc: Exception, endpoint: str = "https://fake"):
        self._exc = exc
        self.endpoint = endpoint
        self.description = f"fake-gpt-4o@{endpoint}"
        self.calls = 0

    def count_tokens(self, messages=()):  # noqa: ARG002
        return 0

    def refresh_credentials(self):
        pass

    async def create(self, *args, **kwargs):  # noqa: ARG002
        self.calls += 1
        raise self._exc

    async def close(self):
        pass


@pytest.mark.parametrize("n_endpoints", [1, 3])
def test_graceful_retry_terminates_on_persistent_auth_error(n_endpoints):
    """Regression: a pool that only ever raises AuthenticationError must make
    create() *terminate* (raising the last error), not spin forever.

    Pre-fix, the AuthenticationError branch neither blocklisted nor consumed
    the retry budget, so a single bad endpoint wedged create() indefinitely
    (the 19h hang). The per-error budget + global ``max_total_attempts`` cap
    now guarantee a bounded number of underlying create() calls.
    """
    from webeval.oai_clients import GracefulRetryClient

    auth_err = openai.AuthenticationError(
        "bad creds",
        response=httpx.Response(401, request=httpx.Request("POST", "https://fake")),
        body=None,
    )
    clients = [
        _AlwaysFailsClient(auth_err, endpoint=f"https://fake-{i}")
        for i in range(n_endpoints)
    ]
    g = GracefulRetryClient(
        clients=clients,
        logger=logging.getLogger(__name__),
        max_retries=3,
    )

    async def _run():
        # Hard timeout so a regression to the old behaviour fails loudly
        # instead of hanging the whole suite.
        return await asyncio.wait_for(g.create(messages=[]), timeout=30)

    with pytest.raises(openai.AuthenticationError):
        asyncio.run(_run())

    total_calls = sum(c.calls for c in clients)
    assert total_calls <= g.max_total_attempts
    # Budget-bounded: max_retries decrements drive termination here.
    assert total_calls <= g.max_retries + 1


def test_graceful_retry_terminates_when_all_endpoints_blocklisted():
    """A pool that only raises NotFoundError must terminate once every
    endpoint is blocklisted (next_client() raises), never looping forever."""
    from webeval.oai_clients import GracefulRetryClient

    not_found = openai.NotFoundError(
        "no such deployment",
        response=httpx.Response(404, request=httpx.Request("POST", "https://fake")),
        body=None,
    )
    clients = [
        _AlwaysFailsClient(not_found, endpoint=f"https://fake-{i}") for i in range(3)
    ]
    g = GracefulRetryClient(
        clients=clients,
        logger=logging.getLogger(__name__),
        max_retries=8,
    )

    async def _run():
        return await asyncio.wait_for(g.create(messages=[]), timeout=30)

    # All endpoints get blocklisted → next_client() raises RuntimeError.
    with pytest.raises(RuntimeError):
        asyncio.run(_run())
    # One create() per endpoint at most before the pool is exhausted.
    assert sum(c.calls for c in clients) <= len(clients)
    assert len(g.blocklist) == len(clients)


def test_openai_wrapper_create_round_trip(monkeypatch):
    """Stub the OpenAI client and verify .create() returns a CreateResult
    with text content and updated usage."""
    from webeval.oai_clients import UserMessage, OpenAIClientWrapper

    captured = {}

    async def fake_chat_create(**kwargs):
        captured["kwargs"] = kwargs
        message = SimpleNamespace(content="hi there", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=3,
            completion_tokens_details=None,
        )
        return SimpleNamespace(choices=[choice], usage=usage)

    client = OpenAIClientWrapper(model="gpt-4o", api_key="sk-test")
    monkeypatch.setattr(
        client.client.chat.completions, "create", fake_chat_create, raising=True
    )

    result = asyncio.run(
        client.create(messages=[UserMessage(content="ping")])
    )
    assert result.content == "hi there"
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 3
    assert client.total_usage().prompt_tokens == 12
    # Verify message conversion routed correctly.
    assert captured["kwargs"]["messages"] == [{"role": "user", "content": "ping"}]
    assert captured["kwargs"]["model"] == "gpt-4o"


def test_orcarouter_provider_round_trip(monkeypatch):
    """The ``orcarouter`` provider resolves to an OpenAI-compatible client
    pointed at the OrcaRouter gateway and marks itself as orcarouter."""
    from webeval.oai_clients import (
        ChatCompletionClient,
        OrcaRouterClientWrapper,
        UserMessage,
        create_client_from_config,
    )

    captured = {}

    async def fake_chat_create(**kwargs):
        captured["kwargs"] = kwargs
        message = SimpleNamespace(content="hello from orca", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(
            prompt_tokens=7,
            completion_tokens=2,
            completion_tokens_details=None,
        )
        return SimpleNamespace(choices=[choice], usage=usage)

    client = create_client_from_config(
        {
            "CHAT_COMPLETION_PROVIDER": "orcarouter",
            "CHAT_COMPLETION_KWARGS_JSON": {"model": "openai/gpt-4o"},
        }
    )
    assert isinstance(client, OrcaRouterClientWrapper)
    assert isinstance(client, ChatCompletionClient)
    assert client.metadata["provider"] == "orcarouter"
    assert str(client.client.base_url).rstrip("/") == "https://api.orcarouter.ai/v1"

    monkeypatch.setattr(
        client.client.chat.completions, "create", fake_chat_create, raising=True
    )
    result = asyncio.run(client.create(messages=[UserMessage(content="ping")]))
    assert result.content == "hello from orca"
    assert captured["kwargs"]["model"] == "openai/gpt-4o"


def test_client_wrapper_from_config_returns_chat_client(monkeypatch):
    """Backwards-compat alias for callers that still use ``ClientWrapper.from_config``."""
    from webeval.oai_clients import ChatCompletionClient, ClientWrapper

    cfg = {
        "CHAT_COMPLETION_PROVIDER": "openai",
        "CHAT_COMPLETION_KWARGS_JSON": {"model": "gpt-4o", "api_key": "sk-test"},
    }
    client = ClientWrapper.from_config(cfg)
    assert isinstance(client, ChatCompletionClient)
    assert client.metadata["model"] == "gpt-4o"
    assert client.endpoint == "gpt-4o"


def test_create_completion_client_from_env_round_trip():
    """The webeval.utils factory delegates to the new oai_clients module."""
    from webeval.utils import create_completion_client_from_env
    from webeval.oai_clients import ChatCompletionClient

    env = {
        "CHAT_COMPLETION_PROVIDER": "openai",
        "CHAT_COMPLETION_KWARGS_JSON": json.dumps(
            {"model": "gpt-4o", "api_key": "sk-test"}
        ),
    }
    client = create_completion_client_from_env(env=env)
    assert isinstance(client, ChatCompletionClient)
    assert client.metadata["provider"] == "openai"
