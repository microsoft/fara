"""Exercise optional search through Fara and the real MCP HTTP transport.

Model responses and HTTP responses are scripted; no model inference is claimed.
"""

import asyncio
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image
from pydantic import ValidationError

from fara import Fara15Agent, Task
from fara.agents.fara.fara15_agent import Fara15AgentConfig, Fara15AgentState
from fara.agents.fara.search import SEARCH_MCP_URL, USER_AGENT
from fara.clients.messages import FunctionCall
from fara.core.run_context import RunContext
from fara.environments.computer import PageContext

pytest.importorskip("mcp")

SOURCE_URL = "https://example.org/research?q=computer%20use&year=2026"
RESULT = f"Computer use research\n{SOURCE_URL}\nRecent research on visual agents."


class McpFixture:
    def __init__(
        self,
        monkeypatch,
        *,
        text=RESULT,
        error=False,
        delay=False,
        malformed=False,
        status=200,
        advertise=True,
    ):
        self.requests = []
        self.clients = []
        self.text = text
        self.error = error
        self.delay = delay
        self.malformed = malformed
        self.status = status
        self.advertise = advertise
        self.cancelled = False
        self.started = asyncio.Event()
        original = httpx.AsyncClient

        def client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(self.handle)
            instance = original(*args, **kwargs)
            self.clients.append(instance)
            return instance

        monkeypatch.setattr(httpx, "AsyncClient", client)
        # Deliberately present unrelated credentials: the caller must ignore them.
        monkeypatch.setenv("PARALLEL_API_KEY", "must-not-be-sent")
        monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-sent")

    async def handle(self, request):
        assert str(request.url) == SEARCH_MCP_URL
        assert request.headers["user-agent"] == USER_AGENT
        assert "authorization" not in request.headers
        assert "x-api-key" not in request.headers
        body = json.loads(request.content)
        self.requests.append(body)
        method = body["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]
                if self.advertise
                else []
            }
        elif method == "tools/call":
            self.started.set()
            if self.delay:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            if self.status != 200:
                return httpx.Response(self.status, text="service unavailable")
            if self.malformed:
                return httpx.Response(
                    200,
                    text="invalid-json",
                    headers={"content-type": "application/json"},
                )
            result = {
                "content": [{"type": "text", "text": self.text}],
                "isError": self.error,
            }
        else:
            raise AssertionError(method)
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": body["id"], "result": result}
        )


class BrowserFixture:
    def __init__(self):
        self.urls = []
        image = io.BytesIO()
        Image.new("RGB", (1440, 900)).save(image, format="PNG")
        self.image = image.getvalue()

    async def get_observation(self):
        return self.image

    async def get_page_context(self):
        return PageContext(
            self.urls[-1] if self.urls else "https://example.org/start", ""
        )

    async def goto_url(self, url):
        self.urls.append(url)

    async def wait_for_load(self):
        pass


def action(name, **kwargs):
    return [
        FunctionCall(
            id="fixture", name="computer_use", arguments={"action": name, **kwargs}
        )
    ]


def agent(provider="parallel", **kwargs):
    result = Fara15Agent(
        Fara15AgentConfig(search_provider=provider, captcha_timeout_limit=0, **kwargs)
    )
    result._state = Fara15AgentState()
    result._get_system_message(Image.new("RGB", (1440, 900)))
    return result


def test_search_consumed_in_real_agent_loop(monkeypatch, tmp_path):
    fixture = McpFixture(monkeypatch)
    browser = BrowserFixture()
    calls = []

    async def scripted_create(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            args = {"action": "web_search", "query": "computer use research"}
        elif len(calls) == 2:
            text = messages[-1].content[-1]
            assert SOURCE_URL in text
            assert "Recent research on visual agents" in text
            assert browser.urls == []
            args = {"action": "visit_url", "url": SOURCE_URL}
        else:
            assert browser.urls == [SOURCE_URL]
            args = {"action": "terminate", "answer": "Found the research source."}
        return SimpleNamespace(
            content=SimpleNamespace(
                content="<tool_call>\n"
                + json.dumps({"name": "computer_use", "arguments": args})
                + "\n</tool_call>"
            )
        )

    async def run():
        model = SimpleNamespace(create=scripted_create)
        instance = agent(client=model, max_rounds=3)
        context = RunContext.create(
            environment=browser,
            output_dir=tmp_path / "run",
            task=Task(
                task_id="search-fixture", instruction="Find computer use research"
            ),
        )
        await instance.initialize(context)
        answer, _, observations = await instance.run(context)
        assert answer == "Found the research source."
        assert SOURCE_URL in observations[0]
        assert instance.config.search_provider == "parallel"
        await instance.close(context)

    asyncio.run(run())
    methods = [r["method"] for r in fixture.requests]
    assert methods == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]
    args = fixture.requests[-1]["params"]["arguments"]
    assert args["objective"] == "computer use research"
    assert args["search_queries"] == ["computer use research"]
    assert len(args["session_id"]) == 32
    assert all(c.is_closed for c in fixture.clients)


def test_default_saved_choice_disabled_and_permissions(monkeypatch):
    fixture = McpFixture(monkeypatch)

    async def run():
        browser = BrowserFixture()
        default = Fara15Agent()
        default._state = Fara15AgentState()
        default._get_system_message(Image.new("RGB", (1440, 900)))
        await default._execute_action(browser, action("web_search", query="a & b?"))
        assert browser.urls == ["https://www.bing.com/search?q=a+%26+b%3F&FORM=QBLH"]
        assert default._pending_observation == ""
        saved = agent()
        saved = Fara15Agent(saved.config.model_dump())
        saved._state = Fara15AgentState()
        saved._get_system_message(Image.new("RGB", (1440, 900)))
        for _ in range(2):
            await saved._execute_action(browser, action("web_search", query="research"))
        sessions = [
            r["params"]["arguments"]["session_id"]
            for r in fixture.requests
            if r["method"] == "tools/call"
        ]
        assert sessions[0] == sessions[1]
        disabled = agent("disabled")
        _, observation = await disabled._execute_action(
            browser, action("web_search", query="research")
        )
        assert "disabled" in observation and "disabled" in disabled._pending_observation
        saved._allowed_actions = frozenset({"terminate"})
        before = len(fixture.requests)
        with pytest.raises(ValueError, match="not allowed"):
            await saved._execute_action(browser, action("web_search", query="research"))
        assert len(fixture.requests) == before
        assert len(browser.urls) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "options, expected",
    [
        ({"error": True, "text": "Rate limit exceeded"}, "Rate limit exceeded"),
        ({"text": ""}, "no search text"),
        ({"advertise": False}, "did not advertise"),
        ({"status": 503}, "503"),
        ({"malformed": True}, ""),
        ({"delay": True}, "timed out"),
    ],
)
def test_errors_are_observations_without_provider_fallback(
    monkeypatch, options, expected
):
    fixture = McpFixture(monkeypatch, **options)

    async def run():
        browser = BrowserFixture()
        instance = agent(search_timeout=0.1 if options.get("delay") else 5)
        stop, text = await instance._execute_action(
            browser, action("web_search", query="research")
        )
        assert not stop and text.startswith("Parallel search failed:")
        assert expected in text
        assert text == instance._pending_observation
        assert browser.urls == []

    asyncio.run(run())
    assert all(c.is_closed for c in fixture.clients)
    if options.get("delay"):
        assert fixture.cancelled


def test_cancellation_closes_transport(monkeypatch):
    fixture = McpFixture(monkeypatch, delay=True)

    async def run():
        browser = BrowserFixture()
        instance = agent()
        task = asyncio.create_task(
            instance._execute_action(browser, action("web_search", query="research"))
        )
        await fixture.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert browser.urls == []

    asyncio.run(run())
    assert fixture.cancelled
    assert all(c.is_closed for c in fixture.clients)


@pytest.mark.parametrize("query", ["", "   ", None])
def test_invalid_query_does_not_open_transport(monkeypatch, query):
    fixture = McpFixture(monkeypatch)

    async def run():
        _, text = await agent()._execute_action(
            BrowserFixture(), action("web_search", query=query)
        )
        assert "nonempty query" in text

    asyncio.run(run())
    assert not fixture.requests


def test_config_and_cli_choices(monkeypatch):
    from fara import run_fara

    with pytest.raises(ValidationError):
        Fara15AgentConfig(search_provider="typo")
    for timeout in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            Fara15AgentConfig(search_timeout=timeout)
    monkeypatch.setattr(
        run_fara, "DEFAULT_ENDPOINT_CONFIG", dict(run_fara.DEFAULT_ENDPOINT_CONFIG)
    )
    run = AsyncMock()
    monkeypatch.setattr(run_fara, "run_fara15_agent", run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fara-cli",
            "--search_provider",
            "parallel",
            "--search_timeout",
            "12",
            "--start_page",
            SOURCE_URL,
            "--api_key",
            "model-key",
        ],
    )
    run_fara.main()
    assert run.call_args.kwargs["search_provider"] == "parallel"
    assert run.call_args.kwargs["search_timeout"] == 12
    assert run.call_args.kwargs["start_page"] == SOURCE_URL
    assert run.call_args.kwargs["endpoint_config"]["api_key"] == "model-key"
    monkeypatch.setattr(
        "sys.argv", ["fara-cli", "--fara-7b", "--search_provider", "parallel"]
    )
    with pytest.raises(SystemExit):
        run_fara.main()


def test_missing_optional_dependency_is_actionable(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_mcp(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("search extra not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mcp)

    async def run():
        browser = BrowserFixture()
        _, text = await agent()._execute_action(
            browser, action("web_search", query="research")
        )
        assert "pip install 'fara[search]'" in text
        assert not browser.urls

    asyncio.run(run())


def test_invalid_python_config_does_not_start_browser(monkeypatch):
    from fara import run_fara

    def unexpected_browser(**kwargs):
        pytest.fail("Invalid search configuration started a browser")

    monkeypatch.setattr(run_fara, "PlaywrightEnvironment", unexpected_browser)
    with pytest.raises(ValidationError):
        asyncio.run(run_fara.run_fara15_agent(search_provider="typo"))
