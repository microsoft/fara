"""Resource ownership and cleanup regression tests."""

import asyncio

from fara.agents.fara.fara15_agent import Fara15Agent, Fara15AgentConfig
from fara.environments.playwright import PlaywrightEnvironment


class FakeClient:
    def __init__(self):
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


def _make_agent():
    return Fara15Agent(
        Fara15AgentConfig(client_config={"model": "m", "base_url": "u", "api_key": "k"})
    )


def test_agent_closes_only_clients_it_owns():
    owned = FakeClient()
    agent = _make_agent()
    agent._client = owned
    agent._owns_client = True
    asyncio.run(agent.close(None))
    assert owned.close_calls == 1

    external = FakeClient()
    agent = _make_agent()
    agent._client = external
    agent._owns_client = False
    asyncio.run(agent.close(None))
    assert external.close_calls == 0


def test_browser_cleanup_continues_after_one_resource_fails():
    class FailingPage:
        def is_closed(self):
            return False

        async def close(self):
            raise RuntimeError("page is already gone")

    class FakeContext:
        closed = False

        async def close(self):
            self.closed = True

    class FakeBrowser:
        closed = False

        def is_connected(self):
            return not self.closed

        async def close(self):
            self.closed = True

    class FakePlaywright:
        stopped = False

        async def stop(self):
            self.stopped = True

    environment = PlaywrightEnvironment(headless=True)
    context = FakeContext()
    browser = FakeBrowser()
    playwright = FakePlaywright()
    environment._page = FailingPage()
    environment._context = context
    environment._browser = browser
    environment._playwright = playwright

    asyncio.run(environment.close())
    asyncio.run(environment.close())

    assert context.closed
    assert browser.closed
    assert playwright.stopped
