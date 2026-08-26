"""Tests for connecting both Fara browser runners to an existing CDP browser."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fara.environments.playwright import PlaywrightEnvironment
from fara.fara_7b.browser.browser_bb import BrowserBB


def _fake_playwright(existing_pages):
    context = SimpleNamespace(pages=existing_pages, new_page=AsyncMock())
    browser = SimpleNamespace(contexts=[context])
    chromium = SimpleNamespace(connect_over_cdp=AsyncMock(return_value=browser))
    return SimpleNamespace(chromium=chromium), browser, context


@pytest.mark.asyncio
async def test_playwright_environment_connects_to_remote_cdp():
    page = object()
    playwright, browser, context = _fake_playwright([page])
    env = PlaywrightEnvironment(cdp_url="wss://cloud.example/cdp")
    env._playwright = playwright

    await env._init_remote_browser()

    playwright.chromium.connect_over_cdp.assert_awaited_once_with(
        "wss://cloud.example/cdp"
    )
    assert env._browser is browser
    assert env._context is context
    assert env._page is page


@pytest.mark.asyncio
async def test_playwright_environment_opens_page_when_remote_context_is_empty():
    playwright, _, context = _fake_playwright([])
    page = object()
    context.new_page.return_value = page
    env = PlaywrightEnvironment(cdp_url="wss://cloud.example/cdp")
    env._playwright = playwright

    await env._init_remote_browser()

    context.new_page.assert_awaited_once_with()
    assert env._page is page


@pytest.mark.asyncio
async def test_fara7b_connects_to_remote_cdp():
    page = object()
    playwright, browser, context = _fake_playwright([page])
    manager = BrowserBB(
        viewport_height=900,
        viewport_width=1440,
        headless=True,
        page_script_path=None,
        cdp_url="wss://cloud.example/cdp",
    )
    manager._playwright = playwright

    await manager._init_remote_browser()

    playwright.chromium.connect_over_cdp.assert_awaited_once_with(
        "wss://cloud.example/cdp"
    )
    assert manager.browser is browser
    assert manager._context is context
    assert manager._page is page
