"""Playwright-based browser environment.

Implements BrowserEnvironment with canonical action names. Old method names
(click, hover, goto, back, keypress, scroll_down, scroll_up) are kept as
backward-compatible aliases. Playwright-specific methods (id-based, CUA)
are also preserved.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import platform
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from playwright.async_api import (
    BrowserContext,
    Download,
    Page,
    Playwright,
    async_playwright,
)

from ...agents.key_mapping import CUA_KEY_TO_PLAYWRIGHT_KEY
from ..computer import BrowserEnvironment, BrowserEnvironmentConfig, OSType, PageContext
from .playwright_controller import PlaywrightController


class PlaywrightEnvironmentConfig(BrowserEnvironmentConfig):
    """Configuration for PlaywrightEnvironment."""

    viewport_width: int = 1440
    viewport_height: int = 900
    headless: bool = True
    browser_channel: str = "chromium"
    start_page: str = "about:blank"
    downloads_folder: str | None = None
    browser_data_dir: str | None = None
    animate_actions: bool = False
    single_tab_mode: bool = True
    default_timeout: int = 60000
    page_script_path: str | None = None
    cdp_url: str | None = None
    use_browserbase: bool = False
    browserbase_project_id: str | None = None
    browserbase_api_key: str | None = None
    enable_find_overlay: bool = True


class PlaywrightEnvironment(BrowserEnvironment):
    """Browser environment using Playwright.

    Supports regular Chromium/Firefox/WebKit, persistent browser contexts,
    remote Chromium browsers over CDP, and BrowserBase cloud sessions.
    """

    os_type = OSType.LINUX

    _BROWSER_CHROME_DISPATCH: Dict[frozenset, str] = {
        frozenset({"Control", "r"}): "refresh",
        frozenset({"Control", "Shift", "r"}): "refresh",
        frozenset({"F5"}): "refresh",
        frozenset({"Alt", "ArrowLeft"}): "go_back",
        frozenset({"Alt", "ArrowRight"}): "forward",
    }
    _FIND_CHORD = frozenset({"Control", "f"})
    _FIND_OVERLAY_JS = (Path(__file__).parent / "find_overlay.js").read_text()

    @staticmethod
    def _normalize_chord(keys: list[str]) -> frozenset:
        """Map CUA/raw keys to Playwright names and lowercase single letters."""
        mapped = [CUA_KEY_TO_PLAYWRIGHT_KEY.get(k.lower(), k) for k in keys]
        return frozenset(k if len(k) > 1 else k.lower() for k in mapped)

    def __init__(
        self,
        config: PlaywrightEnvironmentConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, **kwargs)
        self.config: PlaywrightEnvironmentConfig

        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._browser = None
        self._xvfb_process = None
        self._download_handler: Callable[[Download], None] | None = None
        self._last_download: Download | None = None

        self.logger = logging.getLogger(__name__)
        self._is_linux = platform.system() == "Linux"

        self._controller: PlaywrightController | None = None

        self._bb = None
        self._session = None
        self._captcha_event = asyncio.Event()
        self._captcha_event.set()

        self._task_id: str | None = None

    def _tag(self) -> str:
        """Per-task log prefix; empty when no task_id was supplied."""
        return f"[task={self._task_id}] " if self._task_id else ""

    @classmethod
    def _get_config_class(cls) -> type[PlaywrightEnvironmentConfig]:
        return PlaywrightEnvironmentConfig

    def _default_download_handler(self, download: Download) -> None:
        self._last_download = download

    async def initialize(self, **kwargs) -> None:
        """Initialize the browser environment."""
        self._task_id = kwargs.pop("task_id", None)
        self._playwright = await async_playwright().start()
        self._download_handler = self._default_download_handler

        self._controller = PlaywrightController(
            animate_actions=self.config.animate_actions,
            downloads_folder=self.config.downloads_folder,
            viewport_width=self.config.viewport_width,
            viewport_height=self.config.viewport_height,
            _download_handler=self._download_handler,
            to_resize_viewport=True,
            single_tab_mode=self.config.single_tab_mode,
            logger=self.logger,
        )

        if self.config.cdp_url:
            await self._init_remote_browser()
            await self._setup_browser()
        elif self.config.use_browserbase:
            await self._init_browserbase()
        elif self.config.browser_data_dir:
            await self._init_persistent_browser()
            await self._setup_browser()
        else:
            await self._init_regular_browser()
            await self._setup_browser()

        self._initialized = True

    async def _init_regular_browser(self) -> None:
        """Initialize regular browser."""
        if not self.config.headless and self._is_linux:
            self._start_xvfb()

        launch_args = {"headless": self.config.headless}

        if self.config.browser_channel == "chromium":
            self._browser = await self._playwright.chromium.launch(**launch_args)
        elif self.config.browser_channel == "firefox":
            self._browser = await self._playwright.firefox.launch(**launch_args)
        elif self.config.browser_channel == "webkit":
            self._browser = await self._playwright.webkit.launch(**launch_args)
        else:
            raise ValueError(
                f"Unsupported browser channel: {self.config.browser_channel}"
            )

        self._context = await self._browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
        )
        self._page = await self._context.new_page()

    async def _init_remote_browser(self) -> None:
        """Connect to an existing Chromium browser over CDP."""
        cdp_url = self.config.cdp_url
        if not cdp_url:
            raise ValueError("A CDP URL is required for a remote browser")
        self._browser = await self._playwright.chromium.connect_over_cdp(cdp_url)
        if not self._browser.contexts:
            raise RuntimeError("The remote CDP browser has no browser context")
        self._context = self._browser.contexts[0]
        self._page = (
            self._context.pages[0]
            if self._context.pages
            else await self._context.new_page()
        )

    _BROWSERBASE_MAX_ATTEMPTS = 5
    _BROWSERBASE_RATE_LIMIT_BACKOFF_S = 10
    _BROWSERBASE_CAPTCHA_WAIT_S = 90

    async def _init_browserbase(self) -> None:
        """Initialize BrowserBase connection.

        Retries up to ``_BROWSERBASE_MAX_ATTEMPTS`` times. Each attempt
        creates a fresh session, connects via CDP, runs the standard
        browser setup (which navigates to ``start_page``), and waits
        briefly for any captcha. On failure the session is released and
        the next attempt starts clean.
        """
        import browserbase
        from browserbase import Browserbase

        self.logger.info(f"{self._tag()}Initializing BrowserBase session...")

        api_key = self.config.browserbase_api_key or os.environ.get(
            "BROWSERBASE_API_KEY"
        )
        project_id = self.config.browserbase_project_id or os.environ.get(
            "BROWSERBASE_PROJECT_ID"
        )
        if not api_key or not project_id:
            raise ValueError("BrowserBase API key and project ID are required")

        self._bb = Browserbase(api_key=api_key)

        for attempt in range(1, self._BROWSERBASE_MAX_ATTEMPTS + 1):
            try:
                await self._connect_browserbase_once(project_id)
                if attempt > 1:
                    self.logger.info(
                        f"{self._tag()}BrowserBase init succeeded on attempt {attempt}"
                    )
                return
            except Exception as e:
                await self._teardown_browserbase_session(project_id)
                if attempt == self._BROWSERBASE_MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"BrowserBase init failed after "
                        f"{self._BROWSERBASE_MAX_ATTEMPTS} attempts: "
                        f"{type(e).__name__}: {e}"
                    ) from e
                delay = self._browserbase_retry_delay(e, attempt, browserbase)
                self.logger.warning(
                    f"{self._tag()}BrowserBase init attempt {attempt}/"
                    f"{self._BROWSERBASE_MAX_ATTEMPTS} failed "
                    f"({type(e).__name__}: {e}); retrying in {delay}s"
                )
                await asyncio.sleep(delay)

    async def _connect_browserbase_once(self, project_id: str) -> None:
        """Run one BrowserBase bring-up: session, CDP, setup, captcha wait."""
        self._captcha_event.set()

        self._session = self._bb.sessions.create(
            project_id=project_id,
            proxies=True,
            browser_settings={"advanced_stealth": True, "enablePdfViewer": True},
            keep_alive=True,
            timeout=7200,
            api_timeout=3600,
        )
        assert self._session.id is not None
        assert (
            self._session.status == "RUNNING"
        ), f"Session status is {self._session.status}"

        self._browser = await self._playwright.chromium.connect_over_cdp(
            self._session.connect_url
        )
        self.logger.info(
            f"{self._tag()}Connected to BrowserBase: "
            f"https://browserbase.com/sessions/{self._session.id}"
        )

        self._context = self._browser.contexts[0]
        assert len(self._context.pages) == 1
        self._page = self._context.pages[0]

        self._context.on("console", self._handle_browserbase_console)

        await self._setup_browser()

        try:
            await asyncio.wait_for(
                self._captcha_event.wait(),
                timeout=self._BROWSERBASE_CAPTCHA_WAIT_S,
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                f"{self._tag()}Captcha not resolved within "
                f"{self._BROWSERBASE_CAPTCHA_WAIT_S}s on start_page; proceeding"
            )

    def _handle_browserbase_console(self, msg) -> None:
        """Track BrowserBase captcha-solving lifecycle via console messages."""
        if msg.text == "browserbase-solving-started":
            self.logger.info(f"{self._tag()}Captcha Solving In Progress!")
            self._captcha_event.clear()
        elif msg.text == "browserbase-solving-finished":
            self.logger.info(f"{self._tag()}Captcha Solving Completed!")
            asyncio.create_task(self._resume_after_captcha())

    async def _resume_after_captcha(self) -> None:
        """Release the captcha event after a best-effort post-captcha settle."""
        try:
            await asyncio.sleep(3)
            try:
                await self._page.wait_for_load_state("domcontentloaded", timeout=10000)
            except Exception as e:
                self.logger.warning(
                    f"Post-captcha wait_for_load_state failed "
                    f"(continuing anyway): {e}"
                )
        finally:
            self._captcha_event.set()

    def _browserbase_retry_delay(self, exc: Exception, attempt: int, bb_module) -> int:
        """Pick a backoff delay for a failed BrowserBase init attempt."""
        if isinstance(exc, bb_module.RateLimitError):
            return self._BROWSERBASE_RATE_LIMIT_BACKOFF_S
        return 2 ** (attempt - 1)

    async def _teardown_browserbase_session(self, project_id: str) -> None:
        """Release a partially-built BrowserBase session and reset state."""
        browser_connected = self._browser is not None and self._browser.is_connected()
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        self._context = None
        self._page = None
        if self._session is not None and self._bb is not None:
            session_id = self._session.id
            try:
                self._bb.sessions.update(
                    self._session.id,
                    status="REQUEST_RELEASE",
                    project_id=project_id,
                )
            except Exception:
                pass
            self.logger.info(
                f"[BB-END] task={self._task_id} session={session_id} "
                f"browser_connected_at_teardown={browser_connected}"
            )
            self._session = None

    async def _init_persistent_browser(self) -> None:
        """Initialize persistent browser with data directory."""
        if not self.config.headless and self._is_linux:
            self._start_xvfb()

        launch_args = {"headless": self.config.headless}
        self._context = await self._playwright.chromium.launch_persistent_context(
            self.config.browser_data_dir, **launch_args
        )
        self._page = await self._context.new_page()

    _BING_SAME_TAB_COOKIE = {
        "name": "SRCHHPGUSR",
        "value": "EXLKNT=0",
        "domain": ".bing.com",
        "path": "/",
        "secure": True,
    }

    async def _setup_browser(self) -> None:
        """Set up common browser features."""
        self._context.set_default_timeout(self.config.default_timeout)

        await self._controller.on_new_page(self._page)

        if self._download_handler:
            self._page.on("download", self._download_handler)

        await self._page.set_viewport_size(
            {"width": self.config.viewport_width, "height": self.config.viewport_height}
        )

        if self.config.page_script_path and os.path.exists(
            self.config.page_script_path
        ):
            await self._page.add_init_script(path=self.config.page_script_path)

        try:
            await self._context.add_cookies([self._BING_SAME_TAB_COOKIE])
        except Exception as e:
            self.logger.warning(f"Failed to inject Bing same-tab cookie: {e}")

        await self._page.goto(self.config.start_page, wait_until="commit")

    async def _swap_to_new_page(self, new_page: Page) -> None:
        """Promote a popup the agent's action just opened to the active page.

        Browsers foreground a newly opened tab on `target="_blank"` and
        `window.open()`, so the agent observing the new page matches what
        a real user would see. In ``single_tab_mode`` we also close the
        previous page to preserve the one-page worldview; otherwise we
        leave it on the context for a future multi-tab action space to
        switch back to.
        """
        old_page = self._page
        if new_page is old_page:
            return
        self._page = new_page
        if self.config.single_tab_mode and not old_page.is_closed():
            try:
                await old_page.close()
            except Exception as e:
                self.logger.warning(f"Error closing previous page after swap: {e}")

    def _start_xvfb(self) -> None:
        """Start Xvfb virtual display server."""
        display_num = 99
        self._xvfb_process = subprocess.Popen(
            ["Xvfb", f":{display_num}", "-screen", "0", "1280x1024x24", "-ac"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.environ["DISPLAY"] = f":{display_num}"
        time.sleep(1)
        atexit.register(self._stop_xvfb)

    def _stop_xvfb(self) -> None:
        """Stop the Xvfb process."""
        if self._xvfb_process:
            self._xvfb_process.send_signal(signal.SIGTERM)
            self._xvfb_process.wait()
            self._xvfb_process = None

    async def wait_for_captcha_resolution(self) -> None:
        """Wait for captcha to be resolved if one is being solved."""
        await self._captcha_event.wait()

    async def close(self) -> None:
        """Close the browser and cleanup."""
        self.logger.info("Closing browser...")

        if self._page:
            await self._page.close()
            self._page = None

        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            if self.config.use_browserbase and self._session and self._bb:
                project_id = self.config.browserbase_project_id or os.environ.get(
                    "BROWSERBASE_PROJECT_ID"
                )
                session_id = self._session.id
                browser_connected = self._browser.is_connected()
                self._bb.sessions.update(
                    self._session.id,
                    status="REQUEST_RELEASE",
                    project_id=project_id,
                )
                self.logger.info(
                    f"[BB-END] task={self._task_id} session={session_id} "
                    f"browser_connected_at_teardown={browser_connected}"
                )
                self._session = None
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        if not self.config.headless:
            self._stop_xvfb()

        self._initialized = False

    async def get_observation(self) -> bytes:
        """Get screenshot of current page."""
        return await self.get_screenshot()

    async def left_click(self, x: int, y: int) -> None:
        new_page = await self._controller.click_coords(self._page, x, y)
        if new_page is not None:
            await self._swap_to_new_page(new_page)

    async def right_click(self, x: int, y: int) -> None:
        new_page = await self._controller.cua_click(self._page, x, y, "right")
        if new_page is not None:
            await self._swap_to_new_page(new_page)

    async def double_click(self, x: int, y: int) -> None:
        new_page = await self._controller.cua_double_click(self._page, x, y)
        if new_page is not None:
            await self._swap_to_new_page(new_page)

    async def mouse_move(self, x: int, y: int) -> None:
        await self._controller.cua_move(self._page, x, y)

    async def left_click_drag(self, end_x: int, end_y: int) -> None:
        await self._page.mouse.down()
        await self._page.mouse.move(end_x, end_y)
        await self._page.mouse.up()

    async def key(self, keys: list[str]) -> None:
        chord = self._normalize_chord(keys)
        if self.config.enable_find_overlay and chord == self._FIND_CHORD:
            await self._page.evaluate(self._FIND_OVERLAY_JS)
            return
        handler = self._BROWSER_CHROME_DISPATCH.get(chord)
        if handler is not None:
            await getattr(self, handler)()
            return
        await self._controller.keypress(self._page, keys)

    async def type(self, text: str) -> None:
        await self._controller.cua_type(self._page, text)

    async def scroll(self, pixels: int) -> None:
        """Scroll vertically. Positive=up, negative=down."""
        await self._page.mouse.wheel(0, -pixels)

    async def wait(self, duration: float = 1.0) -> None:
        await self._controller.sleep(self._page, duration)

    async def get_screenshot(self, path: str | None = None) -> bytes:
        """Capture a screenshot of the current page."""
        return await self._controller.get_screenshot(self._page, path=path)

    async def goto_url(self, url: str) -> None:
        await self._controller.visit_page(self._page, url)

    async def go_back(self) -> None:
        await self._controller.back(self._page)

    async def refresh(self) -> None:
        await self._page.reload(wait_until="commit")

    async def middle_click(self, x: int, y: int) -> None:
        await self._page.mouse.click(x, y, button="middle")

    async def triple_click(self, x: int, y: int) -> None:
        await self._page.mouse.click(x, y, click_count=3)

    async def left_mouse_down(self, x: int, y: int) -> None:
        await self._page.mouse.move(x, y)
        await self._page.mouse.down()

    async def left_mouse_up(self, x: int, y: int) -> None:
        await self._page.mouse.move(x, y)
        await self._page.mouse.up()

    async def cursor_position(self) -> tuple[int, int]:
        pos = self._controller.last_cursor_position
        return (int(pos[0]), int(pos[1]))

    async def hscroll(self, pixels: int) -> None:
        """Horizontal scroll. Positive=right, negative=left."""
        await self._page.mouse.wheel(pixels, 0)

    async def click(self, x: float, y: float) -> Dict[str, Any]:
        """Click at coordinates."""
        await self.left_click(int(x), int(y))
        return {"success": True}

    async def hover(self, x: float, y: float) -> Dict[str, Any]:
        """Hover at coordinates."""
        await self._controller.hover_coords(self._page, x, y)
        return {"success": True}

    async def goto(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL."""
        await self.goto_url(url)
        return {"success": True}

    async def back(self) -> Dict[str, Any]:
        """Go back in browser history."""
        await self.go_back()
        return {"success": True}

    async def forward(self) -> Dict[str, Any]:
        """Go forward in browser history."""
        await self._controller.cua_forward(self._page)
        return {"success": True}

    async def keypress(self, keys: List[str]) -> Dict[str, Any]:
        """Press a sequence of keys."""
        await self.key(keys)
        return {"success": True}

    async def scroll_down(self, amount: int = 400) -> Dict[str, Any]:
        """Scroll down the page."""
        await self._controller.page_down(self._page, amount=amount)
        return {"success": True}

    async def scroll_up(self, amount: int = 400) -> Dict[str, Any]:
        """Scroll up the page."""
        await self._controller.page_up(self._page, amount=amount)
        return {"success": True}

    async def click_id(self, identifier: str) -> Dict[str, Any]:
        """Click on an element by its identifier."""
        new_page = await self._controller.click_id(self._page, identifier)
        if new_page is not None:
            await self._swap_to_new_page(new_page)
        return {"success": True}

    async def type_text(
        self,
        x: float,
        y: float,
        text: str,
        press_enter: bool = True,
        clear_first: bool = False,
    ) -> Dict[str, Any]:
        """Type text at coordinates."""
        new_page = await self._controller.fill_coords(
            self._page,
            x,
            y,
            text,
            press_enter=press_enter,
            delete_existing_text=clear_first,
        )
        if new_page is not None:
            await self._swap_to_new_page(new_page)
        return {"success": True}

    async def type_id(
        self,
        identifier: str,
        text: str,
        press_enter: bool = True,
        clear_first: bool = False,
    ) -> Dict[str, Any]:
        """Type text into an element by its identifier."""
        await self._controller.fill_id(
            self._page,
            identifier,
            text,
            press_enter=press_enter,
            delete_existing_text=clear_first,
        )
        return {"success": True}

    async def hover_id(self, identifier: str) -> Dict[str, Any]:
        """Hover over an element by its identifier."""
        await self._controller.hover_id(self._page, identifier)
        return {"success": True}

    async def scroll_id(self, identifier: str, direction: str) -> Dict[str, Any]:
        """Scroll within an element by its identifier."""
        await self._controller.scroll_id(self._page, identifier, direction)
        return {"success": True}

    async def wait_for_load(
        self, state: str = "domcontentloaded", timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Wait for the page to reach a load state."""
        await self._controller.wait_for_load_state(
            self._page, state=state, timeout=timeout
        )
        if self.config.extra_sleep_time > 0:
            await asyncio.sleep(self.config.extra_sleep_time)
        return {"success": True}

    async def get_url(self) -> str:
        """Get the current page URL."""
        return await self._controller.get_page_url(self._page)

    async def get_title(self) -> str:
        """Get the current page title."""
        return await self._controller.get_page_title(self._page)

    async def get_text(self, n_lines: int = 100) -> str:
        """Get the visible text on the page."""
        return await self._controller.get_webpage_text(self._page, n_lines=n_lines)

    async def get_interactive_rects(self) -> Dict[str, Any]:
        """Get interactive element rectangles."""
        return await self._controller.get_interactive_rects(self._page)

    async def get_visual_viewport(self) -> Dict[str, Any]:
        """Get the visual viewport information."""
        return await self._controller.get_visual_viewport(self._page)

    async def get_page_metadata(self) -> Dict[str, Any]:
        """Get page metadata."""
        return await self._controller.get_page_metadata(self._page)

    async def get_focused_rect_id(self) -> str:
        """Get the ID of the currently focused element."""
        return await self._controller.get_focused_rect_id(self._page)

    async def get_page_markdown(self) -> str:
        """Get page content as markdown."""
        return await self._controller.get_page_markdown(self._page)

    async def get_page_context(self) -> PageContext:
        """Return current URL and viewport scroll position."""
        url = await self._controller.get_page_url(self._page)
        parts = [f"Current URL: {url}"]
        vp = await self._controller.get_visual_viewport(self._page)
        page_top = vp.get("pageTop", 0)
        viewport_h = vp.get("height", 0)
        scroll_h = vp.get("scrollHeight", 0)
        if scroll_h > 0:
            bottom = page_top + viewport_h
            pct = min(int(bottom / scroll_h * 100), 100)
            parts.append(
                f"Viewport: scrolled to {pct}% of page "
                f"({int(page_top)}-{int(bottom)}px of {int(scroll_h)}px)"
            )
        return PageContext(url=url, page_info="\n".join(parts))

    async def fill_coords(
        self,
        x: float,
        y: float,
        text: str,
        press_enter: bool = True,
        delete_existing_text: bool = False,
    ) -> Dict[str, Any]:
        """Fill text at coordinates."""
        new_page = await self._controller.fill_coords(
            self._page,
            x,
            y,
            text,
            press_enter=press_enter,
            delete_existing_text=delete_existing_text,
        )
        if new_page is not None:
            await self._swap_to_new_page(new_page)
        return {"success": True}

    async def fill_id(
        self,
        identifier: str,
        text: str,
        press_enter: bool = True,
        delete_existing_text: bool = False,
    ) -> Dict[str, Any]:
        """Fill text into an element by its identifier."""
        await self._controller.fill_id(
            self._page,
            identifier,
            text,
            press_enter=press_enter,
            delete_existing_text=delete_existing_text,
        )
        return {"success": True}

    async def select_option(self, identifier: str) -> Dict[str, Any]:
        """Select an option element by its identifier."""
        new_page = await self._controller.select_option(self._page, identifier)
        if new_page is not None:
            await self._swap_to_new_page(new_page)
        return {"success": True}

    async def cua_click(self, x: int, y: int, button: str = "left") -> Dict[str, Any]:
        """CUA click action."""
        new_page = await self._controller.cua_click(self._page, x, y, button)
        if new_page is not None:
            await self._swap_to_new_page(new_page)
        return {"success": True}

    async def cua_double_click(self, x: int, y: int) -> Dict[str, Any]:
        """CUA double click action."""
        new_page = await self._controller.cua_double_click(self._page, x, y)
        if new_page is not None:
            await self._swap_to_new_page(new_page)
        return {"success": True}

    async def cua_scroll(
        self, x: int, y: int, scroll_x: int, scroll_y: int
    ) -> Dict[str, Any]:
        """CUA scroll action."""
        await self._controller.cua_scroll(self._page, x, y, scroll_x, scroll_y)
        return {"success": True}

    async def cua_type(self, text: str) -> Dict[str, Any]:
        """CUA type action."""
        await self._controller.cua_type(self._page, text)
        return {"success": True}

    async def cua_wait(self, ms: int = 1000) -> Dict[str, Any]:
        """CUA wait action."""
        await self._controller.cua_wait(self._page, ms)
        return {"success": True}

    async def cua_move(self, x: int, y: int) -> Dict[str, Any]:
        """CUA move mouse action."""
        await self._controller.cua_move(self._page, x, y)
        return {"success": True}

    async def cua_drag(self, path: List[Dict[str, int]]) -> Dict[str, Any]:
        """CUA drag action along a path."""
        await self._controller.cua_drag(self._page, path)
        return {"success": True}

    async def evaluate(self, script: str) -> Any:
        """Evaluate JavaScript on the page."""
        return await self._page.evaluate(script)

    async def save_state(self) -> Dict[str, Any]:
        """Save environment state."""
        return {
            "url": self._page.url if self._page else None,
            "viewport_width": self.config.viewport_width,
            "viewport_height": self.config.viewport_height,
        }

    async def load_state(self, state: Dict[str, Any]) -> None:
        """Load environment state."""
        url = state.get("url")
        if url and self._page:
            await self._controller.visit_page(self._page, url)

    @property
    def page(self) -> Page | None:
        """Get the current page."""
        return self._page

    @property
    def context(self) -> BrowserContext | None:
        """Get the browser context."""
        return self._context

    @property
    def controller(self) -> PlaywrightController | None:
        """Get the playwright controller."""
        return self._controller
