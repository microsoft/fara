import asyncio
import atexit
import logging
import os
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional, Callable

import browserbase
from browserbase import Browserbase
from playwright.async_api import (
    BrowserContext,
    Download,
    Page,
    Playwright,
    async_playwright,
)

from playwright.async_api import (
    BrowserContext,
    Download,
    Page,
    Playwright,
    async_playwright,
)
from .playwright_controller import PlaywrightController


class BrowserBB:
    """Manages browser instance, context, and page lifecycle."""

    def __init__(
        self,
        viewport_height: int,
        viewport_width: int,
        headless: bool,
        page_script_path: str,
        browser_channel: str = "firefox",
        browser_data_dir: str | None = None,
        downloads_folder: str | None = None,
        to_resize_viewport: bool = True,
        single_tab_mode: bool = True,
        animate_actions: bool = False,
        use_browser_base: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.headless = headless
        self.page_script_path = page_script_path
        self.browser_channel = browser_channel
        self.browser_data_dir = browser_data_dir
        self.downloads_folder = downloads_folder
        self.to_resize_viewport = to_resize_viewport
        self.animate_actions = animate_actions
        self.single_tab_mode = single_tab_mode
        self.use_browser_base = use_browser_base
        self.logger = logger or logging.getLogger("browser_manager")

        self._viewport_height = viewport_height
        self._viewport_width = viewport_width

        # check _viewport_width and _viewport_height are positive integers
        if not isinstance(self._viewport_width, int) or self._viewport_width <= 0:
            raise ValueError(
                f"Error: Browser_manager.Browser: Invalid viewport width: {self._viewport_width}. Must be a positive integer."
            )
        if not isinstance(self._viewport_height, int) or self._viewport_height <= 0:
            raise ValueError(
                f"Error: Browser_manager.Browser:Invalid viewport height: {self._viewport_height}. Must be a positive integer."
            )
        assert isinstance(
            self.headless, bool
        ), f"Error: Browser_manager.Browser: headless must be a boolean, got {type(self.headless)}"
        if page_script_path is None:
            page_script_path = os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "page_script.js"
            )
            self.page_script_path = page_script_path
        assert isinstance(
            page_script_path, str
        ), f"Error: Browser_manager.Browser: page_script_path must be a string, got {type(self.page_script_path)}"
        assert os.path.exists(
            self.page_script_path
        ), f"Error: Browser_manager.Browser: page_script_path does not exist: {self.page_script_path}"

        assert (
            isinstance(self.browser_channel, str)
            and (self.browser_channel in ["chromium", "firefox", "webkit"])
        ), f"Error: Browser_manager.Browser: browser_channel must be one of ['chromium', 'firefox', 'webkit'], got {self.browser_channel}"

        # Browser-related instances
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self.browser = None
        self.session = None
        self.xvfb_process = None

        # Events and handlers
        self._captcha_event = asyncio.Event()
        self._captcha_event.set()  # Initially set (no captcha)
        self._download_handler: Callable[[Download], None] | None = None

        self._playwright_controller = PlaywrightController(
            animate_actions=self.animate_actions,
            downloads_folder=self.downloads_folder,
            viewport_width=self._viewport_width,
            viewport_height=self._viewport_height,
            _download_handler=self._download_handler,
            to_resize_viewport=self.to_resize_viewport,
            single_tab_mode=self.single_tab_mode,
            logger=self.logger,
        )

    def set_download_handler(self, handler: Callable[[Download], None]) -> None:
        """Set the download handler for the browser."""
        self._download_handler = handler
        self._playwright_controller._download_handler = handler

    def set_captcha_solved_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback to be called when captcha status changes."""
        self._captcha_solved_callback = callback

    async def init(
        self,
        start_page: str,
        shared_data_point=None,  # For captcha tracking
    ) -> None:
        """Initialize the browser, context, and page."""
        self._playwright = await async_playwright().start()
        self.shared_data_point = shared_data_point

        if self.use_browser_base:
            await self._init_browser_base(self.shared_data_point)
        elif self.browser_data_dir is None:
            await self._init_regular_browser(channel=self.browser_channel)
        else:
            await self._init_persistent_browser()

        # Common setup for all browser types
        await self._setup_common_browser_features(start_page)

    async def _init_browser_base(self, shared_data_point) -> None:
        """Initialize BrowserBase connection, defaults to chromium."""

        self.logger.info("Initializing BrowserBase session...")
        self.bb = Browserbase(api_key=os.environ["BROWSERBASE_API_KEY"])

        while True:  # Wait indefinitely until we get a session
            try:
                self.session = self.bb.sessions.create(
                    project_id=os.environ["BROWSERBASE_PROJECT_ID"],
                    proxies=True,
                    browser_settings={"advanced_stealth": True},
                    keep_alive=True,
                    timeout=7200,  # 2 hour timeout
                    region="us-east-1",
                )
                break
            except browserbase.RateLimitError as e:
                self.logger.warning(
                    "Rate limit exceeded while trying to create BrowserBase session. Retrying in 10 seconds..."
                )
                await asyncio.sleep(10)

        assert self.session.id is not None
        assert (
            self.session.status == "RUNNING"
        ), f"Session status is {self.session.status}"

        chromium = self._playwright.chromium
        self.browser = await chromium.connect_over_cdp(self.session.connect_url)
        self.logger.info(
            f"Connected to BrowserBase session: https://browserbase.com/sessions/{self.session.id}"
        )

        self._context = self.browser.contexts[0]
        assert len(self._context.pages) == 1
        self._page = self._context.pages[0]

        # Set up captcha handling
        def handle_console(msg):
            """Handle captcha detection and solving."""
            if msg.text == "browserbase-solving-started":
                self.logger.info("Captcha Solving In Progress!!")
                if shared_data_point:
                    shared_data_point.set_encountered_captcha(True)
                self._captcha_event.clear()  # Block execution
            elif msg.text == "browserbase-solving-finished":
                self.logger.info("Captcha Solving Completed!!")

                async def delayed_resume():
                    await asyncio.sleep(3)  # Wait for navigation to settle
                    await self._page.wait_for_load_state("networkidle")
                    self._captcha_event.set()

                asyncio.create_task(delayed_resume())

        self._context.on("console", handle_console)
        self._page.on("console", handle_console)

    async def _init_regular_browser(self, channel: str = "chromium") -> None:
        """Initialize regular browser according to the specified channel."""
        if not self.headless:
            self.start_xvfb()

        launch_args: Dict[str, Any] = {"headless": self.headless}

        if channel == "chromium":
            self.browser = await self._playwright.chromium.launch(**launch_args)
        elif channel == "firefox":
            self.browser = await self._playwright.firefox.launch(**launch_args)
        elif channel == "webkit":
            self.browser = await self._playwright.webkit.launch(**launch_args)
        else:
            raise ValueError(
                f"Unsupported browser channel: {channel}. Supported channels are 'chromium', 'firefox', and 'webkit'."
            )

        self._context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
        )

        self._page = await self._context.new_page()

    async def _init_persistent_browser(self) -> None:
        """Initialize persistent browser with data directory."""
        if not self.headless:
            self.start_xvfb()

        launch_args: Dict[str, Any] = {"headless": self.headless}
        self._context = await self._playwright.chromium.launch_persistent_context(
            self.browser_data_dir, **launch_args
        )
        self._page = await self._context.new_page()

    async def _setup_common_browser_features(self, start_page: str) -> None:
        """Set up features common to all browser types."""
        self._context.set_default_timeout(60000)  # One minute
        await self._playwright_controller.on_new_page(self._page)
        assert self._page is not None

        # Set up new page handling for single tab mode
        if self.single_tab_mode:
            self._context.on(
                "page", lambda new_pg: self._handle_new_page_safe(new_pg, self._page)
            )

        # Set up download handler
        if self._download_handler:
            self._page.on("download", self._download_handler)

        # Set viewport and add init script
        await self._page.set_viewport_size(
            {"width": self._viewport_width, "height": self._viewport_height}
        )

        await self._page.add_init_script(path=self.page_script_path)

        # Navigate to start page
        await self._page.goto(start_page)
        await self._page.wait_for_load_state()

    async def _handle_new_page_safe(self, new_pg: Page, main_page: Page) -> None:
        """Safely handle new pages in single tab mode."""
        try:
            await new_pg.wait_for_load_state("domcontentloaded")

            # Do not close if new_pg is the current page
            if new_pg == main_page or new_pg.url == main_page.url:
                self.logger.info("New tab is same as current page, not closing.")
                return

            new_url = new_pg.url
            await new_pg.close()
            await self._playwright_controller.visit_page(main_page, new_url)
        except Exception as e:
            self.logger.warning(f"Error in handle_new_page_safe: {e}")

    def start_xvfb(self) -> None:
        """Start Xvfb virtual display server."""
        display_num = 99  # Choose a display number unlikely to be in use
        self.xvfb_process = subprocess.Popen(
            ["Xvfb", f":{display_num}", "-screen", "0", "1280x1024x24", "-ac"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.environ["DISPLAY"] = f":{display_num}"

        # Make sure Xvfb has time to start
        time.sleep(1)

        # Register cleanup function
        atexit.register(self.stop_xvfb)

    def stop_xvfb(self) -> None:
        """Stop the Xvfb process if it's running."""
        if self.xvfb_process:
            self.xvfb_process.send_signal(signal.SIGTERM)
            self.xvfb_process.wait()
            self.xvfb_process = None

    async def wait_for_captcha_resolution(self) -> None:
        """Wait for captcha to be resolved if one is being solved."""
        await self._captcha_event.wait()

    @property
    def page(self) -> Page | None:
        """Get the current page."""
        return self._page

    @page.setter
    def page(self, value):
        self._page = value

    @property
    def context(self) -> BrowserContext | None:
        """Get the browser context."""
        return self._context

    @property
    def playwright_controller(self):
        """Get the playwright controller."""
        return self._playwright_controller

    async def close(self) -> None:
        """Close the browser and clean up resources."""
        self.logger.info("Closing browser...")

        if self._page is not None:
            await self._page.close()
            self._page = None

        if self._context is not None:
            await self._context.close()
            self._context = None

        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

        if self.browser:
            if self.use_browser_base and self.session:
                self.bb.sessions.update(
                    self.session.id,
                    status="REQUEST_RELEASE",
                    project_id=os.environ["BROWSERBASE_PROJECT_ID"],
                )
            await self.browser.close()
            self.browser = None

        if not self.headless:
            self.stop_xvfb()
