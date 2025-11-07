from __future__ import annotations
from types import TracebackType
from typing import AsyncContextManager, Optional, Type, Any, Dict
from typing_extensions import Self
from abc import ABC, abstractmethod
from pathlib import Path
from playwright.async_api import (
    BrowserContext,
    Playwright,
    Browser,
    async_playwright,
)
# Adapted from Magentic-UI


class PlaywrightBrowser(AsyncContextManager["PlaywrightBrowser"], ABC):
    """
    Abstract base class for Playwright browser.
    """

    def __init__(self):
        self._closed: bool = False

    @abstractmethod
    async def _start(self) -> None:
        """
        Start the browser resource.
        """
        pass

    @abstractmethod
    async def _close(self) -> None:
        """
        Close the browser resource.
        """
        pass

    # Expose playwright context
    @property
    @abstractmethod
    def browser_context(self) -> BrowserContext:
        """
        Return the Playwright browser context.
        """
        pass

    async def __aenter__(self) -> Self:
        """
        Start the Playwright browser.

        Returns:
            Self: The current instance of PlaywrightBrowser
        """
        await self._start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Stop the browser.

        This method attempts a graceful termination first by sending SIGTERM,
        and if that fails, it forces termination with SIGKILL. It ensures
        the browser is properly cleaned up.
        """

        if not self._closed:
            # Close the browser resource
            await self._close()
            self._closed = True


class LocalPlaywrightBrowser(PlaywrightBrowser):
    """
    A local Playwright browser implementation that provides flexible browser automation capabilities.
    Supports both persistent and non-persistent browser contexts, with configurable options for
    headless operation and download handling.

    Args:
        headless (bool): Whether to run the browser in headless mode.
        browser_channel (str, optional): The browser channel to use (e.g., 'chrome', 'msedge'). Default: None.
        enable_downloads (bool, optional): Whether to enable file downloads. Default: False.
        persistent_context (bool, optional): Whether to use a persistent browser context. Default: False.
        browser_data_dir (str, optional): Path to the browser user data directory for persistent contexts.
            Required if persistent_context is True. Default: None.

    Properties:
        browser_context (BrowserContext): The active Playwright browser context.
            Raises RuntimeError if accessed before browser is started.

    Example:
        ```python
        # Create a headful Chrome browser with persistent context
        browser = LocalPlaywrightBrowser(
            headless=False,
            browser_channel='chrome',
            persistent_context=True,
            browser_data_dir='./browser_data'
        )
        await browser.start()
        context = browser.browser_context
        # Use the browser for automation
        await browser.close()
        ```
    """

    def __init__(
        self,
        headless: bool = False,
        browser_channel: Optional[str] = None,
        enable_downloads: bool = False,
        persistent_context: bool = False,
        browser_data_dir: Optional[str] = None,
    ):
        super().__init__()
        self._headless = headless
        self._browser_channel = browser_channel
        self._enable_downloads = enable_downloads
        self._persistent_context = persistent_context
        self._browser_data_dir = browser_data_dir
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def _start(self) -> None:
        """
        Start the browser resource.
        """
        self._playwright = await async_playwright().start()

        launch_options: Dict[str, Any] = {"headless": self._headless}
        if self._browser_channel:
            launch_options["channel"] = self._browser_channel

        if self._persistent_context and self._browser_data_dir:
            # Ensure the browser data directory exists
            Path(self._browser_data_dir).mkdir(parents=True, exist_ok=True)

            # Launch persistent context
            self._context = await self._playwright.chromium.launch_persistent_context(
                self._browser_data_dir,
                accept_downloads=self._enable_downloads,
                **launch_options,
                args=["--disable-extensions", "--disable-file-system"],
                env={},
            )
        else:
            # Launch regular browser and create new context
            self._browser = await self._playwright.chromium.launch(
                **launch_options,
                args=["--disable-extensions", "--disable-file-system"],
                env={} if self._headless else {"DISPLAY": ":0"},
            )

            self._context = await self._browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
                accept_downloads=self._enable_downloads,
            )

    async def _close(self) -> None:
        """
        Close the browser resource.
        """
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    @property
    def browser_context(self) -> BrowserContext:
        """
        Return the Playwright browser context.
        """
        if self._context is None:
            raise RuntimeError(
                "Browser context is not initialized. Start the browser first."
            )
        return self._context
