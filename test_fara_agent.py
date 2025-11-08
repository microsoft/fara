import asyncio
import argparse
import os
from fara import FaraAgent
from fara.browser.browser_bb import BrowserBB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_fara_agent(
    task: str,
    start_page: str = "https://www.bing.com/",
    headless: bool = True,
    downloads_folder: str = None,
    save_screenshots: bool = True,
    max_rounds: int = 100,
    use_browser_base: bool = False,
    retries_on_failure: int = 3,
):
    # Config for the local OpenAI-compatible server
    client_config = {
        "model": "gpt-4o-mini-2024-07-18",
        "base_url": "http://localhost:5000/v1",
        "api_key": "not-needed",
    }

    # Create the FaraAgent instance
    for _ in range(retries_on_failure):
        # Initialize browser manager
        browser_manager = BrowserBB(
            headless=headless,
            viewport_height=900,
            viewport_width=1440,
            page_script_path=None,
            browser_channel="firefox",
            browser_data_dir=None,
            downloads_folder=downloads_folder,
            to_resize_viewport=True,
            single_tab_mode=True,
            animate_actions=False,
            use_browser_base=use_browser_base,
            logger=logger,
        )

        agent = FaraAgent(
            browser_manager=browser_manager,
            client_config=client_config,
            start_page=start_page,
            downloads_folder=downloads_folder,
            save_screenshots=save_screenshots,
            max_rounds=max_rounds,
        )

        try:
            await agent.initialize()
            print(f"Running task: {task}")
            await agent.run(task)
            break  # Exit the retry loop if successful
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            # Close the agent and browser
            await agent.close()


async def main():
    parser = argparse.ArgumentParser(description="Run FARA agent with a specified task")
    parser.add_argument(
        "--task", type=str, required=True, help="The task for the FARA agent to perform"
    )
    parser.add_argument(
        "--start_page",
        type=str,
        default="https://www.bing.com/",
        help="The starting page",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run the browser in headful mode (show GUI, default is headless)",
    )
    parser.add_argument(
        "--downloads_folder",
        type=str,
        default=None,
        help="Folder to save screenshots and downloads",
    )
    parser.add_argument(
        "--save_screenshots",
        action="store_true",
        help="Whether to save screenshots during the agent's operation",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=100,
        help="Maximum number of rounds for the agent to run",
    )
    parser.add_argument(
        "--browserbase",
        action="store_true",
        help="Whether to use BrowserBase for browser management",
    )

    args = parser.parse_args()

    if args.browserbase:
        assert os.environ.get("BROWSERBASE_API_KEY"), "BROWSERBASE_API_KEY environment variable must be set to use browserbase"
        assert os.environ.get("BROWSERBASE_PROJECT_ID"), "BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID environment variables must be set to use browserbase"

    await run_fara_agent(
        task=args.task,
        start_page=args.start_page,
        headless=not args.headful,
        downloads_folder=args.downloads_folder,
        save_screenshots=args.save_screenshots,
        max_rounds=args.max_rounds,
        use_browser_base=args.browserbase,
    )


if __name__ == "__main__":
    asyncio.run(main())
