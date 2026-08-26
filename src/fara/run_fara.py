import asyncio
import argparse
import os
import uuid
import logging
from typing import Dict
from pathlib import Path
import json

from .agents.fara.fara15_agent import Fara15Agent, Fara15AgentConfig
from .environments.playwright import PlaywrightEnvironment
from .core.run_context import RunContext
from .core.data_point import Task, SolverStatus, UserMessage, UserMessageType
from .fara_7b import FaraAgent
from .fara_7b.browser.browser_bb import BrowserBB


logging.basicConfig(
    level=logging.CRITICAL,
    format="%(message)s",
)

for _logger_name in ("fara.agents.fara.fara15_agent", "fara.fara_7b.fara_agent"):
    _agent_logger = logging.getLogger(_logger_name)
    _agent_logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _agent_logger.addHandler(_handler)
    _agent_logger.propagate = False

logger = logging.getLogger(__name__)


DEFAULT_ENDPOINT_CONFIG = {
    "model": "Fara1.5-9B",
    "base_url": "http://localhost:5000/v1",
    "api_key": "not-needed",
}

DEFAULT_FARA_7B_ENDPOINT_CONFIG = {
    "model": "microsoft/Fara-7B",
    "base_url": "http://localhost:5000/v1",
    "api_key": "not-needed",
}


async def run_fara15_agent(
    initial_task: str = None,
    endpoint_config: Dict[str, str] = None,
    start_page: str = "https://www.bing.com/",
    headless: bool = True,
    output_folder: str = None,
    save_screenshots: bool = True,
    max_rounds: int = 100,
    cdp_url: str | None = None,
    use_browser_base: bool = False,
) -> None:
    """Interactive loop for the Fara-1.5 agent."""
    output_root = Path(output_folder or "./fara_runs")

    print("Initializing Browser...")
    env = PlaywrightEnvironment(
        viewport_width=1440,
        viewport_height=900,
        headless=headless,
        browser_channel="chromium",
        start_page=start_page,
        single_tab_mode=True,
        cdp_url=cdp_url,
        use_browserbase=use_browser_base,
    )
    await env.initialize()
    print("Browser Running... Starting Fara-1.5 Agent...")

    agent = Fara15Agent(
        Fara15AgentConfig(
            client_config=endpoint_config,
            max_rounds=max_rounds,
            identity="fara_qwen35",
            critical_points="fara-1.5",
            save_screenshots=save_screenshots,
        )
    )

    try:
        task_text = initial_task
        first_round = True
        run_idx = 0

        while True:
            if task_text is None:
                if first_round:
                    task_text = input("Enter task: ").strip()
                else:
                    task_text = input(
                        "\nEnter another task (or press Enter to exit): "
                    ).strip()
                if not task_text:
                    print("Exiting...")
                    break

            print("##########################################")
            print(f"Task: {task_text}")
            print("##########################################")

            run_idx += 1
            output_dir = output_root / f"task_{run_idx}_{uuid.uuid4().hex[:8]}"
            task = Task(task_id=uuid.uuid4().hex, instruction=task_text)
            run_context = RunContext.create(
                environment=env, task=task, output_dir=output_dir
            )
            await agent.initialize(run_context)

            try:
                print("Running Fara...\n")
                final_answer, _, _ = await agent.run(run_context)

                while run_context.solver_log.status == SolverStatus.WAITING_FOR_USER:
                    print(f"\nFara asks: {final_answer}")
                    reply = input("Your response (Enter to abandon): ").strip()
                    if not reply:
                        break
                    run_context.add_observation(
                        UserMessage(
                            content=reply,
                            message_type=UserMessageType.CRITICAL_POINT_RESPONSE,
                        )
                    )
                    final_answer, _, _ = await agent.run(run_context)

                if run_context.solver_log.status == SolverStatus.COMPLETE:
                    print(f"\nFinal Answer: {final_answer}")
                print(f"Trajectory saved to: {output_dir}")
            except Exception as e:
                print(f"Error occurred: {e}")
            finally:
                await agent.close(run_context)

            task_text = None
            first_round = False
    finally:
        await env.close()


async def run_fara7b_agent(
    initial_task: str = None,
    endpoint_config: Dict[str, str] = None,
    start_page: str = "https://www.bing.com/",
    headless: bool = True,
    downloads_folder: str = None,
    save_screenshots: bool = True,
    max_rounds: int = 100,
    cdp_url: str | None = None,
    use_browser_base: bool = False,
) -> None:
    """Interactive loop for the previous-generation Fara-7B agent."""
    print("Initializing Browser...")
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
        cdp_url=cdp_url,
        use_browser_base=use_browser_base,
        logger=logger,
    )
    print("Browser Running... Starting Fara-7B Agent...")

    agent = FaraAgent(
        browser_manager=browser_manager,
        client_config=endpoint_config,
        start_page=start_page,
        downloads_folder=downloads_folder,
        save_screenshots=save_screenshots,
        max_rounds=max_rounds,
    )

    try:
        await agent.initialize()
        task = initial_task
        first_round = True
        while True:
            if task is None:
                if first_round:
                    task = input("Enter task: ").strip()
                else:
                    task = input(
                        "\nEnter another task (or press Enter to exit): "
                    ).strip()
                if not task:
                    print("Exiting...")
                    break

            print("##########################################")
            print(f"Task: {task}")
            print("##########################################")
            try:
                print("Running Fara...\n")
                final_answer, _, _ = await agent.run(task)
                print(f"\nFinal Answer: {final_answer}")
            except Exception as e:
                print(f"Error occurred: {e}")
            task = None
            first_round = False
    finally:
        await agent.close()


def main():
    """CLI entry point for fara command."""
    parser = argparse.ArgumentParser(description="Run FARA agent interactively")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help="Initial task for the FARA agent (optional)",
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
        "--output_folder",
        type=str,
        default=None,
        help="Folder to save screenshots, downloads and the trajectory data_point.json",
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
        "--cdp_url",
        type=str,
        default=os.environ.get("FARA_CDP_URL"),
        help="Connect to an existing Chromium browser over CDP (defaults to FARA_CDP_URL)",
    )
    parser.add_argument(
        "--browserbase",
        action="store_true",
        help="Whether to use BrowserBase for browser management",
    )
    parser.add_argument(
        "--endpoint_config",
        type=Path,
        default=None,
        help="Path to the endpoint configuration JSON file. By default, tries local vllm on 5000 port",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for the model endpoint (overrides endpoint_config)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Base URL for the model endpoint (overrides endpoint_config)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (overrides endpoint_config)",
    )
    parser.add_argument(
        "--fara-7b",
        action="store_true",
        dest="fara_7b",
        help="Run the previous-generation Fara-7B agent instead of Fara-1.5",
    )

    args = parser.parse_args()

    if args.browserbase and args.cdp_url:
        parser.error("--browserbase and --cdp_url cannot be used together")

    if args.browserbase:
        assert os.environ.get(
            "BROWSERBASE_API_KEY"
        ), "BROWSERBASE_API_KEY environment variable must be set to use browserbase"
        assert os.environ.get(
            "BROWSERBASE_PROJECT_ID"
        ), "BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID environment variables must be set to use browserbase"

    endpoint_config = (
        DEFAULT_FARA_7B_ENDPOINT_CONFIG if args.fara_7b else DEFAULT_ENDPOINT_CONFIG
    )
    if args.endpoint_config:
        with open(args.endpoint_config, "r") as f:
            endpoint_config = json.load(f)
            assert (
                "api_key" in endpoint_config
                and "base_url" in endpoint_config
                and "model" in endpoint_config
            ), "endpoint_config file must contain api_key, base_url, and model fields"
    if args.api_key:
        endpoint_config["api_key"] = args.api_key
    if args.base_url:
        endpoint_config["base_url"] = args.base_url
    if args.model:
        endpoint_config["model"] = args.model

    if args.fara_7b:
        asyncio.run(
            run_fara7b_agent(
                initial_task=args.task,
                endpoint_config=endpoint_config,
                start_page=args.start_page,
                headless=not args.headful,
                downloads_folder=args.output_folder,
                save_screenshots=args.save_screenshots,
                max_rounds=args.max_rounds,
                cdp_url=args.cdp_url,
                use_browser_base=args.browserbase,
            )
        )
    else:
        asyncio.run(
            run_fara15_agent(
                initial_task=args.task,
                endpoint_config=endpoint_config,
                start_page=args.start_page,
                headless=not args.headful,
                output_folder=args.output_folder,
                save_screenshots=args.save_screenshots,
                max_rounds=args.max_rounds,
                cdp_url=args.cdp_url,
                use_browser_base=args.browserbase,
            )
        )


if __name__ == "__main__":
    main()
