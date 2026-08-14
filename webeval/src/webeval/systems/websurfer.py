import asyncio
import json
import logging
import math
import os
import random
import re
from logging import Logger
from typing import Any, Dict, List, Optional, Union

from fara import FaraAgent
from fara.fara_7b.browser.browser_bb import BrowserBB

from webeval.basesystem import BaseSystem
from webeval.trajectory import FinalAnswer, Trajectory
from webeval.utils import LogHandler, dict_2_str


class WebSurferSystem(BaseSystem):
    """
        WebSurferSystem that communicates with either a local or hosted WebSurfer model to perform web-based tasks.
        If websurfer_client_cfg is not provided, it defaults to a local vllm server on localhost:5000 which ought to have already been started.
        Otherwise, it can accept a config dict, a list of config dicts (randomly choosing one per run), or a path to a config file to a foundry endpoint.
    """
    def __init__(
        self,
        system_name: str,
        web_surfer_model_type: str = "fara",
        max_rounds: int = 2,
        websurfer_client_cfg: Union[None, Dict[str, Any], List[Dict[str, Any]], str] = None,
        answer_agent_client_cfg: Union[None, Dict[str, Any]] = None,
        start_on_target_url: bool = False,
        browserbase: bool = False,
        web_surfer_kwargs: Dict[str, Any] = None,
        gpt_solver_model_name: Optional[str] = None, # used for gpt_solver to specify the model name
        fn_call_template: str = "default",
        step_budgets: List[int] = None,
        save_env_state: bool = False
    ) -> None:
        super().__init__(system_name)
        self.web_surfer_model_type = web_surfer_model_type
        self.max_rounds = max_rounds
        self.websurfer_client_cfg = websurfer_client_cfg
        self.answer_agent_client_cfg = answer_agent_client_cfg
        self.start_on_target_url = start_on_target_url
        self.use_browserbase = browserbase
        self.web_surfer_kwargs = web_surfer_kwargs or {}
        self.gpt_solver_model_name = gpt_solver_model_name
        self.fn_call_template = fn_call_template
        self.step_budgets = step_budgets or []

        ### add a bool to save env_state
        self.save_env_state=save_env_state

        if not step_budgets:
            self.step_budgets = [
                math.ceil(self.max_rounds * pct) for pct in [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
            ]

        assert self.web_surfer_model_type in {
            "fara"
        }

    ##########################
    async def _capture_env_state(
        self,
        page,                       
        controller,                 
        base: str                   
    ):                
        finish_url = f"{base.rstrip('/')}/finish"        
        if hasattr(self, "logger"):
            self.logger.info(f"[env] visiting finish URL: {finish_url}")

        await controller.visit_page(page, finish_url)
        await controller.wait_for_load_state(page)

        try:
            pre = await page.wait_for_selector("pre", state="visible", timeout=8000)
        except Exception:
            if hasattr(self, "logger"):
                self.logger.warning("[env] <pre> not found on finish page")
            return None, None

        env_text = await pre.inner_text()
        env_json = None
        try:
            import json
            env_json = json.loads(env_text)
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.warning(f"[env] Invalid JSON format: {e}")

        # Stash for downstream consumers
        if getattr(self, "final_answer_store", None) is not None:
            self.final_answer_store.env_state_raw = env_text
            self.final_answer_store.env_state_json = env_json

        return env_text, env_json

    #######################

    def get_answer(self, question_id: str, example_data: Dict[str, Any], output_dir: str, logger: Logger = None) -> Optional[Trajectory]:
        async def _runner():
            final_answer_store = FinalAnswer()


            if self.start_on_target_url:
                start_page = example_data.get("url", example_data.get("web", "https://www.bing.com"))
            else:
                start_page = "https://www.bing.com"

            question_text = example_data.get("question", "")

            # Config for the local OpenAI-compatible server
            if not self.websurfer_client_cfg:
                ### assumes local vllm server on localhost:5000 which should have already been started by this point
                client_config = {
                    "model": "gpt-4o-mini-2024-07-18",
                    "base_url": "http://localhost:5000/v1",
                    "api_key": "not-needed",
                }
            else:
                ### assumes endpoints are loaded from fara/src/fara/webeval/scripts/eval_exp.py:get_foundry_endpoint_configs
                if isinstance(self.websurfer_client_cfg, str):
                    # load from file
                    with open(self.websurfer_client_cfg, "r") as f:
                        client_config = json.load(f)
                elif isinstance(self.websurfer_client_cfg, list):
                    # use a random config from the list
                    client_config = random.choice(self.websurfer_client_cfg)
                elif isinstance(self.websurfer_client_cfg, dict):
                    client_config = self.websurfer_client_cfg
                else:
                    raise ValueError("Invalid websurfer_client_cfg type, must be a valid config with model, base_url, api_key fields")

            # Create the FaraAgent instance
            for _ in range(1):
                # Initialize browser manager
                browser_manager = BrowserBB(
                    headless=True,
                    viewport_height=900,
                    viewport_width=1440,
                    page_script_path=None,
                    browser_channel="firefox",
                    browser_data_dir=None,
                    downloads_folder=output_dir,
                    to_resize_viewport=True,
                    single_tab_mode=True,
                    animate_actions=False,
                    use_browser_base=self.use_browserbase,
                    logger=logger
                )

                agent = FaraAgent(
                    browser_manager=browser_manager,
                    client_config=client_config,
                    start_page=start_page,
                    downloads_folder=output_dir,
                    save_screenshots=True,
                    max_rounds=100,
                    logger = logger
                )

                await agent.initialize()
                logging.info(f"Initialized FaraAgent with start page: {start_page}")
                print(f"Running task: {question_text}")
                print("----------------------------------------")
                final_answer, all_actions, all_observations = await agent.run(question_text)
                final_answer_store.final_answer = final_answer
                break  # Exit the retry loop if successful
                # Close the agent and browser
                await agent.close()


            # look at the output dir for any file that looks like screenshot{i}.png where i is an integer and nothing else
            screenshot_files = [f for f in os.listdir(output_dir) if f.startswith("screenshot") and f.endswith(".png")]
            # Use regex to ensure we only match files with the exact pattern "screenshot{i}.png" where i is an integer
            screenshot_pattern = re.compile(r"^screenshot(\d+)\.png$")
            screenshot_files = [f for f in screenshot_files if screenshot_pattern.match(f)]
            screenshot_files.sort(key=lambda x: int(screenshot_pattern.match(x).group(1)))
            # Store the full absolute paths to the screenshot files
            final_answer_store.screenshots = screenshot_files #[os.path.abspath(os.path.join(output_dir, f)) for f in screenshot_files]
        

            final_answer_store.save(os.path.join(output_dir, f"{question_id}_final_answer.json"))

            return self.load_answer_from_disk(question_id, output_dir)

        log_file = os.path.join(output_dir, "web_surfer.log")    # TODO: use a different mechanism for recording taken actions and use logger for logs
        logger = logger or logging.getLogger("WebSurferLogger")
        handler = LogHandler(filename=log_file)
        # FaraAgent emits Thought/Action/Observation traces via
        # ``self.logger.debug(...)`` (fara_agent.py L382, L394, L446). The
        # Python default level is WARNING, which silently drops them — and
        # downstream evaluators (incl. WebTailBench's rubric verifier) treat
        # an empty web_surfer.log as "no actions in trajectory" → score 0.
        # Force DEBUG on both the logger and the handler so the JSONL log
        # actually captures every step.
        handler.setLevel(logging.DEBUG)
        prev_level = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            logger.addHandler(handler)
            return asyncio.run(_runner())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)
            handler.close()
        
            
    def load_answer_from_disk(self, task_id: str, output_dir: str) -> Any:
        is_gpt_solver = self.web_surfer_model_type == "gpt_solver" or self.web_surfer_model_type == "anthropic"
        return Trajectory.from_folder(output_dir, gpt_solver=is_gpt_solver)
    
    def hash(self) -> str:
        surfer_args = {
            k: v for k, v in self.web_surfer_kwargs.items() if k in {'max_n_images'}}    # TODO: cleanup
        if (self.web_surfer_kwargs is not None) and any(self.web_surfer_kwargs.values()):
            return f'{super().hash()}-{self.web_surfer_model_type}-{self.max_rounds}-{dict_2_str(surfer_args)}'   # TODO: incorporate other hyperparameters?
        return f'{super().hash()}-{self.web_surfer_model_type}-{self.max_rounds}--{self.save_env_state}'
