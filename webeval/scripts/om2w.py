from webeval.systems.websurfer import WebSurferSystem
from webeval.benchmarks import OnlineM2WBenchmark
from pathlib import Path
import numpy as np
import os
import logging
# from aztool.workspace import Workspace, AIF_WORKSPACE
import mlflow
from eval_exp import EvalExp, ModelReference, get_foundry_endpoint_configs
from webeval.oai_clients.graceful_client import GracefulRetryClient
from webeval.eval_result import EvalResult, Stage
from arg_parsing import get_eval_args

DEFAULT_DATA_URL = '../data/om2w/Online_Mind2Web_06042025.json'

class Callback:
    def __init__(self):
        self.scores = []

    def __call__(self, result: EvalResult, mlflow_facade, mlflow_run_id: str):
        if result.stage == Stage.EVALUATED:
            self.scores.append(result.score)
        mlflow_facade.log_metric('score', np.mean(self.scores or [0]), run_id = mlflow_run_id)


def add_om2w_args(parser):
    parser.add_argument('--eval_data_url', type=str, default=DEFAULT_DATA_URL, help='Azure URI to the evaluation data')
    parser.add_argument('--split', type=str, default='*')
    parser.add_argument('--eval_method', type=str, default='WebJudge_Online_Mind2Web_eval', help='Evaluation method (default: WebJudge_Online_Mind2Web_eval, one of: AgentTrek_eval, Autonomous_eval, WebJudge_general_eval, WebJudge_Online_Mind2Web_eval)')


def main():
    args = get_eval_args(add_om2w_args)
    assert args.eval_model == "o4-mini", "only o4-mini can be used for om2w eval"

    if args.browserbase:
        assert os.environ.get("BROWSERBASE_API_KEY"), "BROWSERBASE_API_KEY environment variable must be set to use browserbase"
        assert os.environ.get("BROWSERBASE_PROJECT_ID"), "BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID environment variables must be set to use browserbase"

    experiment = EvalExp(
        ws = None,
        user = args.user,
        seed = args.seed)
  
    with experiment.start_run() as run:
        model_ref = ModelReference(args.model_url, args.model_port, args.device_id, args.web_surfer_kwargs.get('max_n_images', 3), args.gpt_solver_model_name, args.dtype, args.enforce_eager, use_external_endpoint=bool(args.model_endpoint))

        logger = logging.getLogger('om2w-eval')
        logger.setLevel(logging.INFO)

        mlflow.log_param('max_rounds', args.max_rounds)
        if args.web_surfer_model_type == "gpt_solver":
            mlflow.log_param('web_surfer_model_type', args.web_surfer_model_type + "/" + args.gpt_solver_model_name)
        else:
            mlflow.log_param('web_surfer_model_type', args.web_surfer_model_type)
        mlflow.log_param('fn_call_template', args.fn_call_template)

        # If using external endpoint, load all endpoint configs into a list of dicts
        if args.model_endpoint:
            websurfer_client_cfg = get_foundry_endpoint_configs(args.model_endpoint)
            logger.info(f"Loaded {len(websurfer_client_cfg)} external endpoint config(s) from {args.model_endpoint}")
            model_ref.model_url_to_log = websurfer_client_cfg[0]['base_url']  # log the first endpoint URL as
            model_ref.model_to_log = websurfer_client_cfg[0]['base_url']
            mlflow.log_param('using_external_endpoint', True)
            mlflow.log_param('endpoint_config_path', ','.join([x['base_url'] for x in websurfer_client_cfg]))
        else:
            # For local VLLM, use a simple flat config structure that FaraAgent expects
            websurfer_client_cfg = {
                "api_key": "NONE",
                "model": "gpt-4o-mini-2024-07-18",
                "base_url": f"http://0.0.0.0:{args.model_port}/v1"
            }
            if args.web_surfer_client_cfg is not None:
                websurfer_client_cfg = args.web_surfer_client_cfg
        if args.web_surfer_kwargs:
            mlflow.log_params({f'web_surfer_kwargs.{k}': v  for k, v in args.web_surfer_kwargs.items()})

        system = WebSurferSystem(
            system_name="WebSurfer",
            web_surfer_model_type = args.web_surfer_model_type,
            max_rounds = args.max_rounds,
            websurfer_client_cfg = websurfer_client_cfg,
            start_on_target_url=True,
            browserbase=args.browserbase,
            web_surfer_kwargs=args.web_surfer_kwargs,
            gpt_solver_model_name=args.gpt_solver_model_name,
            fn_call_template=args.fn_call_template,
            step_budgets=args.step_budgets,
        )


        mlflow.log_param("eval_data", args.eval_data_url)
        mlflow.log_param("eval_model", args.eval_model)
        mlflow.log_param("eval_method", args.eval_method)
        data_dir = Path(__file__).resolve().parent.parent / "data" / "om2w"
        data_dir.mkdir(parents=True, exist_ok=True)
        benchmark = OnlineM2WBenchmark(
            data_dir=data_dir,
            eval_method = args.eval_method,
            data_az_url = args.eval_data_url,
            model_client = GracefulRetryClient.from_path(args.eval_oai_config, logger=logger, eval_model=args.eval_model))

        mlflow.log_param('subsample', args.subsample)
        mlflow.log_param('processes', args.processes)
        mlflow.log_param('split', args.split)
        mlflow.log_param('max_error_task_retries', args.max_error_task_retries)
        experiment.run(
            model_ref = model_ref,
            system = system,
            benchmark = benchmark,
            out_url = args.out_url,
            subsample = args.subsample,
            redo_eval = args.redo_eval,
            run_id = args.run_id,
            split = args.split,
            processes = args.processes,
            callbacks = [Callback()],
            eval_only = args.eval_only,
            max_error_task_retries = args.max_error_task_retries)


if __name__ == "__main__":
    main()