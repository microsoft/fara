import argparse
import os
# import mlflow

from eval_exp import EvalExp


class ParseKwargs(argparse.Action):
    def __init__(self, option_strings, dest, type_map=None, **kwargs):
        self.type_map = type_map or {}
        super().__init__(option_strings, dest, nargs='*', **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        result = {}
        for item in values:
            if '=' not in item:
                raise argparse.ArgumentError(self, f"Expected key=value format, got '{item}'")
            key, val = item.split('=', 1)
            val_type = self.type_map.get(key, int)
            try:
                if val_type == bool:
                    assert str(val).lower() in {"true", "false", "1", "0", "yes", "no"}
                    tmp_val = str(val).lower() in {"true", "1", "yes"}
                else:
                    tmp_val = val_type(val)

                result[key] = tmp_val
            except ValueError:
                result[key] = str(val)
                raise argparse.ArgumentError(self, f"Invalid value for {key}: expected {val_type.__name__}")
        setattr(namespace, self.dest, result)


def _get_base_eval_arg_parser():
    web_surfer_kwargs_type_map = {
        "max_n_images": int,
        "model_call_timeout": int,
        "enable_guidelines_prompt": bool,
        "include_url": bool,
        "max_url_chars": int,
    }

    parser = argparse.ArgumentParser(description="Evaluate WebSurfer system")
    parser.add_argument("--model_url", type=str, default=None, help="Path to model on either disk or an Azure blobstore if you intend to host the model LOCALLY via vllm")
    parser.add_argument('--model_endpoint', type=str, default=None, help='Path to EXTERNAL Foundry endpoint config JSON file (or directory of configs). If specified, uses external hosted model instead of local vllm. Similar to --endpoint_config in test_fara_agent.py')
    parser.add_argument("--out_url", type=str, default=EvalExp.DEFAULT_OUT, help=f"Output URL, default: {EvalExp.DEFAULT_OUT}")
    parser.add_argument("--user", type=str, default=None, help="User name, default: local user name")
    parser.add_argument("--run_id", type=str, default=None, help="run id")
    parser.add_argument("--subsample", type=float, default=1.0, help="Subsample ratio for evaluation")
    parser.add_argument('--eval_oai_config', type=str, default = "../../agento/endpoint_configs/dev", help='Path to the OpenAI config file(s)')
    parser.add_argument('--processes', type=int, default=1, help='Number of processes to use for evaluation')
    parser.add_argument('--max_rounds', type=int, default=100, help='Maximum trajectory length')
    parser.add_argument('--web_surfer_model_type', type=str, choices=["fara"], default='fara', help='WebSurfer model type (default: fara)')
    parser.add_argument('--model_port', type=int, default=5000, help='Model port (default: 5000), only use if hosting model locally via vllm')
    parser.add_argument('--device_id', type=str, default="0", help='Device ID (default: 0), only use if hosting model locally via vllm')
    parser.add_argument('--redo_eval', action='store_true', help='Redo llm-as-a-judge evaluation even if already exists, but will not re-sample trajectories')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--include_input_text_key_args', action='store_true', help='Include input_text key arguments (press_enter, delete_existing_text)')
    parser.add_argument('--browserbase', action='store_true', help='Use browser-base proxy network to host browser sessions in order to mitigate websites blocking requests (i.e. most airlines/retailers will block traffic they suspect to be from bots)')
    parser.add_argument('--web_surfer_kwargs', action=ParseKwargs, type_map=web_surfer_kwargs_type_map, default={"max_n_images":3, "model_call_timeout":300})
    parser.add_argument('--web_surfer_client_cfg', type=str, default=None, help="Path to the web surfer client configuration file, or a directory of several configs")
    parser.add_argument('--gpt_solver_model_name', type=str,  default=None, help='Model type to use for gpt_solver if gpt_solver is used (* for all models)')
    parser.add_argument('--eval_model', type=str, choices=['gpt-4o', 'o4-mini', 'o3-mini', 'o3', '*'], default='gpt-4o', help='Model type to use for evaluation (default: gpt-4o, * for all models)')
    parser.add_argument('--dtype', type=str, choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'], default='auto', help='Data type for VLLM model (default: auto)')
    parser.add_argument('--enforce_eager', action='store_true', help='Enforce eager execution for VLLM model')
    parser.add_argument('--fn_call_template', type=str, choices=['default', 'qwen', 'thinking', 'with_ci'], default='default', help='Function call template to use for system prompts (default: default)')
    parser.add_argument('--async_mlflow', action='store_true', help='Enable asynchronous logging to MLflow (may lose logs if the program crashes)')
    parser.add_argument('--step_budgets', type=int, nargs='+', default=None,
        help='List of step budgets to compute step budget scores for (e.g., --step_budgets 5 10 15). All values should be between [1, max_rounds]. If not provided, defaults to [5%%, 10%%, 20%%, 25%%, 30%%, 40%%, 50%%, 60%%, 70%%, 75%%, 80%%, 90%%, 100%%] of max_rounds.'
    )
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate existing results without sampling new trajectories from the model')
    parser.add_argument('--max_error_task_retries', type=int, default=5, help='Maximum number of retries for tasks that fail or abort due to errors (default: 5)')
    parser.add_argument('--save_env_state', action='store_true', help='save_env_state arguments')
    parser.add_argument('--save_task_csv', action='store_true', help='Save detailed task results to CSV file with question, answer, refusal status, and trace information')


    return parser


def get_eval_args(benchmark_arg_func):
    parser = _get_base_eval_arg_parser()
    benchmark_arg_func(parser)
    args = parser.parse_args()
    if args.include_input_text_key_args is None:
        # TODO: this should not be needed like this, keeping it like this for now.
        # ATTENTION: this is a hack to make with hardcoding, better than assert
        print("WARNING: include_input_text_key_args is not set, setting it to True")
        args.include_input_text_key_args = True

    ### if both model_url and model_endpoint are None, raise error
    if args.model_url is None and args.model_endpoint is None:
        raise ValueError("Either --model_url or --model_endpoint must be specified to run evaluation")
    ### if both model_url and model_endpoint are set, raise error
    if args.model_url is not None and args.model_endpoint is not None:
        raise ValueError("Only one of --model_url or --model_endpoint can be specified to run evaluation")
    if args.async_mlflow:
        mlflow.config.enable_async_logging()
    # create out_url if not exists
    out_url = os.path.expanduser(args.out_url)
    if not os.path.exists(out_url):
        os.makedirs(out_url, exist_ok=True)
    return args
