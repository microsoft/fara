import mlflow
from typing import Callable, Dict, List
import os
from tqdm import tqdm
import getpass
import requests
import logging
import sys
import dummy_workspace
import random
import json
import numpy as np
import threading
from mlflow_rate_limiter import MlFlowRateLimiter
from joblib import Parallel, delayed
import pandas as pd
import csv
import time
from pathlib import Path
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', None)

# from aztool.azcp import AzFolder
# from aztool.workspace import renew_mlflow_token
from fara.vllm.az_vllm import AzVllm
from webeval.core import reduce_eval_results, run_eval_multiple_examples, run_eval_multiple_examples_with_progress
from webeval.eval_result import EvalResult, Stage
from webeval.post_eval_analysis import aggregate_post_eval_errors, count_web_surfer_log_entries
from webeval.trajectory import Trajectory

def map_folders(path, func: Callable[[os.PathLike], Dict]) -> List[Dict]:
    return Parallel(-1)(delayed(func)(p) for p in tqdm([p for p in path.iterdir() if p.is_dir()], desc=f"Processing folders in {path}..."))


def _is_azure_blob_url(model_path: str) -> bool:
    """Check if a given path is an Azure blob URL."""
    from urllib.parse import urlparse
    parsed = urlparse(model_path)
    return (
        parsed.scheme in ("https", "http")
        and parsed.hostname == "blob.core.windows.net"
    )


def get_default_vllm_model_config(model_port):
    default_vllm_model_config = {
        "CHAT_COMPLETION_PROVIDER": "openai",
        "CHAT_COMPLETION_KWARGS_JSON": {
            "api_key": "NONE",
            "model": "gpt-4o-mini-2024-07-18",
            "base_url": f"http://0.0.0.0:{model_port}/v1/"
        },
        "model_capabilities": {
            "vision": True,
            "function_calling": False,
            "json_output": False
        }
    }
    return default_vllm_model_config

def get_foundry_endpoint_configs(endpoint_config_path: str) -> List[Dict]:
    endpoint_path = Path(endpoint_config_path).resolve()
    if endpoint_path.is_dir():
        config_files = sorted(list(endpoint_path.iterdir()))
    else:
        config_files = [endpoint_path]

    websurfer_client_cfg = []
    try:
        for config_file in config_files:
            with open(config_file, 'r') as f:
                config = json.load(f)
                assert "model" in config and "base_url" in config and "api_key" in config, f"Config file {config_file} is missing required fields: model, base_url, api_key"
                assert config["api_key"], f"API key in config file {config_file} is empty"
                websurfer_client_cfg.append(config)
    except Exception as e:
        raise RuntimeError(f"Error loading endpoint configs from {endpoint_config_path}: {e}")

    return websurfer_client_cfg


class ModelReference:
    def __init__(self, model_url, model_port, device_id, max_n_images, model_name=None, dtype='auto', enforce_eager=False, use_external_endpoint=False):
        self.model_url_to_start = model_url
        self.model_port = model_port
        self.model_url_to_log = model_url
        self.model_to_log = None
        self.model_prefix = None
        self.device_id = device_id
        self.model_name = model_name
        self.max_n_images = max_n_images
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.use_external_endpoint = use_external_endpoint

        if self.use_external_endpoint:
            # If using external endpoint, model_url is expected to be a config dict or path to config file
            self.model_url_to_log = model_url
            self.model_to_log = model_url
            self.model_prefix = "external_model"
            return
        elif model_url is None:
            if self.model_name:
                logging.info(f'Using provided model name: {self.model_name}')
                self.model_prefix = self.model_name.replace('/', '_').replace(':', '_')
            else:
                response = requests.get(f'http://localhost:{model_port}/model')
                if response.status_code == 200:
                    model_name_from_response = response.json()['model']
                    self.model_url_to_log = response.json()['model_url']
                    self.model_to_log = self.model_url_to_log
                    self.model_prefix = model_name_from_response
                else:
                    raise Exception(f"Failed to get model info from VLLM server, status code: {response.status_code}")

        else:
            if _is_azure_blob_url(self.model_url_to_log):
                raise NotImplementedError("Logging Azure Blob URLs is not implemented in this version.")
            else:
                # It's a local directory
                self.model_to_log = self.model_url_to_log
                self.model_prefix = Path(self.model_url_to_log).name.replace('/', '_').replace(':', '_')
    
    def log_2_mlflow(self):
        mlflow.log_param('model', self.model_to_log)
        mlflow.log_param('model_url', self.model_url_to_log)

class Callback:
    def __init__(self, callbacks = None):
        self.evaluated = 0
        self.executed = 0
        self.processed = 0
        self.results = []
        self.run_id = mlflow.active_run().info.run_id
        self.callbacks = callbacks or []
        self.lock = threading.Lock()
        self.mlflow = MlFlowRateLimiter(period_s = 10, thread_safe = False)


    def __call__(self, result: EvalResult):
        with self.lock:
            self.results.append(result)
            try:
                self.processed += 1
                if result.stage >= Stage.EXECUTED:
                    self.executed += 1
                if result.stage == Stage.EVALUATED:
                    self.evaluated += 1
            finally:
                self.mlflow.log_metrics({
                    'evaluated': self.evaluated,
                    'executed': self.executed,
                    'processed': self.processed
                }, run_id = self.run_id)
            for callback in self.callbacks:
                callback(result, self.mlflow, self.run_id)

class EvalExp:
    DEFAULT_OUT = "~/.fara_eval"
    
    def __init__(self, ws = None, user = None, seed = None, max_n_images = 5, save_task_csv = False):
        self.ws = ws or dummy_workspace.Workspace()
        self.experiment_name = 'osagent_eval'
        self.user = user or getpass.getuser().split('@')[0]
        self.seed = seed
        self.max_n_images = max_n_images
        self.save_task_csv = save_task_csv

    def _clean_mlflow_key(self, key: str) -> str:
        """Clean MLflow keys by replacing unsupported characters, Names may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), colon(:) and slashes (/)."""
        return ''.join(c if c.isalnum() or c in ['_', '-', '.', ' ', ':', '/'] else '_' for c in key)[:250]

    def run(self, model_ref, system, benchmark, out_url, subsample = 1.0, redo_eval = False, run_id = '0', split = None, processes = -1, callbacks = None,  eval_only = False, max_error_task_retries = 0):
        # out_az = AzFolder.from_uri(out_url)
        out_context = Path(out_url).expanduser()
        model_ref.log_2_mlflow()
        # with out_az.mount(readonly = False) as out_context, \
        with AzVllm(model_ref.model_url_to_start, model_ref.model_port, model_ref.device_id, model_ref.max_n_images, model_ref.dtype, model_ref.enforce_eager, model_ref.use_external_endpoint) as vllm:
            mlflow.log_param('benchmark', benchmark.name)
            cmd = ' '.join(sys.argv)
            try:
                mlflow.log_param('cmd', cmd)
            except mlflow.exceptions.MlflowException as e:
                log_long_command_as_params(cmd)
            mlflow.log_param('out', out_context)
            mlflow.log_param('out_url', out_url)
            mlflow.log_param('dtype', model_ref.dtype)
            mlflow.log_param('enforce_eager', model_ref.enforce_eager)
            
            # Log fn_call_template mapping if the system has this attribute
            if hasattr(system, 'fn_call_template'):
                log_fn_call_template_as_tag(system.fn_call_template)            

            benchmark.download_dataset()
            benchmark.load_dataset()

            examples = benchmark.get_split_examples(split)

            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
                
            if 0.0 < subsample < 1.0:
                examples = random.sample(examples, int(len(examples) * subsample))

            mlflow.log_param("total examples", len(examples))
            original_run_id = run_id
            run_id = f'runs/{system.hash()}/{model_ref.model_prefix}/{self.user}/{benchmark.exec_hash()}/{run_id or 0}'

            mlflow.log_param('run_id', run_id)
            output_folder = out_context / run_id
            (output_folder / benchmark.eval_hash()).mkdir(parents=True, exist_ok=True)
            (output_folder / 'traj').mkdir(parents=True, exist_ok=True)
            callback = Callback(callbacks = callbacks or [])
            results = run_eval_multiple_examples_with_progress(examples, processes, output_folder, redo_eval, system, benchmark, callback, eval_only,  max_error_task_retries)
            callback.mlflow.flush(force=True)
            metrics = reduce_eval_results(results, benchmark)
            with open(output_folder / benchmark.eval_hash() / 'metrics.json', 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            # renew_mlflow_token() 
            mlflow_compat_metrics = {k: v for k, v in (metrics or {}).items() if isinstance(v, (int, float, str))}        
            mlflow.log_metrics(mlflow_compat_metrics) 

            ### Post-evaluation error analysis of each trajectory's core.log file; output a table of the most frequently-encountered errors
            folders = map_folders(output_folder / 'traj', lambda traj: {'name': traj, 'files': list(traj.iterdir())})
            dataframe = aggregate_post_eval_errors(folders)
            print(dataframe)

            ### log each row of the table individually to mlflow, max 255 characters
            mlflow.log_metrics({self._clean_mlflow_key(row["Error Type"][:254]): row["Count"] for ind_x, row in dataframe.iterrows()})

            ### Count web_surfer.log entries and calculate step statistics
            step_stats = count_web_surfer_log_entries(folders)
            if step_stats is not None:
                # Log aborted statistics
                if step_stats.get('aborted'):
                    mlflow.log_metric("sstats_aborted_avg_steps", step_stats['aborted']['avg_steps'])
                    mlflow.log_metric("sstats_aborted_min_steps", step_stats['aborted']['min_steps'])
                    mlflow.log_metric("sstats_aborted_max_steps", step_stats['aborted']['max_steps'])
                    mlflow.log_metric("sstats_aborted_median_steps", step_stats['aborted']['median_steps'])
                    mlflow.log_metric("sstats_aborted_count_single_step_traj", step_stats['aborted']['count_one_step'])
                    mlflow.log_metric("sstats_aborted_count_max_steps_traj", step_stats['aborted']['count_max_steps'])
                    mlflow.log_metric("sstats_aborted_num_trajs", step_stats['aborted']['total_trajectories'])

                # Log not aborted statistics
                if step_stats.get('not_aborted'):
                    mlflow.log_metric("sstats_avg_steps", step_stats['not_aborted']['avg_steps'])
                    mlflow.log_metric("sstats_min_steps", step_stats['not_aborted']['min_steps'])
                    mlflow.log_metric("sstats_max_steps", step_stats['not_aborted']['max_steps'])
                    mlflow.log_metric("sstats_median_steps", step_stats['not_aborted']['median_steps'])
                    mlflow.log_metric("sstats_count_single_step_traj", step_stats['not_aborted']['count_one_step'])
                    mlflow.log_metric("sstats_count_max_steps_traj", step_stats['not_aborted']['count_max_steps'])
                    mlflow.log_metric("sstats_num_trajs", step_stats['not_aborted']['total_trajectories'])

            else:
                print("No web_surfer.log files found")

            if self.save_task_csv:
                # Save CSV with question, final answer, refusal status, first action
                refusal_map = {}
                has_refusal_data = 'examples_refusal_info' in metrics and metrics['examples_refusal_info']
                if has_refusal_data:
                    refusal_map = {qid: bool(refused) for qid, refused in metrics['examples_refusal_info']}

                # Build a map from task_id to question text from benchmark examples
                question_map = {}
                if hasattr(benchmark, 'examples'):
                    for example in benchmark.examples:
                        task_id = example.get('id', '')
                        question_text = example.get('question', '')
                        question_map[task_id] = question_text

                # Load trajectories to extract first action
                traj_folder = output_folder / 'traj'
                traj_map = {}
                if traj_folder.exists():
                    for traj_path in traj_folder.iterdir():
                        if traj_path.is_dir():
                            traj = Trajectory.from_folder(traj_path, gpt_solver=False)
                            if traj:
                                traj_map[traj_path.name] = traj

                csv_data = []
                for result in results:
                    # Get the actual question text from benchmark, fallback to qid
                    task_id = result.qid
                    question = question_map.get(result.qid, result.qid)
                    final_answer = result.answer if result.answer else ""
                    score = result.score if result.score is not None else ""
                    if has_refusal_data:
                        if result.qid not in refusal_map:
                            # This should not happen - log a warning
                            print(f"Warning: No refusal data found for qid={result.qid}")
                            refused = False
                        else:
                            refused = refusal_map[result.qid]
                    else:
                        refused = False

                    # Get first action and trace from trajectory
                    first_action = ""
                    trace = ""
                    traj = traj_map.get(result.qid)
                    if traj and len(traj.actions) > 0:
                        first_action = traj.actions[0]
                        # Concatenate all thoughts and actions with a separator
                        for i in range(len(traj.actions)):
                            trace += f"{traj.thoughts[i]} {traj.actions[i]} || "
                        trace = trace[:-4]  # Remove trailing separator

                    csv_data.append({
                        'task_id': task_id,
                        'question': question,
                        'final_answer': final_answer,
                        'score': score,
                        'refused': refused,
                        'first_action': first_action,
                        'trace': trace
                    })

                csv_path = output_folder / benchmark.eval_hash() / 'task_answers.csv'
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['task_id', 'question', 'final_answer', 'score', 'refused', 'first_action', 'trace']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_data)

    def start_run(self):
        return self.ws.start_run(self.experiment_name)      

def get_fn_call_template_mapping(template_name):
    """Get the actual template string from the template name using NousFnCallPrompt.
    
    Maps template names to their actual template strings from fncall_prompt.py
    """
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../agento/src'))
    
    try:
        from agento.agents.orca_web_surfer.qwen.fncall_prompt import NousFnCallPrompt
        
        # Instantiate the prompt class with the given template name
        prompt_instance = NousFnCallPrompt(template_name)
        
        # Get the actual template string from the instance
        template_string = prompt_instance.template_map.get(template_name, f"Unknown template: {template_name}")
        
        return template_string
    except ImportError as e:
        return f"Error importing NousFnCallPrompt: {e}"
    except ValueError as e:
        return f"Error creating NousFnCallPrompt: {e}"


def log_fn_call_template_as_tag(template_name, max_tag_length=5000):
    """Log the fn_call_template mapping as an MLflow tag."""
    template_string = get_fn_call_template_mapping(template_name)
    
    # MLflow tags have a length limit, so truncate if necessary
    if len(template_string) > max_tag_length:
        template_string = template_string[:max_tag_length-3] + "..."
    
    mlflow.set_tag('fn_call_template_string', template_string)


def log_long_command_as_params(cmd_string, max_param_length=450):  # Conservative limit
    if len(cmd_string) <= max_param_length:
        mlflow.log_param('cmd', cmd_string)
    else:
        # Split into chunks
        chunks = []
        start = 0
        while start < len(cmd_string):
            end = start + max_param_length
            chunks.append(cmd_string[start:end])
            start = end
        
        mlflow.log_param('cmd_total_parts', len(chunks))
        mlflow.log_param('cmd_full_length', len(cmd_string))
        
        for i, chunk in enumerate(chunks):
            mlflow.log_param(f'cmd_part_{i:02d}', chunk)        



