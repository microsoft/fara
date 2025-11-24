import os
import logging
import multiprocessing
import time
import json
import random
import numpy as np
from .benchmark import load_benchmark_class
from .basesystem import load_system_class
from .eval_result import EvalResult, Stage
from .metric_helpers import calc_step_budget_scores
import rich
import shutil

# ----------------------------------------------------------------------
# Setup Logging
# ----------------------------------------------------------------------
# Configure logging to only write to stdout.txt file, not to actual stdout
# TODO: remove global object
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    filename="stdout.txt",
    filemode="a",
    force=True  # Override any existing configuration
)
logger = logging.getLogger(__name__)
logging.getLogger("autogen_core").setLevel(logging.CRITICAL)

# Global variables for multiprocessing
_GLOBAL_SYSTEM = None
_GLOBAL_BENCHMARK = None

# ----------------------------------------------------------------------
# Helper for parallel tasks in run_benchmark
# ----------------------------------------------------------------------
def run_single_task(question_data: tuple, _logger = None) -> tuple:
    """
    Run a single task in a separate process.

    Args:
        question_data (tuple): (question_id, example_data, output_dir).

    Returns:
        tuple: (question_id, answer, duration).
    """
    global _GLOBAL_SYSTEM
    question_id, example_data, output_dir = question_data
    question_dir = os.path.join(output_dir, str(question_id))
    os.makedirs(question_dir, exist_ok=True)
    _logger = _logger or logger
    _logger.info(f"[Execution {question_id}] Start")  
    try:
        existing_answer = _GLOBAL_SYSTEM.load_answer_from_disk(
            question_id, question_dir
        )
        if existing_answer and not existing_answer.is_aborted:
            times_path = os.path.join(question_dir, "times.json")
            if os.path.exists(times_path):
                with open(times_path, "r") as f:
                    times_data = json.load(f)
                    _logger.info(f"[Execution {question_id}] Skipping (already has answer).")
                    return (
                        question_id,
                        existing_answer,
                        times_data.get("duration", 0),
                    )
            else:
                _logger.warning(f"[Execution {question_id}] Times file not found. Reexecuting...")
        else:
            _logger.info(f"[Execution {question_id}] Trajectory is not complete. Reexecuting...")
    except Exception as e:
        _logger.error(f"[Execution {question_id}] Error loading existing answer: {e}", exc_info=True)

    for elem in os.listdir(question_dir):
        elem_path = os.path.join(question_dir, elem)
        if os.path.isdir(elem_path):
            shutil.rmtree(elem_path, ignore_errors=False) # TODO: proper folder cleanup
        else:
            if elem != 'core.log':
                os.remove(elem_path)

    try:
        start_time = time.time()
        answer = _GLOBAL_SYSTEM.get_answer(question_id, example_data, question_dir, _logger)
        end_time = time.time()
        times_path = os.path.join(question_dir, "times.json")
        with open(times_path, "w") as f:
            json.dump(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                },
                f,
            )
        _logger.info(f"[Execution {question_id}] Completed")
        return question_id, answer, end_time - start_time
    except Exception as e:
        _logger.error(f"[Execution {question_id}] Error running task: {e}", exc_info=True)
        return question_id, None, 0


# ----------------------------------------------------------------------
# Core Functions
# ----------------------------------------------------------------------
def load_or_download_benchmark(
    benchmark_name: str, benchmark_dir: str, benchmark_obj=None
) -> object:
    """
    Load or download the benchmark data.

    Args:
        benchmark_name (str): Benchmark name.
        benchmark_dir (str): Directory for benchmark data.
        benchmark_obj (object, optional): Existing benchmark object.

    Returns:
        object: Benchmark object.
    """
    if benchmark_obj is not None:
        return benchmark_obj
    benchmark_class = load_benchmark_class(benchmark_name)
    data_dir = os.path.join(benchmark_dir, "data", benchmark_name)
    if not os.path.exists(data_dir):
        logger.warning(f"Benchmark data not found in {data_dir}. Downloading...")
        os.makedirs(data_dir, exist_ok=True)
        benchmark = benchmark_class(data_dir=data_dir)
        logger.info(f"Downloading benchmark {benchmark_name} into {data_dir}...")
        benchmark.download_dataset()  # Not implemented in skeleton
        logger.info("Download complete.")
    else:
        benchmark = benchmark_class(data_dir=data_dir)
    benchmark.load_dataset()  # Not implemented, but assume it populates benchmark.examples
    return benchmark


def init_worker(system_name, system_obj=None):
    """Initialize worker process with global system"""
    global _GLOBAL_SYSTEM
    if system_obj is not None:
        _GLOBAL_SYSTEM = system_obj
    else:
        system_class = load_system_class(system_name)
        _GLOBAL_SYSTEM = system_class(system_name)


def run_benchmark_func(
    benchmark_name: str,
    system_name: str,
    parallel: int,
    benchmark_dir: str,
    runs_dir: str,
    split: str = None,
    run_id: int = 0,
    benchmark_obj=None,
    system_obj=None,
    subsample: float = None,
    seed: int = 42,
) -> None:
    """
    Run the benchmark.

    Args:
        benchmark_name (str): Benchmark name.
        system_name (str): System name.
        parallel (int): Number of parallel processes.
        benchmark_dir (str): Directory for benchmark data.
        runs_dir (str): Directory for run data.
        split (str, optional): Data split to use.
        run_id (int, optional): Run ID.
        benchmark_obj (object, optional): Existing benchmark object.
        system_obj (object, optional): Existing system object.
        subsample (float, optional): Fraction of examples to subsample.
        seed (int, optional): Seed for random sampling.

    Returns:
        None
    """
    if subsample is not None and not (0 < subsample <= 1):
        raise ValueError("subsample must be in the range (0, 1].")
    if seed is not None:
        random.seed(seed)

    benchmark = load_or_download_benchmark(benchmark_name, benchmark_dir, benchmark_obj)

    output_dir = os.path.join(
        runs_dir,
        "runs",
        system_name,
        benchmark_name,
        split or "all_benchmark",
        str(run_id),
    )
    os.makedirs(output_dir, exist_ok=True)

    exs = (
        benchmark.get_split_examples(split) if split else benchmark.examples
    )  # Use split examples if specified, otherwise use all examples
    if subsample and 0 < subsample <= 1:
        exs = random.sample(exs, int(len(exs) * subsample))
    
    # Create tasks without system reference
    tasks = [(ex["id"], ex, output_dir) for ex in exs]

    logger.info(f"Starting run_benchmark with {parallel} processes...")
    
    # Use a single system instance in the main process if parallel=1
    if parallel == 1:
        if system_obj is not None:
            system = system_obj
        else:
            system_class = load_system_class(system_name)
            system = system_class(system_name)
        
        # Set global system for the main process
        global _GLOBAL_SYSTEM
        _GLOBAL_SYSTEM = system
        # Process tasks sequentially
        results = [run_single_task(task) for task in tasks]
    else:
        # Use multiprocessing with initialization
        with multiprocessing.Pool(
            processes=parallel, 
            initializer=init_worker,
            initargs=(system_name, system_obj)
        ) as pool:
            results = pool.map(run_single_task, tasks)

    success_count = sum(1 for _, answer, _ in results if answer is not None)
    total_time = sum(t for _, a, t in results if a is not None)
    avg_time = total_time / success_count if success_count else 0
    logger.info(f"Average time per successful task: {avg_time:.4f} seconds")

    fail_count = len(results) - success_count

    logger.info(f"Run completed: {success_count} succeeded, {fail_count} failed.")


def evaluate_single_example(example_data, _logger = None) -> EvalResult:
    """
    Evaluate a single example.

    Args:
        example_data (tuple): (ex, output_dir, redo_eval).

    Returns:
        tuple: (qid, score, duration).
    """
    global _GLOBAL_SYSTEM, _GLOBAL_BENCHMARK
    ex, output_dir, redo_eval = example_data
    qid = ex["id"]
    question_dir = os.path.join(output_dir, str(qid))
    times_path = os.path.join(question_dir, "times.json")
    scores_dir = os.path.join(question_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    _logger = _logger or logger
    _logger.info(f'[Evaluation {qid}] Start')
    score_path = os.path.join(scores_dir, f"{_GLOBAL_BENCHMARK.eval_hash()}.json")
    reasoning = None
    answer = None
    if os.path.exists(times_path):
        with open(times_path, "r") as f:
            times_data = json.load(f)
            duration = times_data.get("duration", 0)
    else:
        duration = 0

    if not redo_eval and os.path.exists(score_path):
        try:
            with open(score_path, "r") as f:
                json_data = json.load(f)
                saved_score = json_data.get("score")
                reasoning = json_data.get("gpt_response_text", None)
                # load answer from answer.json
                candidate = _GLOBAL_SYSTEM.load_answer_from_disk(qid, question_dir)
                answer = candidate.answer.final_answer
                if json_data.get( "step_budget_scores"):
                    step_budget_scores = json_data["step_budget_scores"]
                else:
                    if not hasattr(_GLOBAL_SYSTEM, 'step_budgets') or _GLOBAL_SYSTEM.step_budgets is None:
                        step_budget_scores = {}
                    else:
                        step_budget_scores = calc_step_budget_scores(candidate.answer, saved_score, _GLOBAL_SYSTEM.step_budgets)
            _logger.info(f"[Evaluation {qid}] Loaded existing score.")
            return EvalResult(
                qid = qid,
                score = saved_score,
                duration = duration,
                stage = Stage.EVALUATED,
                reasoning = reasoning,
                answer = answer,
                step_budget_scores = step_budget_scores
            )
        except Exception as e:
            _logger.error(f"[Evaluation {qid}] Error loading existing score: {e}")
            return EvalResult(
                qid = qid,
                score = None,
                duration = duration,
                stage = Stage.EXECUTED,
                step_budget_scores = None
            )

    try:
        candidate = _GLOBAL_SYSTEM.load_answer_from_disk(qid, question_dir)
    except Exception as e:
        _logger.error(f"[Evaluation {qid}] Error loading execution result: {e}", exc_info=True)
        raise e
    if (candidate is None) or (candidate.is_aborted):
        _logger.warning(f"[Evaluation {qid}] Cannot load execution result")
        return EvalResult(qid = qid, score = None, duration = duration, step_budget_scores = None)
    try:
        answer = candidate.answer.final_answer
        evaluate_output = _GLOBAL_BENCHMARK.evaluate_example(ex, candidate)
    except Exception as e:
        _logger.error(f"[Evaluation {qid}] Error evaluating example: {e}", exc_info=True)
        return EvalResult(qid = qid, score = None, duration = duration, stage = Stage.EXECUTED, step_budget_scores = None)
    if isinstance(evaluate_output, tuple):
        score, reasoning = evaluate_output
    else:
        score = evaluate_output
        reasoning = None

    # only do if step_budgets is defined for GLOBAL_SYSTEM
    if not hasattr(_GLOBAL_SYSTEM, 'step_budgets') or _GLOBAL_SYSTEM.step_budgets is None:
        step_budget_scores = {}
    else:
        step_budget_scores = calc_step_budget_scores(candidate.answer, score, _GLOBAL_SYSTEM.step_budgets)
    with open(score_path, "w") as f:
        json.dump({"score": score, "gpt_response_text": reasoning}, f)
    _logger.info(f"[Evaluation {qid}] Completed: score={score}, duration={duration}")
    return EvalResult(
        qid = qid,
        score = score,
        duration = duration,
        stage = Stage.EVALUATED,
        reasoning = reasoning,
        answer = answer,
        step_budget_scores = step_budget_scores
    )


def init_eval_worker(system_name, benchmark, system_obj=None):
    """Initialize worker process with global system and benchmark"""
    global _GLOBAL_SYSTEM, _GLOBAL_BENCHMARK
    _GLOBAL_BENCHMARK = benchmark
    if system_obj is not None:
        _GLOBAL_SYSTEM = system_obj
    else:
        system_class = load_system_class(system_name)
        _GLOBAL_SYSTEM = system_class(system_name)


def evaluate_benchmark_func(
    benchmark_name: str,
    system_name: str,
    benchmark_dir: str,
    runs_dir: str,
    split: str = None,
    run_id: int or list = 0,
    benchmark_obj=None,
    system_obj=None,
    parallel: int = 1,
    redo_eval: bool = False,
    run_path: str = None,
) -> None:
    """
    Evaluate the benchmark.

    Args:
        benchmark_name (str): Benchmark name.
        system_name (str): System name.
        benchmark_dir (str): Directory for benchmark data.
        runs_dir (str): Directory for run data.
        split (str, optional): Data split to use.
        run_id (int or list, optional): Run ID or list of run IDs.
        benchmark_obj (object, optional): Existing benchmark object.
        system_obj (object, optional): Existing system object.
        parallel (int, optional): Number of parallel processes.
        redo_eval (bool, optional): Redo evaluation.
        run_path (str, optional): If provided, use this as the run directory instead of constructing from runs_dir, system_name, etc.

    Returns:
        None
    """
    if isinstance(run_id, int):
        run_ids = [run_id]
    else:
        run_ids = run_id

    all_scores = []
    all_durations = []

    if benchmark_obj is not None:
        benchmark = benchmark_obj
    else:
        benchmark = load_or_download_benchmark(
            benchmark_name, benchmark_dir, benchmark_obj
        )

    for idx, rid in enumerate(run_ids):
        if run_path is not None:
            output_dir = run_path if len(run_ids) == 1 else os.path.join(run_path, str(rid))
        else:
            output_dir = os.path.join(
                runs_dir,
                "runs",
                system_name,
                benchmark_name,
                split or "all_benchmark",
                str(rid),
            )
        if not os.path.exists(output_dir):
            raise FileNotFoundError(
                f"No system output found at {output_dir}. Run the benchmark first."
            )

        exs = (
            benchmark.get_split_examples(split) if split else benchmark.examples
        )  # Use split examples if specified, otherwise use all examples
        
        # Create tasks without system reference
        tasks = [(ex, output_dir, redo_eval) for ex in exs]
        
        # Process tasks based on parallelism
        if parallel == 1:
            # Set up globals for the main process
            global _GLOBAL_SYSTEM, _GLOBAL_BENCHMARK
            _GLOBAL_BENCHMARK = benchmark
            if system_obj is not None:
                _GLOBAL_SYSTEM = system_obj
            else:
                system_class = load_system_class(system_name)
                _GLOBAL_SYSTEM = system_class(system_name)
                
            # Process sequentially
            single_results = [evaluate_single_example(task) for task in tasks]
        else:
            # Use multiprocessing with initialization
            with multiprocessing.Pool(
                processes=parallel, 
                initializer=init_eval_worker,
                initargs=(system_name, benchmark, system_obj)
            ) as pool:
                single_results = pool.map(evaluate_single_example, tasks)
                
        # breakpoint()
        metrics = reduce_eval_results(single_results, benchmark)
        logger.info(f"Average time across evaluated tasks: {metrics['average_time']:.4f} s")
        logger.info(f"Evaluation metrics: {metrics}")

        # Save metrics to a file
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        logger.info(f"Metrics saved to {metrics_path}")

        all_scores.extend(metrics['scores'])
        all_durations.extend(metrics['durations'])

    if len(run_ids) > 1:
        aggregate_metrics = benchmark.compute_aggregate_metrics_multiple_runs(all_scores, all_durations)

        # Save aggregate metrics to a file
        if run_path is not None:
            aggregate_metrics_path = os.path.join(run_path, "aggregate_metrics.json")
        else:
            aggregate_metrics_path = os.path.join(
                runs_dir,
                "runs",
                system_name,
                benchmark_name,
                split or "all_benchmark",
                "aggregate_metrics.json",
            )
        with open(aggregate_metrics_path, "w") as f:
            json.dump(aggregate_metrics, f)
        logger.info(f"Aggregate metrics saved to {aggregate_metrics_path}")
        return aggregate_metrics
    return metrics


def reduce_eval_results(eval_results, benchmark):    
    results_executed = [res for res in eval_results if res.stage == Stage.EXECUTED]
    results_evaluated = [res for res in eval_results if res.stage == Stage.EVALUATED]
    scores = results_evaluated

    durations = [res.duration for res in eval_results]
    avg_time = float(np.mean(durations or [-1]))
    metrics = benchmark.compute_aggregate_metrics(scores)
    metrics["step_budget_scores"] = benchmark.compute_aggregate_step_budget_metrics(scores)
    metrics["average_time"] = avg_time
    metrics['durations'] = durations
    metrics["scores"] = [res.to_json() for res in scores]
    metrics["total examples"] = len(eval_results)
    metrics["executed examples"] = len(results_executed) + len(results_evaluated)
    return metrics


def run_evaluate_benchmark_func(
    benchmark_name: str,
    system_name: str,
    parallel: int,
    benchmark_dir: str,
    runs_dir: str,
    split: str = None,
    run_id: int or list = 0,
    benchmark_obj=None,
    system_obj=None,
    subsample: float = None,
    seed: int = None,
    redo_eval: bool = False,
) -> None:
    """
    Run and evaluate the benchmark.

    Args:
        benchmark_name (str): Benchmark name.
        system_name (str): System name.
        parallel (int): Number of parallel processes.
        benchmark_dir (str): Directory for benchmark data.
        runs_dir (str): Directory for run data.
        split (str, optional): Data split to use.
        run_id (int or list, optional): Run ID or list of run IDs.
        benchmark_obj (object, optional): Existing benchmark object.
        system_obj (object, optional): Existing system object.
        subsample (float, optional): Fraction of examples to subsample.
        seed (int, optional): Seed for random sampling.
        redo_eval (bool, optional): Redo evaluation.
    Returns:
        None
    """
    if not isinstance(run_id, list):
        run_ids = [run_id]
    else:
        run_ids = run_id

    for rid in run_ids:
        run_benchmark_func(
            benchmark_name,
            system_name,
            parallel,
            benchmark_dir,
            runs_dir,
            split,
            rid,
            benchmark_obj,
            system_obj,
            subsample=subsample,
            seed=seed,
        )
    return evaluate_benchmark_func(
        benchmark_name,
        system_name,
        benchmark_dir,
        runs_dir,
        split,
        run_ids,
        benchmark_obj,
        system_obj,
        parallel,
        redo_eval,
    )

def run_eval_single_example(example, folder, redo_eval, progress, task_id, eval_only=False, max_error_task_retries=0, callback=None):
    """
    Run evaluation for a single example with retry logic.

    Args:
        example (dict): Example data.
        folder (str): Output folder.
        redo_eval (bool): Redo evaluation.
        progress (dict): Progress tracking dict.
        task_id: Task ID for progress tracking.
        max_error_task_retries (int, optional): Maximum number of retries for failed/aborted tasks.
        eval_only (bool, optional): Skip execution and only evaluate existing results.
        callback (callable, optional): Callback to report results. If None, returns list of all results for multiprocessing.

    Returns:
        EvalResult or list[EvalResult]: Single result if callback provided, otherwise list of all attempts.
    """
    if eval_only:
        progress[task_id] = {'progress': 0, 'total': 1}
    else:
        progress[task_id] = {'progress': 0, 'total': 2}

    task = (example["id"], example, folder)
    _logger = logger.getChild(str(example["id"]))
    question_dir = folder / str(example["id"])
    question_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(question_dir / "core.log", mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    _logger.addHandler(handler)

    # Collect all results for multiprocessing mode (when callback=None)
    all_results = []

    # Retry logic for failed/aborted tasks
    retry_count = 0
    while retry_count <= max_error_task_retries:
        if retry_count > 0:
            _logger.info(f"[Execution {example['id']}] Retry attempt {retry_count}/{max_error_task_retries}")

        # Skip execution if eval_only is True
        if not eval_only:
            run_single_task(task, _logger)
            progress[task_id] = {'progress': 1, 'total': 2}

        example_data = (example, folder, redo_eval)
        result = evaluate_single_example(example_data, _logger)

        if eval_only:
            progress[task_id] = {'progress': 1, 'total': 1}
            return result


        # Check if the task succeeded (has EVALUATED stage with score)
        if result.stage == Stage.EVALUATED:
            # Task succeeded with evaluation
            progress[task_id] = {'progress': 2, 'total': 2}
            if callback:
                # Single-process mode: report immediately and return single result
                callback(result)
                return result
            else:
                # Multiprocessing mode: collect and return all results (including any prior failures)
                all_results.append(result)
                return all_results if len(all_results) > 1 else result

        # Task failed or aborted
        if retry_count < max_error_task_retries:
            _logger.warning(f"[Execution {example['id']}] Task failed, aborted, or not evaluated (stage={result.stage}). Retrying...")
            if callback:
                # Single-process mode: report intermediate failure immediately
                callback(result)
            else:
                # Multiprocessing mode: collect for later reporting
                all_results.append(result)
            # Reset progress for retry
            progress[task_id] = {'progress': 0, 'total': 2}
            retry_count += 1
        else:
            # Max retries reached
            _logger.error(f"[Execution {example['id']}] Max retries ({max_error_task_retries}) reached. Task failed with stage={result.stage}.")
            progress[task_id] = {'progress': 2, 'total': 2}
            if callback:
                # Single-process mode: report and return single result
                callback(result)
                return result
            else:
                # Multiprocessing mode: return all attempts
                all_results.append(result)
                return all_results if len(all_results) > 1 else result
            

def _make_callback_wrapper(callback):
    """Helper to create a callback wrapper that handles both single results and lists."""
    if not callback:
        return None

    def wrapper(result):
        if isinstance(result, list):
            # Multiple results from retries - report all
            for r in result:
                callback(r)
        else:
            # Single result - report normally
            callback(result)
    return wrapper


class _SeqPool:
    def __init__(self, processes, initializer, initargs):
        assert processes == 1, "Sequential pool should only have one process."
        initializer(*initargs)

    def apply_async(self, func, args, callback):
        callback(func(*args))

    def close(self):
        ...

    def join(self):
        ...

def run_eval_multiple_examples(examples, processes, output_folder, redo_eval, system, benchmark, callback = None, eval_only =False, max_error_task_retries=0):
    pool_class = _SeqPool if processes == 1 else multiprocessing.Pool
    pool = pool_class(processes=processes, initializer = init_eval_worker,  initargs=(None, benchmark, system))
    async_results = []

    # In single-process mode, pass callback directly; in multi-process, use wrapper with apply_async
    worker_callback = callback if processes == 1 else None
    async_callback = _make_callback_wrapper(callback) if processes > 1 else None

    for example in examples:
        if callback:
            callback.on_start(example['id'])
        async_results.append(
            pool.apply_async(
                run_eval_single_example,
                args=(example, output_folder, redo_eval, {}, 0, eval_only, max_error_task_retries, worker_callback),
                callback=async_callback))
    pool.close()
    pool.join()
    return callback.results


def _run_eval_multiple_examples_with_progress_single_proc(examples, output_folder, redo_eval, system, benchmark, callback = None, eval_only=False, max_error_task_retries=0):
    global _GLOBAL_SYSTEM, _GLOBAL_BENCHMARK
    _GLOBAL_BENCHMARK = benchmark
    _GLOBAL_SYSTEM = system
    with rich.progress.Progress() as progress:
        task = progress.add_task('Starting...', total=len(examples))
        for example in examples:
            progress.update(task, description = example['id'])
            run_eval_single_example(example, output_folder, redo_eval, {}, 0, eval_only, max_error_task_retries, callback)
            progress.update(task, advance=1)
    return callback.results


def _run_eval_multiple_examples_with_progress_multi_proc(examples, processes, output_folder, redo_eval, system, benchmark, callback = None, eval_only=False, max_error_task_retries=0):
    pool = multiprocessing.Pool(processes=processes, initializer = init_eval_worker,  initargs=(None, benchmark, system))
    async_results = []

    # Wrap callback to handle lists from retry attempts
    wrapped_callback = _make_callback_wrapper(callback)

    with rich.progress.Progress(
        "[progress.description]{task.description}",
        rich.progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        rich.progress.TimeRemainingColumn(),
        rich.progress.TimeElapsedColumn(),
        refresh_per_second=1,  # bit slower updates
    ) as progress, multiprocessing.Manager() as manager:
        _progress = manager.dict()
        overall_progress_task = progress.add_task("[green]All jobs progress:", total=len(examples))
        for example in examples:
            task_id = progress.add_task(example['id'], visible=False)
            # Pass callback=None to worker (can't pickle), use wrapped callback with apply_async
            async_results.append(
                pool.apply_async(
                    run_eval_single_example,
                    args=(example, output_folder, redo_eval, _progress, task_id, eval_only, max_error_task_retries, None),
                    callback=wrapped_callback))
        while (n_finished := sum([future.ready() for future in async_results])) < len(async_results):
            progress.update(
                overall_progress_task, completed=n_finished, total=len(async_results)
            )
            for task_id, update_data in _progress.items():
                latest = update_data["progress"]
                total = update_data["total"]
                progress.update(
                    task_id,
                    completed=latest,
                    total=total,
                    visible=latest < total,  # Always visible now
                )
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage

    pool.close()
    pool.join()
    return callback.results

def run_eval_multiple_examples_with_progress(examples, processes, output_folder, redo_eval, system, benchmark, callback = None, eval_only=False, max_error_task_retries=0):
    output_folder = output_folder / 'traj'
    output_folder.mkdir(parents=True, exist_ok=True)
    if processes == 1:
        return _run_eval_multiple_examples_with_progress_single_proc(
            examples, output_folder, redo_eval, system, benchmark, callback, eval_only, max_error_task_retries
        )
    return _run_eval_multiple_examples_with_progress_multi_proc(
        examples, processes, output_folder, redo_eval, system, benchmark, callback, eval_only, max_error_task_retries
    )