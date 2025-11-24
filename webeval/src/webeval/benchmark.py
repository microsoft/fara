from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
import importlib


class Benchmark:
    """
    Base benchmark class. Responsible for:
      - Downloading/loading data into data/<benchmark_name>/.
      - Storing examples in self.examples (each has a unique "id").
      - Evaluating a single question-answer pair and computing aggregate metrics.

    Attributes:
        name (str): The name of the benchmark.
        data_dir (str): The directory where the dataset is stored.
        examples (list): A list of examples, where each example is a dictionary with the following keys:
            - "id": Unique identifier for the example.
            - "question": The question or task.
            - "ground_truth": The correct answer or expected output.
            - "set": The dataset split to which the example belongs (e.g., 'dev' or 'test').
            - [...additional keys as needed].
    """

    def __init__(self, name: str, data_dir: str, examples: List[Dict[str, Any]] = None):
        self.name = name
        self.data_dir = data_dir
        self.examples = examples if examples is not None else []

    def download_dataset(self) -> None:
        """
        Download or load the dataset into data/<benchmark_name>/.
        """
        raise NotImplementedError("Implement dataset downloading/loading.")

    def load_dataset(self) -> None:
        """
        Loads the previously downloaded dataset from self.data_dir
        into self.examples.
        """
        raise NotImplementedError("Implement dataset loading from disk.")

    def evaluate_example(self, task_data: Dict[str, Any], candidate: Any) -> Any:
        """
        Evaluate a single example
        """
        return self.evaluator(task_data, candidate)

    def evaluator(self, task_data: Dict[str, Any], candidate: Any) -> Any:
        """
        Implement the evaluator logic here
        """
        raise NotImplementedError("Implement the evaluator logic here.")

    def compute_aggregate_metrics(self, scores: List[Any]) -> Dict[str, Any]:
        """
        Compute aggregate metrics (e.g. average score).
        Must produce a dictionary.

        You can override this method to compute custom metrics.
        """
        scores = [score.score for score in scores]
        if not scores:
            return {"mean_score": 0.0, "max_score": 0.0, "num_examples": 0}
        
        if isinstance(scores[0], dict):
            score_sums = {}
            score_counts = {}
            for score_dict in scores:
                for key, value in score_dict.items():
                    if key not in score_sums:
                        score_sums[key] = 0
                        score_counts[key] = 0
                    score_sums[key] += value
                    score_counts[key] += 1
            mean_scores = {f"mean_score_{key}": (score_sums[key] / score_counts[key]) if score_counts[key] > 0 else 0.0 for key in score_sums}
            max_scores = {f"max_score_{key}": max(score_dict[key] for score_dict in scores) for key in score_sums}
            return {**mean_scores, **max_scores, "num_examples": len(scores)}
        else:
            mean_score = sum(scores) / len(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0
            return {"mean_score": mean_score, "max_score": max_score, "num_examples": len(scores)}

    def compute_aggregate_metrics_multiple_runs(self, all_scores: List[Dict[str, Any]], all_durations: List[float]) -> Dict[str, Any]:
        """
        Compute aggregate metrics for multiple runs.
        """
        if not all_scores:
            return {"mean_score": 0.0, "max_score": 0.0, "num_examples": 0, "average_time": -1}

        if isinstance(all_scores[0], dict):
            score_sums = {}
            score_counts = {}
            for score_dict in all_scores:
                for key, value in score_dict.items():
                    if key not in score_sums:
                        score_sums[key] = 0
                        score_counts[key] = 0
                    score_sums[key] += value
                    score_counts[key] += 1
            mean_scores = {f"mean_score_{key}": (score_sums[key] / score_counts[key]) if score_counts[key] > 0 else 0.0 for key in score_sums}
            max_scores = {f"max_score_{key}": max(score_dict[key] for score_dict in all_scores) for key in score_sums}
            avg_time = sum(all_durations) / len(all_durations) if all_durations else -1
            return {**mean_scores, **max_scores, "num_examples": len(all_scores), "average_time": avg_time}
        else:
            mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            max_score = max(all_scores) if all_scores else 0.0
            avg_time = sum(all_durations) / len(all_durations) if all_durations else -1
            return {"mean_score": mean_score, "max_score": max_score, "num_examples": len(all_scores), "average_time": avg_time}

    def compute_aggregate_step_budget_metrics(self, scores: List[Any]) -> Dict[str, Any]:
        num_examples = Counter()
        aggregated_scores = Counter()
        max_scores = defaultdict(float)

        for score in scores:
            step_budget_scores = score.step_budget_scores
            for budget, val in step_budget_scores.items():
                num_examples[budget] += 1
                aggregated_scores[budget] += val
                max_scores[budget] = max(max_scores[budget], val)

        result = {
            budget: {
                "mean_score": (aggregated_scores[budget] / float(num_examples[budget])) if num_examples[budget] > 0 else 0.0,
                "max_score": max_scores[budget],
                "num_examples": n_ex,
            } for budget, n_ex in num_examples.items()
        }
        return result

    def exec_hash(self) -> str:
        """
        Returns a string hash of the benchmark execution configuration suitable to be a path.
        This is used to route results with different parameters to different locations.
        """
        return f"{self.name}"
    
    def eval_hash(self) -> str:
        """
        Returns a string hash of the benchmark evaluation configuration suitable to be a path.
        This is used to route results with different parameters to different locations.
        """
        raise NotImplementedError("Implement eval_hash to return a unique identifier for the evaluation configuration.")


def load_benchmark_class(benchmark_name: str) -> Any:
    """
    Dynamically load a benchmark class based on the benchmark name.
    """
    module_name = f"webeval.benchmarks"
    class_name = f"{benchmark_name}Benchmark"
    module = importlib.import_module(module_name)
    benchmark_class = getattr(module, class_name)
    return benchmark_class

