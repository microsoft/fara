
from ...benchmark import Benchmark
from typing import Optional, Dict, Any, List
from pathlib import Path
import os
from autogen_core.components.models import UserMessage, SystemMessage, LLMMessage, ChatCompletionClient
from autogen_core.components import Image as AGImage
from autogen_core.components.models import ChatCompletionClient
from webeval.benchmarks.om2w.impl.src.methods import agenttrek_eval, automomous_eval, webjudge_general_eval, webjudge_online_mind2web, webvoyager_eval
from webeval.benchmarks.om2w.impl.src.utils import extract_predication
from webeval.trajectory import Trajectory, FinalAnswer
from ...evaluators import compute_aggregate_metrics_gpt_evaluator, safe_mean, compute_refusal_metrics
import asyncio
import json

def _content_2_agento(content):
    result = []
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        if content['type'] == 'text':
            return content['text']
        elif content['type'] == 'image_url':
            return AGImage.from_uri(content['image_url']['url'])
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")
    elif isinstance(content, list):
        return [_content_2_agento(c) for c in content]
    else:
        raise ValueError(f"Unsupported content type: {type(content)}")  

def _messages_2_agento(messages):
    result = []
    for m in messages:
        if m['role'] == 'user':
            result.append(UserMessage(content=_content_2_agento(m['content']), source='user'))
        elif m['role'] == 'system':
            result.append(SystemMessage(content=_content_2_agento(m['content'])))
    return result

class _ModelWrapper:
    def __init__(self, model_client: ChatCompletionClient):
        self.model_client = model_client

    def generate(self, messages, **kwargs) -> LLMMessage:
        response = asyncio.run(self.model_client.create(_messages_2_agento(messages), **kwargs))
        return [response.content]
    
    async def agenerate(self, messages, **kwargs) -> LLMMessage:
        response = await self.model_client.create(_messages_2_agento(messages), **kwargs)
        return [response.content]

class OnlineM2WBenchmark(Benchmark):
    def __init__(self,
                data_dir: Path,
                eval_method = 'AgentTrek_eval',
                model_client: Optional[ChatCompletionClient] = None,
                data_az_url: Optional[str] = None,
                score_threshold: float = 3):
        self.eval_method = eval_method
        self.original_model_client = model_client
        self.model_client = _ModelWrapper(model_client)
        data_dir = Path(data_dir).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = data_dir / 'Online_Mind2Web_06042025.json'
        super().__init__(
            name='OnlineM2W',
            data_dir=data_dir)
        self.data_az_url = "Online_Mind2Web_06042025.json" if data_az_url is None else data_az_url
        self.score_threshold = score_threshold

    def download_dataset(self) -> None:
        print("Updated datasets for om2w are stored locally in webeval/data/om2w")
        pass

    def load_dataset(self) -> None:
        with open(self.data_file, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        self.examples = [
            {
                'id': ex['task_id'],
                'level': ex['level'],
                'question': ex['confirmed_task'],
                'web': ex['website'],
                'reference_length': ex['reference_length']
                } for ex in examples
        ]

    def evaluator(self, task_data: Dict[str, Any], candidate: Trajectory) -> Any:
        if self.eval_method == 'AgentTrek_eval':
            messages, _, __ = agenttrek_eval.AgentTrek_eval(
                task=task_data['question'],
                last_actions = candidate.actions,
                thoughts = candidate.thoughts,
                images_path = candidate.latest_screenshot
                )
        elif self.eval_method == 'Autonomous_eval':
            messages, _, __ = automomous_eval.Autonomous_eval(
                task=task_data['question'],
                last_actions = candidate.actions,
                images_path = candidate.latest_screenshot       
            )
        elif self.eval_method == "WebJudge_general_eval":
            messages, _, __, ___, ____ = asyncio.run(
                webjudge_general_eval.WebJudge_general_eval(
                    task_data['question'],
                    None,  # TODO: input image path
                    candidate.thoughts,
                    candidate.actions,
                    candidate.screenshots,
                    self.model_client,
                    self.score_threshold))
        elif self.eval_method == "WebJudge_Online_Mind2Web_eval":
            messages, _, __, ___, ____ = asyncio.run(
                webjudge_online_mind2web.WebJudge_Online_Mind2Web_eval(
                    task_data['question'],
                    candidate.actions,
                    candidate.screenshots,
                    self.model_client,
                    self.score_threshold))
        else:
            raise NotImplementedError(f"Evaluation method {self.eval_method} is not implemented.")
        verdict, gpt_response_text = asyncio.run(
            self.gpt_evaluator_async(messages)
        )
        return verdict, gpt_response_text
        
    async def gpt_evaluator_async(self, messages) -> float:
        response = await self.model_client.agenerate(messages)
        verdict = extract_predication(response[0], self.eval_method)
        return verdict, response[0]
    

    def get_split_examples(self, split: str) -> List[Dict[str, Any]]:
        exs = None
        if split == 'easy':
            return [ex for ex in self.examples if ex["level"] == "easy"]
        elif split == 'medium':
            return [ex for ex in self.examples if ex["level"] == "medium"]
        elif split == 'hard':
            return [ex for ex in self.examples if ex["level"] == "hard"]
        elif split == '*':
            return self.examples
        
        raise ValueError(f"Unsupported split: {split}")
    
    def compute_aggregate_metrics(self, results: List[Any]) -> Dict[str, float]:
        # Build a mapping from id to level for all examples
        id_to_level = {
            ex["id"]: ex["level"]
            for ex in self.examples
        }

        # Group scores by level (using all results, not filtered)
        level_to_scores = {}
        for score in results:
            level = id_to_level[score.qid]
            level_to_scores.setdefault(level, []).append(score.score)

        # Compute mean score (accuracy) for each level
        accuracy_by_level = {
            k: (safe_mean(v), len(v)) for k, v in level_to_scores.items()
        }

        # Add individual accuracy metrics for each level
        accuracy_for_key = {}
        samples_by_key = {}
        for level, level_scores in level_to_scores.items():
            accuracy_for_key[f"accuracy_{level}"] = safe_mean(level_scores)
            samples_by_key[f"samples_{level}"] = len(level_scores)

        # Compute global metrics
        metrics = compute_aggregate_metrics_gpt_evaluator(results)

        if self.original_model_client is not None:
            refusal_metrics = compute_refusal_metrics(results, self.original_model_client)
            metrics.update(refusal_metrics)

        metrics["accuracy_by_level"] = accuracy_by_level
        metrics.update(accuracy_for_key)
        metrics.update(samples_by_key)
        return metrics


    def exec_hash(self) -> str:
        return f"{super().exec_hash()}_{(self.data_az_url or '').split('/')[-1]}"
    
    def eval_hash(self) -> str:
        return f"{self.eval_method}-{self.score_threshold}"

