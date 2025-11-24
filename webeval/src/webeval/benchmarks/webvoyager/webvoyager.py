import json
import os
import base64
import asyncio
import re
from typing import List, Dict, Any
from autogen_core.components.models import UserMessage, LLMMessage, ChatCompletionClient
from autogen_core.components import Image as AGImage
from ...benchmark import Benchmark
from ...eval_result import EvalResult
from ...utils import download_file, load_jsonl, load_json
from ...evaluators import compute_aggregate_metrics_gpt_evaluator, filter_no_answer_no_error, safe_mean, compute_refusal_metrics
from webeval.trajectory import FinalAnswer, Trajectory

import openai


SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'."""
USER_PROMPT = """TASK: <task>
Result Response: <answer>
<num> screenshots at the end: """


def encode_image(image_path: str) -> str:
    """
    Encodes an image file into a base64 string.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class WebVoyagerBenchmark(Benchmark):
    """
    Loads the WebVoyager dataset, stores it locally,
    and evaluates predictions using the GAIA evaluator.
    """

    DATA_URL = "https://raw.githubusercontent.com/MinorJerry/WebVoyager/main/data/WebVoyager_data.jsonl"
    REFERENCE_URL = "https://raw.githubusercontent.com/MinorJerry/WebVoyager/main/data/reference_answer.json"
    GAIA_DATA_URL = "https://raw.githubusercontent.com/MinorJerry/WebVoyager/main/data/GAIA_web.jsonl"

    def __init__(
        self,
        data_dir: str,
        eval_method: str = "exact_match",
        model_client: ChatCompletionClient | None = None,
        data_az_folder=None,
        max_images: int = 30,
    ):
        super().__init__(
            name="WebVoyager",
            data_dir=data_dir,
        )
        if eval_method not in ["exact_match", "gpt_eval"]:
            raise ValueError("eval_method must be 'exact_match' or 'gpt_eval'")
        self.eval_method = eval_method
        if eval_method == "gpt_eval" and model_client is None:
            raise ValueError("model_client must be provided for gpt_eval")
        self.model_client = model_client
        self.data_file = os.path.join(self.data_dir, "WebVoyager_data_08312025.jsonl")
        self.reference_file = os.path.join(self.data_dir, "reference_answer.json")
        self.examples = []
        self.data_az_url = data_az_folder
        self.max_images = max_images

    def download_dataset(self) -> None:
        """
        Download the dataset files into self.data_dir.
        """
        print("Updated datasets for webvoyager are stored locally in webeval/data/webvoyager")
        pass
        # if not os.path.exists(self.data_dir):
        #     os.makedirs(self.data_dir, exist_ok=True)
        # if self.data_az_url:
        #     AzFile.from_uri(f"{self.data_az_url}/WebVoyager_data.jsonl").copy(
        #         LocalFile(self.data_file)
        #     )
        #     AzFile.from_uri(f"{self.data_az_url}/reference_answer.json").copy(
        #         LocalFile(self.reference_file)
        #     )
        # else:
        #     download_file(self.DATA_URL, self.data_file)
        #     download_file(self.REFERENCE_URL, self.reference_file)

    def load_dataset(self):
        """
        Loads the data from a JSONL file and the references from a JSON file.
        Matches entries by:
          - item["web_name"] (e.g. 'Allrecipes')
          - numeric_id from item["id"] (e.g. 'Allrecipes--0' -> 0)
        Then retrieves the reference answer from reference_data[web_name]["answers"].
        Also adds 'web' (website URL) and 'answer_type' (the type of the referenced answer).
        """        
        data = load_jsonl(self.data_file)

        reference_data = load_json(self.reference_file)

        examples = []
        for item in data:
            numeric_id_str = item["id"].split("--")[-1]
            try:
                numeric_id = int(numeric_id_str)
            except ValueError:
                numeric_id = None

            web_name = item.get("web_name", "")
            ref_answer = None
            answer_type = None

            if web_name in reference_data:
                answers_list = reference_data[web_name].get("answers", [])
                for ans_obj in answers_list:
                    if ans_obj.get("id") == numeric_id:
                        ref_answer = ans_obj.get("ans", None)
                        answer_type = ans_obj.get("type", None)
                        break

            example = {
                "id": item["id"],
                "web_name": web_name,
                "web": item.get("web", ""),
                "question": item.get("ques", ""),
                "ground_truth": ref_answer,
                "answer_type": answer_type,
                "metadata": item.get("metadata", {}),
                "set": "webvoyager",
            }
            examples.append(example)

        print(f"Loaded {len(examples)} examples for WebVoyager benchmark.")
        self.examples = examples

    def get_split_examples(self, split: str) -> List[Dict[str, Any]]:
        """
        Returns a list of examples that belong to the specified set (e.g. 'dev', 'test', or 'gaia').
        """
        if split not in ["webvoyager", "gaia"]:
            raise ValueError("split must be 'webvoyager' or 'gaia'")
        return [ex for ex in self.examples if ex["set"] == split]

    def evaluator(self, question_data: Dict[str, Any], candidate: Trajectory) -> float:
        """
        Evaluate how 'correct' the candidate answer is relative to the gold_answer.
        """
        if self.eval_method == "exact_match":
            raise NotImplementedError("Exact match evaluation is not implemented.")
            # ground_truth = question_data.get("ground_truth", "")
            # candidate_answer = candidate.answer.final_answer
            # if not candidate_answer:
            #     raise ValueError(
            #         "Candidate does not have an answer. It should be a dict with key 'answer'."
            #     )
            # return gaia_evaluator(ground_truth, candidate_answer)
        elif self.eval_method == "gpt_eval":
            verdict, gpt_response_text = asyncio.run(
                self.gpt_evaluator_async(question_data, candidate.answer)
            )
            return verdict, gpt_response_text

    async def gpt_evaluator_async(self, question_data: Dict[str, Any], candidate: FinalAnswer) -> float:
        """
        Adapted from https://github.com/MinorJerry/WebVoyager/blob/main/evaluation/auto_eval.py
        Evaluates the candidate answer by calling GPT-based auto-eval.

        Args:
            question_data: dict containing the original question, any ground-truth info, etc.
            candidate: FinalAnswer containing the predicted/produced answer and screenshots

        Returns:
            1.0 if GPT decides the result is "SUCCESS",
            0.0 if GPT decides "NOT SUCCESS",
            or 0.0 if the verdict is missing or unclear.
        """
        # Extract data
        task_content = question_data.get("question", "")
        answer_content = candidate.final_answer
        screenshot_paths = candidate.screenshots
        # Suppose we only attach up to <num> screenshots
        num_screens = len(screenshot_paths)

        # Build user content from the template
        user_prompt_tmp = USER_PROMPT.replace("<task>", task_content)
        user_prompt_tmp = user_prompt_tmp.replace("<answer>", answer_content)
        user_prompt_tmp = user_prompt_tmp.replace("<num>", str(num_screens))

        images = []
        for path in screenshot_paths:
            # from_file
            image = AGImage.from_file(path)
            images.append(image)

        # Need this to avoid tokens over limit
        if len(images) > self.max_images:
            images = images[-self.max_images:]

        # The system prompt explains how to evaluate correctness
        user_message = ""
        if len(images) > 0:
            user_message = [
                user_prompt_tmp,
            ]
            user_message.extend(images)
        else:
            user_message = user_prompt_tmp

        messages = [
            UserMessage(
                source="system",
                content=SYSTEM_PROMPT,
            ),
            UserMessage(
                source="user",
                content=user_message,
            ),
            UserMessage(
                source="user",
                content="Your verdict:\n.",
            ),
        ]

        # Now call the GPT model
        response = await self.model_client.create(messages)

        assert isinstance(response.content, str)
        gpt_response_text = response.content
        # Parse out the text from the model
        verdict = 0.0
        if "NOT SUCCESS" in gpt_response_text:
            verdict = 0.0
        elif "SUCCESS" in gpt_response_text:
            verdict = 1.0
        else:
            verdict = 0.0  # Could not decide

        return verdict, gpt_response_text

    def compute_aggregate_metrics(self, scores: List[EvalResult]) -> Dict[str, Any]:
        """
        Compute aggregate metrics for WebVoyager using the shared utility function, and also compute accuracy by web_name.
        """
        # Filter scores to exclude empty answers and errors
        filtered_scores = filter_no_answer_no_error(scores)
        # Build a mapping from id to web_name for all examples
        id_to_web_name = {
            ex["id"]: ex.get("web_name", None)
            for ex in self.examples
            if "web_name" in ex
        }
        # Group scores by web_name (filtered)
        web_name_to_scores = {}
        for score in scores:
            web_name = id_to_web_name.get(score.qid, None)
            if web_name is not None:
                web_name_to_scores.setdefault(web_name, []).append(score.score)
        # Compute mean score (accuracy) for each web_name
        accuracy_by_web_name = {
            k: (safe_mean(v), len(v)) for k, v in web_name_to_scores.items()
        }

        # Add individual accuracy metrics for each web domain
        accuracy_for_key = {}
        samples_by_key = {}
        for web_name, web_scores in web_name_to_scores.items():
            accuracy_for_key[f"accuracy_{web_name}"] = safe_mean(web_scores)
            samples_by_key[f"samples_{web_name}"] = len(web_scores)

        # Compute global metrics
        metrics = compute_aggregate_metrics_gpt_evaluator(scores)

        # Add refusal detection if model_client is available
        if self.model_client is not None:
            refusal_metrics = compute_refusal_metrics(scores, self.model_client)
            metrics.update(refusal_metrics)

        metrics["accuracy_by_web_name"] = accuracy_by_web_name
        metrics.update(accuracy_for_key)
        metrics.update(samples_by_key)
        return metrics
    
    def exec_hash(self) -> str:
        return f"{super().exec_hash()}_{(self.data_az_url or '').split('/')[-1]}"
    
    def eval_hash(self) -> str:
        return f"{self.eval_method}"

    
