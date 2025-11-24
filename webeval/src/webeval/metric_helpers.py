from .trajectory import FinalAnswer
from typing import Dict, List


def calc_step_budget_scores(answer: FinalAnswer, score: float, budget_vals: List[int]) -> Dict[str, float]:
    budget_scores = {}
    for t in budget_vals:
        if len(answer.screenshots) - 1 <= t:
            budget_scores[str(t)] = score
        else:
            budget_scores[str(t)] = 0
    return budget_scores
