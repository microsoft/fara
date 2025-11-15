from typing import Optional, Dict
from dataclasses import dataclass, asdict
from enum import IntEnum
import json


class Stage(IntEnum):
    INIT = 0
    EXECUTED = 1
    EVALUATED = 2


@dataclass
class EvalResult:
    qid: str
    score: Optional[float] = None
    duration: Optional[float] = None
    stage: Stage = Stage.INIT
    reasoning: Optional[str] = None
    answer: Optional[str] = None
    step_budget_scores: Optional[dict] = None

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict())
