"""RunContext - shared context for a single data generation run."""

from dataclasses import dataclass, field
from typing import Any, Dict, TYPE_CHECKING
from pathlib import Path
import json
import os

from .data_point import (
    Action,
    DataPoint,
    ObservationEvent,
    SolverLog,
    Task,
    VerificationResult,
)
from .data_point_io import DataPointWriter

if TYPE_CHECKING:
    from ..environments.base import Environment


@dataclass
class RunContext:
    """
    Shared context for a single data generation run.

    Contains the environment and the data point being generated.
    All agents share this context and write to data_point.
    A DataPointWriter handles incremental persistence to output_dir.
    """

    environment: "Environment"
    output_dir: Path
    _writer: DataPointWriter = field(repr=False)
    config: Dict[str, Any] = field(default_factory=dict)
    _state: Dict[str, Any] = field(default_factory=dict)

    @property
    def data_point(self) -> DataPoint:
        return self._writer.data_point

    @classmethod
    def create(
        cls,
        environment: "Environment",
        task: Task,
        output_dir: os.PathLike,
        run_id: str = None,
        config: Dict[str, Any] = None,
    ) -> "RunContext":
        """Create a new RunContext. Auto-detects create vs resume."""
        output_path = Path(output_dir)
        if output_path.exists() and (output_path / "task.json").exists():
            writer = DataPointWriter.resume(output_path)
            if writer.data_point.task.task_id != task.task_id:
                raise ValueError(
                    "Cannot resume task: checkpoint task_id "
                    f"{writer.data_point.task.task_id!r} does not match requested "
                    f"task_id {task.task_id!r}"
                )
        else:
            writer = DataPointWriter.create(output_path, task, run_id=run_id)
        ctx = cls(
            environment=environment,
            output_dir=output_path,
            config=config or {},
            _writer=writer,
        )
        state_path = output_path / "run_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                ctx._state = json.load(f)
        return ctx

    def add_action(self, action: Action) -> None:
        """Append an action event, writing to disk immediately."""
        self._writer.add_event(action)

    def add_observation(self, observation: ObservationEvent) -> None:
        """Append an observation event, writing to disk immediately."""
        self._writer.add_event(observation)

    def add_verification_result(self, result: VerificationResult) -> None:
        """Append a verification result, writing to disk immediately."""
        key = result.verifier_name
        if key in self.data_point.verification:
            count = sum(
                1
                for k in self.data_point.verification
                if k == key or k.startswith(f"{key}_")
            )
            key = f"{key}_{count}"
            result.verifier_name = key
        self._writer.add_verification_result(result)

    def save(self) -> None:
        """Save the current state to output_dir."""
        self._writer.finalize()
        self.data_point.save(self.output_dir / "data_point.json")
        with open(self.output_dir / "run_state.json", "w") as f:
            json.dump(self._state, f, indent=2)

    def checkpoint(self) -> None:
        """Save a checkpoint of current progress."""
        self.save()

    @property
    def task(self) -> Task:
        return self.data_point.task

    @property
    def solver_log(self) -> SolverLog:
        return self.data_point.solver_log

    def set_state(self, key: str, value: Any) -> None:
        """Set a state value (for resuming runs)."""
        self._state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self._state.get(key, default)

    def is_resuming(self) -> bool:
        """Check if this is a resumed run with existing progress."""
        return len(self.solver_log.events) > 0
