"""Native, no-browser tests for resume task identity invariants."""
import sys
from pathlib import Path
from types import ModuleType

import pytest

# Load first-party core modules without Fara's package-level agent/OpenAI imports.
source = Path(__file__).resolve().parents[1] / "src" / "fara"
fara_package = ModuleType("fara")
fara_package.__path__ = [str(source)]
core_package = ModuleType("fara.core")
core_package.__path__ = [str(source / "core")]
sys.modules.setdefault("fara", fara_package)
sys.modules.setdefault("fara.core", core_package)
from fara.core.data_point import Task
from fara.core.run_context import RunContext


def test_resume_rejects_different_task_id(tmp_path):
    original = Task(task_id="experiment-1", instruction="collect A")
    different = Task(task_id="experiment-2", instruction="collect B")
    RunContext.create(environment=object(), task=original, output_dir=tmp_path)

    with pytest.raises(ValueError, match="task"):
        RunContext.create(environment=object(), task=different, output_dir=tmp_path)

    # Denial must not overwrite the recorded identity.
    assert Task.load(tmp_path / "task.json").task_id == original.task_id


def test_resume_same_task_id_keeps_original_checkpoint(tmp_path):
    task = Task(task_id="experiment-1", instruction="collect A")
    first = RunContext.create(environment=object(), task=task, output_dir=tmp_path)
    first.set_state("stage", "complete")
    first.checkpoint()

    resumed = RunContext.create(environment=object(), task=task, output_dir=tmp_path)
    assert resumed.task.task_id == task.task_id
    assert resumed.get_state("stage") == "complete"


def test_new_directory_keeps_requested_task_identity(tmp_path):
    task = Task(task_id="experiment-2", instruction="new task")
    created = RunContext.create(environment=object(), task=task, output_dir=tmp_path)
    assert created.task.task_id == "experiment-2"
