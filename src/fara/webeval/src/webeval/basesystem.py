import os
import json
import importlib
from typing import Dict, Any
from logging import Logger


# ----------------------------------------------------------------------
# System Base Class
# ----------------------------------------------------------------------
class BaseSystem:
    """
    All systems should implement this interface.
    """

    def __init__(self, system_name: str):
        self.system_name = system_name

    def get_answer(self, task_id: str, task_data: Dict[str, Any], output_dir: str, logger: Logger) -> Any:
        """
        Return an answer for the question. Should use save_answer_to_disk to save the answer.
        """
        raise NotImplementedError("Implement your system's logic here.")

    def load_answer_from_disk(self, task_id: str, output_dir: str) -> Any:
        """
        Helper to load an answer from disk if it exists.
        """
        answer_path = os.path.join(output_dir, f"{task_id}_answer.json")
        if not os.path.exists(answer_path):
            return None
        with open(answer_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def hash(self) -> str:
        """
        Returns a string hash of the system configuration applicable to be a path.
        This is route results with different parameters to different locations.
        """
        return f"{self.system_name}"


def load_system_class(system_name: str):
    """
    Dynamically load a system class based on the system name.
    """
    module_name = f"webeval.systems.{system_name.lower()}"
    class_name = f"{system_name}System"
    module = importlib.import_module(module_name)
    system_class = getattr(module, class_name)
    return system_class
