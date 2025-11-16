from pathlib import Path
import datetime
from typing import List, Callable, Dict
import os
from tqdm.auto import tqdm
from joblib import Parallel, delayed


def _dt_now_str():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


class AzRef:
    def __init__(self, az_folder, local_path, tag = None):
        # Check if az_folder is a local path (string or Path) or an Azure folder object
        if isinstance(az_folder, (str, Path)):
            # Local path mode
            self.az_folder = None
            self.az_context = None
            self.path = Path(az_folder)
            self.is_local = True
        else:
            # Azure mode
            tag = tag or _dt_now_str()
            local_path = Path(local_path / tag)
            local_path.mkdir(parents = True, exist_ok = True)
            self.az_folder = az_folder
            self.az_context = az_folder.mount(local_path)
            self.path = self.az_context.path
            self.is_local = False

    @property
    def url(self):
        if self.is_local:
            return str(self.path)
        return self.az_folder.url()

    def mount(self):
        if not self.is_local:
            self.az_context.mount()

    def unmount(self):
        if not self.is_local:
            self.az_context.unmount()

    def __enter__(self):
        self.mount()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unmount()
        return False

    def __repr__(self):
        if self.is_local:
            return f'Local: {self.path}'
        return f'{repr(self.az_folder)} -> {self.path}'

    def map_folders(self, func: Callable[[os.PathLike], Dict], subdir: str = None) -> List[Dict]:
        path = self.path / subdir if subdir else self.path
        return Parallel(-1)(delayed(func)(p) for p in tqdm([p for p in path.iterdir() if p.is_dir()], desc=f"Processing folders in {path}..."))

    def map_files(self, func: Callable[[os.PathLike], Dict], subdir: str = None) -> List[Dict]:
        path = self.path / subdir if subdir else self.path
        return Parallel(-1)(delayed(func)(p) for p in tqdm([p for p in path.iterdir() if not p.is_dir()], desc=f"Processing files in {path}..."))

class EvalLogs(AzRef):
    def __init__(self, az_folder, local_path = None, tag = None):
        local_path = Path(local_path or Path(os.getcwd()) / 'osagent_data' / "eval")
        super().__init__(az_folder, local_path, tag)

    # @staticmethod
    # def from_experiment(workspace, experiment_name, local_path = None):
    #     job = EvalJob.from_run_id(workspace, experiment_name)
    #     return EvalLogs(job.logs(), local_path, tag = f'{job.job_name}')
    
    @property
    def traj_path(self):
        return self.path / 'traj'

