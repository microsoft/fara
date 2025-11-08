import subprocess
from enum import Enum
import os
import logging
import threading
import time

class Status(Enum):
    NotStarted = 0
    Running = 1
    Stopped = 2

class VLLM:
    cmd_template = ' '.join([
        "python -O -u -m vllm.entrypoints.openai.api_server",
        "--host={host}",
        "--port={port}",
        "--model={model_dir}",
        "--served-model-name {model_name}",
        "--tensor-parallel-size {tensor_parallel_size}",
        "--gpu-memory-utilization 0.95",
        "--trust-remote-code",
        "--dtype {dtype}"
    ])
    def __init__(self,
                 model_path,
                 max_n_images,
                 device_id = "0",
                 host = "0.0.0.0",
                 port = 5000,
                 model_name = "gpt-4o-mini-2024-07-18",
                 dtype = "auto",
                 enforce_eager = False):
        self.model_path = model_path
        self.device_id = device_id
        self.host = host
        self.port = port
        self.max_n_images = int(max_n_images)
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.cmd = VLLM.cmd_template
        if self.max_n_images > 1:
            # new versions of vllm require dictionary-like arguments for this
            # see https://docs.vllm.ai/en/latest/configuration/engine_args.html#multimodalconfig
            self.cmd += f" --limit-mm-per-prompt.image {self.max_n_images}"
        if enforce_eager:  # Most helpful for float32 cases when attention backends are incompatible
            self.cmd += " --enforce-eager"
        self.model_name = model_name
        self.tensor_parallel_size = len(str(device_id).split(','))
        self.status = Status.NotStarted
        self.process = None
        self.logs = []

    @property
    def endpoint(self):
        return f"http://{self.host}:{self.port}/v1/"

    def start(self):
        def _drain(pipe):
            for line in iter(pipe.readline, ''):
                self.logs.append(line)
                print(line, end='')          
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = self.device_id
        env['NCCL_DEBUG'] = "TRACE"
        self.process = subprocess.Popen(
            self.cmd.format(
                host=self.host,
                port=self.port,
                model_dir=self.model_path,
                model_name=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype=self.dtype
            ).split(),
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            shell = False,
            env = env
        )
        t = threading.Thread(target=_drain, args=(self.process.stdout,), daemon=True)
        t.start()

        while True:
            while not any(self.logs):
                time.sleep(1)
            line = self.logs.pop(0)
            if "Application startup complete." in line:
                logging.info("VLLM process started successfully.")
                self.status = Status.Running
                return True        
    
    def stop(self):
        if self.process:
            self.process.terminate()
        self.status = Status.Stopped