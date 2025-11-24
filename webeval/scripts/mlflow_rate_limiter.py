import mlflow
import time
import threading
# from aztool.workspace import renew_mlflow_token

class _Lock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

class MlFlowRateLimiter:
    def __init__(self, period_s = 60, thread_safe = True, retries = 2, timeout = 5):
        self.period_s = period_s
        self.metrics = {}
        self.metrics_with_run_id = {}
        self.last_ts = None
        self.lock = threading.Lock() if thread_safe else _Lock()
        self.retries = retries
        self.timeout = timeout

    def __enter__(self):
        return self

    def log_metric(self, key, value, run_id = None):
        with self.lock:
            if run_id is not None:
                self.metrics_with_run_id[run_id] = self.metrics_with_run_id.get(run_id, {})
                self.metrics_with_run_id[run_id][key] = value
            else:
                self.metrics[key] = value
            self.flush()

    def log_metrics(self, metrics, run_id = None):
        with self.lock:
            if run_id is not None:
                self.metrics_with_run_id[run_id] = self.metrics_with_run_id.get(run_id, {})
                self.metrics_with_run_id[run_id].update(metrics)
            else:
                self.metrics.update(metrics)
            self.flush()

    def _flush(self, force = False):
        if force or (self.last_ts is None) or (self.last_ts + self.period_s) < time.time():
            if self.metrics:
                mlflow.log_metrics(self.metrics)
            self.metrics = {}                
            for k, v in self.metrics_with_run_id.items():
                if v:
                    mlflow.log_metrics(v, run_id=k)
            self.metrics_with_run_id = {}
            self.last_ts = time.time()

    def flush(self, force = False):
        for _ in range(self.retries):
            try:
                self._flush(force)
                return
            except Exception as e:
                print(f"Failed to log metrics: {e}")
                print("Renewing token")
                # renew_mlflow_token()
                time.sleep(self.timeout)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.metrics:
            self.flush(True)
