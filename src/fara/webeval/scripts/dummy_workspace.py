class Workspace:
    def renew_mlflow_token(self):
        ...

    def start_run(self, _):
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...