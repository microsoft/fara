from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
import httpx
import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .vllm_facade import VLLM, Status

try:
    from aztool.azcp import AzFolder, LocalFolder
except ImportError:  # keep old behaviour when aztool missing
    AzFolder = None
    LocalFolder = None


def _is_azure_blob_url(model_path: str) -> bool:
    return model_path.startswith(("https://", "http://")) and "blob.core.windows.net" in model_path


def _extract_model_name(model_url: str) -> str:
    """Extract model name from URL for consistent naming."""
    url_parts = model_url.rstrip('/').split('/')
    return url_parts[-1] if url_parts else model_url


def _cache_model(model_url: str) -> str:
    if AzFolder is None:
        raise RuntimeError("Azure support not available. Install aztool or run without --cache.")

    cache_root = Path(args.cache_dir or os.path.expanduser("~/.cache/vllm_models"))
    cache_root.mkdir(parents=True, exist_ok=True)

    model_name = _extract_model_name(model_url)
    cache_key = f"{model_name}_{hashlib.md5(model_url.encode()).hexdigest()[:8]}"
    cached_model_path = cache_root / cache_key

    if cached_model_path.exists() and cached_model_path.is_dir():
        return str(cached_model_path)

    model_az = AzFolder.from_uri(model_url)
    context = model_az.mount()
    context.mount()
    copy_error: Optional[Exception] = None
    try:
        if cached_model_path.exists():
            shutil.rmtree(cached_model_path)
        shutil.copytree(context.path, cached_model_path)
    except OSError as exc:
        copy_error = exc
    finally:
        context.unmount()

    if copy_error is None:
        return str(cached_model_path)

    if LocalFolder is None:
        raise copy_error

    if cached_model_path.exists():
        shutil.rmtree(cached_model_path)
    cached_model_path.mkdir(parents=True, exist_ok=True)
    try:
        model_az.copy(LocalFolder(cached_model_path, create=False))
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to copy model using azcopy") from exc

    return str(cached_model_path)


def _prepare_cached_model(model_url: str) -> str:
    if _is_azure_blob_url(model_url):
        return _cache_model(model_url)

    model_path = Path(model_url).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Local model directory not found: {model_url}")
    return str(model_path.resolve())

class AzVllm:
    def __init__(self, model_url, port, device_id, max_n_images, dtype='auto', enforce_eager=False, use_external_endpoint=False):
        self.model_az = None
        self.local_model_path = None
        self.vllm = None
        self.context = None
        self.device_id = device_id
        self.max_n_images = max_n_images
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.use_external_endpoint = use_external_endpoint
        if model_url and not use_external_endpoint:
            # Check if model_url is an Azure blob URL or a local directory
            if _is_azure_blob_url(model_url):
                self.model_az = AzFolder.from_uri(model_url)
            else:
                # It's a local directory
                model_path = Path(model_url).expanduser()
                if not model_path.exists():
                    raise FileNotFoundError(f"Local model directory not found: {model_url}")
                self.local_model_path = str(model_path.resolve())
            self.port = port

    def __enter__(self):
        # No-op if using external endpoint
        if self.use_external_endpoint:
            print('Using external endpoint, skipping VLLM startup')
            return self

        if self.model_az:
            self.context = self.model_az.mount()
            self.context.mount()
            print(f"VLLM has mounted model at {self.context.path}, contents:")
            ### sometimes need to ls the directory or else huggingface will complain a config.json doesn't exist
            for root, dirs, files in os.walk(self.context.path):
                for file in files:
                    print(f"\t{os.path.join(root, file)}")
            self.vllm = VLLM(
                model_path = self.context.path,
                port = self.port,
                device_id = self.device_id,
                max_n_images = self.max_n_images,
                dtype = self.dtype,
                enforce_eager = self.enforce_eager
            )
            self.vllm.start()
            print('VLLM has started')
        elif self.local_model_path:
            print(f"VLLM using on-disk model at path {self.local_model_path}, contents:")
            ### sometimes need to ls the directory or else huggingface will complain a config.json doesn't exist
            for root, dirs, files in os.walk(self.local_model_path):
                for file in files:
                    print(f"\t{os.path.join(root, file)}")
            self.vllm = VLLM(
                model_path = self.local_model_path,
                port = self.port,
                device_id = self.device_id,
                max_n_images = self.max_n_images,
                dtype = self.dtype,
                enforce_eager = self.enforce_eager
            )
            self.vllm.start()
            print('VLLM has started')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.vllm:
            if self.vllm and (self.vllm.status == Status.Running):
                self.vllm.stop()
        if self.context:
            self.context.unmount()

@asynccontextmanager
async def lifespan(app: FastAPI):
    cached_vllm: Optional[VLLM] = None
    az_vllm: Optional[AzVllm] = None

    try:
        app.state.resolved_model_path = None
        app.state.model_name = None
        if args.cache and args.model_url:
            model_path = _prepare_cached_model(args.model_url)
            app.state.resolved_model_path = model_path
            app.state.model_name = _extract_model_name(args.model_url)
            cached_vllm = VLLM(
                model_path=model_path,
                port=args.vllm_port,
                device_id=args.device_id,
                max_n_images=args.max_n_images,
                dtype=args.dtype,
                enforce_eager=args.enforce_eager
            )
            cached_vllm.start()
        else:
            az_vllm = AzVllm(
                model_url = args.model_url,
                port = args.vllm_port,
                device_id = args.device_id,
                max_n_images = args.max_n_images,
                dtype = args.dtype,
                enforce_eager = args.enforce_eager)
            az_vllm.__enter__()
            app.state.resolved_model_path = args.model_url
            app.state.model_name = _extract_model_name(args.model_url)

        yield
    finally:
        if cached_vllm and cached_vllm.status == Status.Running:
            cached_vllm.stop()
        if az_vllm:
            az_vllm.__exit__(None, None, None)
        app.state.resolved_model_path = None
        app.state.model_name = None


app = FastAPI(lifespan = lifespan)


@app.post("/v1/chat/completions")
async def post_v1_chat_completions(request: Request):
    body = await request.body()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f'http://localhost:{args.vllm_port}/v1/chat/completions',
            content=body,
            headers=dict(request.headers),
            timeout=None
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp.headers
    )


@app.get("/model")
async def get_model():

    return {"model": _extract_model_name(args.model_url), "model_url": args.model_url}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vllm from azure blob model")
    parser.add_argument("--model_url", type=str, default=None, help="Model URL")
    parser.add_argument("--port", type=int, default=5000, help="port")
    parser.add_argument("--vllm_port", type=int, default=5001, help="vllm port")
    parser.add_argument("--device_id", type=str, default="0", help="device id")
    parser.add_argument("--max_n_images", type=int, default=3, help="Maximum number of images to process")
    parser.add_argument('--dtype', type=str, choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'], default='auto', help='Data type for VLLM model (default: auto)')
    parser.add_argument('--enforce_eager', action='store_true', help='Enforce eager execution mode for compatibility')
    parser.add_argument('--cache', action='store_true', help='Enable caching / local path serving instead of Azure mount')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to cache downloaded models (default: ~/.cache/vllm_models)')
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)