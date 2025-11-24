#!/usr/bin/env python3
"""
Script to download FARA-7B model weights from HuggingFace Hub.

This script downloads the model to the model_checkpoints/ directory in the repository root.
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download FARA-7B model weights from HuggingFace Hub"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="microsoft/fara-7b",
        help="HuggingFace model ID (default: microsoft/fara-7b)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model weights (default: model_checkpoints/fara-7b in repo root)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for authentication (optional, can also use HF_TOKEN env var)",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        # Default to model_checkpoints/fara-7b in the repository root
        repo_root = Path(__file__).parent.parent
        output_dir = repo_root / "model_checkpoints" / "fara-7b"
    else:
        output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model_id} to {output_dir}")
    print("This may take a while depending on your internet connection...")

    try:
        snapshot_download(
            repo_id=args.model_id,
            local_dir=str(output_dir),
            token=args.token,
            local_dir_use_symlinks=False,
        )
        print(f"\n✓ Successfully downloaded model to {output_dir}")
        print(f"\nYou can now use this model with:")
        print(f"  python az_vllm.py --model_url {output_dir} --device_id 0,1")
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nIf you're getting authentication errors, you may need to:")
        print("  1. Install huggingface-cli: pip install -U huggingface_hub")
        print("  2. Login: huggingface-cli login")
        print("  3. Or provide a token: --token YOUR_HF_TOKEN")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
