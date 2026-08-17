#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LandFlow release artifacts.")
    parser.add_argument("--model-repo", default=os.environ.get("HF_MODEL_REPO", ""))
    parser.add_argument("--data-repo", default=os.environ.get("HF_DATA_REPO", ""))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_repo or not args.data_repo:
        raise SystemExit(
            "Set HF_MODEL_REPO and HF_DATA_REPO, or pass --model-repo and --data-repo."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface-hub before downloading artifacts.") from exc

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model_repo,
        repo_type="model",
        local_dir=root,
        allow_patterns=["pretrained_models/*", "checkpoints/*"],
    )
    snapshot_download(
        repo_id=args.data_repo,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=["artifacts/*", "data/*"],
    )
    print(f"Downloaded LandFlow artifacts under {root}")


if __name__ == "__main__":
    main()
