#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload LandFlow artifacts to Hugging Face.")
    parser.add_argument("--model-repo", default=os.environ.get("HF_MODEL_REPO", ""))
    parser.add_argument("--data-repo", default=os.environ.get("HF_DATA_REPO", ""))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("LANDFLOW_SOURCE_ROOT", PROJECT_ROOT)),
        help="Root containing pretrained_models/, checkpoints/, artifacts/, and data/.",
    )
    parser.add_argument(
        "--landflow-checkpoint",
        type=Path,
        default=os.environ.get("LANDFLOW_CHECKPOINT_SOURCE"),
    )
    parser.add_argument(
        "--trajectory-bank",
        type=Path,
        default=os.environ.get("LANDFLOW_BANK_SOURCE"),
    )
    parser.add_argument("--public", action="store_true", help="Create public repositories.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_repo or not args.data_repo:
        raise SystemExit("Set HF_MODEL_REPO and HF_DATA_REPO before uploading.")
    source_root = args.source_root.resolve()
    sources = {
        "pretrained_models/pretrained_flow.pt": source_root / "pretrained_models/pretrained_flow.pt",
        "pretrained_models/atom_num_model.pt": source_root / "pretrained_models/atom_num_model.pt",
        "checkpoints/landflow.pt": args.landflow_checkpoint or source_root / "checkpoints/landflow.pt",
        "artifacts/trajectory_bank.pt": args.trajectory_bank or source_root / "artifacts/trajectory_bank.pt",
        "data/crossdocked_pocket10_pose_split.pt": source_root / "data/crossdocked_pocket10_pose_split.pt",
        "data/atom_num_dataset.pkl": source_root / "data/atom_num_dataset.pkl",
        "data/reference_metrics.pt": source_root / "data/reference_metrics.pt",
    }
    manifest = json.loads((PROJECT_ROOT / "artifacts.json").read_text())
    entries = {
        item["path"]: item
        for group in ("model_repo_files", "dataset_repo_files")
        for item in manifest[group]
    }
    for path_in_repo, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        item = entries[path_in_repo]
        if source.stat().st_size != int(item["size"]) or sha256(source) != item["sha256"]:
            raise RuntimeError(f"Source does not match release manifest: {source}")
        print(f"[ready] {source} -> {path_in_repo}")
    if args.dry_run:
        print("Dry run passed; no repositories or files were changed.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install huggingface-hub before uploading artifacts.") from exc
    api = HfApi()
    private = not args.public
    api.create_repo(args.model_repo, repo_type="model", private=private, exist_ok=True)
    api.create_repo(args.data_repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=PROJECT_ROOT / "huggingface/model/README.md",
        path_in_repo="README.md",
        repo_id=args.model_repo,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=PROJECT_ROOT / "huggingface/dataset/README.md",
        path_in_repo="README.md",
        repo_id=args.data_repo,
        repo_type="dataset",
    )
    for path_in_repo, source in sources.items():
        repo_id = args.model_repo if path_in_repo in {
            item["path"] for item in manifest["model_repo_files"]
        } else args.data_repo
        repo_type = "model" if repo_id == args.model_repo else "dataset"
        api.upload_file(
            path_or_fileobj=source,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
        )
    print("Hugging Face upload completed.")


if __name__ == "__main__":
    main()
