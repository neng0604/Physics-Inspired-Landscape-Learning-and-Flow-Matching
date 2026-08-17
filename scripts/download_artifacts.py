#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LandFlow release artifacts.")
    parser.add_argument("--model-repo", default=os.environ.get("HF_MODEL_REPO", ""))
    parser.add_argument("--data-repo", default=os.environ.get("HF_DATA_REPO", ""))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--include-crossdocked",
        action="store_true",
        help="Also download the 7.7 GB PAFlow-ready CrossDocked pocket data.",
    )
    parser.add_argument(
        "--extract-test-set",
        action="store_true",
        help="Extract data/test_set.tar.gz after downloading it.",
    )
    return parser.parse_args()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive links are not allowed: {member.name}")
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
        handle.extractall(destination)


def main() -> None:
    args = parse_args()
    if not args.model_repo or not args.data_repo:
        raise SystemExit(
            "Set HF_MODEL_REPO and HF_DATA_REPO, or pass --model-repo and --data-repo."
        )
    if args.extract_test_set and not args.include_crossdocked:
        raise SystemExit("--extract-test-set requires --include-crossdocked.")
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
    data_patterns = [
        "artifacts/trajectory_bank.pt",
        "data/crossdocked_pocket10_pose_split.pt",
        "data/atom_num_dataset.pkl",
        "data/reference_metrics.pt",
    ]
    if args.include_crossdocked:
        data_patterns.extend(
            [
                "data/crossdocked_pocket10_with_protein.tar.gz",
                "data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb",
                "data/test_set.tar.gz",
                "data/CrossDocked2020_LICENSE.txt",
            ]
        )
    snapshot_download(
        repo_id=args.data_repo,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=data_patterns,
    )
    if args.extract_test_set:
        archive = root / "data/test_set.tar.gz"
        safe_extract(archive, root / "data")
        print(f"Extracted {archive} under {root / 'data'}")
    print(f"Downloaded LandFlow artifacts under {root}")


if __name__ == "__main__":
    main()
