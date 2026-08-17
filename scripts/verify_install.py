#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.hjb_value_model import build_hjb_value_model_from_checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> tuple[int, int]:
    manifest = json.loads((root / "artifacts.json").read_text())
    checked = 0
    missing = 0
    for group in ("model_repo_files", "dataset_repo_files", "crossdocked_repo_files"):
        for item in manifest[group]:
            path = root / item["path"]
            if not path.is_file():
                print(f"[missing] {item['path']}")
                missing += 1
                continue
            actual_size = path.stat().st_size
            actual_hash = sha256(path)
            if actual_size != int(item["size"]) or actual_hash != item["sha256"]:
                raise RuntimeError(f"Artifact checksum mismatch: {path}")
            print(f"[ok] {item['path']}")
            checked += 1
    return checked, missing


def verify_gradient(checkpoint_path: Path) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_hjb_value_model_from_checkpoint(payload, "cpu").eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 115842:
        raise RuntimeError(f"Expected 115842 parameters, found {parameter_count}")

    torch.manual_seed(7)
    ligand_pos = torch.randn(6, 3, requires_grad=True)
    ligand_v = torch.randint(0, 16, (6,))
    protein_pos = torch.randn(12, 3) * 2.0
    protein_v = torch.randn(12, 27)
    batch_ligand = torch.zeros(6, dtype=torch.long)
    batch_protein = torch.zeros(12, dtype=torch.long)
    time_fraction = torch.tensor([0.6])
    value = model(
        ligand_pos,
        ligand_v,
        protein_pos,
        protein_v,
        batch_ligand,
        batch_protein,
        time_fraction,
    )
    gradient = torch.autograd.grad(value.sum(), ligand_pos)[0]
    if not torch.isfinite(value).all() or not torch.isfinite(gradient).all():
        raise RuntimeError("Potential value or coordinate gradient is non-finite")
    if float(gradient.norm()) <= 0:
        raise RuntimeError("Potential returned a zero coordinate gradient")
    print(
        f"[ok] LandFlow checkpoint: parameters={parameter_count}, "
        f"value={float(value.item()):.6f}, gradient_norm={float(gradient.norm()):.6f}"
    )
    return parameter_count


def verify_crossdocked(root: Path, required: bool) -> None:
    lmdb_path = root / "data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
    test_set_path = root / "data/test_set"
    missing = []
    if not lmdb_path.is_file():
        missing.append(str(lmdb_path.relative_to(root)))
    if not test_set_path.is_dir():
        missing.append(str(test_set_path.relative_to(root)))
    if missing:
        message = "Missing external CrossDocked resources: " + ", ".join(missing)
        if required:
            raise SystemExit(message)
        print(f"[external data not checked] {message}")
        return
    print("[ok] processed CrossDocked LMDB and test_set receptor directory")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a LandFlow installation.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--require-crossdocked",
        action="store_true",
        help="Fail unless the external processed LMDB and test receptor directory exist.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    checked, missing = verify_manifest(root)
    checkpoint = root / "checkpoints/landflow.pt"
    if not checkpoint.is_file():
        raise SystemExit("Missing checkpoints/landflow.pt; run download_artifacts.py first.")
    verify_gradient(checkpoint)
    verify_crossdocked(root, required=args.require_crossdocked)
    print(f"Verification passed ({checked} artifacts checked, {missing} optional files missing).")


if __name__ == "__main__":
    main()
