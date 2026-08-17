#!/usr/bin/env python
"""Run PAFlow evaluation on a subset of result_*.pt files.

This wrapper keeps scripts/evaluate_diffusion.py unchanged. It creates a
chunk-specific directory containing symlinks to selected result files, then
calls the original evaluator on that directory. Multiple chunks can therefore
run in parallel without writing to the same eval_results directory.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_diffusion.py"


def result_index(path: Path) -> int:
    match = re.search(r"result_(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Cannot parse result index from {path}")
    return int(match.group(1))


def safe_suffix(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a chunk of result_*.pt files by symlinking them into a "
            "separate directory and invoking scripts/evaluate_diffusion.py."
        )
    )
    parser.add_argument("sample_path", type=Path)
    parser.add_argument("--eval_start_index", type=int, default=0)
    parser.add_argument("--eval_end_index", type=int, default=None)
    parser.add_argument("--eval_output_suffix", type=str, default=None)
    parser.add_argument("--chunk_root", type=Path, default=None)
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_known_args()


def prepare_chunk_dir(args: argparse.Namespace) -> tuple[Path, list[Path], int, int]:
    sample_path = args.sample_path.resolve()
    result_files = sorted(sample_path.glob("result_*.pt"), key=result_index)
    total = len(result_files)
    start = max(args.eval_start_index, 0)
    end = args.eval_end_index if args.eval_end_index is not None else total
    end = min(end, total)
    if end <= start:
        raise ValueError(f"Invalid chunk range: start={start}, end={end}, total={total}")

    selected = result_files[start:end]
    suffix = args.eval_output_suffix or f"chunk{start}_{end}"
    suffix = safe_suffix(suffix)
    chunk_root = args.chunk_root.resolve() if args.chunk_root else sample_path / "eval_chunks"
    chunk_dir = chunk_root / suffix

    if chunk_dir.exists() and args.force:
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for existing in chunk_dir.glob("result_*.pt"):
        if existing.is_symlink():
            existing.unlink()
        else:
            raise RuntimeError(
                f"Refusing to overwrite non-symlink result file in chunk dir: {existing}"
            )

    for source in selected:
        target = chunk_dir / source.name
        target.symlink_to(source)

    return chunk_dir, selected, start, end


def main() -> None:
    args, passthrough = parse_args()
    chunk_dir, selected, start, end = prepare_chunk_dir(args)

    cmd = [
        args.python_bin,
        str(EVALUATOR),
        str(chunk_dir),
        *passthrough,
    ]

    print(f"Selected {len(selected)} result files: [{start}, {end})")
    print(f"Chunk directory: {chunk_dir}")
    print(f"Evaluation output: {chunk_dir / 'eval_results'}")
    print("Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not pythonpath else str(PROJECT_ROOT) + os.pathsep + pythonpath
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
