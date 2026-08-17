#!/usr/bin/env python
"""Merge metrics produced by scripts/evaluate_diffusion_chunk.py.

The chunk evaluator intentionally keeps the original PAFlow evaluator unchanged.
This script collects chunk-level metrics and recomputes the paper-facing summary
statistics from the concatenated evaluated molecules.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from rdkit import Chem, DataStructs


def mean_pairwise_diversity(mols):
    fps = []
    for mol in mols:
        try:
            fps.append(Chem.RDKFingerprint(mol))
        except Exception:
            continue
    if len(fps) < 2:
        return None

    values = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            values.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return float(np.mean(values)) if values else None


def get_vina_affinity(vina_results, key):
    if vina_results is None or key not in vina_results:
        return None
    try:
        return float(vina_results[key][0]["affinity"])
    except Exception:
        return None


def stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"mean": None, "median": None, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "n": len(values),
    }


def fmt(value, ndigits=3):
    if value is None:
        return "N/A"
    return f"{value:.{ndigits}f}"


def parse_num_samples(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    match = re.search(r"Evaluate done!\s+(\d+)\s+samples in total", text)
    if not match:
        return None
    return int(match.group(1))


def expected_metric_paths(chunk_root: Path, start: int, end: int, chunk_size: int):
    paths = []
    for chunk_start in range(start, end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end)
        chunk_name = f"chunk{chunk_start}_{chunk_end}"
        paths.append(chunk_root / chunk_name / "eval_results" / "metrics_-1.pt")
    return paths


def weighted_stability(chunk_items):
    keys = ["mol_stable", "atm_stable", "recon_success", "eval_success", "complete"]
    total_weight = 0
    weighted = {key: 0.0 for key in keys}

    for item in chunk_items:
        stability = item["metrics"].get("stability") or {}
        n_samples = item.get("num_samples")
        if n_samples is None:
            complete = stability.get("complete")
            n_results = len(item["metrics"].get("all_results", []))
            if complete and complete > 0:
                n_samples = int(round(n_results / complete))
            else:
                n_samples = n_results
        if n_samples <= 0:
            continue
        total_weight += n_samples
        for key in keys:
            if key in stability and stability[key] is not None:
                weighted[key] += float(stability[key]) * n_samples

    if total_weight == 0:
        return {}
    return {key: weighted[key] / total_weight for key in keys}


def summarize(all_results, chunk_items, docking_mode, success_qed, success_sa, success_vina):
    summary = {
        "num_chunks": len(chunk_items),
        "num_samples": int(sum(item.get("num_samples") or 0 for item in chunk_items)),
        "num_evaluated_mols": len(all_results),
        "stability": weighted_stability(chunk_items),
    }

    qed = [r["chem_results"]["qed"] for r in all_results if r.get("chem_results")]
    sa = [r["chem_results"]["sa"] for r in all_results if r.get("chem_results")]
    summary["qed"] = stats(qed)
    summary["sa"] = stats(sa)

    grouped = {}
    for result in all_results:
        grouped.setdefault(result["ligand_filename"], []).append(result["mol"])
    diversity_by_target = [
        value
        for value in (mean_pairwise_diversity(mols) for mols in grouped.values())
        if value is not None
    ]
    summary["diversity"] = stats(diversity_by_target)

    if docking_mode == "qvina":
        vina = []
        for result in all_results:
            try:
                vina.append(float(result["vina"][0]["affinity"]))
            except Exception:
                pass
        summary["vina"] = stats(vina)
    elif docking_mode in {"vina_score", "vina_dock"}:
        affinity_keys = [("score_only", "vina_score"), ("minimize", "vina_min")]
        if docking_mode == "vina_dock":
            affinity_keys.append(("dock", "vina_dock"))

        high_affinity = {}
        reference = {}
        for key, out_key in affinity_keys:
            gen_values = []
            ref_values = []
            comparisons = []
            for result in all_results:
                gen = get_vina_affinity(result.get("vina"), key)
                ref = get_vina_affinity(result.get("reference_vina"), key)
                if gen is not None:
                    gen_values.append(gen)
                if ref is not None:
                    ref_values.append(ref)
                if gen is not None and ref is not None:
                    comparisons.append(float(gen < ref))
            summary[out_key] = stats(gen_values)
            reference[out_key] = stats(ref_values)
            high_affinity[out_key] = stats(comparisons)["mean"]

        summary["reference_vina"] = reference
        summary["high_affinity"] = high_affinity

        if docking_mode == "vina_dock":
            success_flags = []
            for result in all_results:
                chem = result.get("chem_results") or {}
                dock = get_vina_affinity(result.get("vina"), "dock")
                success_flags.append(
                    bool(
                        chem.get("qed", -np.inf) > success_qed
                        and chem.get("sa", -np.inf) > success_sa
                        and dock is not None
                        and dock < success_vina
                    )
                )
            summary["success_rate"] = stats(success_flags)["mean"]

    reference_chem = [
        result.get("reference_chem")
        for result in all_results
        if result.get("reference_chem") is not None
    ]
    summary["reference_qed"] = stats([r["qed"] for r in reference_chem])
    summary["reference_sa"] = stats([r["sa"] for r in reference_chem])

    ring_counters = [
        result["chem_results"]["ring_size"]
        for result in all_results
        if result.get("chem_results") and "ring_size" in result["chem_results"]
    ]
    ring_ratio = {}
    if ring_counters:
        for ring_size in range(3, 10):
            ring_ratio[ring_size] = float(
                sum(1 for counter in ring_counters if ring_size in counter)
                / len(ring_counters)
            )
    summary["ring_ratio"] = ring_ratio
    summary["diversity_by_target"] = diversity_by_target
    return summary


def attach_reference_metrics(all_results, reference_metrics):
    if not reference_metrics:
        return
    records = reference_metrics.get("records", {})
    if not records:
        return
    for result in all_results:
        ligand_filename = result.get("ligand_filename")
        if not ligand_filename:
            continue
        record = records.get(ligand_filename)
        if record is None:
            continue
        result["reference_vina"] = record.get("vina")
        result["reference_chem"] = record.get("chem_results")


def write_log(summary, output_dir: Path, docking_mode: str):
    lines = []
    lines.append(f"Merged chunks: {summary['num_chunks']}")
    lines.append(f"Evaluate done! {summary['num_samples']} samples in total.")
    lines.append(f"Evaluated molecules: {summary['num_evaluated_mols']}")
    for key, value in summary.get("stability", {}).items():
        lines.append(f"{key}:\t{value:.4f}")
    lines.append("")
    lines.append(
        f"QED:   Mean: {fmt(summary['qed']['mean'])} Median: {fmt(summary['qed']['median'])}"
    )
    lines.append(
        f"SA:    Mean: {fmt(summary['sa']['mean'])} Median: {fmt(summary['sa']['median'])}"
    )
    if summary.get("diversity", {}).get("n", 0):
        lines.append(
            "Diversity: Mean: "
            f"{fmt(summary['diversity']['mean'])} Median: {fmt(summary['diversity']['median'])}"
        )

    if docking_mode == "qvina" and "vina" in summary:
        lines.append(
            f"Vina:  Mean: {fmt(summary['vina']['mean'])} Median: {fmt(summary['vina']['median'])}"
        )
    elif docking_mode in {"vina_score", "vina_dock"}:
        label_map = {
            "vina_score": "Vina Score",
            "vina_min": "Vina Min  ",
            "vina_dock": "Vina Dock ",
        }
        for key in ["vina_score", "vina_min", "vina_dock"]:
            if key in summary:
                lines.append(
                    f"{label_map[key]}:  Mean: {fmt(summary[key]['mean'])} "
                    f"Median: {fmt(summary[key]['median'])}"
                )
        for key, value in summary.get("high_affinity", {}).items():
            lines.append(f"High Affinity ({label_map.get(key, key).strip()}): Mean: {fmt(value)}")
        if "success_rate" in summary:
            lines.append(f"Success Rate: Mean: {fmt(summary['success_rate'])}")

    for ring_size, ratio in summary.get("ring_ratio", {}).items():
        lines.append(f"ring size: {ring_size} ratio: {ratio:.3f}")

    log_text = "\n".join(lines) + "\n"
    (output_dir / "log.txt").write_text(log_text)
    (output_dir / "summary.md").write_text("```text\n" + log_text + "```\n")


def write_csv(summary, output_dir: Path):
    rows = []
    rows.append(["num_samples", summary["num_samples"]])
    rows.append(["num_evaluated_mols", summary["num_evaluated_mols"]])
    for key, value in summary.get("stability", {}).items():
        rows.append([key, value])
    for metric in ["qed", "sa", "diversity", "vina_score", "vina_min", "vina_dock"]:
        if metric in summary:
            rows.append([f"{metric}_mean", summary[metric]["mean"]])
            rows.append([f"{metric}_median", summary[metric]["median"]])
    for key, value in summary.get("high_affinity", {}).items():
        rows.append([f"high_affinity_{key}", value])
    if "success_rate" in summary:
        rows.append(["success_rate", summary["success_rate"]])

    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_path", type=Path)
    parser.add_argument("--chunk_root", type=Path, default=None)
    parser.add_argument("--eval_start_index", type=int, default=0)
    parser.add_argument("--eval_end_index", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--docking_mode", choices=["none", "qvina", "vina_score", "vina_dock"], default="vina_dock")
    parser.add_argument("--success_qed_threshold", type=float, default=0.25)
    parser.add_argument("--success_sa_threshold", type=float, default=0.59)
    parser.add_argument("--success_vina_dock_threshold", type=float, default=-8.18)
    parser.add_argument("--reference_metrics_path", type=Path, default=None)
    parser.add_argument("--allow_missing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sample_path = args.sample_path.resolve()
    chunk_root = args.chunk_root.resolve() if args.chunk_root else sample_path / "eval_chunks"
    output_dir = args.output_dir.resolve() if args.output_dir else sample_path / "eval_results_parallel_merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_paths = expected_metric_paths(
        chunk_root,
        args.eval_start_index,
        args.eval_end_index,
        args.chunk_size,
    )

    chunk_items = []
    missing = []
    for metrics_path in metric_paths:
        if not metrics_path.exists():
            missing.append(metrics_path)
            continue
        metrics = torch.load(metrics_path, map_location="cpu")
        log_path = metrics_path.parent / "log.txt"
        chunk_items.append({
            "path": metrics_path,
            "metrics": metrics,
            "num_samples": parse_num_samples(log_path),
        })

    if missing and not args.allow_missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing chunk metrics:\n{missing_text}")
    if not chunk_items:
        raise RuntimeError(f"No chunk metrics found under {chunk_root}")

    all_results = []
    all_bond_lengths = []
    for item in chunk_items:
        all_results.extend(item["metrics"].get("all_results", []))
        all_bond_lengths.extend(item["metrics"].get("bond_length", []))

    reference_metrics = None
    if args.reference_metrics_path:
        reference_metrics = torch.load(args.reference_metrics_path, map_location="cpu")
        attach_reference_metrics(all_results, reference_metrics)

    summary = summarize(
        all_results,
        chunk_items,
        args.docking_mode,
        args.success_qed_threshold,
        args.success_sa_threshold,
        args.success_vina_dock_threshold,
    )

    torch.save(
        {
            "stability": summary.get("stability", {}),
            "bond_length": all_bond_lengths,
            "diversity_by_target": summary.get("diversity_by_target", []),
            "all_results": all_results,
            "summary": summary,
            "chunk_metric_paths": [str(item["path"]) for item in chunk_items],
        },
        output_dir / "metrics_-1.pt",
    )
    write_log(summary, output_dir, args.docking_mode)
    write_csv(summary, output_dir)

    print(f"Merged {len(chunk_items)} chunks")
    print(f"Output: {output_dir}")
    print((output_dir / "log.txt").read_text())


if __name__ == "__main__":
    main()
