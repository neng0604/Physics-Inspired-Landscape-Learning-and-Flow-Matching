from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)

INTERACTION_GROUPS = {
    "hbond": {"HBAcceptor", "HBDonor"},
    "hydrophobic": {"Hydrophobic"},
    "salt_bridge": {"Anionic", "Cationic", "Ionic"},
    "pi": {"PiStacking", "PiCation", "CationPi"},
}


def safe_float(value, default=float("nan")):
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def nan_summary(values):
    arr = np.asarray([safe_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "n": int(arr.size),
    }


def parse_labeled_path(item: str):
    if "=" not in item:
        raise ValueError(f"Expected LABEL=/path, got {item!r}")
    label, path = item.split("=", 1)
    return label, Path(path)


def find_metrics_path(result_dir: Path):
    for rel in (
        "eval_results_with_reference/metrics_-1.pt",
        "eval_vina_results/metrics_-1.pt",
        "eval_results/metrics_-1.pt",
    ):
        path = result_dir / rel
        if path.exists():
            return path
    return None


def load_reference_records(reference_pt: Path):
    payload = torch.load(reference_pt, map_location="cpu", weights_only=False)
    records = list((payload.get("records") or {}).values())
    by_index = {int(record["index"]): record for record in records if record.get("index") is not None}
    by_ligand = {str(record["ligand_filename"]): record for record in records if record.get("ligand_filename")}
    return records, by_index, by_ligand


def reference_for_row(row: dict, by_index: dict[int, dict], by_ligand: dict[str, dict]):
    source_id = row.get("source_data_id", row.get("dataset_index", row.get("subset_index")))
    if source_id is not None:
        try:
            rec = by_index.get(int(source_id))
            if rec is not None:
                return rec
        except Exception:
            pass
    ligand_filename = row.get("ligand_filename")
    if ligand_filename is not None:
        return by_ligand.get(str(ligand_filename))
    return None


def reference_protein_path(root: Path, record: dict):
    protein_filename = record.get("protein_filename")
    if protein_filename:
        path = root / str(protein_filename)
        if path.exists():
            return path
    ligand_filename = str(record.get("ligand_filename", ""))
    if ligand_filename:
        ligand_path = Path(ligand_filename)
        fallback = root / ligand_path.parent / f"{ligand_path.name[:10]}.pdb"
        if fallback.exists():
            return fallback
    return root / str(protein_filename or "")


def reference_ligand_path(root: Path, record: dict):
    return root / str(record.get("ligand_filename", ""))


def load_reference_mol(ligand_path: Path):
    suppl = Chem.SDMolSupplier(str(ligand_path), removeHs=False)
    mol = next(iter(suppl), None)
    if mol is None:
        raise ValueError(f"Could not load reference ligand {ligand_path}")
    return mol


def mol_with_positions(mol, pos):
    if mol is None or pos is None:
        return None
    arr = np.asarray(pos, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    out = Chem.Mol(mol)
    n = min(out.GetNumAtoms(), int(arr.shape[0]))
    if n <= 0:
        return None
    if out.GetNumConformers() == 0:
        out.AddConformer(Chem.Conformer(out.GetNumAtoms()), assignId=True)
    conf = out.GetConformer()
    for idx in range(n):
        conf.SetAtomPosition(idx, tuple(float(x) for x in arr[idx]))
    return out


def row_vina(row: dict, key: str):
    value = (row.get("vina") or {}).get(key)
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, dict):
        return safe_float(value.get("affinity"))
    return safe_float(value)


def row_property(row: dict, key: str):
    return safe_float((row.get("chem_results") or {}).get(key))


def make_generated_items(label: str, result_dir: Path, ref_by_index, ref_by_ligand, allow_missing: bool):
    metrics_path = find_metrics_path(result_dir)
    if metrics_path is None:
        if allow_missing:
            return []
        raise FileNotFoundError(f"No metrics_-1.pt found under {result_dir}")
    payload = torch.load(metrics_path, map_location="cpu", weights_only=False)
    items = []
    for row_idx, row in enumerate(payload.get("all_results", [])):
        rec = reference_for_row(row, ref_by_index, ref_by_ligand)
        if rec is None:
            continue
        mol = mol_with_positions(row.get("mol"), row.get("pred_pos"))
        if mol is None:
            continue
        source_id = int(rec["index"])
        items.append(
            {
                "method": label,
                "row_idx": row_idx,
                "source_data_id": source_id,
                "source_sample_idx": row.get("source_sample_idx", ""),
                "ligand_filename": rec.get("ligand_filename", ""),
                "mol": mol,
                "vina_score": row_vina(row, "score_only"),
                "vina_min": row_vina(row, "minimize"),
                "vina_dock": row_vina(row, "dock"),
                "qed": row_property(row, "qed"),
                "sa": row_property(row, "sa"),
            }
        )
    return items


def make_generated_items_from_metrics(metrics_path: Path, ref_by_ligand: dict[str, dict]):
    """Load a mixed-method reconstruction file without assuming dataset-index IDs.

    Online matched rollouts use ``source_data_id`` for the paired anchor (0..N-1),
    whereas the official reference file uses a pocket index (0..P-1).  Matching
    those numeric fields would silently associate most anchors with the wrong
    receptor.  The ligand filename is the stable key shared by both artifacts.
    """
    payload = torch.load(metrics_path, map_location="cpu", weights_only=False)
    items = []
    for row_idx, row in enumerate(payload.get("all_results", [])):
        ligand_filename = str(row.get("ligand_filename") or "")
        rec = ref_by_ligand.get(ligand_filename)
        if rec is None:
            continue
        mol = mol_with_positions(row.get("mol"), row.get("pred_pos"))
        if mol is None:
            continue
        items.append(
            {
                "method": str(row.get("branch_family") or "Generated"),
                "row_idx": row_idx,
                "anchor_id": int(row.get("source_data_id", row_idx)),
                "source_data_id": int(rec["index"]),
                # Keep the paired online anchor as the sample key.  The original
                # source_sample_idx is only the within-anchor method slot (0..3)
                # and repeats across the ten anchors in each pocket.
                "source_sample_idx": int(row.get("source_data_id", row_idx)),
                "branch_idx": row.get("source_sample_idx", ""),
                "ligand_filename": ligand_filename,
                "mol": mol,
                "vina_score": row_vina(row, "score_only"),
                "vina_min": row_vina(row, "minimize"),
                "vina_dock": row_vina(row, "dock"),
                "qed": row_property(row, "qed"),
                "sa": row_property(row, "sa"),
            }
        )
    return items


def make_reference_items(label: str, records: list[dict], ligand_root: Path):
    items = []
    for rec in records:
        try:
            mol = load_reference_mol(reference_ligand_path(ligand_root, rec))
        except Exception:
            continue
        row = {"vina": rec.get("vina") or {}}
        items.append(
            {
                "method": label,
                "row_idx": int(rec.get("index", -1)),
                "source_data_id": int(rec.get("index", -1)),
                "source_sample_idx": "reference",
                "ligand_filename": rec.get("ligand_filename", ""),
                "mol": mol,
                "vina_score": row_vina(row, "score_only"),
                "vina_min": row_vina(row, "minimize"),
                "vina_dock": row_vina(row, "dock"),
                "qed": safe_float((rec.get("chem_results") or {}).get("qed")),
                "sa": safe_float((rec.get("chem_results") or {}).get("sa")),
            }
        )
    return items


def filter_max_sources(items: list[dict], max_sources: int | None):
    if max_sources is None:
        return items
    keep = []
    seen = []
    seen_set = set()
    for item in items:
        source_id = int(item["source_data_id"])
        if source_id not in seen_set:
            if len(seen) >= int(max_sources):
                continue
            seen.append(source_id)
            seen_set.add(source_id)
        keep.append(item)
    return keep


def filter_source_shard(items: list[dict], modulus: int | None, remainder: int | None):
    if modulus is None:
        return items
    if int(modulus) <= 0 or remainder is None or not 0 <= int(remainder) < int(modulus):
        raise ValueError("source shard requires modulus > 0 and 0 <= remainder < modulus")
    return [
        item for item in items
        if int(item["source_data_id"]) % int(modulus) == int(remainder)
    ]


def bool_columns(df: pd.DataFrame):
    return [col for col in df.columns if df[col].dtype == bool]


def summarize_posebusters_row(pb_row: pd.Series):
    checks = [col for col in pb_row.index if isinstance(pb_row.get(col), (bool, np.bool_))]
    if not checks:
        return {"pb_pass_count": float("nan"), "pb_total": 0, "pb_pass_fraction": float("nan"), "pb_valid": float("nan")}
    passed = int(sum(bool(pb_row[col]) for col in checks))
    return {
        "pb_pass_count": passed,
        "pb_total": len(checks),
        "pb_pass_fraction": float(passed / len(checks)),
        "pb_valid": float(passed == len(checks)),
    }


def posebusters_for_group(buster, mols, protein_path: Path):
    try:
        df = buster.bust(mols, mol_cond=str(protein_path), full_report=False)
        return [summarize_posebusters_row(row) for _, row in df.iterrows()]
    except Exception:
        out = []
        for mol in mols:
            try:
                df = buster.bust(mol, mol_cond=str(protein_path), full_report=False)
                out.append(summarize_posebusters_row(df.iloc[0]))
            except Exception:
                out.append({"pb_pass_count": float("nan"), "pb_total": 0, "pb_pass_fraction": float("nan"), "pb_valid": float("nan")})
        return out


def interaction_set(df: pd.DataFrame, row_idx: int = 0):
    if df is None or df.empty or row_idx >= len(df):
        return set()
    row = df.iloc[row_idx]
    out = set()
    for col, value in row.items():
        if bool(value):
            out.add(tuple(str(x) for x in (col if isinstance(col, tuple) else (col,))))
    return out


def interaction_type(item: tuple[str, ...]):
    return item[-1] if item else ""


def filter_interaction_group(items: set, group_name: str):
    allowed = INTERACTION_GROUPS[group_name]
    return {item for item in items if interaction_type(item) in allowed}


def interaction_scores(gen_set: set, ref_set: set):
    shared = gen_set & ref_set
    union = gen_set | ref_set
    return {
        "interaction_gen_n": len(gen_set),
        "interaction_ref_n": len(ref_set),
        "interaction_shared_n": len(shared),
        "interaction_recovery": float(len(shared) / len(ref_set)) if ref_set else float("nan"),
        "interaction_precision": float(len(shared) / len(gen_set)) if gen_set else float("nan"),
        "interaction_jaccard": float(len(shared) / len(union)) if union else float("nan"),
    }


def prefixed_interaction_scores(prefix: str, gen_set: set, ref_set: set):
    scores = interaction_scores(gen_set, ref_set)
    return {
        f"{prefix}_gen_n": scores["interaction_gen_n"],
        f"{prefix}_ref_n": scores["interaction_ref_n"],
        f"{prefix}_shared_n": scores["interaction_shared_n"],
        f"{prefix}_recovery": scores["interaction_recovery"],
        f"{prefix}_precision": scores["interaction_precision"],
        f"{prefix}_jaccard": scores["interaction_jaccard"],
    }


def per_type_interaction_scores(gen_set: set, ref_set: set):
    out = {}
    for group_name in INTERACTION_GROUPS:
        prefix = f"interaction_{group_name}"
        out.update(
            prefixed_interaction_scores(
                prefix,
                filter_interaction_group(gen_set, group_name),
                filter_interaction_group(ref_set, group_name),
            )
        )
    return out


def posecheck_for_group(
    items,
    protein_path: Path,
    reference_mol,
    reduce_path: str,
    calculate_strain: bool = False,
    calculate_clashes: bool = True,
    calculate_interactions: bool = True,
):
    from posecheck import PoseCheck

    base = {
        "posecheck_clashes": float("nan"),
        "posecheck_strain": float("nan"),
        "posecheck_interaction_count": float("nan"),
        "interaction_gen_n": float("nan"),
        "interaction_ref_n": float("nan"),
        "interaction_shared_n": float("nan"),
        "interaction_recovery": float("nan"),
        "interaction_precision": float("nan"),
        "interaction_jaccard": float("nan"),
    }
    for group_name in INTERACTION_GROUPS:
        prefix = f"interaction_{group_name}"
        base.update(
            {
                f"{prefix}_gen_n": float("nan"),
                f"{prefix}_ref_n": float("nan"),
                f"{prefix}_shared_n": float("nan"),
                f"{prefix}_recovery": float("nan"),
                f"{prefix}_precision": float("nan"),
                f"{prefix}_jaccard": float("nan"),
            }
        )
    mols = [item["mol"] for item in items]
    ref_set = set()
    if calculate_interactions:
        try:
            pc_ref = PoseCheck(reduce_path=reduce_path)
            pc_ref.load_protein_from_pdb(str(protein_path))
            pc_ref.load_ligands_from_mols([reference_mol])
            ref_interactions = pc_ref.calculate_interactions()
            ref_set = interaction_set(ref_interactions, 0)
        except Exception:
            ref_set = set()

    try:
        pc = PoseCheck(reduce_path=reduce_path)
        pc.load_protein_from_pdb(str(protein_path))
        pc.load_ligands_from_mols(mols)
    except Exception:
        return [dict(base) for _ in items]

    clashes = [float("nan")] * len(items)
    if calculate_clashes:
        try:
            clashes = pc.calculate_clashes()
        except Exception:
            clashes = [float("nan")] * len(items)

    if calculate_strain:
        try:
            strains = pc.calculate_strain_energy()
        except Exception:
            strains = [float("nan")] * len(items)
    else:
        strains = [float("nan")] * len(items)

    interactions = pd.DataFrame()
    if calculate_interactions:
        try:
            interactions = pc.calculate_interactions()
        except Exception:
            interactions = pd.DataFrame()

    out = []
    for idx in range(len(items)):
        gen_set = interaction_set(interactions, idx)
        row = dict(base)
        row.update(
            {
                "posecheck_clashes": safe_float(clashes[idx] if idx < len(clashes) else float("nan")),
                "posecheck_strain": safe_float(strains[idx] if idx < len(strains) else float("nan")),
                "posecheck_interaction_count": len(gen_set),
            }
        )
        row.update(interaction_scores(gen_set, ref_set))
        row.update(per_type_interaction_scores(gen_set, ref_set))
        out.append(row)
    return out


def grouped_by_source(items):
    groups = defaultdict(list)
    for item in items:
        groups[int(item["source_data_id"])].append(item)
    return groups


def evaluate_items(
    items,
    records_by_index,
    ligand_root: Path,
    protein_root: Path,
    reduce_path: str,
    calculate_posecheck_strain: bool,
    metrics_mode: str = "all",
):
    run_posebusters = metrics_mode in {"all", "posebusters"}
    run_clashes = metrics_mode in {"all", "clashes"}
    run_interactions = metrics_mode in {"all", "interactions"}
    if run_posebusters:
        from posebusters import PoseBusters

        buster = PoseBusters(config="dock", max_workers=0)
    else:
        buster = None
    rows = []
    for source_id, group in grouped_by_source(items).items():
        record = records_by_index.get(int(source_id))
        if record is None:
            continue
        protein_path = reference_protein_path(protein_root, record)
        ligand_path = reference_ligand_path(ligand_root, record)
        if not protein_path.exists() or not ligand_path.exists():
            continue
        try:
            reference_mol = load_reference_mol(ligand_path)
        except Exception:
            continue
        mols = [item["mol"] for item in group]
        if run_posebusters:
            pb_rows = posebusters_for_group(buster, mols, protein_path)
        else:
            pb_rows = [{} for _ in group]
        if run_clashes or run_interactions:
            pc_rows = posecheck_for_group(
                group,
                protein_path,
                reference_mol,
                reduce_path,
                calculate_posecheck_strain and run_clashes,
                calculate_clashes=run_clashes,
                calculate_interactions=run_interactions,
            )
        else:
            pc_rows = [{} for _ in group]
        for item, pb, pc in zip(group, pb_rows, pc_rows):
            row = {k: v for k, v in item.items() if k != "mol"}
            row["protein_path"] = str(protein_path)
            row.update(pb)
            row.update(pc)
            rows.append(row)
    return rows


def write_outputs(rows: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "official_pose_quality_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = [
        "pb_pass_fraction",
        "pb_valid",
        "posecheck_clashes",
        "posecheck_strain",
        "posecheck_interaction_count",
        "interaction_recovery",
        "interaction_precision",
        "interaction_jaccard",
        "vina_score",
        "vina_min",
        "vina_dock",
        "qed",
        "sa",
    ]
    for group_name in INTERACTION_GROUPS:
        metric_keys.extend(
            [
                f"interaction_{group_name}_recovery",
                f"interaction_{group_name}_precision",
                f"interaction_{group_name}_jaccard",
                f"interaction_{group_name}_gen_n",
                f"interaction_{group_name}_ref_n",
                f"interaction_{group_name}_shared_n",
            ]
        )
    methods = list(dict.fromkeys(row["method"] for row in rows))
    summary = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {"n": len(subset)}
        for key in metric_keys:
            summary[method][key] = nan_summary([row.get(key) for row in subset])
    (output_dir / "official_pose_quality_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Official Pose-Quality Metrics",
        "",
        "| Method | n | PB pass Avg. (↑) | PB-valid Avg. (↑) | PoseCheck clash Avg. (↓) | Interaction recovery Avg. (↑) | Interaction precision Avg. (↑) | Interaction Jaccard Avg. (↑) | VinaDock Avg. (↓) | QED Avg. (↑) | SA Avg. (↑) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        item = summary[method]
        lines.append(
            f"| {method} | {item['n']} | "
            f"{item['pb_pass_fraction']['mean']:.3f} | "
            f"{item['pb_valid']['mean']:.3f} | "
            f"{item['posecheck_clashes']['mean']:.3f} | "
            f"{item['interaction_recovery']['mean']:.3f} | "
            f"{item['interaction_precision']['mean']:.3f} | "
            f"{item['interaction_jaccard']['mean']:.3f} | "
            f"{item['vina_dock']['mean']:.3f} | "
            f"{item['qed']['mean']:.3f} | "
            f"{item['sa']['mean']:.3f} |"
        )
    lines += [
        "",
        f"- CSV: `{csv_path}`",
        "- PoseBusters uses the official `dock` configuration, which is appropriate for generated ligands that are not required to match the reference ligand identity.",
        "- PoseCheck clashes and interactions are computed with the official PoseCheck API.",
        "- PoseCheck strain is written to CSV/JSON only when `--posecheck_strain` is enabled. The default official run omits it because PoseCheck performs expensive conformer relaxation; PoseBusters internal-energy checks are still included in the PB pass statistics.",
        "- Interaction recovery compares PoseCheck/ProLIF interaction fingerprints of generated ligands against the corresponding test-set reference ligand in the same receptor.",
        "",
        "## Per-Type Interaction Recovery",
        "",
        "| Method | H-bond R/P/J (↑) | Hydrophobic R/P/J (↑) | Salt bridge R/P/J (↑) | Pi R/P/J (↑) |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        item = summary[method]

        def group_triplet(group_name: str):
            rec = item[f"interaction_{group_name}_recovery"]["mean"]
            prec = item[f"interaction_{group_name}_precision"]["mean"]
            jac = item[f"interaction_{group_name}_jaccard"]["mean"]
            return f"{rec:.3f}/{prec:.3f}/{jac:.3f}"

        lines.append(
            f"| {method} | "
            f"{group_triplet('hbond')} | "
            f"{group_triplet('hydrophobic')} | "
            f"{group_triplet('salt_bridge')} | "
            f"{group_triplet('pi')} |"
        )
    lines += [
        "",
        "- R/P/J denotes recovery, precision, and Jaccard similarity against the corresponding reference-ligand interaction fingerprint.",
        "- H-bond merges ProLIF `HBAcceptor` and `HBDonor`; salt bridge merges `Anionic`, `Cationic`, and `Ionic`; pi merges `PiStacking`, `PiCation`, and `CationPi`.",
        "",
    ]
    (output_dir / "official_pose_quality_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", default=[], help="LABEL=/path/to/result_dir")
    parser.add_argument(
        "--metrics_file",
        action="append",
        default=[],
        help=(
            "Mixed-method metrics .pt file. Its branch_family labels methods and "
            "ligand_filename, rather than source_data_id, selects the reference pocket."
        ),
    )
    parser.add_argument("--reference_label", default="Reference")
    parser.add_argument("--include_reference", action="store_true")
    parser.add_argument("--reference_pt", required=True)
    parser.add_argument("--ligand_root", default="./data/test_set")
    parser.add_argument("--protein_root", default="./data/test_set")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--reduce_path", default="hydride")
    parser.add_argument("--max_sources", type=int, default=None)
    parser.add_argument("--source_modulus", type=int, default=None)
    parser.add_argument("--source_remainder", type=int, default=None)
    parser.add_argument("--posecheck_strain", action="store_true")
    parser.add_argument(
        "--metrics_mode",
        choices=("all", "posebusters", "clashes", "interactions"),
        default="all",
    )
    args = parser.parse_args()

    ligand_root = Path(args.ligand_root)
    protein_root = Path(args.protein_root)
    records, ref_by_index, ref_by_ligand = load_reference_records(Path(args.reference_pt))
    items = []
    if args.include_reference:
        items.extend(make_reference_items(args.reference_label, records, ligand_root))
    for label, result_dir in [parse_labeled_path(item) for item in args.method]:
        items.extend(make_generated_items(label, result_dir, ref_by_index, ref_by_ligand, args.allow_missing))
    for metrics_path in args.metrics_file:
        items.extend(make_generated_items_from_metrics(Path(metrics_path), ref_by_ligand))
    items = filter_max_sources(items, args.max_sources)
    items = filter_source_shard(items, args.source_modulus, args.source_remainder)
    if not items:
        raise RuntimeError("No valid molecules were found for official pose-quality analysis.")
    rows = evaluate_items(
        items,
        ref_by_index,
        ligand_root,
        protein_root,
        args.reduce_path,
        args.posecheck_strain,
        metrics_mode=args.metrics_mode,
    )
    if not rows:
        raise RuntimeError("Official pose-quality analysis produced no rows.")
    write_outputs(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
