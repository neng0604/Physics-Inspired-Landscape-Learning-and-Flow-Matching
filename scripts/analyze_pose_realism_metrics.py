from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


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
    mask = np.isfinite(arr)
    if int(mask.sum()) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "n": 0,
        }
    valid = arr[mask]
    return {
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "std": float(valid.std()),
        "q25": float(np.percentile(valid, 25)),
        "q75": float(np.percentile(valid, 75)),
        "n": int(valid.size),
    }


def parse_method(item: str):
    if "=" not in item:
        raise ValueError(f"Expected LABEL=/path/to/result_dir, got {item!r}")
    label, path = item.split("=", 1)
    return label, Path(path)


def find_metrics_path(result_dir: Path) -> Path | None:
    candidates = [
        result_dir / "eval_results_with_reference" / "metrics_-1.pt",
        result_dir / "eval_results" / "metrics_-1.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def row_affinity(row: dict, key: str):
    value = (row.get("vina") or {}).get(key)
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, dict):
        return safe_float(value.get("affinity"))
    return safe_float(value)


def ligand_positions(row: dict):
    pos = row.get("pred_pos")
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr


def mol_positions(mol):
    if mol is None or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    coords = []
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(idx)
        coords.append((float(pos.x), float(pos.y), float(pos.z)))
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float32)


def pdb_positions(path: Path):
    coords = []
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            if atom_name.upper().startswith("H") or element.upper() == "H":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue
            coords.append((x, y, z))
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float32)


def load_result(result_dir: Path, row: dict, cache: dict[str, dict]):
    source = row.get("source_result_file")
    if source:
        result_path = result_dir / str(source)
    else:
        source_id = row.get("source_data_id", row.get("subset_index", 0))
        result_path = result_dir / f"result_{int(source_id)}.pt"
    key = str(result_path)
    if key not in cache:
        cache[key] = torch.load(result_path, map_location="cpu", weights_only=False)
    return cache[key]


def protein_positions(result: dict):
    data = result.get("data")
    if data is None or not hasattr(data, "protein_pos"):
        return None
    pos = data.protein_pos.detach().cpu().numpy() if torch.is_tensor(data.protein_pos) else np.asarray(data.protein_pos)
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        return None
    return pos


def clash_metrics(lig_pos: np.ndarray, prot_pos: np.ndarray, clash_distance: float, severe_distance: float):
    if lig_pos is None or prot_pos is None or lig_pos.size == 0 or prot_pos.size == 0:
        return {
            "min_pl_distance": float("nan"),
            "clash_count": float("nan"),
            "severe_clash_count": float("nan"),
            "clash_per_atom": float("nan"),
            "severe_clash_per_atom": float("nan"),
            "soft_clash_energy": float("nan"),
            "overburied_fraction": float("nan"),
        }
    diff = lig_pos[:, None, :] - prot_pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    n_lig = max(int(lig_pos.shape[0]), 1)
    clash = dist < float(clash_distance)
    severe = dist < float(severe_distance)
    nearest = dist.min(axis=1)
    return {
        "min_pl_distance": float(np.min(dist)),
        "clash_count": int(clash.sum()),
        "severe_clash_count": int(severe.sum()),
        "clash_per_atom": float(clash.sum() / n_lig),
        "severe_clash_per_atom": float(severe.sum() / n_lig),
        "soft_clash_energy": float(np.square(np.maximum(float(clash_distance) - dist, 0.0)).sum() / n_lig),
        "overburied_fraction": float(np.mean(nearest < 2.4)),
    }


def mol_with_positions(mol, pos: np.ndarray):
    if mol is None or pos is None:
        return None
    out = Chem.Mol(mol)
    n = min(out.GetNumAtoms(), int(pos.shape[0]))
    if n == 0:
        return None
    if out.GetNumConformers() == 0:
        conf = Chem.Conformer(out.GetNumAtoms())
        out.AddConformer(conf, assignId=True)
    conf = out.GetConformer()
    for idx in range(n):
        conf.SetAtomPosition(idx, tuple(float(x) for x in pos[idx]))
    return out


def force_field(mol):
    for ff_name in ("MMFF94s", "MMFF94", "UFF"):
        try:
            if ff_name.startswith("MMFF"):
                props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=ff_name)
                if props is None:
                    continue
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0)
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
            if ff is not None:
                return ff_name, ff
        except Exception:
            continue
    return "", None


def molcraft_uff_energy(mol):
    # MolCRAFT evaluates PoseCheck commit 57a1938. Its strain energy uses
    # UFF with explicit hydrogens and returns total energy, not per-atom energy.
    work = Chem.Mol(mol)
    work = Chem.AddHs(work, addCoords=True)
    ff = AllChem.UFFGetMoleculeForceField(work, confId=0)
    if ff is None:
        return float("nan")
    return round(float(ff.CalcEnergy()), 2)


def molcraft_relax_mol(mol):
    work = deepcopy(mol)
    try:
        Chem.GetSSSR(work)
    except Exception:
        pass
    work = Chem.AddHs(work, addCoords=True)
    AllChem.EmbedMolecule(work, randomSeed=0xF00D)
    AllChem.UFFOptimizeMolecule(work)
    return work


def molcraft_strain_energy(mol):
    try:
        before = molcraft_uff_energy(mol)
        relaxed = molcraft_relax_mol(mol)
        after = molcraft_uff_energy(relaxed)
        if not (math.isfinite(before) and math.isfinite(after)):
            return float("nan")
        return float(before - after)
    except Exception:
        return float("nan")


def strain_metrics(mol, pos: np.ndarray, max_iters: int):
    placed = mol_with_positions(mol, pos)
    if placed is None:
        return {
            "strain_energy": float("nan"),
            "strain_per_atom": float("nan"),
            "ff_energy_per_atom": float("nan"),
            "ff_name": "",
            "molcraft_se": float("nan"),
            "molcraft_se_per_atom": float("nan"),
        }
    try:
        Chem.SanitizeMol(placed)
    except Exception:
        pass
    n_atoms = max(int(placed.GetNumAtoms()), 1)
    molcraft_se = molcraft_strain_energy(placed)
    molcraft_se_per_atom = molcraft_se / n_atoms if math.isfinite(molcraft_se) else float("nan")
    ff_name, ff = force_field(placed)
    if ff is None:
        return {
            "strain_energy": float("nan"),
            "strain_per_atom": float("nan"),
            "ff_energy_per_atom": float("nan"),
            "ff_name": "",
            "molcraft_se": float(molcraft_se),
            "molcraft_se_per_atom": float(molcraft_se_per_atom),
        }
    try:
        energy_before = float(ff.CalcEnergy())
        opt = Chem.Mol(placed)
        _, opt_ff = force_field(opt)
        if opt_ff is None:
            return {
                "strain_energy": float("nan"),
                "strain_per_atom": float("nan"),
                "ff_energy_per_atom": energy_before / n_atoms,
                "ff_name": ff_name,
                "molcraft_se": float(molcraft_se),
                "molcraft_se_per_atom": float(molcraft_se_per_atom),
            }
        opt_ff.Minimize(maxIts=int(max_iters))
        energy_after = float(opt_ff.CalcEnergy())
        strain = max(energy_before - energy_after, 0.0)
        return {
            "strain_energy": float(strain),
            "strain_per_atom": float(strain / n_atoms),
            "ff_energy_per_atom": float(energy_before / n_atoms),
            "ff_name": ff_name,
            "molcraft_se": float(molcraft_se),
            "molcraft_se_per_atom": float(molcraft_se_per_atom),
        }
    except Exception:
        return {
            "strain_energy": float("nan"),
            "strain_per_atom": float("nan"),
            "ff_energy_per_atom": float("nan"),
            "ff_name": ff_name,
            "molcraft_se": float(molcraft_se),
            "molcraft_se_per_atom": float(molcraft_se_per_atom),
        }


def parse_pdbqt_pose_coords(pose: str):
    coords = []
    for line in str(pose).splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except Exception:
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                x, y, z = map(float, parts[5:8])
            except Exception:
                continue
        parts = line.split()
        atom_name = parts[2] if len(parts) > 2 else ""
        atom_type = parts[-1] if parts else ""
        if atom_name.upper().startswith("H") or atom_type.upper().startswith("H"):
            continue
        coords.append((x, y, z))
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float32)


def redocked_rmsd(row: dict, lig_pos: np.ndarray):
    dock = (row.get("vina") or {}).get("dock")
    if not isinstance(dock, (list, tuple)) or not dock:
        return float("nan")
    pose = dock[0].get("pose") if isinstance(dock[0], dict) else None
    coords = parse_pdbqt_pose_coords(pose or "")
    if coords is None or lig_pos is None:
        return float("nan")
    n = min(int(coords.shape[0]), int(lig_pos.shape[0]))
    if n == 0:
        return float("nan")
    diff = lig_pos[:n] - coords[:n]
    return float(np.sqrt(np.mean(np.square(diff).sum(axis=1))))


def analyze_method(label: str, result_dir: Path, args):
    metrics_path = find_metrics_path(result_dir)
    if metrics_path is None:
        if args.allow_missing:
            return []
        raise FileNotFoundError(f"No metrics_-1.pt under {result_dir}")
    metrics = torch.load(metrics_path, map_location="cpu", weights_only=False)
    rows = metrics.get("all_results", [])
    result_cache = {}
    out_rows = []
    for row in tqdm(rows, desc=label):
        if args.max_rows_per_method is not None and len(out_rows) >= args.max_rows_per_method:
            break
        mol = row.get("mol")
        lig_pos = ligand_positions(row)
        if mol is None or lig_pos is None:
            continue
        try:
            result = load_result(result_dir, row, result_cache)
            prot_pos = protein_positions(result)
        except Exception:
            prot_pos = None
        if prot_pos is None:
            prot_pos = pdb_positions(reference_protein_path(Path(args.protein_root), row))
        item = {
            "method": label,
            "result_dir": str(result_dir),
            "source_data_id": row.get("source_data_id"),
            "source_sample_idx": row.get("source_sample_idx"),
            "ligand_filename": row.get("ligand_filename", ""),
            "num_atoms": int(lig_pos.shape[0]),
            "vina_score": row_affinity(row, "score_only"),
            "vina_min": row_affinity(row, "minimize"),
            "vina_dock": row_affinity(row, "dock"),
            "qed": safe_float((row.get("chem_results") or {}).get("qed")),
            "sa": safe_float((row.get("chem_results") or {}).get("sa")),
            "redocked_rmsd": redocked_rmsd(row, lig_pos),
        }
        item.update(clash_metrics(lig_pos, prot_pos, args.clash_distance, args.severe_clash_distance))
        item.update(strain_metrics(mol, lig_pos, args.strain_max_iters))
        out_rows.append(item)
    return out_rows


def load_reference_mol(ligand_root: Path, ligand_filename: str):
    path = ligand_root / ligand_filename
    suppl = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = next(iter(suppl), None)
    if mol is None:
        raise ValueError(f"Failed to load reference ligand: {path}")
    return mol


def reference_protein_path(protein_root: Path, record: dict):
    protein_filename = record.get("protein_filename")
    if protein_filename:
        path = protein_root / str(protein_filename)
        if path.exists():
            return path
    ligand_filename = str(record.get("ligand_filename", ""))
    if ligand_filename:
        fallback = protein_root / Path(ligand_filename).parent / f"{Path(ligand_filename).name[:10]}.pdb"
        if fallback.exists():
            return fallback
    return protein_root / str(protein_filename or "")


def analyze_reference(label: str, reference_path: Path, args):
    payload = torch.load(reference_path, map_location="cpu", weights_only=False)
    records = list((payload.get("records") or {}).values())
    out_rows = []
    ligand_root = Path(args.ligand_root)
    protein_root = Path(args.protein_root)
    for record in tqdm(records, desc=label):
        if args.max_rows_per_method is not None and len(out_rows) >= args.max_rows_per_method:
            break
        if record.get("chem_results") is None or record.get("vina") is None:
            continue
        try:
            mol = load_reference_mol(ligand_root, str(record.get("ligand_filename", "")))
        except Exception:
            continue
        lig_pos = mol_positions(mol)
        if lig_pos is None:
            continue
        prot_pos = pdb_positions(reference_protein_path(protein_root, record))
        row = {"vina": record.get("vina")}
        item = {
            "method": label,
            "result_dir": str(reference_path),
            "source_data_id": record.get("index"),
            "source_sample_idx": "",
            "ligand_filename": record.get("ligand_filename", ""),
            "num_atoms": int(lig_pos.shape[0]),
            "vina_score": row_affinity(row, "score_only"),
            "vina_min": row_affinity(row, "minimize"),
            "vina_dock": row_affinity(row, "dock"),
            "qed": safe_float((record.get("chem_results") or {}).get("qed")),
            "sa": safe_float((record.get("chem_results") or {}).get("sa")),
            "redocked_rmsd": redocked_rmsd(row, lig_pos),
        }
        item.update(clash_metrics(lig_pos, prot_pos, args.clash_distance, args.severe_clash_distance))
        item.update(strain_metrics(mol, lig_pos, args.strain_max_iters))
        out_rows.append(item)
    return out_rows


def write_outputs(rows: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pose_realism_metrics.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    metric_keys = [
        "min_pl_distance",
        "clash_count",
        "severe_clash_count",
        "clash_per_atom",
        "severe_clash_per_atom",
        "soft_clash_energy",
        "overburied_fraction",
        "strain_energy",
        "strain_per_atom",
        "molcraft_se",
        "molcraft_se_per_atom",
        "ff_energy_per_atom",
        "redocked_rmsd",
        "vina_dock",
        "vina_min",
        "vina_score",
        "qed",
        "sa",
    ]
    methods = list(dict.fromkeys(row["method"] for row in rows))
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {"n": len(subset)}
        for key in metric_keys:
            summary[method][key] = nan_summary([row.get(key) for row in subset])

    (output_dir / "pose_realism_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Pose Realism Metrics",
        "",
        "| Method | n | clash/atom Avg. | severe/atom Avg. | MolCRAFT SE 25% | MolCRAFT SE 50% | MolCRAFT SE 75% | strain/atom Avg. | strain/atom Med. | redocked RMSD Avg. | redocked RMSD Med. | VinaDock Avg. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        item = summary[method]
        lines.append(
            f"| {method} | {item['n']} | "
            f"{item['clash_per_atom']['mean']:.4f} | "
            f"{item['severe_clash_per_atom']['mean']:.4f} | "
            f"{item['molcraft_se']['q25']:.1f} | {item['molcraft_se']['median']:.1f} | {item['molcraft_se']['q75']:.1f} | "
            f"{item['strain_per_atom']['mean']:.4f} | {item['strain_per_atom']['median']:.4f} | "
            f"{item['redocked_rmsd']['mean']:.4f} | {item['redocked_rmsd']['median']:.4f} | "
            f"{item['vina_dock']['mean']:.3f} |"
        )
    lines += [
        "",
        f"- CSV: `{csv_path}`",
        "- MolCRAFT SE follows the PoseCheck-style total strain-energy statistic used by MolCRAFT: UFF energy of the input pose minus UFF energy of a relaxed embedded conformer, reported as 25/50/75 percentiles.",
        "- Lower clash, severe clash, strain, and redocked RMSD are better.",
        "- `strain/atom` is the local RDKit force-field relaxation diagnostic retained for internal comparison; it is not the MolCRAFT SE metric.",
        "- Redocked RMSD compares generated coordinates to Vina redocked coordinates in the receptor frame without alignment.",
        "",
    ]
    (output_dir / "pose_realism_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", default=[], help="LABEL=/path/to/result_dir")
    parser.add_argument("--reference_pt", action="append", default=[], help="LABEL=/path/to/reference_metrics.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--clash_distance", type=float, default=2.0)
    parser.add_argument("--severe_clash_distance", type=float, default=1.5)
    parser.add_argument("--strain_max_iters", type=int, default=200)
    parser.add_argument("--ligand_root", default="./data/test_set")
    parser.add_argument("--protein_root", default="./data/test_set")
    parser.add_argument("--max_rows_per_method", type=int, default=None)
    args = parser.parse_args()

    rows = []
    for label, reference_path in [parse_method(item) for item in args.reference_pt]:
        reference_rows = analyze_reference(label, reference_path, args)
        rows.extend(reference_rows)
    for label, result_dir in [parse_method(item) for item in args.method]:
        method_rows = analyze_method(label, result_dir, args)
        rows.extend(method_rows)
    if not rows:
        raise RuntimeError("No pose realism rows were produced.")
    write_outputs(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
