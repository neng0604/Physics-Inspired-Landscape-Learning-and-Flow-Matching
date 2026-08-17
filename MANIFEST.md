# Release manifest

## Main LandFlow path

- `models/hjb_value_model.py`: pairwise scalar route potential.
- `models/molopt_score_model_guide.py`: calibrated potential-gradient residual
  integrated into frozen PAFlow sampling.
- `scripts/sample_flow_VP_paflow_prior_guide.py`: base and LandFlow sampling.
- `scripts/build_affinity_hjb_bank.py`: state cost and future-quality target
  construction.
- `scripts/train_hjb_value.py`: route-potential training objective.
- `workflows/01_collect_trajectories.sh` through
  `workflows/05_evaluate.sh`: ordered reproduction entry points.

## Released artifacts

Exact byte sizes and SHA-256 values are recorded in `artifacts.json`.

- model repository: frozen PAFlow, atom-number predictor, LandFlow potential;
- dataset repository: 3,000-trajectory bank, fixed split, atom-number metadata,
  and reference metric cache;
- external download: processed CrossDocked LMDB and test receptor files.

Generated results, logs, caches, and environment-specific paths are excluded
from GitHub by `.gitignore`.
