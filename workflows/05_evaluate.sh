#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

RESULT_PATH="${RESULT_PATH:-results/landflow/test}"
DOCKING_MODE="${DOCKING_MODE:-vina_dock}"
EXHAUSTIVENESS="${EXHAUSTIVENESS:-16}"
PROTEIN_ROOT="${PROTEIN_ROOT:-data/test_set}"

require_dir "$RESULT_PATH"
if [[ "$DOCKING_MODE" != "none" ]]; then
  require_dir "$PROTEIN_ROOT"
fi

args=(
  "$RESULT_PATH"
  --docking_mode "$DOCKING_MODE"
  --exhaustiveness "$EXHAUSTIVENESS"
  --protein_root "$PROTEIN_ROOT"
)
if [[ -f data/reference_metrics.pt ]]; then
  args+=(--reference_metrics_path data/reference_metrics.pt)
fi

"$PYTHON_BIN" scripts/evaluate_diffusion.py "${args[@]}"
