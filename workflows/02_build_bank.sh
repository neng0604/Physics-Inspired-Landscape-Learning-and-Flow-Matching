#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

CONFIG="${CONFIG:-configs/sampling_train.yml}"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-outputs/trajectories}"
OUTPUT_BANK="${OUTPUT_BANK:-artifacts/trajectory_bank.pt}"
GAMMAS="${GAMMAS:-0 150 350}"

require_file "$CONFIG"
args=()
for gamma in $GAMMAS; do
  result_dir="$TRAJECTORY_ROOT/gamma_${gamma}/train"
  require_dir "$result_dir"
  args+=(--gamma_result "${gamma}=${result_dir}")
done

mkdir -p "$(dirname "$OUTPUT_BANK")"
"$PYTHON_BIN" scripts/build_affinity_hjb_bank.py \
  "${args[@]}" \
  --output "$OUTPUT_BANK" \
  --config "$CONFIG" \
  --subset train \
  --terminal_source expert_affinity \
  --expert_batch_size 10 \
  --device "$DEVICE"
