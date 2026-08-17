#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

CONFIG="${CONFIG:-configs/sampling_train.yml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/trajectories}"
GAMMAS="${GAMMAS:-0 150 350}"
DATA_START="${DATA_START:-0}"
DATA_END="${DATA_END:-99}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
BATCH_SIZE="${BATCH_SIZE:-5}"
SEED="${SEED:-2021}"

require_file "$CONFIG"
require_file pretrained_models/pretrained_flow.pt
require_file pretrained_models/atom_num_model.pt
require_file data/crossdocked_pocket10_pose_split.pt
require_file data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb

for gamma in $GAMMAS; do
  result_dir="$OUTPUT_ROOT/gamma_${gamma}/train"
  mkdir -p "$result_dir"
  for data_id in $(seq "$DATA_START" "$DATA_END"); do
    result_file="$result_dir/result_${data_id}.pt"
    if [[ -f "$result_file" && "${FORCE:-0}" != "1" ]]; then
      echo "[skip] $result_file"
      continue
    fi
    "$PYTHON_BIN" scripts/sample_flow_VP_paflow_prior_guide.py \
      --config "$CONFIG" \
      --subset train \
      --result_path "$result_dir" \
      --batch_size "$BATCH_SIZE" \
      --num_samples "$NUM_SAMPLES" \
      --seed "$SEED" \
      --pos_grad_w "$gamma" \
      --device "$DEVICE" \
      -i "$data_id"
  done
done
