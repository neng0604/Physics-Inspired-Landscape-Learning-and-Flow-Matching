#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

CONFIG="${CONFIG:-configs/sampling_test.yml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/landflow.pt}"
RESULT_PATH="${RESULT_PATH:-results/landflow/test}"
DATA_START="${DATA_START:-0}"
DATA_END="${DATA_END:-99}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
BATCH_SIZE="${BATCH_SIZE:-5}"
SEED="${SEED:-2021}"
PAFLOW_AFFINITY_GAMMA="${PAFLOW_AFFINITY_GAMMA:-250}"
RHO="${RHO:-0.30}"

require_file "$CONFIG"
require_file "$CHECKPOINT"
require_file pretrained_models/pretrained_flow.pt
require_file pretrained_models/atom_num_model.pt
require_file data/crossdocked_pocket10_pose_split.pt
require_file data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
mkdir -p "$RESULT_PATH"

for data_id in $(seq "$DATA_START" "$DATA_END"); do
  result_file="$RESULT_PATH/result_${data_id}.pt"
  if [[ -f "$result_file" && "${FORCE:-0}" != "1" ]]; then
    echo "[skip] $result_file"
    continue
  fi
  "$PYTHON_BIN" scripts/sample_flow_VP_paflow_prior_guide.py \
    --config "$CONFIG" \
    --subset test \
    --result_path "$RESULT_PATH" \
    --batch_size "$BATCH_SIZE" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --pos_grad_w "$PAFLOW_AFFINITY_GAMMA" \
    --hjb_value_checkpoint "$CHECKPOINT" \
    --hjb_value_component total \
    --hjb_value_time_mode vp_time \
    --hjb_target_base_ratio "$RHO" \
    --hjb_target_base_ratio_max_scale 100.0 \
    --hjb_control_mode normalized \
    --hjb_control_cap_ratio "$RHO" \
    --hjb_projection_mode none \
    --hjb_t0 0.50 \
    --hjb_late_taper_start 1.0 \
    --trace_velocity_components \
    --device "$DEVICE" \
    -i "$data_id"
done
