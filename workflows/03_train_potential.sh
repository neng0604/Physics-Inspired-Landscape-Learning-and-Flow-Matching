#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

BANK="${BANK:-artifacts/trajectory_bank.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/landflow_training}"
EPOCHS="${EPOCHS:-30}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:--1}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-50}"

require_file "$BANK"
mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" scripts/train_hjb_value.py \
  --bank "$BANK" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --model_arch pairwise \
  --hidden_dim 128 \
  --num_layers 3 \
  --dropout 0.10 \
  --epochs "$EPOCHS" \
  --max_train_batches "$MAX_TRAIN_BATCHES" \
  --batch_size 8 \
  --lr 1e-4 \
  --target_key G_affinity_direct \
  --hjb_u_key U_affinity_local \
  --boundary_key trajectory_affinity_cost \
  --hjb_mode residual \
  --lambda_value 0.70 \
  --lambda_hjb 0.20 \
  --lambda_boundary 1.00 \
  --lambda_rank 0.25 \
  --lambda_action_geom 0.05 \
  --action_geom_clash_weight 1.0 \
  --action_geom_center_weight 0.05 \
  --action_geom_contact_weight 0.25 \
  --best_metric val_spearman_s_G \
  --best_mode max \
  --val_fraction 0.20 \
  --val_max_batches "$VAL_MAX_BATCHES" \
  --val_balance_mode none \
  --val_time_bins 5
