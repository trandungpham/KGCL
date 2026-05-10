#!/usr/bin/env bash
# Run both training phases for the CGCL multi-task model.
#
# Phase 1 (pretrain)  — chaos_and_clues/   1826 images, clue+chaos labels
# Phase 2 (finetune)  — Annotated_data/    325 train / 70 val / 70 test, with diagnosis
#
# Usage:
#   bash run_training.sh                          # full two-phase run
#   bash run_training.sh --quick_test             # smoke test (10 batches each phase)
#   bash run_training.sh --logger csv             # skip W&B, write CSV logs
#   bash run_training.sh --backbone_name resnet50
#   bash run_training.sh --max_epochs 50 --batch_size 8

set -euo pipefail

# ── Config (edit to taste) ─────────────────────────────────────────────────────
BACKBONE="convnext_base"
MAX_EPOCHS=100
BATCH_SIZE=16
LR=1e-4
WEIGHT_DECAY=1e-4
LAMBDA_DIAG=2.0
LAMBDA_CLUE=1.0
LAMBDA_CHAOS=1.0
LAMBDA_ALIGN=0.5
SEED=42
LOGGER="wandb"    # "wandb" | "csv"
PRECISION="32"
NUM_WORKERS=4
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load W&B key from .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

echo "========================================================"
echo "  CGCL Two-Phase Training"
echo "  Backbone : $BACKBONE"
echo "  Epochs   : $MAX_EPOCHS   Batch : $BATCH_SIZE   LR : $LR"
echo "  Logger   : $LOGGER"
echo "========================================================"

cd "$SCRIPT_DIR"

python train_multitask.py \
    --phase                   all \
    --backbone_name           "$BACKBONE" \
    --max_epochs              "$MAX_EPOCHS" \
    --batch_size              "$BATCH_SIZE" \
    --learning_rate           "$LR" \
    --weight_decay            "$WEIGHT_DECAY" \
    --lambda_diag             "$LAMBDA_DIAG" \
    --lambda_clue             "$LAMBDA_CLUE" \
    --lambda_chaos            "$LAMBDA_CHAOS" \
    --lambda_align            "$LAMBDA_ALIGN" \
    --seed                    "$SEED" \
    --logger                  "$LOGGER" \
    --precision               "$PRECISION" \
    --num_workers             "$NUM_WORKERS" \
    --auto_finetune_task_mode multitask \
    "$@"
