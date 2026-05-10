#!/usr/bin/env bash
# Run all transformer-based backbone experiments for the ISIC multi-task framework.
#
# All runs enable --use_agentic_aux (per-concept PerConceptAuxiliaryHub) and
# --use_weighted_sampler, matching the sweep.yaml command section.
#
# Backbones:
#   swin_base_patch4_window7_224        (from sweep.yaml)
#   swin_small_patch4_window7_224
#   swin_large_patch4_window7_224_22k
#   vit_base_patch16_224
#   deit3_base_patch16_224
#
# Each backbone is run against 6 representative configs derived from sweep.yaml:
#   low_lr / mid_lr / high_lr — LR sweep, 4 agents, 128 hidden
#   diag_heavy — diag-focused loss, low λ_confidence
#   clue_heavy — clue-focused loss, high λ_confidence
#   aux_light  — 2 agents instead of 4 (capacity ablation)
#
# Usage:
#   bash run_transformer_experiments.sh                        # all backbones, all configs
#   bash run_transformer_experiments.sh --dry_run              # print commands only
#   bash run_transformer_experiments.sh --logger csv           # skip W&B
#   bash run_transformer_experiments.sh --quick_test           # limited batches
#   bash run_transformer_experiments.sh --max_epochs 50
#   bash run_transformer_experiments.sh --backbones "swin_base_patch4_window7_224 vit_base_patch16_224"
#   bash run_transformer_experiments.sh --configs "low_lr mid_lr"

set -euo pipefail

# ── Defaults (override via flags below) ───────────────────────────────────────
MAX_EPOCHS=100
LOGGER="wandb"
WANDB_PROJECT="KGCL-transformers"
WANDB_ENTITY=""
SEED=42
NUM_WORKERS=4
PRECISION="32"
QUICK_TEST=false
DRY_RUN=false
STOP_ON_FAILURE=false
LOG_FILE=""

BACKBONES=(
    "swin_base_patch4_window7_224"
    "swin_small_patch4_window7_224"
    "swin_large_patch4_window7_224_22k"
    "vit_base_patch16_224"
    "deit3_base_patch16_224"
)

# Config columns:
#   id  lr  wd  bs  λ_diag  λ_clue  λ_chaos  λ_align  clue_sc  diag_sc  max_pw  n_agents  hidden_dim  λ_conf
CONFIGS=(
    "low_lr     1e-5   1e-4   16  1.0   1.0   1.0   0.1   1.0   0.5   15.0  4  128  0.1"
    "mid_lr     3e-5   1e-4   16  1.0   2.0   1.0   0.1   1.0   0.5   15.0  4  128  0.1"
    "high_lr    1e-4   1e-4   16  1.0   2.0   0.5   0.3   1.0   0.5   15.0  4  256  0.1"
    "diag_heavy 3e-5   5e-4    8  1.5   1.0   0.5   0.0   0.75  0.25  10.0  4  128  0.05"
    "clue_heavy 3e-5   1e-5   16  0.75  3.0   1.0   0.3   1.25  0.75  20.0  4  128  0.2"
    "aux_light  3e-5   1e-4   16  1.0   2.0   1.0   0.1   1.0   0.5   15.0  2  128  0.1"
)

ALL_CONFIG_IDS=("low_lr" "mid_lr" "high_lr" "diag_heavy" "clue_heavy" "aux_light")
SELECTED_CONFIG_IDS=()

# ── Parse arguments ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run)          DRY_RUN=true;             shift ;;
        --quick_test)       QUICK_TEST=true;           shift ;;
        --stop_on_failure)  STOP_ON_FAILURE=true;      shift ;;
        --max_epochs)       MAX_EPOCHS="$2";           shift 2 ;;
        --logger)           LOGGER="$2";               shift 2 ;;
        --wandb_project)    WANDB_PROJECT="$2";        shift 2 ;;
        --wandb_entity)     WANDB_ENTITY="$2";         shift 2 ;;
        --seed)             SEED="$2";                 shift 2 ;;
        --num_workers)      NUM_WORKERS="$2";          shift 2 ;;
        --precision)        PRECISION="$2";            shift 2 ;;
        --log_file)         LOG_FILE="$2";             shift 2 ;;
        --backbones)
            IFS=' ' read -r -a BACKBONES <<< "$2";    shift 2 ;;
        --configs)
            IFS=' ' read -r -a SELECTED_CONFIG_IDS <<< "$2"; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: bash $0 [--dry_run] [--quick_test] [--max_epochs N] [--logger wandb|csv]" >&2
            echo "              [--backbones \"backbone1 backbone2\"] [--configs \"low_lr mid_lr\"]" >&2
            exit 1 ;;
    esac
done

# Filter configs if --configs was specified
if [[ ${#SELECTED_CONFIG_IDS[@]} -gt 0 ]]; then
    FILTERED_CONFIGS=()
    for cfg_line in "${CONFIGS[@]}"; do
        cfg_id=$(echo "$cfg_line" | awk '{print $1}')
        for sel in "${SELECTED_CONFIG_IDS[@]}"; do
            if [[ "$cfg_id" == "$sel" ]]; then
                FILTERED_CONFIGS+=("$cfg_line")
                break
            fi
        done
    done
    CONFIGS=("${FILTERED_CONFIGS[@]}")
fi

# ── Setup ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="$SCRIPT_DIR/../transformer_results_${TIMESTAMP}.log"
fi

TOTAL=$(( ${#BACKBONES[@]} * ${#CONFIGS[@]} ))
RUN_IDX=0
FAILED=0

log() { echo "$*" | tee -a "$LOG_FILE"; }

log "========================================================================"
log "  Transformer Backbone Sweep — ${#BACKBONES[@]} backbone(s) × ${#CONFIGS[@]} config(s) = $TOTAL run(s)"
log "  Started   : $(date)"
log "  Log file  : $LOG_FILE"
log "  Max epochs: $MAX_EPOCHS   Logger: $LOGGER   Seed: $SEED"
log "  Dry run   : $DRY_RUN   Quick test: $QUICK_TEST"
log "========================================================================"

cd "$SCRIPT_DIR"

# ── Main loop ──────────────────────────────────────────────────────────────────
for BACKBONE in "${BACKBONES[@]}"; do
    for CFG_LINE in "${CONFIGS[@]}"; do
        read -r CFG_ID LR WD BS L_DIAG L_CLUE L_CHAOS L_ALIGN CLUE_SCALE DIAG_SCALE MAX_PW N_AGENTS HIDDEN_DIM L_CONF <<< "$CFG_LINE"

        RUN_IDX=$(( RUN_IDX + 1 ))
        RUN_NAME="transformer-${BACKBONE}-${CFG_ID}"
        LABEL="[${RUN_IDX}/${TOTAL}] ${BACKBONE} / ${CFG_ID}"

        EXTRA_ARGS=""
        if $QUICK_TEST; then EXTRA_ARGS="$EXTRA_ARGS --quick_test"; fi
        if [[ -n "$WANDB_ENTITY" ]]; then EXTRA_ARGS="$EXTRA_ARGS --wandb_entity $WANDB_ENTITY"; fi

        CMD="python train_multitask.py \
  --phase                    all \
  --backbone_name            $BACKBONE \
  --learning_rate            $LR \
  --weight_decay             $WD \
  --batch_size               $BS \
  --max_epochs               $MAX_EPOCHS \
  --lambda_diag              $L_DIAG \
  --lambda_clue              $L_CLUE \
  --lambda_chaos             $L_CHAOS \
  --lambda_align             $L_ALIGN \
  --sampler_clue_weight_scale $CLUE_SCALE \
  --sampler_diag_weight_scale $DIAG_SCALE \
  --max_pos_weight           $MAX_PW \
  --use_agentic_aux \
  --num_aux_agents           $N_AGENTS \
  --aux_agent_hidden_dim     $HIDDEN_DIM \
  --lambda_confidence        $L_CONF \
  --use_weighted_sampler \
  --seed                     $SEED \
  --num_workers              $NUM_WORKERS \
  --precision                $PRECISION \
  --logger                   $LOGGER \
  --wandb_project            $WANDB_PROJECT \
  --wandb_run_name           $RUN_NAME \
  --auto_finetune_task_mode  multitask \
  $EXTRA_ARGS"

        log ""
        log "--------------------------------------------------------------------"
        log "Starting : $LABEL"
        log "Run name : $RUN_NAME"
        log "Command  : $CMD"
        log "--------------------------------------------------------------------"

        if $DRY_RUN; then
            log "[DRY-RUN] Skipping execution."
            continue
        fi

        START_TIME=$(date +%s)

        set +e
        eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
        RC=${PIPESTATUS[0]}
        set -e

        END_TIME=$(date +%s)
        ELAPSED=$(( END_TIME - START_TIME ))
        ELAPSED_MIN=$(echo "scale=1; $ELAPSED / 60" | bc)

        if [[ $RC -eq 0 ]]; then
            log "Result   : OK  (${ELAPSED_MIN} min)"
        else
            FAILED=$(( FAILED + 1 ))
            log "Result   : FAILED rc=$RC  (${ELAPSED_MIN} min)"
            if $STOP_ON_FAILURE; then
                log "Aborting sweep after failure (--stop_on_failure)."
                exit 1
            fi
        fi
    done
done

# ── Summary ────────────────────────────────────────────────────────────────────
log ""
log "========================================================================"
log "  Sweep complete: $RUN_IDX run(s), $FAILED failure(s)"
log "  Finished: $(date)"
log "  Log file: $LOG_FILE"
log "========================================================================"

exit $FAILED
