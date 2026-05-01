# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-task deep learning framework for skin cancer diagnosis on the ISIC-2019 dataset. The model predicts:
- **Diagnosis**: MEL (melanoma, malignant) vs. NV (melanocytic nevus, benign) — binary
- **9 dermoscopic clues**: Spatial (segmentation masks) + presence (binary multi-label)
- **2 chaos indicators**: `structure_is_chaotic` and `colour_is_chaotic` — binary each

Based on the MGCA paper (Wang et al., NeurIPS 2022). Training runs from `CGCL/`.

## Setup

```bash
pip install torch torchvision  # Install CUDA build first if needed
pip install -r requirements.txt
```

Set `WANDB_API_KEY` in `CGCL/.env` for experiment logging (or use `--logger csv` to skip W&B).

## Training Commands

All commands run from `CGCL/`:

```bash
# Phase 1 — pretrain (no diagnosis labels)
python train_multitask.py --phase pretrain --max_epochs 100 --batch_size 16 \
  --backbone_name convnext_base --lambda_align 0.5

# Phase 2 — finetune from pretrain checkpoint
python train_multitask.py --phase finetune --backbone_name convnext_base \
  --lambda_diag 2.0 --task_mode multitask \
  --phase1_ckpt checkpoints/pretrain/convnext_base/best.ckpt

# Run both phases sequentially
python train_multitask.py --phase all --max_epochs 100 --batch_size 16

# Diagnosis-only ablation (no clue/chaos auxiliary tasks)
python train_multitask.py --phase finetune --task_mode diag_only

# Quick smoke test
python train_multitask.py --phase pretrain --max_epochs 1 --batch_size 8 --quick_test
```

Key arguments:
- `--backbone_name`: `resnet50` (default), `convnext_base`, `efficientnet_b3`, `swin_base_patch4_window7_224`
- `--task_mode`: `multitask` (default) | `diag_only`
- `--logger`: `wandb` (default) | `csv`
- `--lambda_diag/clue/chaos/align`: Loss weights for each task head

## Hyperparameter Sweep

```bash
# Initialize and run a W&B sweep using sweep.yaml from repo root
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

## Architecture

**Two-phase training** with `PretrainModule` (Phase 1) and `FinetuneModule` (Phase 2), both defined in `CGCL/models/cgcl/multitask_module.py`.

**`MultiTaskNet`** (`multitask_module.py:25`):
```
Input (3, 224, 224)
  → CNN backbone (timm) with feature pyramid
  → Global Average Pool → feature vector
  → [Clue Presence Head]  : 9-dim BCE (presence of each clue)
  → [Clue Area Head]      : Conv2d → upsample → (9, 224, 224) spatial masks
  → [Chaos Head]          : 2-dim BCE (structure/colour chaos)
  → [Diagnosis Head]*     : concat(features, pooled_clue_area, chaos_logits) → 2-dim CE
```
*Diagnosis head only exists in Phase 2 (`FinetuneModule`).

**Loss functions:**
- Clue presence: BCE with logits + `pos_weight` balancing
- Chaos: BCE with logits
- Clue area alignment: BCE on spatial predictions vs. ground-truth masks
- Diagnosis: Cross-entropy with label smoothing (0.1)
- Phase 1: `L = λ_clue·L_clue + λ_chaos·L_chaos + λ_align·L_align`
- Phase 2: `L = λ_diag·L_diag + λ_clue·L_clue + λ_chaos·L_chaos + λ_align·L_align`

## Data

**Annotated dataset** (465 images) lives in `Annotated_data/`:
- `Images/` — 467 JPGs
- `GroundTruthMasks/` — 9-channel numpy masks (one per clue)
- `Vectors/` — 9-dim binary clue presence vectors
- `train.csv` / `val.csv` / `test.csv` — 325 / 70 / 70 split

**Dataset classes** (`CGCL/datasets/dataset.py`):
- `PretrainDataset` — returns `{imgs, clue_present, chaos_labels, clue_masks}` (no diagnosis)
- `FinetuneDataset` — adds `diagnosis_labels` (1 = MEL, 0 = NV)

**Transforms** (`CGCL/datasets/transforms.py`): `SpatialClueDataTransforms` applies synchronized augmentations to image + clue masks (random crop train, center crop val/test; normalize to `[-1,1]`).

**Raw data preprocessing** is handled by `Dataprocess/preprocess.py` (XML parsing → masks + vectors → CSVs). This only needs to be rerun if re-processing original annotations.

## Key Files

| File | Purpose |
|------|---------|
| `CGCL/train_multitask.py` | Entry point — CLI args, phase orchestration, trainer setup |
| `CGCL/models/cgcl/multitask_module.py` | `MultiTaskNet`, `PretrainModule`, `FinetuneModule` |
| `CGCL/datasets/dataset.py` | `PretrainDataset`, `FinetuneDataset`, `BaseDataset` |
| `CGCL/datasets/transforms.py` | `SpatialClueDataTransforms` — synced image+mask augmentation |
| `CGCL/datasets/constants.py` | `CLUES_NAMES`, `CHAOS_LABELS`, `DIAGNOSIS_LABELS` |
| `CGCL/datasets/datamodule.py` | PyTorch Lightning `DataModule` |
| `Dataprocess/preprocess.py` | One-time annotation preprocessing |
| `sweep.yaml` | W&B Bayesian hyperparameter sweep config |
