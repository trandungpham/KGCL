# MGCA-ISIC: Multi-Granularity Cross-modal Alignment for Skin Cancer

This package adapts the MGCA framework for the ISIC-2019 skin cancer dataset, adding **dual classification heads** for chaos and dermoscopic clues prediction.

## Architecture

```
                         ┌─────────────────┐
                         │   Input Image   │
                         │   (224×224×3)   │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
           ┌────────▼────────┐         ┌────────▼────────┐
           │  Image Encoder  │         │  Text Encoder   │
           │   (ResNet-50)   │         │ (BioClinicalBERT)│
           └────────┬────────┘         └────────┬────────┘
                    │                           │
     ┌──────────────┼──────────────┬────────────┼──────────────┐
     │              │              │            │              │
     ▼              ▼              ▼            ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│   ITA   │   │   CTA   │   │   CPA   │   │  CHAOS  │   │  CLUES  │
│ λ₁=1.0  │   │ λ₂=0.7  │   │ λ₃=0.5  │   │  HEAD   │   │  HEAD   │
│ InfoNCE │   │ Attn    │   │Sinkhorn │   │(2 out)  │   │(10 out) │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

## Classification Tasks

### Chaos Head (2 binary outputs)
- `structure_is_chaotic`: Structural chaos in the lesion
- `colour_is_chaotic`: Color chaos in the lesion

### Clues Head (10 multi-label outputs)
| # | Clue | Description |
|---|------|-------------|
| 1 | Eccentric structureless area | Asymmetric pigmentation |
| 2 | Thick lines | Irregular thick lines |
| 3 | Grey blue structures | Blue-gray coloration |
| 4 | Black dots & clods | Irregular pigmented dots |
| 5 | Radial lines / pseudopods | Streaming patterns |
| 6 | White lines | Regression structures |
| 7 | Polymorphous vessels | Atypical vascular patterns |
| 8 | Parallel ridge lines | Acral patterns |
| 9 | Angulated lines | Zigzag patterns |
| 10 | No clues | Absence of specific findings |

## Training Paradigms

### 1. Joint Training (`train_joint.py`)
Pre-training and classification simultaneously:
```bash
# Quick test
python train_joint.py --max_epochs 1 --batch_size 8 --quick_test

# Full training
python train_joint.py --max_epochs 50 --batch_size 32

# With ViT encoder
python train_joint.py --img_encoder vit_base --batch_size 16
```

### 2. Two-Stage Fine-tuning (`train_finetune.py`)
Following the MGCA paper's approach:
```bash
# Fine-tune from checkpoint
python train_finetune.py --checkpoint checkpoints/joint/xxx/last.ckpt

# With ImageNet backbone (no pre-training)
python train_finetune.py --max_epochs 50 --batch_size 32

# Full fine-tuning (unfreeze backbone)
python train_finetune.py --no_freeze_backbone
```

## Loss Function

```
Total Loss = λ₁·L_ITA + λ₂·L_CTA + λ₃·L_CPA + λ_chaos·L_chaos + λ_clues·L_clues
```

| Component | Description | Default Weight |
|-----------|-------------|----------------|
| L_ITA | Instance-wise Text-Image Alignment (InfoNCE) | λ₁ = 1.0 |
| L_CTA | Cross-modal Token Alignment | λ₂ = 0.7 |
| L_CPA | Cross-modal Prototype Alignment | λ₃ = 0.5 |
| L_chaos | BCE loss for chaos prediction | λ_chaos = 1.0 |
| L_clues | BCE loss for clues prediction | λ_clues = 1.0 |

## Directory Structure

```
MGCA-ISIC/
├── __init__.py
├── README.md
├── requirements.txt
├── train_joint.py          # Joint training script
├── train_finetune.py       # Two-stage fine-tuning script
├── configs/
│   └── bert_config.json
├── datasets/
│   ├── __init__.py
│   ├── constants.py        # Label definitions
│   ├── data_module.py
│   ├── isic_dataset.py     # ISIC dataset with text generation
│   ├── pretrain_dataset.py
│   ├── transforms.py
│   └── utils.py
└── models/
    ├── __init__.py
    ├── ssl_finetuner.py
    ├── backbones/
    │   ├── encoder.py      # Image & Text encoders
    │   ├── cnn_backbones.py
    │   ├── med.py
    │   ├── transformer_model.py
    │   └── vits.py
    └── kgcl/
        ├── __init__.py
        ├── kgcl_module.py      # MGCA_ISIC (joint training)
        └── isic_finetuner.py   # ISICFineTuner (two-stage)
```

## Requirements

```bash
pip install -r requirements.txt
```

## Reference

Based on the MGCA paper:
> Wang et al., "Multi-Granularity Cross-modal Alignment for Generalized Medical Visual Representation Learning", NeurIPS 2022
