"""
MGCA-ISIC: Multi-Granularity Cross-modal Alignment for ISIC Skin Cancer Dataset
================================================================================

This package implements MGCA adapted for the ISIC-2019 skin cancer dataset,
adding classification heads for chaos and dermoscopic clues prediction.

Training Paradigms:
    1. Joint Training (train_joint.py): MGCA pre-training + classification together
    2. Two-Stage (train_finetune.py): Pre-train first, then fine-tune heads

Classification Tasks:
    - Chaos Head (2 outputs): structure_is_chaotic, colour_is_chaotic
    - Clues Head (10 outputs): 10 dermoscopic clue indicators

Modules:
    - datasets: Data loading and augmentation
    - models: Network architectures and training modules  
    - configs: Configuration files

Usage:
    # Joint training
    python train_joint.py --max_epochs 50 --batch_size 32
    
    # Two-stage fine-tuning
    python train_finetune.py --checkpoint checkpoints/joint/xxx/last.ckpt
"""

__version__ = "0.2.0"
