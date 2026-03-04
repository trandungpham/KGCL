"""
MGCA-ISIC: Multi-Granularity Cross-modal Alignment for ISIC Skin Cancer Dataset
with Chaos and Clues Classification Heads

Two training paradigms are supported:
1. Joint Training (MGCA_ISIC): Pre-train + classify simultaneously
2. Two-Stage (ISICFineTuner): Pre-train first, then fine-tune classification heads
"""

from .kgcl_module import MGCA_ISIC
from .isic_finetuner import ISICFineTuner, DualHeadClassifier
from ...datasets.constants import CHAOS_LABELS, CLUE_LABELS

__all__ = [
    "MGCA_ISIC",
    "ISICFineTuner",
    "DualHeadClassifier",
    "CHAOS_LABELS", 
    "CLUE_LABELS"
]
