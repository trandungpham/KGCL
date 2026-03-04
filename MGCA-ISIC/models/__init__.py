"""
MGCA-ISIC Models Module
========================
Contains all model implementations for MGCA-ISIC:

- MGCA_ISIC: Joint pre-training + classification
- ISICFineTuner: Two-stage fine-tuning approach
- SSLFineTuner: Generic SSL fine-tuner
"""

from .ssl_finetuner import SSLFineTuner, SSLEvaluator
from .kgcl import MGCA_ISIC, ISICFineTuner, DualHeadClassifier

__all__ = [
    # Joint training
    "MGCA_ISIC",
    # Two-stage fine-tuning
    "ISICFineTuner",
    "DualHeadClassifier",
    # Generic SSL
    "SSLFineTuner",
    "SSLEvaluator",
]
