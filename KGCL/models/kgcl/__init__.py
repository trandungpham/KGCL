"""Exports for the joint-training KGCL module."""

from .kgcl_module import MGCA_ISIC
from .image_module import ISICImageOnly
from .segmenatation_module import SpatialClueAlignment
from datasets.constants import NUM_DIAGNOSIS_CLASSES

__all__ = ["MGCA_ISIC", "NUM_DIAGNOSIS_CLASSES", "ISICImageOnly", "SpatialClueAlignment"]
