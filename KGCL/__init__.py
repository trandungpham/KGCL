"""KGCL package for MGCA-style training on the ISIC-2019 dataset."""

from .models.kgcl import MGCA_ISIC, ISICImageOnly
__version__ = "0.2.1"

__all__ = ["MGCA_ISIC", "ISICImageOnly", "__version__"]
