"""CGCL package for ISIC-2019 training experiments."""

from .models.cgcl import FinetuneModule, MultiTaskNet, PretrainModule

__version__ = "0.2.2"

__all__ = ["MultiTaskNet", "PretrainModule", "FinetuneModule", "__version__"]
