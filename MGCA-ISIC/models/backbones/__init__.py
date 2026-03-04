"""
MGCA-ISIC Backbone Models
"""

from .cnn_backbones import resnet_18, resnet_34, resnet_50, Identity
from .encoder import ImageEncoder, BertEncoder, GlobalEmbedding, LocalEmbedding

__all__ = [
    # CNN backbones
    "resnet_18",
    "resnet_34", 
    "resnet_50",
    "Identity",
    # Encoders
    "ImageEncoder",
    "BertEncoder",
    "GlobalEmbedding",
    "LocalEmbedding",
]
