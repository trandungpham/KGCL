"""
================================================================================
Encoders: Image and Text Encoder Implementations
================================================================================
This module implements the encoder backbones for the MGCA framework:

1. IMAGE ENCODER (ImageEncoder)
   - Supports ResNet-50 and Vision Transformer (ViT) backbones
   - Extracts both GLOBAL features (image-level) and LOCAL features (patch-level)
   - Global features: Used for instance-wise alignment (ITA)
   - Local features: Used for token-wise alignment (CTA)

Projection Heads:
- GlobalEmbedding: Projects global features to joint embedding space
- LocalEmbedding: Projects local features to joint embedding space
  (Uses 1D convolution to process sequences efficiently)

Architecture:
┌─────────────┐     ┌─────────────────┐
│   Image     │     │ Medical Report  │
│ [224x224]   │     │   [Tokens]      │
└──────┬──────┘     └───────┬─────────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────────┐
│ CNN/ViT      │     │ BioClinicalBERT  │
│ Backbone     │     │ Backbone         │
└──────┬───────┘     └───────┬──────────┘
       │                     │
       ├── Global Pool ──┐   ├── [CLS] Token ──┐
       │                 │   │                  │
       │                 ▼   │                  ▼
       │          ┌───────────┐          ┌───────────┐
       │          │GlobalEmbed│          │GlobalEmbed│
       │          │  (MLP)    │          │  (MLP)    │
       │          └─────┬─────┘          └─────┬─────┘
       │                │                      │
       │                └──── Global Feats ────┘  → ITA Loss
       │                                  
       ├── Patch/Grid ───┐   ├── Word Tokens ──┐
       │                 │   │                  │
       │                 ▼   │                  ▼
       │          ┌───────────┐          ┌───────────┐
       │          │LocalEmbed │          │LocalEmbed │
       │          │ (Conv1D)  │          │ (Conv1D)  │
       │          └─────┬─────┘          └─────┬─────┘
       │                │                      │
       └────────────────┴──── Local Feats ─────┘  → CTA Loss
================================================================================
"""

import os

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoTokenizer, BertConfig, BertTokenizer, logging

# Relative imports for standalone package
from . import cnn_backbones
from .vits import create_vit, interpolate_pos_embed

# Suppress transformer warnings for cleaner output
logging.set_verbosity_error()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class GlobalEmbedding(nn.Module):
    """
    Global Embedding Projection Head.
    
    Projects global features (image [CLS] or report [CLS]) to the joint 
    embedding space for instance-wise contrastive learning.
    
    Architecture:
        Linear → BatchNorm → ReLU → Linear → BatchNorm (no affine)
    
    The final BatchNorm without affine parameters ensures the output
    has zero mean and unit variance, which is beneficial for contrastive
    learning with L2-normalized features.
    
    Args:
        input_dim: Input feature dimension (from backbone)
        hidden_dim: Hidden layer dimension (2048 default, following SimCLR)
        output_dim: Output embedding dimension (128 default)
    """
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            # First projection layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            # Second projection layer (output)
            nn.Linear(hidden_dim, output_dim),
            # Final BatchNorm without learnable affine parameters
            # This normalizes outputs to unit variance
            nn.BatchNorm1d(output_dim, affine=False)
        )

    def forward(self, x):
        """
        Args:
            x: Global features [batch_size, input_dim]
            
        Returns:
            Projected embeddings [batch_size, output_dim]
        """
        return self.head(x)


class LocalEmbedding(nn.Module):
    """
    Local Embedding Projection Head.
    
    Projects local features (image patches or text tokens) to the joint
    embedding space for token-wise contrastive learning.
    
    Uses 1D convolution (kernel_size=1) instead of Linear layers to 
    efficiently process sequences while maintaining the sequence dimension.
    This is equivalent to applying the same Linear layer to each position.
    
    Architecture:
        Conv1D → BatchNorm1D → ReLU → Conv1D → BatchNorm1D (no affine)
    
    Args:
        input_dim: Input feature dimension (from backbone)
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
    """
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            # 1D convolution acts as position-wise linear transformation
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)
        )

    def forward(self, x):
        """
        Args:
            x: Local features [batch_size, seq_len, input_dim]
            
        Returns:
            Projected embeddings [batch_size, seq_len, output_dim]
        """
        # Conv1d expects [B, C, L] format, so we permute
        x = x.permute(0, 2, 1)  # [B, input_dim, seq_len]
        x = self.head(x)        # [B, output_dim, seq_len]
        return x.permute(0, 2, 1)  # [B, seq_len, output_dim]


class ImageEncoder(nn.Module):
    """
    Image Encoder for MGCA.
    
    Extracts visual features at multiple granularities:
    - GLOBAL features: Single vector representing entire image
    - LOCAL features: Grid of patch/region features
    
    Supports two backbone architectures:
    
    1. ResNet-50 (model_name="resnet_50"):
       - Global: Average pooled features from final layer [B, 2048]
       - Local: Feature map from layer3 [B, 1024, H, W] → [B, H*W, 1024]
       
    2. Vision Transformer (model_name="vit_base"):
       - Global: [CLS] token embedding [B, 768]
       - Local: Patch token embeddings [B, 196, 768] (14x14 patches)
       
    Args:
        model_name: Backbone architecture ("resnet_50" or "vit_base")
        text_feat_dim: Not used (legacy parameter)
        output_dim: Output embedding dimension after projection
        hidden_dim: Hidden dimension in projection heads
        pretrained: Whether to use pretrained weights
    """
    def __init__(self,
                 model_name: str = "resnet_50",
                 text_feat_dim: int = 768,
                 output_dim: int = 768,
                 hidden_dim: int = 2048,
                 pretrained: bool = True,
                 image_size: int = 224,
                 ):
        super(ImageEncoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.text_feat_dim = text_feat_dim
        self.image_size = image_size

        if "vit" in model_name:
            # =========================================================
            # VISION TRANSFORMER BACKBONE
            # =========================================================
            # ViT divides image into patches and processes them with
            # a transformer encoder. Returns:
            # - [CLS] token: aggregated global representation
            # - Patch tokens: local region representations
            # =========================================================
            
            vit_grad_ckpt = False  # Gradient checkpointing (saves memory)
            vit_ckpt_layer = 0     # Checkpoint layer
            # Extract ViT variant name (e.g., "vit_base" → "base")
            vit_name = model_name[4:]
            self.model, vision_width = create_vit(
                vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

            self.feature_dim = vision_width  # 768 for ViT-Base

            # Load pretrained DeiT weights (Data-efficient Image Transformer)
            # DeiT is ViT trained with improved training recipe
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            if "pos_embed" in state_dict:
                state_dict["pos_embed"] = interpolate_pos_embed(
                    state_dict["pos_embed"], self.model
                )
            msg = self.model.load_state_dict(state_dict, strict=False)

            # Global embedding projection: [CLS] token → joint space
            self.global_embed = GlobalEmbedding(
                vision_width, hidden_dim, output_dim
            )

            # Local embedding projection: Patch tokens → joint space
            self.local_embed = LocalEmbedding(
                vision_width, hidden_dim, output_dim
            )

        else:
            # =========================================================
            # CNN (RESNET) BACKBONE
            # =========================================================
            # ResNet extracts hierarchical features through conv layers.
            # We use:
            # - layer4 output (pooled): Global representation
            # - layer3 output: Local/regional features (before final pooling)
            # =========================================================
            
            # Dynamically get the CNN backbone function
            model_function = getattr(cnn_backbones, model_name)
            self.model, self.feature_dim, self.interm_feature_dim = model_function(
                pretrained=pretrained
            )
            # feature_dim: Final layer channels (2048 for ResNet-50)
            # interm_feature_dim: Intermediate layer channels (1024)

            # Global average pooling for global features
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            # Global embedding: pooled features → joint space
            self.global_embed = GlobalEmbedding(
                self.feature_dim, hidden_dim, output_dim
            )

            # Local embedding: intermediate features → joint space
            self.local_embed = LocalEmbedding(
                self.interm_feature_dim, hidden_dim, output_dim
            )

    def resnet_forward(self, x, get_local=True):
        """
        Forward pass through ResNet backbone.
        
        Args:
            x: Input images [batch_size, 3, H, W]
            get_local: Whether to return local features (always True in MGCA)
            
        Returns:
            tuple:
                - global_features: [batch_size, 2048] - pooled final layer
                - local_features: [batch_size, H'*W', 1024] - flattened layer3
        
        ResNet Architecture:
            Input: [B, 3, 224, 224]
            ↓ conv1 + bn + relu + maxpool
            ↓ layer1: [B, 256, 75, 75]
            ↓ layer2: [B, 512, 38, 38]  
            ↓ layer3: [B, 1024, 19, 19] ← LOCAL FEATURES (361 regions)
            ↓ layer4: [B, 2048, 10, 10]
            ↓ avgpool: [B, 2048, 1, 1]
            ↓ flatten: [B, 2048] ← GLOBAL FEATURES
        """
        # Upsample to 299x299 (originally for Inception compatibility)
        x = nn.Upsample(size=(299, 299), mode="bilinear",
                        align_corners=True)(x)
        
        # Early layers
        x = self.model.conv1(x)    # [B, 64, 150, 150]
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # Residual blocks
        x = self.model.layer1(x)   # [B, 256, 75, 75]
        x = self.model.layer2(x)   # [B, 512, 38, 38]
        x = self.model.layer3(x)   # [B, 1024, 19, 19]
        
        # Save intermediate features for local alignment
        local_features = x
        
        x = self.model.layer4(x)   # [B, 2048, 10, 10]

        # Global average pooling
        x = self.pool(x)           # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 2048]

        # Reshape local features: [B, C, H, W] → [B, H*W, C]
        # This creates a sequence of 361 region features
        local_features = rearrange(local_features, "b c w h -> b (w h) c")

        return x, local_features.contiguous()

    def vit_forward(self, x):
        """
        Forward pass through Vision Transformer backbone.
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            
        Returns:
            All tokens [batch_size, 197, 768]
            - First token (index 0): [CLS] token (global)
            - Remaining tokens: Patch tokens (local)
            
        ViT Architecture:
            Input: [B, 3, 224, 224]
            ↓ Patch embedding (16x16 patches): [B, 196, 768]
            ↓ Prepend [CLS] token: [B, 197, 768]
            ↓ Add position embeddings
            ↓ Transformer blocks × 12
            Output: [B, 197, 768]
        """
        # register_blk=11 saves attention map from block 11 (last block)
        # for visualization and importance weighting
        return self.model(x, register_blk=11)

    def forward(self, x, get_local=False):
        """
        Main forward pass.
        
        Args:
            x: Input images [batch_size, 3, H, W]
            get_local: Legacy parameter (always returns local features in MGCA)
            
        Returns:
            tuple:
                - global_features: [batch_size, feature_dim]
                - local_features: [batch_size, num_regions, feature_dim]
        """
        if "resnet" in self.model_name:
            return self.resnet_forward(x, get_local=get_local)
        elif "vit" in self.model_name:
            img_feat = self.vit_forward(x)
            # Split into [CLS] (global) and patch tokens (local)
            return img_feat[:, 0].contiguous(), img_feat[:, 1:].contiguous()


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))