"""
================================================================================
ISIC Fine-tuner: Two-Head Classification from Pre-trained MGCA
================================================================================
This module implements fine-tuning for ISIC-2019 skin cancer classification
using a pre-trained MGCA backbone.

Following the MGCA paper (Wang et al., NeurIPS 2022), this implements a
TWO-STAGE training approach:
1. Pre-train with MGCA (ITA + CTA + CPA) on image-text pairs
2. Fine-tune classification heads on labeled data (this module)

Classification Tasks:
─────────────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRE-TRAINED BACKBONE                                │
│                    (MGCA Image Encoder - Frozen)                            │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Feature Extraction   │
                    │      (2048-dim)        │
                    └────────────┬───────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
    ┌─────────────────┐                   ┌─────────────────┐
    │   CHAOS HEAD    │                   │   CLUES HEAD    │
    │   (2 outputs)   │                   │  (10 outputs)   │
    │                 │                   │                 │
    │ - structure     │                   │ - eccentric     │
    │ - colour        │                   │ - thick lines   │
    └─────────────────┘                   │ - grey blue     │
                                          │ - black dots    │
                                          │ - radial lines  │
                                          │ - white lines   │
                                          │ - vessels       │
                                          │ - parallel      │
                                          │ - angulated     │
                                          │ - no clues      │
                                          └─────────────────┘
─────────────────────────────────────────────────────────────────────────────────

Key Differences from Joint Training (MGCA_ISIC):
- Backbone is FROZEN (only heads are trained)
- No MGCA losses (ITA, CTA, CPA) during fine-tuning
- Faster training, uses pre-learned representations
- Better for limited labeled data scenarios
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import AUROC, Accuracy, F1Score
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

# Relative imports for standalone package
from ...datasets.constants import CHAOS_LABELS, CLUE_LABELS


class DualHeadClassifier(nn.Module):
    """
    Two-head classifier for chaos and clues prediction.
    
    Architecture:
        Input (2048) → Shared Layer (512) → [Chaos Head (2), Clues Head (10)]
    """
    
    def __init__(self, 
                 in_features: int = 2048,
                 hidden_dim: int = 512,
                 num_chaos: int = 2,
                 num_clues: int = 10,
                 dropout: float = 0.1,
                 use_shared_layer: bool = True):
        super().__init__()
        
        self.use_shared_layer = use_shared_layer
        
        if use_shared_layer:
            # Shared representation layer
            self.shared = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
            head_in_features = hidden_dim
        else:
            self.shared = nn.Identity()
            head_in_features = in_features
        
        # Chaos head
        self.chaos_head = nn.Sequential(
            nn.Linear(head_in_features, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_chaos)
        )
        
        # Clues head
        self.clues_head = nn.Sequential(
            nn.Linear(head_in_features, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_clues)
        )
    
    def forward(self, x):
        """
        Args:
            x: Feature tensor of shape (batch, in_features)
            
        Returns:
            chaos_logits: (batch, 2)
            clues_logits: (batch, 10)
        """
        x = x.view(x.size(0), -1)  # Flatten
        shared_features = self.shared(x)
        
        chaos_logits = self.chaos_head(shared_features)
        clues_logits = self.clues_head(shared_features)
        
        return chaos_logits, clues_logits


class ISICFineTuner(LightningModule):
    """
    Fine-tuner for ISIC classification with pre-trained MGCA backbone.
    
    This follows the two-stage approach from the MGCA paper:
    1. Pre-train backbone with MGCA
    2. Freeze backbone and train classification heads
    
    Args:
        backbone: Pre-trained image encoder (from MGCA)
        in_features: Feature dimension from backbone
        hidden_dim: Hidden layer dimension for classifiers
        dropout: Dropout rate
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        freeze_backbone: Whether to freeze the backbone
        chaos_weight: Loss weight for chaos classification
        clues_weight: Loss weight for clues classification
    """
    
    def __init__(self,
                 backbone: nn.Module,
                 model_name: str = "resnet_50",
                 in_features: int = 2048,
                 hidden_dim: int = 512,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 freeze_backbone: bool = True,
                 chaos_weight: float = 1.0,
                 clues_weight: float = 1.0,
                 use_shared_layer: bool = True,
                 **kwargs):
        super().__init__()
        
        self.save_hyperparameters(ignore=['backbone'])
        
        # Backbone (pre-trained MGCA image encoder)
        self.backbone = backbone
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification heads
        self.classifier = DualHeadClassifier(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_chaos=len(CHAOS_LABELS),
            num_clues=len(CLUE_LABELS),
            dropout=dropout,
            use_shared_layer=use_shared_layer
        )
        
        # Losses
        self.chaos_loss_fn = nn.BCEWithLogitsLoss()
        self.clues_loss_fn = nn.BCEWithLogitsLoss()
        
        # Metrics - Chaos
        self.train_chaos_auroc = AUROC(task="multilabel", num_labels=len(CHAOS_LABELS))
        self.val_chaos_auroc = AUROC(task="multilabel", num_labels=len(CHAOS_LABELS))
        self.test_chaos_auroc = AUROC(task="multilabel", num_labels=len(CHAOS_LABELS))
        
        self.train_chaos_f1 = F1Score(task="multilabel", num_labels=len(CHAOS_LABELS), average="macro")
        self.val_chaos_f1 = F1Score(task="multilabel", num_labels=len(CHAOS_LABELS), average="macro")
        self.test_chaos_f1 = F1Score(task="multilabel", num_labels=len(CHAOS_LABELS), average="macro")
        
        # Metrics - Clues
        self.train_clues_auroc = AUROC(task="multilabel", num_labels=len(CLUE_LABELS))
        self.val_clues_auroc = AUROC(task="multilabel", num_labels=len(CLUE_LABELS))
        self.test_clues_auroc = AUROC(task="multilabel", num_labels=len(CLUE_LABELS))
        
        self.train_clues_f1 = F1Score(task="multilabel", num_labels=len(CLUE_LABELS), average="macro")
        self.val_clues_f1 = F1Score(task="multilabel", num_labels=len(CLUE_LABELS), average="macro")
        self.test_clues_f1 = F1Score(task="multilabel", num_labels=len(CLUE_LABELS), average="macro")
    
    def on_train_batch_start(self, batch, batch_idx) -> None:
        """Ensure backbone is in eval mode if frozen."""
        if self.freeze_backbone:
            self.backbone.eval()
    
    def extract_features(self, x):
        """Extract features from backbone."""
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            features, _ = self.backbone(x)
        return features
    
    def forward(self, x):
        """Forward pass."""
        features = self.extract_features(x)
        chaos_logits, clues_logits = self.classifier(features)
        return chaos_logits, clues_logits
    
    def shared_step(self, batch):
        """Shared step for train/val/test."""
        # Handle different batch formats
        if isinstance(batch, dict):
            x = batch["imgs"]
            chaos_labels = batch["chaos_labels"]
            clues_labels = batch["clues_labels"]
        else:
            x, chaos_labels, clues_labels = batch
        
        # Forward
        chaos_logits, clues_logits = self(x)
        
        # Losses
        chaos_loss = self.chaos_loss_fn(chaos_logits, chaos_labels.float())
        clues_loss = self.clues_loss_fn(clues_logits, clues_labels.float())
        
        total_loss = (self.hparams.chaos_weight * chaos_loss + 
                      self.hparams.clues_weight * clues_loss)
        
        return total_loss, chaos_logits, clues_logits, chaos_labels, clues_labels, chaos_loss, clues_loss
    
    def training_step(self, batch, batch_idx):
        total_loss, chaos_logits, clues_logits, chaos_labels, clues_labels, chaos_loss, clues_loss = self.shared_step(batch)
        
        # Predictions
        chaos_preds = torch.sigmoid(chaos_logits)
        clues_preds = torch.sigmoid(clues_logits)
        
        # Metrics
        self.train_chaos_auroc(chaos_preds, chaos_labels.int())
        self.train_clues_auroc(clues_preds, clues_labels.int())
        
        # Logging
        bz = chaos_logits.size(0)
        self.log("train_loss", total_loss, prog_bar=True, batch_size=bz)
        self.log("train_chaos_loss", chaos_loss, batch_size=bz)
        self.log("train_clues_loss", clues_loss, batch_size=bz)
        self.log("train_chaos_auroc", self.train_chaos_auroc, prog_bar=True, batch_size=bz)
        self.log("train_clues_auroc", self.train_clues_auroc, prog_bar=True, batch_size=bz)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        total_loss, chaos_logits, clues_logits, chaos_labels, clues_labels, chaos_loss, clues_loss = self.shared_step(batch)
        
        # Predictions
        chaos_preds = torch.sigmoid(chaos_logits)
        clues_preds = torch.sigmoid(clues_logits)
        
        # Metrics
        self.val_chaos_auroc(chaos_preds, chaos_labels.int())
        self.val_chaos_f1(chaos_preds, chaos_labels.int())
        self.val_clues_auroc(clues_preds, clues_labels.int())
        self.val_clues_f1(clues_preds, clues_labels.int())
        
        # Logging
        bz = chaos_logits.size(0)
        self.log("val_loss", total_loss, prog_bar=True, batch_size=bz, sync_dist=True)
        self.log("val_chaos_loss", chaos_loss, batch_size=bz, sync_dist=True)
        self.log("val_clues_loss", clues_loss, batch_size=bz, sync_dist=True)
        self.log("val_chaos_auroc", self.val_chaos_auroc, prog_bar=True, batch_size=bz)
        self.log("val_chaos_f1", self.val_chaos_f1, batch_size=bz)
        self.log("val_clues_auroc", self.val_clues_auroc, prog_bar=True, batch_size=bz)
        self.log("val_clues_f1", self.val_clues_f1, batch_size=bz)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        total_loss, chaos_logits, clues_logits, chaos_labels, clues_labels, chaos_loss, clues_loss = self.shared_step(batch)
        
        # Predictions
        chaos_preds = torch.sigmoid(chaos_logits)
        clues_preds = torch.sigmoid(clues_logits)
        
        # Metrics
        self.test_chaos_auroc(chaos_preds, chaos_labels.int())
        self.test_chaos_f1(chaos_preds, chaos_labels.int())
        self.test_clues_auroc(clues_preds, clues_labels.int())
        self.test_clues_f1(clues_preds, clues_labels.int())
        
        # Logging
        bz = chaos_logits.size(0)
        self.log("test_loss", total_loss, batch_size=bz, sync_dist=True)
        self.log("test_chaos_auroc", self.test_chaos_auroc, batch_size=bz)
        self.log("test_chaos_f1", self.test_chaos_f1, batch_size=bz)
        self.log("test_clues_auroc", self.test_clues_auroc, batch_size=bz)
        self.log("test_clues_f1", self.test_clues_f1, batch_size=bz)
        
        return total_loss
    
    def configure_optimizers(self):
        # Only optimize classifier parameters (backbone is frozen)
        if self.freeze_backbone:
            params = self.classifier.parameters()
        else:
            params = self.parameters()
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.1),
            gamma=1.0
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    @property
    def training_steps(self) -> int:
        if hasattr(self, "_training_steps"):
            return self._training_steps
        return 1000
    
    @training_steps.setter
    def training_steps(self, value: int):
        self._training_steps = value
    
    @staticmethod
    def num_training_steps(trainer, datamodule) -> int:
        """Calculate total training steps."""
        dataset_size = len(datamodule.train_dataloader())
        num_devices = max(1, trainer.num_devices)
        max_epochs = trainer.max_epochs
        
        steps = dataset_size * max_epochs
        return steps
