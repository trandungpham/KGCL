import datetime
import os
from argparse import ArgumentParser
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
import torchmetrics
from torchmetrics import AUROC, F1Score

# Import from sibling top-level packages because train_joint.py adds KGCL/ to sys.path.
from ..backbones.encoder import BertEncoder, ImageEncoder
from datasets.constants import NUM_DIAGNOSIS_CLASSES

# Disable anomaly detection in production (slows down training)
# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ClassificationHead(nn.Module):
    """Classification head with hidden layer."""
    
    def __init__(self, in_features: int, num_classes: int, 
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)


class MGCA_ISIC(LightningModule):
    """
    MGCA-ISIC: Multi-Granularity Cross-modal Alignment for ISIC Skin Cancer
    
    Extends MGCA with binary diagnosis classification head (NV vs MEL).
    
    Args:
        img_encoder: Image encoder type ("resnet_50" or "vit_base")
        freeze_bert: Whether to freeze BERT text encoder
        emb_dim: Joint embedding dimension
        softmax_temperature: Temperature for ITA loss
        local_temperature: Temperature for CTA loss
        proto_temperature: Temperature for CPA loss
        num_prototypes: Number of disease prototypes
        lambda_1: Weight for ITA loss
        lambda_2: Weight for CTA loss  
        lambda_3: Weight for CPA loss
        lambda_diagnosis: Weight for diagnosis classification loss
        learning_rate: Learning rate
        weight_decay: Weight decay
        batch_size: Batch size
        num_workers: Number of data workers
        hidden_dim: Hidden dimension for classification head
        dropout: Dropout rate for classification head
    """
    
    def __init__(self,
                 img_encoder: str = "resnet_50",
                 freeze_bert: bool = False,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 local_temperature: float = 0.1,
                 proto_temperature: float = 0.2,
                 num_prototypes: int = 100,
                 num_heads: int = 1,
                 lambda_1: float = 0.5,
                 lambda_2: float = 0.3,
                 lambda_3: float = 0.2,
                 lambda_diagnosis: float = 2.0,
                 sinkhorn_iterations: int = 3,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # =====================================================================
        # ENCODERS
        # =====================================================================
        
        # Image encoder
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder,
            output_dim=emb_dim,
            pretrained=True
        )
        
        # Text encoder (BioClinicalBERT)
        self.text_encoder_q = BertEncoder(
            output_dim=emb_dim,
            freeze_bert=freeze_bert
        )
        
        # Get feature dimensions
        if "resnet" in img_encoder:
            self.img_feat_dim = 2048
        else:  # vit
            self.img_feat_dim = 768
        
        self.text_feat_dim = 768  # BERT hidden size
        
        # =====================================================================
        # CROSS-ATTENTION LAYERS (for Token-wise Alignment)
        # =====================================================================
        
        self.patch_local_atten_layer = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.word_local_atten_layer = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # =====================================================================
        # PROTOTYPE LAYER (for Prototype Alignment)
        # =====================================================================
        
        self.prototype_layer = nn.Linear(emb_dim, num_prototypes, bias=False)
        
        # =====================================================================
        # CLASSIFICATION HEAD - Binary Diagnosis (NV vs MEL)
        # =====================================================================
        
        self.diagnosis_head = ClassificationHead(
            in_features=self.img_feat_dim,
            num_classes=NUM_DIAGNOSIS_CLASSES,  # 2 classes
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # CrossEntropyLoss for 2-class classification
        self.diagnosis_loss_fn = nn.CrossEntropyLoss()
        
        # =====================================================================
        # METRICS - Binary Classification
        # =====================================================================
        
        self.train_diagnosis_acc = torchmetrics.Accuracy(task="binary")
        self.val_diagnosis_acc = torchmetrics.Accuracy(task="binary")
        self.train_diagnosis_auroc = torchmetrics.AUROC(task="binary")
        self.val_diagnosis_auroc = torchmetrics.AUROC(task="binary")
        self.val_diagnosis_f1 = torchmetrics.F1Score(task="binary")
    
    # =========================================================================
    # FORWARD METHODS
    # =========================================================================
    
    def forward(self, batch):
        """
        Forward pass through all components.
        
        Returns:
            Dictionary with all embeddings and logits
        """
        # Unpack batch
        imgs = batch["imgs"]
        caption_ids = batch["caption_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        
        # Image encoding
        img_feat_q, patch_feat_q = self.img_encoder_q(imgs)
        patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)
        
        # Text encoding
        report_feat_q, word_feat_q, word_attn_q, sents, _ = self.text_encoder_q(
            caption_ids, attention_mask, token_type_ids
        )
        word_emb_q = self.text_encoder_q.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)
        
        # Image features for classification (reuse img_feat_q)
        if len(img_feat_q.shape) == 4:
            img_feat_raw = F.adaptive_avg_pool2d(img_feat_q, 1).flatten(1)
        elif len(img_feat_q.shape) == 3:
            img_feat_raw = img_feat_q[:, 0]  # CLS token for ViT
        else:
            img_feat_raw = img_feat_q
        
        # Diagnosis classification head
        diagnosis_logits = self.diagnosis_head(img_feat_raw)
        
        return {
            "img_emb_q": img_emb_q,
            "report_emb_q": report_emb_q,
            "patch_emb_q": patch_emb_q,
            "word_emb_q": word_emb_q,
            "word_attn_q": word_attn_q,
            "sents": sents,
            "diagnosis_logits": diagnosis_logits,
        }
    
    # =========================================================================
    # LOSS FUNCTIONS
    # =========================================================================
    
    def compute_ita_loss(self, img_emb_q, report_emb_q):
        """Instance-wise Text-Image Alignment loss (InfoNCE)."""
        bz = img_emb_q.size(0)
        labels = torch.arange(bz).type_as(img_emb_q).long()
        
        scores = img_emb_q.mm(report_emb_q.t())
        scores /= self.hparams.softmax_temperature
        scores1 = scores.transpose(0, 1)
        
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        
        return (loss0 + loss1) / 2, scores
    
    def compute_local_loss(self, patch_emb_q, word_emb_q, word_attn_q, sents):
        """Token-wise Cross-modal Alignment loss."""
        bz = patch_emb_q.size(0)
        
        # Patch-to-word attention
        patch_atten_out, _ = self.patch_local_atten_layer(
            patch_emb_q, word_emb_q, word_emb_q
        )
        
        # Word-to-patch attention
        word_atten_out, _ = self.word_local_atten_layer(
            word_emb_q, patch_emb_q, patch_emb_q
        )
        
        # Normalize
        patch_atten_out = F.normalize(patch_atten_out, dim=-1)
        word_atten_out = F.normalize(word_atten_out, dim=-1)
        
        # Average pool
        word_atten_weights = word_attn_q.unsqueeze(-1)
        word_atten_avg = (word_atten_out * word_atten_weights).sum(1)
        word_atten_avg = F.normalize(word_atten_avg, dim=-1)
        
        patch_atten_avg = patch_atten_out.mean(1)
        patch_atten_avg = F.normalize(patch_atten_avg, dim=-1)
        
        # Contrastive loss
        labels = torch.arange(bz).type_as(patch_emb_q).long()
        
        scores = patch_atten_avg.mm(word_atten_avg.t())
        scores /= self.hparams.local_temperature
        scores1 = scores.transpose(0, 1)
        
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        
        return (loss0 + loss1) / 2
    
    def compute_proto_loss(self, img_emb_q, report_emb_q):
        """Prototype (Disease-level) Alignment loss."""
        # Project to prototype space
        img_proto = self.prototype_layer(img_emb_q)
        text_proto = self.prototype_layer(report_emb_q)
        
        # Softmax
        img_proto = F.softmax(img_proto / self.hparams.proto_temperature, dim=-1)
        text_proto = F.softmax(text_proto / self.hparams.proto_temperature, dim=-1)
        
        # Get cluster assignments using Sinkhorn-Knopp
        with torch.no_grad():
            img_codes = self.sinkhorn(img_proto)
            text_codes = self.sinkhorn(text_proto)
        
        # Cross-entropy loss
        loss = - 0.5 * (
            (text_codes * torch.log(img_proto + 1e-8)).sum(dim=-1).mean() +
            (img_codes * torch.log(text_proto + 1e-8)).sum(dim=-1).mean()
        )
        
        return loss
    
    def sinkhorn(self, Q, nmb_iters=None):
        """Sinkhorn-Knopp algorithm for optimal transport."""
        with torch.no_grad():
            Q = Q.t()  # [K, B]
            K, B = Q.shape
            
            sum_Q = Q.sum()
            if sum_Q > 0:
                Q = Q / sum_Q
            
            if nmb_iters is None:
                nmb_iters = self.hparams.sinkhorn_iterations
            
            for _ in range(nmb_iters):
                Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
                Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)
            
            Q = Q * B
            return Q.t()
    
    def compute_diagnosis_loss(self, diagnosis_logits, diagnosis_labels):
        """Compute diagnosis classification loss."""
        return self.diagnosis_loss_fn(diagnosis_logits, diagnosis_labels)
    
    # =========================================================================
    # TRAINING / VALIDATION STEPS
    # =========================================================================
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self(batch)
        
        # MGCA losses
        loss_ita, scores = self.compute_ita_loss(
            outputs["img_emb_q"], outputs["report_emb_q"]
        )
        
        loss_local = self.compute_local_loss(
            outputs["patch_emb_q"], outputs["word_emb_q"],
            outputs["word_attn_q"], outputs["sents"]
        )
        
        loss_proto = self.compute_proto_loss(
            outputs["img_emb_q"], outputs["report_emb_q"]
        )
        
        # Diagnosis classification loss
        diagnosis_loss = self.compute_diagnosis_loss(
            outputs["diagnosis_logits"], batch["diagnosis_labels"]
        )
        
        # Total loss
        loss_mgca = (self.hparams.lambda_1 * loss_ita +
                     self.hparams.lambda_2 * loss_local +
                     self.hparams.lambda_3 * loss_proto)
        
        total_loss = loss_mgca + self.hparams.lambda_diagnosis * diagnosis_loss
        
        # Compute contrastive accuracy
        bz = batch["imgs"].size(0)
        labels = torch.arange(bz).type_as(scores).long()
        acc1, acc5 = self.precision_at_k(scores, labels, top_k=(1, 5))
        
        # Diagnosis metrics - P(MEL) is softmax[:, 1]
        diagnosis_probs = F.softmax(outputs["diagnosis_logits"], dim=1)[:, 1]
        # diagnosis_preds = (diagnosis_probs > 0.5).long()
        diagnosis_preds = outputs["diagnosis_logits"].argmax(dim=1)
        
        self.train_diagnosis_acc(diagnosis_preds, batch["diagnosis_labels"])
        self.train_diagnosis_auroc(diagnosis_probs, batch["diagnosis_labels"])
        
        # Logging
        self.log("train_loss", total_loss, prog_bar=True, batch_size=bz)
        self.log("train_loss_mgca", loss_mgca, batch_size=bz)
        self.log("train_loss_ita", loss_ita, batch_size=bz)
        self.log("train_loss_local", loss_local, batch_size=bz)
        self.log("train_loss_proto", loss_proto, batch_size=bz)
        self.log("train_loss_diagnosis", diagnosis_loss, batch_size=bz)
        self.log("train_retrieval_acc", acc1, prog_bar=True, batch_size=bz)
        self.log("train_diagnosis_acc", self.train_diagnosis_acc, prog_bar=True, batch_size=bz)
        self.log("train_diagnosis_auroc", self.train_diagnosis_auroc, batch_size=bz)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        outputs = self(batch)
        
        # MGCA losses
        loss_ita, scores = self.compute_ita_loss(
            outputs["img_emb_q"], outputs["report_emb_q"]
        )
        
        loss_local = self.compute_local_loss(
            outputs["patch_emb_q"], outputs["word_emb_q"],
            outputs["word_attn_q"], outputs["sents"]
        )
        
        loss_proto = self.compute_proto_loss(
            outputs["img_emb_q"], outputs["report_emb_q"]
        )
        
        # Diagnosis classification loss
        diagnosis_loss = self.compute_diagnosis_loss(
            outputs["diagnosis_logits"], batch["diagnosis_labels"]
        )
        
        # Total loss
        loss_mgca = (self.hparams.lambda_1 * loss_ita +
                     self.hparams.lambda_2 * loss_local +
                     self.hparams.lambda_3 * loss_proto)
        
        total_loss = loss_mgca + self.hparams.lambda_diagnosis * diagnosis_loss
        
        # Compute contrastive accuracy
        bz = batch["imgs"].size(0)
        labels = torch.arange(bz).type_as(scores).long()
        acc1, acc5 = self.precision_at_k(scores, labels, top_k=(1, 5))
        
        # Diagnosis metrics - P(MEL) is softmax[:, 1]
        diagnosis_probs = F.softmax(outputs["diagnosis_logits"], dim=1)[:, 1]
        diagnosis_preds = (diagnosis_probs > 0.5).long()
        
        self.val_diagnosis_acc(diagnosis_preds, batch["diagnosis_labels"])
        self.val_diagnosis_auroc(diagnosis_probs, batch["diagnosis_labels"])
        self.val_diagnosis_f1(diagnosis_preds, batch["diagnosis_labels"])
        
        # Logging
        self.log("val_loss", total_loss, prog_bar=True, batch_size=bz, sync_dist=True)
        self.log("val_loss_mgca", loss_mgca, batch_size=bz, sync_dist=True)
        self.log("val_loss_ita", loss_ita, batch_size=bz, sync_dist=True)
        self.log("val_loss_local", loss_local, batch_size=bz, sync_dist=True)
        self.log("val_loss_proto", loss_proto, batch_size=bz, sync_dist=True)
        self.log("val_loss_diagnosis", diagnosis_loss, batch_size=bz, sync_dist=True)
        self.log("val_retrieval_acc", acc1, prog_bar=True, batch_size=bz, sync_dist=True)
        self.log("val_diagnosis_acc", self.val_diagnosis_acc, prog_bar=True, batch_size=bz)
        self.log("val_diagnosis_auroc", self.val_diagnosis_auroc, prog_bar=True, batch_size=bz)
        self.log("val_diagnosis_f1", self.val_diagnosis_f1, batch_size=bz)
        
        return total_loss
    
    @staticmethod
    def precision_at_k(output, target, top_k=(1,)):
        """Compute precision@k."""
        maxk = max(top_k)
        batch_size = target.size(0)
        
        _, pred = output.topk(min(maxk, output.size(1)), 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in top_k:
            if k <= output.size(1):
                correct_k = correct[:k].reshape(-1).float().sum(0)
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                res.append(torch.tensor(0.0).type_as(output))
        return res
    
    # =========================================================================
    # OPTIMIZER
    # =========================================================================
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
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
    
    def num_training_steps(self, trainer, datamodule) -> int:
        train_loader = datamodule.train_dataloader()
        steps_per_epoch = len(train_loader)

        if trainer.limit_train_batches != 1.0:
            if isinstance(trainer.limit_train_batches, int):
                steps_per_epoch = min(steps_per_epoch, trainer.limit_train_batches)
            else:
                steps_per_epoch = int(steps_per_epoch * trainer.limit_train_batches)

        effective_accum = max(1, trainer.accumulate_grad_batches)
        steps_per_epoch = steps_per_epoch // effective_accum

        return steps_per_epoch * trainer.max_epochs
