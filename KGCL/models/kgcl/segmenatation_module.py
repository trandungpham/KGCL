import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from pytorch_lightning import LightningModule
import torchmetrics

from ..backbones.encoder import BertEncoder, ImageEncoder
from datasets.constants import NUM_DIAGNOSIS_CLASSES

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ClassificationHead(nn.Module):
    """Classification head with hidden layer."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class SegmentationHead(nn.Module):
    """
    Lightweight segmentation decoder.
    Input:  spatial feature map [B, C, Hf, Wf]
    Output: segmentation logits [B, out_channels, H, W]
    """

    def __init__(self, in_channels: int, mid_channels: int = 256, out_channels: int = 1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels // 2, out_channels, kernel_size=1),
        )

    def forward(self, x, output_size):
        x = self.decoder(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Binary Dice loss.
    logits:  [B, 1, H, W]
    targets: [B, 1, H, W]
    """
    probs = torch.sigmoid(logits)
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class SpatialClueAlignment(LightningModule):
    """
    MGCA-ISIC with:
    - ITA / CTA / CPA
    - binary diagnosis classification
    - lesion segmentation supervision
    - clue-specific masked patch-token alignment

    Expected batch keys
    -------------------
    imgs:               [B, 3, H, W]
    caption_ids:        [B, L]
    attention_mask:     [B, L]
    token_type_ids:     [B, L]
    diagnosis_labels:   [B]
    seg_masks:          [B, 1, H, W]         binary lesion mask
    clue_masks:         [B, C, H, W]         binary mask per clue
    clue_token_masks:   [B, C, L]            binary token mask per clue phrase
    clue_present:       [B, C]               1 if clue exists in sample else 0
    """

    def __init__(
        self,
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
        lambda_seg: float = 1.0,
        lambda_seg_dice: float = 1.0,
        lambda_clue_align: float = 1.0,
        sinkhorn_iterations: int = 3,
        learning_rate: float = 2e-5,
        momentum: float = 0.9,
        weight_decay: float = 0.05,
        batch_size: int = 32,
        num_workers: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        seg_mid_dim: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ============================================================
        # ENCODERS
        # ============================================================
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder,
            output_dim=emb_dim,
            pretrained=True,
        )

        self.text_encoder_q = BertEncoder(
            output_dim=emb_dim,
            freeze_bert=freeze_bert,
        )

        if "resnet" in img_encoder:
            self.img_feat_dim = 2048
            self.patch_feat_dim = 2048
            self.is_vit = False
        else:
            self.img_feat_dim = 768
            self.patch_feat_dim = 768
            self.is_vit = True

        self.text_feat_dim = 768

        # ============================================================
        # CTA
        # ============================================================
        self.patch_local_atten_layer = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.word_local_atten_layer = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # ============================================================
        # CPA
        # ============================================================
        self.prototype_layer = nn.Linear(emb_dim, num_prototypes, bias=False)

        # ============================================================
        # DIAGNOSIS HEAD
        # ============================================================
        self.diagnosis_head = ClassificationHead(
            in_features=self.img_feat_dim,
            num_classes=NUM_DIAGNOSIS_CLASSES,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.diagnosis_loss_fn = nn.CrossEntropyLoss()

        # ============================================================
        # SEGMENTATION HEAD
        # ============================================================
        self.seg_head = SegmentationHead(
            in_channels=self.patch_feat_dim,
            mid_channels=seg_mid_dim,
            out_channels=1,
        )
        self.seg_bce_loss_fn = nn.BCEWithLogitsLoss()

        # ============================================================
        # METRICS
        # ============================================================
        self.train_diagnosis_acc = torchmetrics.Accuracy(task="binary")
        self.val_diagnosis_acc = torchmetrics.Accuracy(task="binary")
        self.test_diagnosis_acc = torchmetrics.Accuracy(task="binary")
        self.train_diagnosis_auroc = torchmetrics.AUROC(task="binary")
        self.val_diagnosis_auroc = torchmetrics.AUROC(task="binary")
        self.test_diagnosis_auroc = torchmetrics.AUROC(task="binary")
        self.val_diagnosis_f1 = torchmetrics.F1Score(task="binary")
        self.test_diagnosis_f1 = torchmetrics.F1Score(task="binary")

        self.train_seg_iou = torchmetrics.JaccardIndex(task="binary")
        self.val_seg_iou = torchmetrics.JaccardIndex(task="binary")
        self.test_seg_iou = torchmetrics.JaccardIndex(task="binary")

    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================

    def to_patch_tokens(self, patch_feat: torch.Tensor) -> torch.Tensor:
        """
        Convert raw patch feature to tokens [B, N, C].
        """
        if patch_feat.dim() == 4:
            # [B, C, H, W] -> [B, N, C]
            return patch_feat.flatten(2).transpose(1, 2).contiguous()
        if patch_feat.dim() == 3:
            return patch_feat
        raise ValueError(f"Unsupported patch_feat shape: {patch_feat.shape}")

    def to_spatial_map(self, patch_feat: torch.Tensor) -> torch.Tensor:
        """
        Convert raw patch feature to spatial map [B, C, Hf, Wf].
        Supports CNN maps and ViT patch tokens (with or without CLS token).
        """
        if patch_feat.dim() == 4:
            return patch_feat

        if patch_feat.dim() != 3:
            raise ValueError(f"Unsupported patch_feat shape: {patch_feat.shape}")

        B, N, C = patch_feat.shape
        side = int(np.sqrt(N))

        if side * side != N:
            # likely includes CLS token
            if N > 1:
                patch_feat = patch_feat[:, 1:, :]
                N = patch_feat.size(1)
                side = int(np.sqrt(N))

        if side * side != N:
            raise ValueError(f"Cannot reshape patch tokens of length {N} into square map.")

        return patch_feat.transpose(1, 2).contiguous().view(B, C, side, side)

    def get_patch_grid_size(self, patch_feat: torch.Tensor):
        """
        Infer feature-grid size from raw patch features.
        """
        if patch_feat.dim() == 4:
            _, _, h, w = patch_feat.shape
            return h, w

        if patch_feat.dim() == 3:
            _, n, _ = patch_feat.shape
            side = int(np.sqrt(n))
            if side * side != n:
                n = n - 1
                side = int(np.sqrt(n))
            if side * side != n:
                raise ValueError(f"Cannot infer square grid from patch count {n}")
            return side, side

        raise ValueError(f"Unsupported patch_feat shape: {patch_feat.shape}")

    def get_global_image_feature_for_cls(self, img_feat_q: torch.Tensor) -> torch.Tensor:
        """
        Convert raw image features into global vector for diagnosis head.
        """
        if img_feat_q.dim() == 4:
            return F.adaptive_avg_pool2d(img_feat_q, 1).flatten(1)
        if img_feat_q.dim() == 3:
            return img_feat_q[:, 0]  # CLS token
        return img_feat_q

    def clue_masks_to_patch_masks(self, clue_masks: torch.Tensor, patch_feat_q: torch.Tensor) -> torch.Tensor:
        """
        clue_masks:  [B, C, H, W]
        returns:     [B, C, N]
        """
        grid_h, grid_w = self.get_patch_grid_size(patch_feat_q)
        resized = F.interpolate(
            clue_masks.float(),
            size=(grid_h, grid_w),
            mode="nearest",
        )
        return resized.flatten(2)

    def masked_patch_pool(
        self,
        patch_emb: torch.Tensor,
        patch_masks: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        patch_emb:   [B, N, D]
        patch_masks: [B, C, N]
        returns:     [B, C, D]
        """
        patch_masks = patch_masks.float()
        denom = patch_masks.sum(dim=-1, keepdim=True) + eps
        weights = patch_masks / denom
        pooled = torch.einsum("bcn,bnd->bcd", weights, patch_emb)
        return F.normalize(pooled, dim=-1)

    def masked_token_pool(
        self,
        word_emb: torch.Tensor,
        token_masks: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        word_emb:    [B, L, D]
        token_masks: [B, C, L]
        returns:     [B, C, D]
        """
        token_masks = token_masks.float()
        denom = token_masks.sum(dim=-1, keepdim=True) + eps
        weights = token_masks / denom
        pooled = torch.einsum("bcl,bld->bcd", weights, word_emb)
        return F.normalize(pooled, dim=-1)

    def remap_clue_token_masks(
        self,
        clue_token_masks: torch.Tensor,
        raw_to_agg_token_map: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """
        Remap raw tokenizer-space clue masks into the aggregated token space
        produced by BertEncoder.word_feat.

        clue_token_masks:     [B, C, L_raw]
        raw_to_agg_token_map: [B, L_raw]  indices in aggregated sequence incl. CLS
        target_length:        length of word_emb_q (aggregated sequence without CLS)
        """
        batch_size, num_clues, _ = clue_token_masks.shape
        remapped = clue_token_masks.new_zeros(batch_size, num_clues, target_length)

        for batch_idx in range(batch_size):
            mapping = raw_to_agg_token_map[batch_idx]
            for raw_idx in range(mapping.size(0)):
                agg_idx = int(mapping[raw_idx].item())
                if agg_idx <= 0:
                    continue
                word_idx = agg_idx - 1  # word_emb excludes CLS
                if word_idx >= target_length:
                    continue
                remapped[batch_idx, :, word_idx] = torch.maximum(
                    remapped[batch_idx, :, word_idx],
                    clue_token_masks[batch_idx, :, raw_idx],
                )

        return remapped

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(self, batch):
        imgs = batch["imgs"]
        caption_ids = batch["caption_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        # ------------------------------
        # Image encoding
        # ------------------------------
        img_feat_q, patch_feat_q = self.img_encoder_q(imgs)

        patch_tokens_q = self.to_patch_tokens(patch_feat_q)
        patch_emb_q = self.img_encoder_q.local_embed(patch_tokens_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)

        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        img_feat_raw = self.get_global_image_feature_for_cls(img_feat_q)
        diagnosis_logits = self.diagnosis_head(img_feat_raw)

        # ------------------------------
        # Text encoding
        # ------------------------------
        report_feat_q, word_feat_q, word_attn_q, sents, raw_to_agg_token_map = self.text_encoder_q(
            caption_ids, attention_mask, token_type_ids
        )
        word_emb_q = self.text_encoder_q.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)

        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)

        # ------------------------------
        # Segmentation branch
        # ------------------------------
        patch_feat_map = self.to_spatial_map(patch_feat_q)
        seg_logits = self.seg_head(patch_feat_map, output_size=imgs.shape[-2:])

        return {
            "img_emb_q": img_emb_q,
            "report_emb_q": report_emb_q,
            "patch_emb_q": patch_emb_q,
            "word_emb_q": word_emb_q,
            "word_attn_q": word_attn_q,
            "sents": sents,
            "raw_to_agg_token_map": raw_to_agg_token_map,
            "diagnosis_logits": diagnosis_logits,
            "seg_logits": seg_logits,
            "patch_feat_q": patch_feat_q,
        }

    # ============================================================
    # LOSSES
    # ============================================================

    def compute_ita_loss(self, img_emb_q, report_emb_q):
        """Instance-wise text-image alignment loss."""
        bz = img_emb_q.size(0)
        labels = torch.arange(bz, device=img_emb_q.device).long()

        scores = img_emb_q.mm(report_emb_q.t())
        scores = scores / self.hparams.softmax_temperature
        scores_t = scores.transpose(0, 1)

        loss_i2t = F.cross_entropy(scores, labels)
        loss_t2i = F.cross_entropy(scores_t, labels)
        return (loss_i2t + loss_t2i) / 2, scores

    def compute_local_loss(self, patch_emb_q, word_emb_q, word_attn_q, sents):
        """Original CTA loss."""
        bz = patch_emb_q.size(0)

        patch_atten_out, _ = self.patch_local_atten_layer(
            patch_emb_q, word_emb_q, word_emb_q
        )
        word_atten_out, _ = self.word_local_atten_layer(
            word_emb_q, patch_emb_q, patch_emb_q
        )

        patch_atten_out = F.normalize(patch_atten_out, dim=-1)
        word_atten_out = F.normalize(word_atten_out, dim=-1)

        word_atten_weights = word_attn_q.unsqueeze(-1)
        word_atten_avg = (word_atten_out * word_atten_weights).sum(1)
        word_atten_avg = F.normalize(word_atten_avg, dim=-1)

        patch_atten_avg = patch_atten_out.mean(1)
        patch_atten_avg = F.normalize(patch_atten_avg, dim=-1)

        labels = torch.arange(bz, device=patch_emb_q.device).long()

        scores = patch_atten_avg.mm(word_atten_avg.t())
        scores = scores / self.hparams.local_temperature
        scores_t = scores.transpose(0, 1)

        loss_p2w = F.cross_entropy(scores, labels)
        loss_w2p = F.cross_entropy(scores_t, labels)

        return (loss_p2w + loss_w2p) / 2

    def compute_proto_loss(self, img_emb_q, report_emb_q):
        """Prototype alignment loss."""
        img_proto = self.prototype_layer(img_emb_q)
        text_proto = self.prototype_layer(report_emb_q)

        img_proto = F.softmax(img_proto / self.hparams.proto_temperature, dim=-1)
        text_proto = F.softmax(text_proto / self.hparams.proto_temperature, dim=-1)

        with torch.no_grad():
            img_codes = self.sinkhorn(img_proto)
            text_codes = self.sinkhorn(text_proto)

        loss = -0.5 * (
            (text_codes * torch.log(img_proto + 1e-8)).sum(dim=-1).mean()
            + (img_codes * torch.log(text_proto + 1e-8)).sum(dim=-1).mean()
        )
        return loss

    def compute_diagnosis_loss(self, diagnosis_logits, diagnosis_labels):
        return self.diagnosis_loss_fn(diagnosis_logits, diagnosis_labels)

    def compute_segmentation_loss(self, seg_logits, seg_masks):
        seg_masks = seg_masks.float()
        loss_bce = self.seg_bce_loss_fn(seg_logits, seg_masks)
        loss_dice = dice_loss(seg_logits, seg_masks)
        loss_seg = loss_bce + self.hparams.lambda_seg_dice * loss_dice
        return loss_seg, loss_bce, loss_dice

    def compute_clue_alignment_loss(
        self,
        patch_emb_q: torch.Tensor,
        word_emb_q: torch.Tensor,
        clue_masks: torch.Tensor,
        clue_token_masks: torch.Tensor,
        clue_present: torch.Tensor,
        patch_feat_q: torch.Tensor,
        raw_to_agg_token_map: torch.Tensor,
    ):
        """
        Clue-specific mask-guided alignment.

        patch_emb_q:       [B, N, D]
        word_emb_q:        [B, L, D]
        clue_masks:        [B, C, H, W]
        clue_token_masks:  [B, C, L]
        clue_present:      [B, C]
        patch_feat_q: raw patch features for grid inference
        """
        patch_masks = self.clue_masks_to_patch_masks(clue_masks, patch_feat_q)  # [B, C, N]

        clue_visual_emb = self.masked_patch_pool(patch_emb_q, patch_masks)       # [B, C, D]
        remapped_clue_token_masks = self.remap_clue_token_masks(
            clue_token_masks,
            raw_to_agg_token_map,
            target_length=word_emb_q.size(1),
        )
        clue_text_emb = self.masked_token_pool(word_emb_q, remapped_clue_token_masks)  # [B, C, D]

        valid = (
            clue_present.bool()
            & (patch_masks.sum(dim=-1) > 0)
            & (remapped_clue_token_masks.sum(dim=-1) > 0)
        )

        v = clue_visual_emb[valid]
        t = clue_text_emb[valid]

        if v.numel() == 0:
            zero = patch_emb_q.new_tensor(0.0)
            return zero

        labels = torch.arange(v.size(0), device=v.device).long()
        logits = torch.matmul(v, t.t()) / self.hparams.local_temperature

        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_v2t + loss_t2v)

    def sinkhorn(self, Q, nmb_iters=None):
        """Sinkhorn-Knopp algorithm."""
        with torch.no_grad():
            Q = Q.t()  # [K, B]
            _, B = Q.shape

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

    # ============================================================
    # TRAIN / VAL
    # ============================================================

    def _shared_step(self, batch, stage: str):
        outputs = self(batch)

        # MGCA losses
        loss_ita, scores = self.compute_ita_loss(
            outputs["img_emb_q"], outputs["report_emb_q"]
        )

        loss_local = self.compute_local_loss(
            outputs["patch_emb_q"],
            outputs["word_emb_q"],
            outputs["word_attn_q"],
            outputs["sents"],
        )

        loss_proto = self.compute_proto_loss(
            outputs["img_emb_q"], outputs["report_emb_q"]
        )

        # Supervised losses
        diagnosis_loss = self.compute_diagnosis_loss(
            outputs["diagnosis_logits"], batch["diagnosis_labels"]
        )

        seg_loss, seg_bce, seg_dice = self.compute_segmentation_loss(
            outputs["seg_logits"], batch["seg_masks"]
        )

        clue_align_loss = self.compute_clue_alignment_loss(
            outputs["patch_emb_q"],
            outputs["word_emb_q"],
            batch["clue_masks"],
            batch["clue_token_masks"],
            batch["clue_present"],
            outputs["patch_feat_q"],
            outputs["raw_to_agg_token_map"],
        )

        loss_mgca = (
            self.hparams.lambda_1 * loss_ita
            + self.hparams.lambda_2 * loss_local
            + self.hparams.lambda_3 * loss_proto
        )

        total_loss = (
            loss_mgca
            + self.hparams.lambda_diagnosis * diagnosis_loss
            + self.hparams.lambda_seg * seg_loss
            + self.hparams.lambda_clue_align * clue_align_loss
        )

        bz = batch["imgs"].size(0)
        labels = torch.arange(bz, device=scores.device).long()
        acc1, _ = self.precision_at_k(scores, labels, top_k=(1, 5))

        # Diagnosis metrics
        diagnosis_probs = F.softmax(outputs["diagnosis_logits"], dim=1)[:, 1]
        diagnosis_preds = outputs["diagnosis_logits"].argmax(dim=1)

        seg_probs = torch.sigmoid(outputs["seg_logits"])
        seg_preds = (seg_probs > 0.5).long()
        seg_targets = batch["seg_masks"].long()

        sync_dist = stage != "train"

        if stage == "train":
            self.train_diagnosis_acc(diagnosis_preds, batch["diagnosis_labels"])
            self.train_diagnosis_auroc(diagnosis_probs, batch["diagnosis_labels"])
            self.train_seg_iou(seg_preds, seg_targets)

            self.log("train_loss", total_loss, prog_bar=True, batch_size=bz)
            self.log("train_loss_mgca", loss_mgca, batch_size=bz)
            self.log("train_loss_ita", loss_ita, batch_size=bz)
            self.log("train_loss_local", loss_local, batch_size=bz)
            self.log("train_loss_proto", loss_proto, batch_size=bz)
            self.log("train_loss_diagnosis", diagnosis_loss, batch_size=bz)
            self.log("train_loss_seg", seg_loss, batch_size=bz)
            self.log("train_loss_seg_bce", seg_bce, batch_size=bz)
            self.log("train_loss_seg_dice", seg_dice, batch_size=bz)
            self.log("train_loss_clue_align", clue_align_loss, batch_size=bz)

            self.log("train_retrieval_acc", acc1, prog_bar=True, batch_size=bz)
            self.log("train_diagnosis_acc", self.train_diagnosis_acc, prog_bar=True, batch_size=bz)
            self.log("train_diagnosis_auroc", self.train_diagnosis_auroc, batch_size=bz)
            self.log("train_seg_iou", self.train_seg_iou, prog_bar=True, batch_size=bz)

        elif stage == "val":
            self.val_diagnosis_acc(diagnosis_preds, batch["diagnosis_labels"])
            self.val_diagnosis_auroc(diagnosis_probs, batch["diagnosis_labels"])
            self.val_diagnosis_f1(diagnosis_preds, batch["diagnosis_labels"])
            self.val_seg_iou(seg_preds, seg_targets)

            self.log("val_loss", total_loss, prog_bar=True, batch_size=bz, sync_dist=sync_dist)
            self.log("val_loss_mgca", loss_mgca, batch_size=bz, sync_dist=sync_dist)
            self.log("val_loss_ita", loss_ita, batch_size=bz, sync_dist=sync_dist)
            self.log("val_loss_local", loss_local, batch_size=bz, sync_dist=sync_dist)
            self.log("val_loss_proto", loss_proto, batch_size=bz, sync_dist=sync_dist)
            self.log("val_loss_diagnosis", diagnosis_loss, batch_size=bz, sync_dist=sync_dist)
            self.log("val_loss_seg", seg_loss, batch_size=bz, sync_dist=sync_dist)
            self.log("val_loss_seg_bce", seg_bce, batch_size=bz, sync_dist=sync_dist)
            self.log("val_loss_seg_dice", seg_dice, batch_size=bz, sync_dist=sync_dist)
            self.log("val_loss_clue_align", clue_align_loss, batch_size=bz, sync_dist=sync_dist)

            self.log("val_retrieval_acc", acc1, prog_bar=True, batch_size=bz, sync_dist=sync_dist)
            self.log("val_diagnosis_acc", self.val_diagnosis_acc, prog_bar=True, batch_size=bz)
            self.log("val_diagnosis_auroc", self.val_diagnosis_auroc, prog_bar=True, batch_size=bz)
            self.log("val_diagnosis_f1", self.val_diagnosis_f1, batch_size=bz)
            self.log("val_seg_iou", self.val_seg_iou, prog_bar=True, batch_size=bz)

        elif stage == "test":
            self.test_diagnosis_acc(diagnosis_preds, batch["diagnosis_labels"])
            self.test_diagnosis_auroc(diagnosis_probs, batch["diagnosis_labels"])
            self.test_diagnosis_f1(diagnosis_preds, batch["diagnosis_labels"])
            self.test_seg_iou(seg_preds, seg_targets)

            self.log("test_loss", total_loss, batch_size=bz, sync_dist=sync_dist)
            self.log("test_loss_mgca", loss_mgca, batch_size=bz, sync_dist=sync_dist)
            self.log("test_loss_ita", loss_ita, batch_size=bz, sync_dist=sync_dist)
            self.log("test_loss_local", loss_local, batch_size=bz, sync_dist=sync_dist)
            self.log("test_loss_proto", loss_proto, batch_size=bz, sync_dist=sync_dist)
            self.log("test_loss_diagnosis", diagnosis_loss, batch_size=bz, sync_dist=sync_dist)
            self.log("test_loss_seg", seg_loss, batch_size=bz, sync_dist=sync_dist)
            self.log("test_loss_seg_bce", seg_bce, batch_size=bz, sync_dist=sync_dist)
            self.log("test_loss_seg_dice", seg_dice, batch_size=bz, sync_dist=sync_dist)
            self.log("test_loss_clue_align", clue_align_loss, batch_size=bz, sync_dist=sync_dist)

            self.log("test_retrieval_acc", acc1, batch_size=bz, sync_dist=sync_dist)
            self.log("test_diagnosis_acc", self.test_diagnosis_acc, batch_size=bz)
            self.log("test_diagnosis_auroc", self.test_diagnosis_auroc, batch_size=bz)
            self.log("test_diagnosis_f1", self.test_diagnosis_f1, batch_size=bz)
            self.log("test_seg_iou", self.test_seg_iou, batch_size=bz)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="test")

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
                res.append(torch.tensor(0.0, device=output.device))
        return res

    # ============================================================
    # OPTIMIZER
    # ============================================================

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.1),
            gamma=1.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
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
