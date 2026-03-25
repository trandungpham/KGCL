from typing import Optional, Dict, Any, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import Accuracy, F1Score, MultilabelF1Score


# -----------------------------
# Small building blocks
# -----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleFPNDecoder(nn.Module):
    """
    Simple multi-scale decoder for segmentation.
    Assumes encoder returns 4 feature maps from low->high resolution stages.
    """

    def __init__(self, encoder_channels, decoder_channels=256, num_classes=9):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels

        self.lateral4 = nn.Conv2d(c4, decoder_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(c3, decoder_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(c2, decoder_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(c1, decoder_channels, kernel_size=1)

        self.smooth3 = ConvBNReLU(decoder_channels, decoder_channels)
        self.smooth2 = ConvBNReLU(decoder_channels, decoder_channels)
        self.smooth1 = ConvBNReLU(decoder_channels, decoder_channels)

        self.seg_head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, features, out_size):
        f1, f2, f3, f4 = features  # low -> high level

        p4 = self.lateral4(f4)
        p3 = self.lateral3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.smooth3(p3)

        p2 = self.lateral2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.smooth2(p2)

        p1 = self.lateral1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.smooth1(p1)

        seg_logits = self.seg_head(p1)
        seg_logits = F.interpolate(seg_logits, size=out_size, mode="bilinear", align_corners=False)
        return seg_logits


class GlobalMLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.head(x)


# -----------------------------
# Loss helpers
# -----------------------------
def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_loss(seg_logits, seg_targets, bce_weight=0.5, dice_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(seg_logits, seg_targets)
    dice = dice_loss_with_logits(seg_logits, seg_targets)
    return bce_weight * bce + dice_weight * dice


# -----------------------------
# Segmentation metric helpers
# -----------------------------
def compute_multilabel_segmentation_stats(
    seg_logits: torch.Tensor,
    seg_targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
):
    """
    seg_logits:  [B, C, H, W]
    seg_targets: [B, C, H, W]
    Returns:
        dice_per_class: [C]
        iou_per_class:  [C]
        macro_dice: scalar
        macro_iou: scalar
    """
    seg_probs = torch.sigmoid(seg_logits)
    seg_preds = (seg_probs > threshold).float()

    dims = (0, 2, 3)

    intersection = (seg_preds * seg_targets).sum(dim=dims)                  # [C]
    pred_sum = seg_preds.sum(dim=dims)                                      # [C]
    target_sum = seg_targets.sum(dim=dims)                                  # [C]
    union = pred_sum + target_sum - intersection                            # [C]

    dice_per_class = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou_per_class = (intersection + eps) / (union + eps)

    macro_dice = dice_per_class.mean()
    macro_iou = iou_per_class.mean()

    return {
        "dice_per_class": dice_per_class,
        "iou_per_class": iou_per_class,
        "macro_dice": macro_dice,
        "macro_iou": macro_iou,
    }


# -----------------------------
# Backbone + multitask base
# -----------------------------
class MultiTaskNet(nn.Module):
    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        num_clues=9,
        num_chaos=2,
        num_diag=2,
        decoder_channels=256,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        encoder_channels = self.encoder.feature_info.channels()
        last_dim = encoder_channels[-1]

        self.decoder = SimpleFPNDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_clues,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.clue_head = GlobalMLPHead(last_dim, num_clues)
        self.chaos_head = GlobalMLPHead(last_dim, num_chaos)
        self.diagnosis_head = GlobalMLPHead(last_dim, num_diag)

    def forward(self, x):
        feats = self.encoder(x)
        seg_logits = self.decoder(feats, out_size=x.shape[-2:])

        g = self.global_pool(feats[-1]).flatten(1)
        clue_logits = self.clue_head(g)
        chaos_logits = self.chaos_head(g)
        diagnosis_logits = self.diagnosis_head(g)

        return {
            "seg_logits": seg_logits,
            "clue_logits": clue_logits,
            "chaos_logits": chaos_logits,
            "diagnosis_logits": diagnosis_logits,
        }


# -----------------------------
# Phase 1 LightningModule
# -----------------------------
class PretrainModule(pl.LightningModule):
    """
    Phase 1:
    train encoder using
    - clue segmentation
    - clue presence
    - chaos classification
    """

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        lr=1e-4,
        weight_decay=1e-4,
        num_clues=9,
        num_chaos=2,
        lambda_seg=1.0,
        lambda_clue=1.0,
        lambda_chaos=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MultiTaskNet(
            backbone_name=backbone_name,
            pretrained=pretrained,
            num_clues=num_clues,
            num_chaos=num_chaos,
            num_diag=2,
        )

        self.lr = lr
        self.weight_decay = weight_decay

        self.lambda_seg = lambda_seg
        self.lambda_clue = lambda_clue
        self.lambda_chaos = lambda_chaos

        self.train_clue_f1 = MultilabelF1Score(
            num_labels=num_clues, average="macro"
        )
        self.train_chaos_f1 = MultilabelF1Score(
            num_labels=num_chaos, average="macro"
        )

    def forward(self, x):
        return self.model(x)

    def compute_losses(self, batch):
        imgs = batch["imgs"]
        clue_masks = batch["clue_masks"]
        clue_present = batch["clue_present"]
        chaos_labels = batch["chaos_labels"]

        outputs = self(imgs)

        seg_logits = outputs["seg_logits"]
        clue_logits = outputs["clue_logits"]
        chaos_logits = outputs["chaos_logits"]

        loss_seg = segmentation_loss(seg_logits, clue_masks)
        loss_clue = F.binary_cross_entropy_with_logits(clue_logits, clue_present)
        loss_chaos = F.binary_cross_entropy_with_logits(chaos_logits, chaos_labels)

        total_loss = (
            self.lambda_seg * loss_seg
            + self.lambda_clue * loss_clue
            + self.lambda_chaos * loss_chaos
        )

        return {
            "loss": total_loss,
            "loss_seg": loss_seg,
            "loss_clue": loss_clue,
            "loss_chaos": loss_chaos,
            "outputs": outputs,
        }

    def training_step(self, batch, batch_idx):
        results = self.compute_losses(batch)
        outputs = results["outputs"]

        clue_preds = (torch.sigmoid(outputs["clue_logits"]) > 0.5).int()
        chaos_preds = (torch.sigmoid(outputs["chaos_logits"]) > 0.5).int()

        self.train_clue_f1.update(clue_preds, batch["clue_present"].int())
        self.train_chaos_f1.update(chaos_preds, batch["chaos_labels"].int())

        self.log("train_loss", results["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_seg", results["loss_seg"], on_step=True, on_epoch=True)
        self.log("train_loss_clue", results["loss_clue"], on_step=True, on_epoch=True)
        self.log("train_loss_chaos", results["loss_chaos"], on_step=True, on_epoch=True)

        return results["loss"]

    def on_train_epoch_end(self):
        self.log("train_clue_f1", self.train_clue_f1.compute(), prog_bar=True)
        self.log("train_chaos_f1", self.train_chaos_f1.compute(), prog_bar=True)
        self.train_clue_f1.reset()
        self.train_chaos_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer


# -----------------------------
# Phase 2 LightningModule
# -----------------------------
class FinetuneModule(pl.LightningModule):
    """
    Phase 2:
    image-only fine-tuning with
    - diagnosis classification
    - clue segmentation
    - clue presence
    - chaos classification

    Added:
    - macro Dice / IoU for 9 clue masks
    - per-class Dice / IoU for 9 clue masks
    - per-class clue presence F1
    """

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        pretrained_phase1_ckpt: Optional[str] = None,
        clue_names: Optional[List[str]] = None,
        lr=1e-4,
        weight_decay=1e-4,
        num_clues=9,
        num_chaos=2,
        seg_threshold=0.5,
        lambda_diag=1.0,
        lambda_seg=1.0,
        lambda_clue=1.0,
        lambda_chaos=1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["clue_names"])

        self.model = MultiTaskNet(
            backbone_name=backbone_name,
            pretrained=pretrained,
            num_clues=num_clues,
            num_chaos=num_chaos,
            num_diag=2,
        )

        if pretrained_phase1_ckpt is not None:
            ckpt = torch.load(pretrained_phase1_ckpt, map_location="cpu")
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

            filtered_state_dict = {
                k.replace("model.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
            self.model.load_state_dict(filtered_state_dict, strict=False)

        self.lr = lr
        self.weight_decay = weight_decay
        self.seg_threshold = seg_threshold

        self.lambda_diag = lambda_diag
        self.lambda_seg = lambda_seg
        self.lambda_clue = lambda_clue
        self.lambda_chaos = lambda_chaos

        self.num_clues = num_clues
        self.num_chaos = num_chaos
        self.clue_names = clue_names or [f"clue_{i}" for i in range(num_clues)]

        # classification metrics
        self.val_diag_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_diag_f1 = F1Score(task="multiclass", num_classes=2, average="macro")
        self.val_clue_f1 = MultilabelF1Score(num_labels=num_clues, average="macro")
        self.val_chaos_f1 = MultilabelF1Score(num_labels=num_chaos, average="macro")

        self.test_diag_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_diag_f1 = F1Score(task="multiclass", num_classes=2, average="macro")
        self.test_clue_f1 = MultilabelF1Score(num_labels=num_clues, average="macro")
        self.test_chaos_f1 = MultilabelF1Score(num_labels=num_chaos, average="macro")

        # per-class clue presence F1
        self.val_clue_f1_per_class = MultilabelF1Score(
            num_labels=num_clues, average=None
        )
        self.test_clue_f1_per_class = MultilabelF1Score(
            num_labels=num_clues, average=None
        )

        # accumulators for segmentation metrics
        self._reset_val_seg_accumulators()
        self._reset_test_seg_accumulators()

    def _reset_val_seg_accumulators(self):
        device = self.device if hasattr(self, "device") else torch.device("cpu")
        self.val_dice_sum = torch.zeros(self.num_clues, device=device)
        self.val_iou_sum = torch.zeros(self.num_clues, device=device)
        self.val_seg_batches = 0

    def _reset_test_seg_accumulators(self):
        device = self.device if hasattr(self, "device") else torch.device("cpu")
        self.test_dice_sum = torch.zeros(self.num_clues, device=device)
        self.test_iou_sum = torch.zeros(self.num_clues, device=device)
        self.test_seg_batches = 0

    def forward(self, x):
        return self.model(x)

    def compute_losses(self, batch):
        imgs = batch["imgs"]
        diagnosis_labels = batch["diagnosis_labels"]
        clue_masks = batch["clue_masks"]
        clue_present = batch["clue_present"]
        chaos_labels = batch["chaos_labels"]

        outputs = self(imgs)

        diagnosis_logits = outputs["diagnosis_logits"]
        seg_logits = outputs["seg_logits"]
        clue_logits = outputs["clue_logits"]
        chaos_logits = outputs["chaos_logits"]

        loss_diag = F.cross_entropy(diagnosis_logits, diagnosis_labels)
        loss_seg = segmentation_loss(seg_logits, clue_masks)
        loss_clue = F.binary_cross_entropy_with_logits(clue_logits, clue_present)
        loss_chaos = F.binary_cross_entropy_with_logits(chaos_logits, chaos_labels)

        total_loss = (
            self.lambda_diag * loss_diag
            + self.lambda_seg * loss_seg
            + self.lambda_clue * loss_clue
            + self.lambda_chaos * loss_chaos
        )

        return {
            "loss": total_loss,
            "loss_diag": loss_diag,
            "loss_seg": loss_seg,
            "loss_clue": loss_clue,
            "loss_chaos": loss_chaos,
            "outputs": outputs,
        }

    def training_step(self, batch, batch_idx):
        results = self.compute_losses(batch)

        self.log("train_loss", results["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_diag", results["loss_diag"], on_step=True, on_epoch=True)
        self.log("train_loss_seg", results["loss_seg"], on_step=True, on_epoch=True)
        self.log("train_loss_clue", results["loss_clue"], on_step=True, on_epoch=True)
        self.log("train_loss_chaos", results["loss_chaos"], on_step=True, on_epoch=True)

        return results["loss"]

    def validation_step(self, batch, batch_idx):
        results = self.compute_losses(batch)
        outputs = results["outputs"]

        diag_preds = torch.argmax(outputs["diagnosis_logits"], dim=1)
        clue_preds = (torch.sigmoid(outputs["clue_logits"]) > 0.5).int()
        chaos_preds = (torch.sigmoid(outputs["chaos_logits"]) > 0.5).int()

        self.val_diag_acc.update(diag_preds, batch["diagnosis_labels"])
        self.val_diag_f1.update(diag_preds, batch["diagnosis_labels"])
        self.val_clue_f1.update(clue_preds, batch["clue_present"].int())
        self.val_chaos_f1.update(chaos_preds, batch["chaos_labels"].int())
        self.val_clue_f1_per_class.update(clue_preds, batch["clue_present"].int())

        seg_stats = compute_multilabel_segmentation_stats(
            outputs["seg_logits"],
            batch["clue_masks"],
            threshold=self.seg_threshold,
        )
        self.val_dice_sum += seg_stats["dice_per_class"].detach()
        self.val_iou_sum += seg_stats["iou_per_class"].detach()
        self.val_seg_batches += 1

        self.log("val_loss", results["loss"], prog_bar=True, on_epoch=True)
        self.log("val_loss_diag", results["loss_diag"], on_epoch=True)
        self.log("val_loss_seg", results["loss_seg"], on_epoch=True)
        self.log("val_loss_clue", results["loss_clue"], on_epoch=True)
        self.log("val_loss_chaos", results["loss_chaos"], on_epoch=True)

    def on_validation_epoch_start(self):
        self._reset_val_seg_accumulators()

    def on_validation_epoch_end(self):
        # classification metrics
        self.log("val_diag_acc", self.val_diag_acc.compute(), prog_bar=True)
        self.log("val_diag_f1", self.val_diag_f1.compute(), prog_bar=True)
        self.log("val_clue_f1", self.val_clue_f1.compute(), prog_bar=True)
        self.log("val_chaos_f1", self.val_chaos_f1.compute(), prog_bar=True)

        # segmentation metrics
        if self.val_seg_batches > 0:
            val_dice_per_class = self.val_dice_sum / self.val_seg_batches
            val_iou_per_class = self.val_iou_sum / self.val_seg_batches

            self.log("val_clue_seg_dice", val_dice_per_class.mean(), prog_bar=True)
            self.log("val_clue_seg_iou", val_iou_per_class.mean(), prog_bar=True)

            for i, clue_name in enumerate(self.clue_names):
                safe_name = clue_name.lower().replace(" ", "_").replace("/", "_")
                self.log(f"val_dice_{safe_name}", val_dice_per_class[i])
                self.log(f"val_iou_{safe_name}", val_iou_per_class[i])

        # per-class clue presence F1
        val_clue_f1_per_class = self.val_clue_f1_per_class.compute()
        for i, clue_name in enumerate(self.clue_names):
            safe_name = clue_name.lower().replace(" ", "_").replace("/", "_")
            self.log(f"val_clue_f1_{safe_name}", val_clue_f1_per_class[i])

        self.val_diag_acc.reset()
        self.val_diag_f1.reset()
        self.val_clue_f1.reset()
        self.val_chaos_f1.reset()
        self.val_clue_f1_per_class.reset()

    def test_step(self, batch, batch_idx):
        results = self.compute_losses(batch)
        outputs = results["outputs"]

        diag_preds = torch.argmax(outputs["diagnosis_logits"], dim=1)
        clue_preds = (torch.sigmoid(outputs["clue_logits"]) > 0.5).int()
        chaos_preds = (torch.sigmoid(outputs["chaos_logits"]) > 0.5).int()

        self.test_diag_acc.update(diag_preds, batch["diagnosis_labels"])
        self.test_diag_f1.update(diag_preds, batch["diagnosis_labels"])
        self.test_clue_f1.update(clue_preds, batch["clue_present"].int())
        self.test_chaos_f1.update(chaos_preds, batch["chaos_labels"].int())
        self.test_clue_f1_per_class.update(clue_preds, batch["clue_present"].int())

        seg_stats = compute_multilabel_segmentation_stats(
            outputs["seg_logits"],
            batch["clue_masks"],
            threshold=self.seg_threshold,
        )
        self.test_dice_sum += seg_stats["dice_per_class"].detach()
        self.test_iou_sum += seg_stats["iou_per_class"].detach()
        self.test_seg_batches += 1

        self.log("test_loss", results["loss"], on_epoch=True)
        self.log("test_loss_diag", results["loss_diag"], on_epoch=True)
        self.log("test_loss_seg", results["loss_seg"], on_epoch=True)
        self.log("test_loss_clue", results["loss_clue"], on_epoch=True)
        self.log("test_loss_chaos", results["loss_chaos"], on_epoch=True)

    def on_test_epoch_start(self):
        self._reset_test_seg_accumulators()

    def on_test_epoch_end(self):
        # classification metrics
        self.log("test_diag_acc", self.test_diag_acc.compute(), prog_bar=True)
        self.log("test_diag_f1", self.test_diag_f1.compute(), prog_bar=True)
        self.log("test_clue_f1", self.test_clue_f1.compute(), prog_bar=True)
        self.log("test_chaos_f1", self.test_chaos_f1.compute(), prog_bar=True)

        # segmentation metrics
        if self.test_seg_batches > 0:
            test_dice_per_class = self.test_dice_sum / self.test_seg_batches
            test_iou_per_class = self.test_iou_sum / self.test_seg_batches

            self.log("test_clue_seg_dice", test_dice_per_class.mean(), prog_bar=True)
            self.log("test_clue_seg_iou", test_iou_per_class.mean(), prog_bar=True)

            for i, clue_name in enumerate(self.clue_names):
                safe_name = clue_name.lower().replace(" ", "_").replace("/", "_")
                self.log(f"test_dice_{safe_name}", test_dice_per_class[i])
                self.log(f"test_iou_{safe_name}", test_iou_per_class[i])

        # per-class clue presence F1
        test_clue_f1_per_class = self.test_clue_f1_per_class.compute()
        for i, clue_name in enumerate(self.clue_names):
            safe_name = clue_name.lower().replace(" ", "_").replace("/", "_")
            self.log(f"test_clue_f1_{safe_name}", test_clue_f1_per_class[i])

        self.test_diag_acc.reset()
        self.test_diag_f1.reset()
        self.test_clue_f1.reset()
        self.test_chaos_f1.reset()
        self.test_clue_f1_per_class.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer is not None else 20,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    