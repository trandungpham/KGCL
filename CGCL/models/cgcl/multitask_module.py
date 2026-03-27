from typing import List, Optional

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score, MultilabelF1Score


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


class MultiTaskNet(nn.Module):
    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        num_clues=9,
        num_chaos=2,
        num_diag=2,
        out_indices=(1, 2, 3, 4),
        diag_hidden_dim=512,
        diag_dropout=0.3,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        last_dim = self.encoder.feature_info.channels()[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.clue_area_head = nn.Conv2d(last_dim, num_clues, kernel_size=1)
        self.clue_head = GlobalMLPHead(last_dim, num_clues)
        self.chaos_head = GlobalMLPHead(last_dim, num_chaos)

        self.diagnosis_head = GlobalMLPHead(
            last_dim + num_clues + num_chaos,
            num_diag,
            hidden_dim=diag_hidden_dim,
            dropout=diag_dropout,
        )

    def forward(self, x):
        feats = self.encoder(x)
        pooled = self.global_pool(feats[-1]).flatten(1)

        clue_area_logits = self.clue_area_head(feats[-1])
        clue_area_logits = F.interpolate(
            clue_area_logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        clue_area_pooled = F.adaptive_avg_pool2d(clue_area_logits, 1).flatten(1)

        clue_logits = self.clue_head(pooled) + clue_area_pooled
        chaos_logits = self.chaos_head(pooled)

        diag_input = torch.cat([
            pooled,
            clue_logits.detach(),
            chaos_logits.detach()
        ], dim=1)

        diagnosis_logits = self.diagnosis_head(diag_input)

        return {
            "clue_logits": clue_logits,
            "clue_area_logits": clue_area_logits,
            "chaos_logits": chaos_logits,
            "diagnosis_logits": diagnosis_logits,
        }


def clue_area_alignment_loss(
    area_logits: torch.Tensor,
    area_targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
):
    if pos_weight is not None:
        pos_weight = pos_weight.view(1, -1, 1, 1)
    return F.binary_cross_entropy_with_logits(
        area_logits,
        area_targets,
        pos_weight=pos_weight,
    )


class PretrainModule(pl.LightningModule):
    """
    Phase 1:
    train encoder using
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
        lambda_clue=1.0,
        lambda_chaos=1.0,
        lambda_align=1.0,
        clue_pos_weight: Optional[torch.Tensor] = None,
        area_pos_weight: Optional[torch.Tensor] = None,
        out_indices=(1, 2, 3, 4),
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MultiTaskNet(
            backbone_name=backbone_name,
            pretrained=pretrained,
            num_clues=num_clues,
            num_chaos=num_chaos,
            num_diag=2,
            out_indices=out_indices,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_clue = lambda_clue
        self.lambda_chaos = lambda_chaos
        self.lambda_align = lambda_align
        self.register_buffer("clue_pos_weight", clue_pos_weight)
        self.register_buffer("area_pos_weight", area_pos_weight)

        self.train_clue_f1 = MultilabelF1Score(num_labels=num_clues, average="macro")
        self.train_chaos_f1 = MultilabelF1Score(num_labels=num_chaos, average="macro")

    def forward(self, x):
        return self.model(x)

    def compute_losses(self, batch):
        outputs = self(batch["imgs"])

        loss_clue = F.binary_cross_entropy_with_logits(
            outputs["clue_logits"],
            batch["clue_present"],
            pos_weight=self.clue_pos_weight,
        )
        loss_chaos = F.binary_cross_entropy_with_logits(
            outputs["chaos_logits"],
            batch["chaos_labels"],
        )
        loss_align = clue_area_alignment_loss(
            outputs["clue_area_logits"],
            batch["clue_masks"],
            pos_weight=self.area_pos_weight,
        )
        total_loss = (
            self.lambda_clue * loss_clue
            + self.lambda_chaos * loss_chaos
            + self.lambda_align * loss_align
        )

        return {
            "loss": total_loss,
            "loss_clue": loss_clue,
            "loss_chaos": loss_chaos,
            "loss_align": loss_align,
            "outputs": outputs,
        }

    def training_step(self, batch, batch_idx):
        results = self.compute_losses(batch)
        outputs = results["outputs"]

        clue_preds = (torch.sigmoid(outputs["clue_logits"]) >= 0.5).int()
        chaos_preds = (torch.sigmoid(outputs["chaos_logits"]) >= 0.5).int()

        self.train_clue_f1.update(clue_preds, batch["clue_present"].int())
        self.train_chaos_f1.update(chaos_preds, batch["chaos_labels"].int())

        self.log("train_loss", results["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_clue", results["loss_clue"], on_step=True, on_epoch=True)
        self.log("train_loss_chaos", results["loss_chaos"], on_step=True, on_epoch=True)
        self.log("train_loss_align", results["loss_align"], on_step=True, on_epoch=True)
        return results["loss"]

    def on_train_epoch_end(self):
        self.log("train_clue_f1", self.train_clue_f1.compute(), prog_bar=True)
        self.log("train_chaos_f1", self.train_chaos_f1.compute(), prog_bar=True)
        self.train_clue_f1.reset()
        self.train_chaos_f1.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


class FinetuneModule(pl.LightningModule):
    """
    Phase 2:
    image-only fine-tuning with
    - diagnosis classification
    - clue presence
    - chaos classification
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
        lambda_diag=1.0,
        lambda_clue=1.0,
        lambda_chaos=1.0,
        lambda_align=1.0,
        clue_pos_weight: Optional[torch.Tensor] = None,
        area_pos_weight: Optional[torch.Tensor] = None,
        initial_clue_thresholds: Optional[torch.Tensor] = None,
        out_indices=(1, 2, 3, 4),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["clue_names"])

        self.model = MultiTaskNet(
            backbone_name=backbone_name,
            pretrained=pretrained,
            num_clues=num_clues,
            num_chaos=num_chaos,
            num_diag=2,
            out_indices=out_indices,
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
        self.lambda_diag = lambda_diag
        self.lambda_clue = lambda_clue
        self.lambda_chaos = lambda_chaos
        self.lambda_align = lambda_align
        self.register_buffer("clue_pos_weight", clue_pos_weight)
        self.register_buffer("area_pos_weight", area_pos_weight)

        self.num_clues = num_clues
        self.num_chaos = num_chaos
        self.clue_names = clue_names or [f"clue_{i}" for i in range(num_clues)]
        if initial_clue_thresholds is None:
            initial_clue_thresholds = torch.full((num_clues,), 0.5, dtype=torch.float32)
        self.register_buffer("clue_thresholds", initial_clue_thresholds.float())

        self.val_diag_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_diag_f1 = F1Score(task="multiclass", num_classes=2, average="macro")
        self.val_clue_f1 = MultilabelF1Score(num_labels=num_clues, average="macro")
        self.val_chaos_f1 = MultilabelF1Score(num_labels=num_chaos, average="macro")

        self.test_diag_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_diag_f1 = F1Score(task="multiclass", num_classes=2, average="macro")
        self.test_clue_f1 = MultilabelF1Score(num_labels=num_clues, average="macro")
        self.test_chaos_f1 = MultilabelF1Score(num_labels=num_chaos, average="macro")

        self.val_clue_f1_per_class = MultilabelF1Score(num_labels=num_clues, average=None)
        self.test_clue_f1_per_class = MultilabelF1Score(num_labels=num_clues, average=None)
        self._reset_val_clue_threshold_buffers()

    def _reset_val_clue_threshold_buffers(self):
        self.val_clue_probs = []
        self.val_clue_targets = []

    def _compute_best_clue_thresholds(self):
        if not self.val_clue_probs:
            return self.clue_thresholds, torch.tensor(0.0, device=self.device)

        probs = torch.cat(self.val_clue_probs, dim=0)
        targets = torch.cat(self.val_clue_targets, dim=0).int()
        grid = torch.linspace(0.05, 0.95, steps=19, device=probs.device)

        best_thresholds = self.clue_thresholds.detach().clone().to(probs.device)
        best_scores = torch.zeros(self.num_clues, device=probs.device)

        for clue_idx in range(self.num_clues):
            clue_probs = probs[:, clue_idx]
            clue_targets = targets[:, clue_idx]
            for threshold in grid:
                preds = (clue_probs >= threshold).int()
                tp = ((preds == 1) & (clue_targets == 1)).sum().float()
                fp = ((preds == 1) & (clue_targets == 0)).sum().float()
                fn = ((preds == 0) & (clue_targets == 1)).sum().float()
                score = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
                if score > best_scores[clue_idx]:
                    best_scores[clue_idx] = score
                    best_thresholds[clue_idx] = threshold

        return best_thresholds.detach(), best_scores.mean()

    def forward(self, x):
        return self.model(x)

    def compute_losses(self, batch):
        outputs = self(batch["imgs"])

        loss_diag = F.cross_entropy(outputs["diagnosis_logits"], batch["diagnosis_labels"], label_smoothing=0.1)
        loss_clue = F.binary_cross_entropy_with_logits(
            outputs["clue_logits"],
            batch["clue_present"],
            pos_weight=self.clue_pos_weight,
        )
        loss_chaos = F.binary_cross_entropy_with_logits(
            outputs["chaos_logits"],
            batch["chaos_labels"],
        )
        loss_align = clue_area_alignment_loss(
            outputs["clue_area_logits"],
            batch["clue_masks"],
            pos_weight=self.area_pos_weight,
        )
        total_loss = (
            self.lambda_diag * loss_diag
            + self.lambda_clue * loss_clue
            + self.lambda_chaos * loss_chaos
            + self.lambda_align * loss_align
        )

        return {
            "loss": total_loss,
            "loss_diag": loss_diag,
            "loss_clue": loss_clue,
            "loss_chaos": loss_chaos,
            "loss_align": loss_align,
            "outputs": outputs,
        }

    def training_step(self, batch, batch_idx):
        results = self.compute_losses(batch)
        self.log("train_loss", results["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_diag", results["loss_diag"], on_step=True, on_epoch=True)
        self.log("train_loss_clue", results["loss_clue"], on_step=True, on_epoch=True)
        self.log("train_loss_chaos", results["loss_chaos"], on_step=True, on_epoch=True)
        self.log("train_loss_align", results["loss_align"], on_step=True, on_epoch=True)
        return results["loss"]

    def validation_step(self, batch, batch_idx):
        results = self.compute_losses(batch)
        outputs = results["outputs"]

        diag_preds = torch.argmax(outputs["diagnosis_logits"], dim=1)
        chaos_preds = (torch.sigmoid(outputs["chaos_logits"]) >= 0.5).int()

        self.val_diag_acc.update(diag_preds, batch["diagnosis_labels"])
        self.val_diag_f1.update(diag_preds, batch["diagnosis_labels"])
        self.val_chaos_f1.update(chaos_preds, batch["chaos_labels"].int())
        self.val_clue_probs.append(torch.sigmoid(outputs["clue_logits"]).detach())
        self.val_clue_targets.append(batch["clue_present"].detach())

        self.log("val_loss", results["loss"], prog_bar=True, on_epoch=True)
        self.log("val_loss_diag", results["loss_diag"], on_epoch=True)
        self.log("val_loss_clue", results["loss_clue"], on_epoch=True)
        self.log("val_loss_chaos", results["loss_chaos"], on_epoch=True)
        self.log("val_loss_align", results["loss_align"], on_epoch=True)

    def on_validation_epoch_start(self):
        self._reset_val_clue_threshold_buffers()

    def on_validation_epoch_end(self):
        best_thresholds, tuned_macro_f1 = self._compute_best_clue_thresholds()
        self.clue_thresholds.copy_(best_thresholds.to(self.clue_thresholds.device))

        all_probs = torch.cat(self.val_clue_probs, dim=0)
        all_targets = torch.cat(self.val_clue_targets, dim=0).int()
        tuned_preds = (all_probs >= self.clue_thresholds.unsqueeze(0)).int()
        self.val_clue_f1.update(tuned_preds, all_targets)
        self.val_clue_f1_per_class.update(tuned_preds, all_targets)

        self.log("val_diag_acc", self.val_diag_acc.compute(), prog_bar=True)
        self.log("val_diag_f1", self.val_diag_f1.compute(), prog_bar=True)
        self.log("val_clue_f1", self.val_clue_f1.compute(), prog_bar=True)
        self.log("val_chaos_f1", self.val_chaos_f1.compute(), prog_bar=True)
        self.log("val_clue_f1_tuned", tuned_macro_f1, prog_bar=True)

        val_clue_f1_per_class = self.val_clue_f1_per_class.compute()
        for i, clue_name in enumerate(self.clue_names):
            safe_name = clue_name.lower().replace(" ", "_").replace("/", "_")
            self.log(f"val_clue_f1_{safe_name}", val_clue_f1_per_class[i])
            self.log(f"val_clue_threshold_{safe_name}", self.clue_thresholds[i])

        self.val_diag_acc.reset()
        self.val_diag_f1.reset()
        self.val_clue_f1.reset()
        self.val_chaos_f1.reset()
        self.val_clue_f1_per_class.reset()

    def test_step(self, batch, batch_idx):
        results = self.compute_losses(batch)
        outputs = results["outputs"]

        diag_preds = torch.argmax(outputs["diagnosis_logits"], dim=1)
        clue_preds = (torch.sigmoid(outputs["clue_logits"]) >= self.clue_thresholds.unsqueeze(0)).int()
        chaos_preds = (torch.sigmoid(outputs["chaos_logits"]) >= 0.5).int()

        self.test_diag_acc.update(diag_preds, batch["diagnosis_labels"])
        self.test_diag_f1.update(diag_preds, batch["diagnosis_labels"])
        self.test_clue_f1.update(clue_preds, batch["clue_present"].int())
        self.test_chaos_f1.update(chaos_preds, batch["chaos_labels"].int())
        self.test_clue_f1_per_class.update(clue_preds, batch["clue_present"].int())

        self.log("test_loss", results["loss"], on_epoch=True)
        self.log("test_loss_diag", results["loss_diag"], on_epoch=True)
        self.log("test_loss_clue", results["loss_clue"], on_epoch=True)
        self.log("test_loss_chaos", results["loss_chaos"], on_epoch=True)
        self.log("test_loss_align", results["loss_align"], on_epoch=True)

    def on_test_epoch_end(self):
        self.log("test_diag_acc", self.test_diag_acc.compute(), prog_bar=True)
        self.log("test_diag_f1", self.test_diag_f1.compute(), prog_bar=True)
        self.log("test_clue_f1", self.test_clue_f1.compute(), prog_bar=True)
        self.log("test_chaos_f1", self.test_chaos_f1.compute(), prog_bar=True)

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
        backbone_params = list(self.model.encoder.parameters())
        head_params = [p for n, p in self.model.named_parameters() if "encoder" not in n]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.lr * 0.1},
            {"params": head_params, "lr": self.lr},
        ], weight_decay=self.weight_decay)

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
