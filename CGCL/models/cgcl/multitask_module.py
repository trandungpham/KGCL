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


class ConceptConfidenceEstimator(nn.Module):
    """
    Dedicated agentic confidence estimator for a single clue or chaos concept.

    A pool of num_agents specialist networks with feature-conditioned softmax
    routing. Each specialist independently outputs a confidence in [0, 1];
    the final score is their routing-weighted mixture. Because each instance
    covers exactly one concept, the agents can specialise on concept-specific
    visual cues rather than sharing capacity across all T concepts.
    """

    def __init__(self, in_dim: int, num_agents: int = 4, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_agents),
        )
        self.agents = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_agents)
        ])

    def forward(self, pooled: torch.Tensor):
        weights = torch.softmax(self.router(pooled), dim=-1)          # [B, E]
        scores  = torch.cat([a(pooled) for a in self.agents], dim=-1) # [B, E]
        return (weights * scores).sum(dim=-1), weights                 # [B], [B, E]


class PerConceptAuxiliaryHub(nn.Module):
    """
    One ConceptConfidenceEstimator per clue/chaos concept.

    Replaces the shared AgenticAuxiliaryLayer: each of the T = num_clues +
    num_chaos concepts owns a fully independent agent pool, allowing
    concept-specific confidence cues to be learned without interference.
    """

    def __init__(
        self,
        in_dim: int,
        num_clues: int,
        num_chaos: int,
        num_agents: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_clues = num_clues
        self.num_chaos = num_chaos
        self.estimators = nn.ModuleList([
            ConceptConfidenceEstimator(in_dim, num_agents, hidden_dim, dropout)
            for _ in range(num_clues + num_chaos)
        ])

    def forward(self, pooled: torch.Tensor):
        confidences, route_weights = zip(*[est(pooled) for est in self.estimators])
        confidences   = torch.stack(confidences,   dim=1)  # [B, T]
        route_weights = torch.stack(route_weights, dim=1)  # [B, T, E]
        return {
            "clue_confidence":  confidences[:, :self.num_clues],
            "chaos_confidence": confidences[:, self.num_clues:],
            "route_weights":    route_weights,
        }


class MultiTaskNet(nn.Module):
    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        num_clues=9,
        num_chaos=2,
        num_diag=2,
        out_indices=(1, 2, 3, 4),
        use_agentic_aux=False,
        num_aux_agents=4,
        aux_agent_hidden_dim=256,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        last_dim = self.encoder.feature_info.channels()[-1]
        self.last_dim = last_dim
        self.use_agentic_aux = use_agentic_aux
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.clue_area_head = nn.Conv2d(last_dim, num_clues, kernel_size=1)
        # Main prediction heads — always present; agentic layer does not replace them.
        self.clue_head = GlobalMLPHead(last_dim, num_clues)
        self.chaos_head = GlobalMLPHead(last_dim, num_chaos)
        if use_agentic_aux:
            # One dedicated estimator per clue/chaos concept.
            self.auxiliary_layer = PerConceptAuxiliaryHub(
                in_dim=last_dim,
                num_clues=num_clues,
                num_chaos=num_chaos,
                num_agents=num_aux_agents,
                hidden_dim=aux_agent_hidden_dim,
            )
        # Only used in FinetuneModule; stored here so weights load correctly from phase1 ckpt.
        self.diagnosis_head = GlobalMLPHead(last_dim + num_clues + num_chaos, num_diag)
        self._run_diagnosis_head = False

    def forward(self, x):
        feats = self.encoder(x)
        last_feat = feats[-1]
        # Swin outputs NHWC [B,H,W,C]; check both that C is in the last dim
        # and not already in dim-1 before permuting, to avoid mis-permuting
        # NCHW tensors whose spatial size happens to equal the channel count.
        if (last_feat.ndim == 4
                and last_feat.shape[-1] == self.last_dim
                and last_feat.shape[1] != self.last_dim):
            last_feat = last_feat.permute(0, 3, 1, 2).contiguous()
        pooled = self.global_pool(last_feat).flatten(1)

        clue_area_logits = self.clue_area_head(last_feat)
        clue_area_logits = F.interpolate(
            clue_area_logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        clue_area_pooled = F.adaptive_avg_pool2d(clue_area_logits, 1).flatten(1)

        # Main predictions — always from the fixed heads, never modified by agentic layer.
        clue_logits = self.clue_head(pooled) + clue_area_pooled
        chaos_logits = self.chaos_head(pooled)

        # Confidence scores — parallel readout, does not touch logits.
        if self.use_agentic_aux:
            aux = self.auxiliary_layer(pooled)
            clue_confidence = aux["clue_confidence"]
            chaos_confidence = aux["chaos_confidence"]
            route_weights = aux["route_weights"]
        else:
            # Fallback: certainty derived from prediction margin (no learned router).
            clue_probs = torch.sigmoid(clue_logits)
            chaos_probs = torch.sigmoid(chaos_logits)
            clue_confidence = torch.maximum(clue_probs, 1.0 - clue_probs)
            chaos_confidence = torch.maximum(chaos_probs, 1.0 - chaos_probs)
            route_weights = None

        out = {
            "clue_logits": clue_logits,
            "clue_area_logits": clue_area_logits,
            "chaos_logits": chaos_logits,
            "clue_confidence": clue_confidence,
            "chaos_confidence": chaos_confidence,
            "aux_route_weights": route_weights,
            "diagnosis_logits": None,
        }
        if self._run_diagnosis_head:
            out["diagnosis_logits"] = self.diagnosis_head(
                torch.cat([pooled, clue_area_pooled, chaos_logits.detach()], dim=1)
            )
        return out


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


def auxiliary_confidence_loss(
    outputs: dict,
    clue_targets: torch.Tensor,
    chaos_targets: torch.Tensor,
):
    clue_probs = torch.sigmoid(outputs["clue_logits"]).detach()
    chaos_probs = torch.sigmoid(outputs["chaos_logits"]).detach()
    clue_confidence_targets = torch.where(
        clue_targets.bool(),
        clue_probs,
        1.0 - clue_probs,
    )
    chaos_confidence_targets = torch.where(
        chaos_targets.bool(),
        chaos_probs,
        1.0 - chaos_probs,
    )

    return 0.5 * (
        F.mse_loss(outputs["clue_confidence"], clue_confidence_targets)
        + F.mse_loss(outputs["chaos_confidence"], chaos_confidence_targets)
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
        lambda_confidence=0.1,
        clue_pos_weight: Optional[torch.Tensor] = None,
        area_pos_weight: Optional[torch.Tensor] = None,
        out_indices=(1, 2, 3, 4),
        use_agentic_aux=False,
        num_aux_agents=4,
        aux_agent_hidden_dim=256,
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
            use_agentic_aux=use_agentic_aux,
            num_aux_agents=num_aux_agents,
            aux_agent_hidden_dim=aux_agent_hidden_dim,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_clue = lambda_clue
        self.lambda_chaos = lambda_chaos
        self.lambda_align = lambda_align
        self.lambda_confidence = lambda_confidence
        self.use_agentic_aux = use_agentic_aux
        self.register_buffer("clue_pos_weight", clue_pos_weight)
        self.register_buffer("area_pos_weight", area_pos_weight)

        self.train_clue_f1 = MultilabelF1Score(num_labels=num_clues, average="macro")
        self.train_chaos_f1 = MultilabelF1Score(num_labels=num_chaos, average="macro")
        self._train_clue_conf: list = []
        self._train_chaos_conf: list = []

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
        if self.use_agentic_aux:
            loss_confidence = auxiliary_confidence_loss(
                outputs,
                batch["clue_present"],
                batch["chaos_labels"],
            )
        else:
            loss_confidence = loss_clue.new_zeros(())

        total_loss = (
            self.lambda_clue * loss_clue
            + self.lambda_chaos * loss_chaos
            + self.lambda_align * loss_align
            + self.lambda_confidence * loss_confidence
        )

        return {
            "loss": total_loss,
            "loss_clue": loss_clue,
            "loss_chaos": loss_chaos,
            "loss_align": loss_align,
            "loss_confidence": loss_confidence,
            "outputs": outputs,
        }

    def training_step(self, batch, batch_idx):
        results = self.compute_losses(batch)
        outputs = results["outputs"]

        clue_preds = (torch.sigmoid(outputs["clue_logits"]) >= 0.5).int()
        chaos_preds = (torch.sigmoid(outputs["chaos_logits"]) >= 0.5).int()

        self.train_clue_f1.update(clue_preds, batch["clue_present"].int())
        self.train_chaos_f1.update(chaos_preds, batch["chaos_labels"].int())
        if self.use_agentic_aux:
            self._train_clue_conf.append(outputs["clue_confidence"].detach())
            self._train_chaos_conf.append(outputs["chaos_confidence"].detach())

        self.log("train_loss", results["loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss_clue", results["loss_clue"], on_step=False, on_epoch=True)
        self.log("train_loss_chaos", results["loss_chaos"], on_step=False, on_epoch=True)
        self.log("train_loss_align", results["loss_align"], on_step=False, on_epoch=True)
        if self.use_agentic_aux:
            self.log("train_loss_confidence", results["loss_confidence"], on_step=False, on_epoch=True)
        return results["loss"]

    def on_train_epoch_end(self):
        self.log("train_clue_f1", self.train_clue_f1.compute(), prog_bar=True)
        self.log("train_chaos_f1", self.train_chaos_f1.compute(), prog_bar=True)
        self.train_clue_f1.reset()
        self.train_chaos_f1.reset()
        if self.use_agentic_aux and self._train_clue_conf:
            self.log("train_clue_conf_mean", torch.cat(self._train_clue_conf).mean())
            self.log("train_chaos_conf_mean", torch.cat(self._train_chaos_conf).mean())
            self._train_clue_conf.clear()
            self._train_chaos_conf.clear()

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
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }


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
        chaos_names: Optional[List[str]] = None,
        lr=1e-4,
        weight_decay=1e-4,
        num_clues=9,
        num_chaos=2,
        lambda_diag=1.0,
        lambda_clue=1.0,
        lambda_chaos=1.0,
        lambda_align=1.0,
        lambda_confidence=0.1,
        task_mode: str = "multitask",
        clue_pos_weight: Optional[torch.Tensor] = None,
        area_pos_weight: Optional[torch.Tensor] = None,
        diag_class_weight: Optional[torch.Tensor] = None,
        initial_clue_thresholds: Optional[torch.Tensor] = None,
        out_indices=(1, 2, 3, 4),
        use_agentic_aux=False,
        num_aux_agents=4,
        aux_agent_hidden_dim=256,
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
            use_agentic_aux=use_agentic_aux,
            num_aux_agents=num_aux_agents,
            aux_agent_hidden_dim=aux_agent_hidden_dim,
        )

        self.model._run_diagnosis_head = True

        if pretrained_phase1_ckpt is not None:
            ckpt = torch.load(pretrained_phase1_ckpt, map_location="cpu", weights_only=True)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            filtered_state_dict = {
                k.replace("model.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
            self.model.load_state_dict(filtered_state_dict, strict=False)

        self.lr = lr
        self.weight_decay = weight_decay
        self.task_mode = task_mode
        self.lambda_diag = lambda_diag
        self.lambda_clue = lambda_clue
        self.lambda_chaos = lambda_chaos
        self.lambda_align = lambda_align
        self.lambda_confidence = lambda_confidence
        self.use_agentic_aux = use_agentic_aux
        self.register_buffer("clue_pos_weight", clue_pos_weight)
        self.register_buffer("area_pos_weight", area_pos_weight)
        self.register_buffer("diag_class_weight", diag_class_weight)

        self.num_clues = num_clues
        self.num_chaos = num_chaos
        self.clue_names  = clue_names  or [f"clue_{i}"  for i in range(num_clues)]
        self.chaos_names = chaos_names or [f"chaos_{i}" for i in range(num_chaos)]
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
        self._val_clue_conf: list = []
        self._val_chaos_conf: list = []
        self._val_chaos_probs: list = []
        self._val_chaos_targets: list = []
        self._test_clue_conf: list = []
        self._test_chaos_conf: list = []
        self._test_clue_probs: list = []
        self._test_clue_targets: list = []
        self._test_chaos_probs: list = []
        self._test_chaos_targets: list = []

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

        loss_diag = F.cross_entropy(
            outputs["diagnosis_logits"],
            batch["diagnosis_labels"],
            weight=self.diag_class_weight,
            label_smoothing=0.1,
        )
        if self.task_mode == "diag_only":
            zero = loss_diag.new_zeros(())
            return {
                "loss": loss_diag,
                "loss_diag": loss_diag,
                "loss_clue": zero,
                "loss_chaos": zero,
                "loss_align": zero,
                "loss_confidence": zero,
                "outputs": outputs,
            }

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
        if self.use_agentic_aux:
            loss_confidence = auxiliary_confidence_loss(
                outputs,
                batch["clue_present"],
                batch["chaos_labels"],
            )
        else:
            loss_confidence = loss_clue.new_zeros(())

        total_loss = (
            self.lambda_diag * loss_diag
            + self.lambda_clue * loss_clue
            + self.lambda_chaos * loss_chaos
            + self.lambda_align * loss_align
            + self.lambda_confidence * loss_confidence
        )

        return {
            "loss": total_loss,
            "loss_diag": loss_diag,
            "loss_clue": loss_clue,
            "loss_chaos": loss_chaos,
            "loss_align": loss_align,
            "loss_confidence": loss_confidence,
            "outputs": outputs,
        }

    def training_step(self, batch, batch_idx):
        results = self.compute_losses(batch)
        self.log("train_loss", results["loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss_diag", results["loss_diag"], on_step=False, on_epoch=True)
        if self.task_mode != "diag_only":
            self.log("train_loss_clue", results["loss_clue"], on_step=False, on_epoch=True)
            self.log("train_loss_chaos", results["loss_chaos"], on_step=False, on_epoch=True)
            self.log("train_loss_align", results["loss_align"], on_step=False, on_epoch=True)
            if self.use_agentic_aux:
                self.log("train_loss_confidence", results["loss_confidence"], on_step=False, on_epoch=True)
        return results["loss"]

    def validation_step(self, batch, batch_idx):
        results = self.compute_losses(batch)
        outputs = results["outputs"]

        diag_preds = torch.argmax(outputs["diagnosis_logits"], dim=1)

        self.val_diag_acc.update(diag_preds, batch["diagnosis_labels"])
        self.val_diag_f1.update(diag_preds, batch["diagnosis_labels"])
        if self.task_mode != "diag_only":
            chaos_preds = (torch.sigmoid(outputs["chaos_logits"]) >= 0.5).int()
            self.val_chaos_f1.update(chaos_preds, batch["chaos_labels"].int())
            self.val_clue_probs.append(torch.sigmoid(outputs["clue_logits"]).detach())
            self.val_clue_targets.append(batch["clue_present"].detach())
            if self.use_agentic_aux:
                self._val_clue_conf.append(outputs["clue_confidence"].detach())
                self._val_chaos_conf.append(outputs["chaos_confidence"].detach())
                self._val_chaos_probs.append(torch.sigmoid(outputs["chaos_logits"]).detach())
                self._val_chaos_targets.append(batch["chaos_labels"].detach())

        self.log("val_loss", results["loss"], prog_bar=True, on_epoch=True)
        self.log("val_loss_diag", results["loss_diag"], on_epoch=True)
        if self.task_mode != "diag_only":
            self.log("val_loss_clue", results["loss_clue"], on_epoch=True)
            self.log("val_loss_chaos", results["loss_chaos"], on_epoch=True)
            self.log("val_loss_align", results["loss_align"], on_epoch=True)
            if self.use_agentic_aux:
                self.log("val_loss_confidence", results["loss_confidence"], on_epoch=True)

    def on_validation_epoch_start(self):
        if self.task_mode != "diag_only":
            self._reset_val_clue_threshold_buffers()

    def on_validation_epoch_end(self):
        self.log("val_diag_acc", self.val_diag_acc.compute(), prog_bar=True)
        self.log("val_diag_f1", self.val_diag_f1.compute(), prog_bar=True)
        if self.task_mode != "diag_only":
            best_thresholds, tuned_macro_f1 = self._compute_best_clue_thresholds()
            self.clue_thresholds.copy_(best_thresholds.to(self.clue_thresholds.device))

            all_probs = torch.cat(self.val_clue_probs, dim=0)
            all_targets = torch.cat(self.val_clue_targets, dim=0).int()
            tuned_preds = (all_probs >= self.clue_thresholds.unsqueeze(0)).int()
            self.val_clue_f1.update(tuned_preds, all_targets)
            self.val_clue_f1_per_class.update(tuned_preds, all_targets)

            self.log("val_clue_f1", self.val_clue_f1.compute(), prog_bar=True)
            self.log("val_chaos_f1", self.val_chaos_f1.compute(), prog_bar=True)
            self.log("val_clue_f1_tuned", tuned_macro_f1, prog_bar=True)

            val_clue_f1_per_class = self.val_clue_f1_per_class.compute()
            for i, clue_name in enumerate(self.clue_names):
                safe_name = clue_name.lower().replace(" ", "_").replace("/", "_")
                self.log(f"val_clue_f1_{safe_name}", val_clue_f1_per_class[i])
                self.log(f"val_clue_threshold_{safe_name}", self.clue_thresholds[i])

            if self.use_agentic_aux and self._val_clue_conf:
                all_clue_conf    = torch.cat(self._val_clue_conf,     dim=0)
                all_chaos_conf   = torch.cat(self._val_chaos_conf,    dim=0)
                all_clue_probs   = torch.cat(self.val_clue_probs,     dim=0)
                all_clue_tgts    = torch.cat(self.val_clue_targets,   dim=0)
                all_chaos_probs  = torch.cat(self._val_chaos_probs,   dim=0)
                all_chaos_tgts   = torch.cat(self._val_chaos_targets, dim=0)
                for m, v in self._confidence_metrics(
                    all_clue_conf,  all_clue_probs,  all_clue_tgts,  self.clue_names,  "val", "clue",
                ).items():
                    self.log(m, v, prog_bar=False)
                for m, v in self._confidence_metrics(
                    all_chaos_conf, all_chaos_probs, all_chaos_tgts, self.chaos_names, "val", "chaos",
                ).items():
                    self.log(m, v, prog_bar=False)
                self._val_clue_conf.clear()
                self._val_chaos_conf.clear()
                self._val_chaos_probs.clear()
                self._val_chaos_targets.clear()

        self.val_diag_acc.reset()
        self.val_diag_f1.reset()
        if self.task_mode != "diag_only":
            self.val_clue_f1.reset()
            self.val_chaos_f1.reset()
            self.val_clue_f1_per_class.reset()

    def test_step(self, batch, batch_idx):
        results = self.compute_losses(batch)
        outputs = results["outputs"]

        diag_preds = torch.argmax(outputs["diagnosis_logits"], dim=1)

        self.test_diag_acc.update(diag_preds, batch["diagnosis_labels"])
        self.test_diag_f1.update(diag_preds, batch["diagnosis_labels"])
        if self.task_mode != "diag_only":
            clue_preds = (torch.sigmoid(outputs["clue_logits"]) >= self.clue_thresholds.unsqueeze(0)).int()
            chaos_preds = (torch.sigmoid(outputs["chaos_logits"]) >= 0.5).int()
            self.test_clue_f1.update(clue_preds, batch["clue_present"].int())
            self.test_chaos_f1.update(chaos_preds, batch["chaos_labels"].int())
            self.test_clue_f1_per_class.update(clue_preds, batch["clue_present"].int())
            if self.use_agentic_aux:
                self._test_clue_conf.append(outputs["clue_confidence"].detach())
                self._test_chaos_conf.append(outputs["chaos_confidence"].detach())
                self._test_clue_probs.append(torch.sigmoid(outputs["clue_logits"]).detach())
                self._test_clue_targets.append(batch["clue_present"].detach())
                self._test_chaos_probs.append(torch.sigmoid(outputs["chaos_logits"]).detach())
                self._test_chaos_targets.append(batch["chaos_labels"].detach())

        self.log("test_loss", results["loss"], on_epoch=True)
        self.log("test_loss_diag", results["loss_diag"], on_epoch=True)
        if self.task_mode != "diag_only":
            self.log("test_loss_clue", results["loss_clue"], on_epoch=True)
            self.log("test_loss_chaos", results["loss_chaos"], on_epoch=True)
            self.log("test_loss_align", results["loss_align"], on_epoch=True)
            if self.use_agentic_aux:
                self.log("test_loss_confidence", results["loss_confidence"], on_epoch=True)

    def on_test_epoch_end(self):
        self.log("test_diag_acc", self.test_diag_acc.compute(), prog_bar=True)
        self.log("test_diag_f1", self.test_diag_f1.compute(), prog_bar=True)
        if self.task_mode != "diag_only":
            self.log("test_clue_f1", self.test_clue_f1.compute(), prog_bar=True)
            self.log("test_chaos_f1", self.test_chaos_f1.compute(), prog_bar=True)

            test_clue_f1_per_class = self.test_clue_f1_per_class.compute()
            for i, clue_name in enumerate(self.clue_names):
                safe_name = clue_name.lower().replace(" ", "_").replace("/", "_")
                self.log(f"test_clue_f1_{safe_name}", test_clue_f1_per_class[i])

            if self.use_agentic_aux and self._test_clue_conf:
                all_clue_conf    = torch.cat(self._test_clue_conf,     dim=0)
                all_chaos_conf   = torch.cat(self._test_chaos_conf,    dim=0)
                all_clue_probs   = torch.cat(self._test_clue_probs,    dim=0)
                all_clue_tgts    = torch.cat(self._test_clue_targets,  dim=0)
                all_chaos_probs  = torch.cat(self._test_chaos_probs,   dim=0)
                all_chaos_tgts   = torch.cat(self._test_chaos_targets, dim=0)
                for m, v in self._confidence_metrics(
                    all_clue_conf,  all_clue_probs,  all_clue_tgts,  self.clue_names,  "test", "clue",
                ).items():
                    self.log(m, v)
                for m, v in self._confidence_metrics(
                    all_chaos_conf, all_chaos_probs, all_chaos_tgts, self.chaos_names, "test", "chaos",
                ).items():
                    self.log(m, v)
                self._test_clue_conf.clear()
                self._test_chaos_conf.clear()
                self._test_clue_probs.clear()
                self._test_clue_targets.clear()
                self._test_chaos_probs.clear()
                self._test_chaos_targets.clear()

        self.test_diag_acc.reset()
        self.test_diag_f1.reset()
        if self.task_mode != "diag_only":
            self.test_clue_f1.reset()
            self.test_chaos_f1.reset()
            self.test_clue_f1_per_class.reset()

    @staticmethod
    def _confidence_metrics(
        conf: torch.Tensor,
        probs: torch.Tensor,
        targets: torch.Tensor,
        names: List[str],
        split: str,
        prefix: str,
    ) -> dict:
        """
        Compute three groups of confidence-quality metrics per concept.

        Args:
            conf:    [N, C] — confidence scores in [0, 1]
            probs:   [N, C] — main head predicted probabilities
            targets: [N, C] — binary ground-truth labels (float)
            names:   C concept names (clue or chaos)
            split:   "val" or "test"
            prefix:  "clue" or "chaos"

        Returns dict with:
          Brier score  — MSE(conf, binary_correctness) per concept + mean
                         Lower → better; 0 = perfect calibration
          Calib gap    — |mean(conf) − accuracy| per concept + mean
                         Lower → better; 0 = perfectly calibrated
          Separation   — mean_conf_correct vs mean_conf_wrong (overall)
                         conf_correct > conf_wrong = good discrimination
        """
        metrics: dict = {}
        preds   = (probs >= 0.5).float()
        correct = (preds == targets).float()            # [N, C]  1 = correct

        brier     = ((conf - correct) ** 2).mean(dim=0)                      # [C]
        calib_gap = (conf.mean(dim=0) - correct.mean(dim=0)).abs()           # [C]

        metrics[f"{split}_{prefix}_conf_brier_mean"]    = brier.mean()
        metrics[f"{split}_{prefix}_conf_calib_gap_mean"] = calib_gap.mean()

        correct_mask = correct.bool()
        if correct_mask.any():
            metrics[f"{split}_{prefix}_conf_when_correct"] = conf[correct_mask].mean()
        if (~correct_mask).any():
            metrics[f"{split}_{prefix}_conf_when_wrong"]   = conf[~correct_mask].mean()

        for i, name in enumerate(names):
            safe = name.lower().replace(" ", "_").replace("/", "_")
            metrics[f"{split}_{prefix}_conf_brier_{safe}"]    = brier[i]
            metrics[f"{split}_{prefix}_conf_calib_gap_{safe}"] = calib_gap[i]

        return metrics

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
