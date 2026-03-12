import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule

from ..backbones.encoder import ImageEncoder
from .kgcl_module import ClassificationHead

class ISICImageOnly(LightningModule):
    def __init__(self,
                 img_encoder: str = "resnet_50",
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.05,
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder,
            output_dim=128,
            pretrained=True
        )

        if "resnet" in img_encoder:
            self.img_feat_dim = 2048
        else:
            self.img_feat_dim = 768

        self.diagnosis_head = ClassificationHead(
            in_features=self.img_feat_dim,
            num_classes=2,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_auroc = torchmetrics.AUROC(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")

    def forward(self, batch):
        imgs = batch["imgs"]
        img_feat_q, patch_feat_q = self.img_encoder_q(imgs)

        if len(img_feat_q.shape) == 4:
            img_feat_raw = F.adaptive_avg_pool2d(img_feat_q, 1).flatten(1)
        elif len(img_feat_q.shape) == 3:
            img_feat_raw = img_feat_q[:, 0]
        else:
            img_feat_raw = img_feat_q

        diagnosis_logits = self.diagnosis_head(img_feat_raw)
        return diagnosis_logits

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["diagnosis_labels"]
        loss = self.loss_fn(logits, labels)

        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        self.train_acc(preds, labels)

        self.log("train_loss", loss, prog_bar=True, batch_size=labels.size(0))
        self.log("train_diagnosis_acc", self.train_acc, prog_bar=True, batch_size=labels.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["diagnosis_labels"]
        loss = self.loss_fn(logits, labels)

        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        self.val_acc(preds, labels)
        self.val_auroc(probs, labels)
        self.val_f1(preds, labels)

        self.log("val_loss", loss, prog_bar=True, batch_size=labels.size(0), sync_dist=True)
        self.log("val_diagnosis_acc", self.val_acc, prog_bar=True, batch_size=labels.size(0))
        self.log("val_diagnosis_auroc", self.val_auroc, prog_bar=True, batch_size=labels.size(0))
        self.log("val_diagnosis_f1", self.val_f1, batch_size=labels.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer