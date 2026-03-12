"""
================================================================================
Train Image-Only ISIC Diagnosis Model
================================================================================
This script trains an image-only baseline for binary diagnosis classification:

- Input: dermoscopic image
- Output: binary diagnosis (NV=0, MEL=1)

This baseline is intended for ablation against the full MGCA-ISIC framework
with text-guided alignment.

Usage:
    # Quick test
    python train_image_only.py --max_epochs 1 --batch_size 8 --quick_test

    # Standard training
    python train_image_only.py --max_epochs 50 --batch_size 32

    # ViT backbone
    python train_image_only.py --max_epochs 50 --batch_size 16 --img_encoder vit_base
================================================================================
"""

import datetime
import os
import sys
from argparse import ArgumentParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from datasets.constants import NUM_DIAGNOSIS_CLASSES
from datasets.isic_dataset import ISICPretrainingDataset, isic_collate_fn
from datasets.transforms import DataTransforms
from models.backbones.encoder import ImageEncoder

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class ClassificationHead(nn.Module):
    """Simple diagnosis classification head."""

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


class ISICDataModule(LightningDataModule):
    """DataModule for ISIC-2019 dataset."""

    def __init__(self, csv_path, img_dir, batch_size, num_workers,
                 data_pct=1.0, crop_size=224):
        super().__init__()
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_pct = data_pct
        self.crop_size = crop_size

    def train_dataloader(self):
        transform = DataTransforms(is_train=True, crop_size=self.crop_size)
        dataset = ISICPretrainingDataset(
            csv_path=self.csv_path,
            img_dir=self.img_dir,
            split="train",
            transform=transform,
            data_pct=self.data_pct
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=isic_collate_fn,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        transform = DataTransforms(is_train=False, crop_size=self.crop_size)
        dataset = ISICPretrainingDataset(
            csv_path=self.csv_path,
            img_dir=self.img_dir,
            split="valid",
            transform=transform,
            data_pct=1.0
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=isic_collate_fn,
            drop_last=True,
            pin_memory=True
        )

    def test_dataloader(self):
        transform = DataTransforms(is_train=False, crop_size=self.crop_size)
        dataset = ISICPretrainingDataset(
            csv_path=self.csv_path,
            img_dir=self.img_dir,
            split="test",
            transform=transform,
            data_pct=1.0
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=isic_collate_fn,
            pin_memory=True
        )


class ISICImageOnly(LightningModule):
    """
    Image-only baseline for NV vs MEL diagnosis.
    """

    def __init__(self,
                 img_encoder: str = "resnet_50",
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.05,
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 seed: int = 42,
                 class_weights=None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Image encoder
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder,
            output_dim=128,
            pretrained=True
        )

        if "resnet" in img_encoder:
            self.img_feat_dim = 2048
        else:
            self.img_feat_dim = 768

        # Diagnosis head
        self.diagnosis_head = ClassificationHead(
            in_features=self.img_feat_dim,
            num_classes=NUM_DIAGNOSIS_CLASSES,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Loss
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            self.register_buffer("class_weights", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.train_diagnosis_acc = torchmetrics.Accuracy(task="binary")
        self.val_diagnosis_acc = torchmetrics.Accuracy(task="binary")
        self.test_diagnosis_acc = torchmetrics.Accuracy(task="binary")

        self.train_diagnosis_auroc = torchmetrics.AUROC(task="binary")
        self.val_diagnosis_auroc = torchmetrics.AUROC(task="binary")
        self.test_diagnosis_auroc = torchmetrics.AUROC(task="binary")

        self.val_diagnosis_f1 = torchmetrics.F1Score(task="binary")
        self.test_diagnosis_f1 = torchmetrics.F1Score(task="binary")

    def forward(self, batch):
        imgs = batch["imgs"]

        img_feat_q, patch_feat_q = self.img_encoder_q(imgs)

        if len(img_feat_q.shape) == 4:
            img_feat_raw = F.adaptive_avg_pool2d(img_feat_q, 1).flatten(1)
        elif len(img_feat_q.shape) == 3:
            img_feat_raw = img_feat_q[:, 0]  # CLS token for ViT
        else:
            img_feat_raw = img_feat_q

        diagnosis_logits = self.diagnosis_head(img_feat_raw)
        return diagnosis_logits

    def _shared_step(self, batch, stage: str):
        logits = self(batch)
        labels = batch["diagnosis_labels"]

        loss = self.loss_fn(logits, labels)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        bz = labels.size(0)

        if stage == "train":
            self.train_diagnosis_acc(preds, labels)
            self.train_diagnosis_auroc(probs, labels)

            self.log("train_loss", loss, prog_bar=True, batch_size=bz)
            self.log("train_diagnosis_acc", self.train_diagnosis_acc, prog_bar=True, batch_size=bz)
            self.log("train_diagnosis_auroc", self.train_diagnosis_auroc, batch_size=bz)

        elif stage == "val":
            self.val_diagnosis_acc(preds, labels)
            self.val_diagnosis_auroc(probs, labels)
            self.val_diagnosis_f1(preds, labels)

            self.log("val_loss", loss, prog_bar=True, batch_size=bz, sync_dist=True)
            self.log("val_diagnosis_acc", self.val_diagnosis_acc, prog_bar=True, batch_size=bz)
            self.log("val_diagnosis_auroc", self.val_diagnosis_auroc, prog_bar=True, batch_size=bz)
            self.log("val_diagnosis_f1", self.val_diagnosis_f1, prog_bar=True, batch_size=bz)

        elif stage == "test":
            self.test_diagnosis_acc(preds, labels)
            self.test_diagnosis_auroc(probs, labels)
            self.test_diagnosis_f1(preds, labels)

            self.log("test_loss", loss, batch_size=bz, sync_dist=True)
            self.log("test_diagnosis_acc", self.test_diagnosis_acc, batch_size=bz)
            self.log("test_diagnosis_auroc", self.test_diagnosis_auroc, batch_size=bz)
            self.log("test_diagnosis_f1", self.test_diagnosis_f1, batch_size=bz)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-8
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


def main():
    parser = ArgumentParser(description="Train Image-Only ISIC Baseline")

    # Data paths
    default_csv = os.path.join(BASE_DIR, "../Dataset/annotated_combined_with_descriptions.csv")
    default_img = os.path.join(BASE_DIR, "../Dataset/Images")

    parser.add_argument("--csv_path", type=str, default=default_csv)
    parser.add_argument("--img_dir", type=str, default=default_img)

    # Model args
    parser.add_argument("--img_encoder", type=str, default="resnet_50",
                        help="Image encoder: resnet_50 or vit_base")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for diagnosis head")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training args
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_pct", type=float, default=1.0,
                        help="Fraction of training data to use")

    # Trainer args
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Optional class weighting
    parser.add_argument("--mel_weight", type=float, default=None,
                        help="Optional positive-class weight for MEL. Example: 2.0")
    parser.add_argument("--nv_weight", type=float, default=1.0,
                        help="Weight for NV class if class weighting is used")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick_test", action="store_true",
                        help="Run quick test with limited batches")

    args = parser.parse_args()

    seed_everything(args.seed)

    print("=" * 70)
    print("ISIC Image-Only Baseline: Binary Diagnosis Classification")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Image Encoder: {args.img_encoder}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Max Epochs:    {args.max_epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Quick Test:    {args.quick_test}")

    # Data module
    print("\n📊 Creating data module...")
    datamodule = ISICDataModule(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_pct=args.data_pct
    )

    # Optional class weights
    class_weights = None
    if args.mel_weight is not None:
        class_weights = [args.nv_weight, args.mel_weight]
        print(f"  Class weights: NV={args.nv_weight}, MEL={args.mel_weight}")

    # Model
    print("🔧 Creating image-only model...")
    model = ISICImageOnly(
        img_encoder=args.img_encoder,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        class_weights=class_weights
    )

    # Checkpoint directory
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(BASE_DIR, f"checkpoints/image_only/{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"📁 Checkpoint directory: {ckpt_dir}")

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            monitor="val_diagnosis_acc",
            dirpath=ckpt_dir,
            save_last=True,
            mode="max",
            save_top_k=3,
            filename="image-only-{epoch:02d}-{val_loss:.4f}-{val_diagnosis_acc:.4f}"
        ),
        EarlyStopping(
            monitor="val_diagnosis_acc",
            min_delta=0.001,
            patience=10,
            verbose=True,
            mode="max"
        )
    ]

    logger = CSVLogger(
        save_dir=os.path.join(BASE_DIR, "logs"),
        name="image_only_training"
    )

    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "devices": args.gpus,
        "precision": args.precision,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "deterministic": True,
        "callbacks": callbacks,
        "logger": logger,
        "enable_progress_bar": True,
    }

    if args.quick_test:
        trainer_kwargs["limit_train_batches"] = 10
        trainer_kwargs["limit_val_batches"] = 5
        print("\n⚡ Quick test mode: Limited batches")

    trainer = Trainer(**trainer_kwargs)

    print("\n🚀 Starting training...")
    print("-" * 70)
    trainer.fit(model, datamodule=datamodule)
    print("-" * 70)

    print("\n🧪 Running test with best checkpoint...")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print("\n✅ Training completed!")
    print(f"📁 Best checkpoint saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()