"""
================================================================================
Train ISIC Multi-Task Model
================================================================================
This script runs the experiments defined in `models/kgcl/multitask_module.py`.

Supported phases:
- `pretrain`: train the encoder with clue presence, clue-area alignment, and chaos
- `finetune`: fine-tune with diagnosis, clue presence, clue-area alignment, and chaos

Usage:
    # Quick phase-1 smoke test
    python train_multitask.py --phase pretrain --max_epochs 1 --batch_size 8 --quick_test

    # Standard phase-1 run
    python train_multitask.py --phase pretrain --max_epochs 100 --batch_size 16 --backbone_name convnext_base

    # Phase-2 run initialized from phase-1 checkpoint
    python train_multitask.py --phase finetune --lambda_diag 2.0 --lambda_align 0.5 --pretrained_phase1_ckpt checkpoints/multitask/pretrain/2026_03_27_14_46_53/best.ckpt --backbone_name convnext_base
================================================================================
"""

import datetime
import os
import sys
from argparse import ArgumentParser
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from datasets.constants import CLUES_NAMES
from datasets.dataset import (
    FinetuneDataset,
    PretrainDataset,
    finetune_collate_fn,
    pretrain_collate_fn,
)
from datasets.transforms import SpatialClueDataTransforms
from models.cgcl.multitask_module import FinetuneModule, PretrainModule

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)


class MultiTaskDataModule(LightningDataModule):
    """DataModule for phase-1 pretraining and phase-2 finetuning."""

    def __init__(
        self,
        phase,
        train_csv,
        test_csv,
        val_csv,
        img_dir,
        mask_dir,
        vector_dir,
        batch_size,
        num_workers,
        data_pct=1.0,
        crop_size=224,
        use_weighted_sampler=False,
        sampler_clue_weight_scale=1.0,
        sampler_diag_weight_scale=0.5,
    ):
        super().__init__()
        self.phase = phase
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.val_csv = val_csv
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.vector_dir = vector_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_pct = data_pct
        self.crop_size = crop_size
        self.use_weighted_sampler = use_weighted_sampler
        self.sampler_clue_weight_scale = sampler_clue_weight_scale
        self.sampler_diag_weight_scale = sampler_diag_weight_scale

    def _transform(self, is_train):
        return SpatialClueDataTransforms(
            is_train=is_train,
            crop_size=self.crop_size,
        )

    def train_dataloader(self):
        transform = self._transform(is_train=True)

        if self.phase == "pretrain":
            dataset = PretrainDataset(
                csv_path=self.train_csv,
                img_dir=self.img_dir,
                mask_dir=self.mask_dir,
                vector_dir=self.vector_dir,
                transform=transform,
                data_pct=self.data_pct,
            )
            collate_fn = pretrain_collate_fn
        else:
            dataset = FinetuneDataset(
                csv_path=self.train_csv,
                img_dir=self.img_dir,
                mask_dir=self.mask_dir,
                vector_dir=self.vector_dir,
                split="train",
                transform=transform,
                data_pct=self.data_pct,
            )
            collate_fn = finetune_collate_fn

        sampler = None
        shuffle = True
        if self.phase == "finetune" and self.use_weighted_sampler:
            sample_weights = dataset.get_sample_weights(
                clue_weight_scale=self.sampler_clue_weight_scale,
                diagnosis_weight_scale=self.sampler_diag_weight_scale,
            )
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.phase == "pretrain":
            return None

        dataset = FinetuneDataset(
            csv_path=self.val_csv,
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            vector_dir=self.vector_dir,
            split="valid",
            transform=self._transform(is_train=False),
            data_pct=1.0,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=finetune_collate_fn,
            # drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.phase == "pretrain":
            return None

        dataset = FinetuneDataset(
            csv_path=self.test_csv,
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            vector_dir=self.vector_dir,
            split="test",
            transform=self._transform(is_train=False),
            data_pct=1.0,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=finetune_collate_fn,
            pin_memory=True,
        )


def build_parser():
    parser = ArgumentParser(description="Train multi-task model")

    default_csv = os.path.join(BASE_DIR, "../Annotated_data/annotations_index.csv")
    train_csv = os.path.join(BASE_DIR, "../Annotated_data/train.csv")
    test_csv = os.path.join(BASE_DIR, "../Annotated_data/test.csv")
    val_csv = os.path.join(BASE_DIR, "../Annotated_data/val.csv")
    default_img = os.path.join(BASE_DIR, "../Annotated_data/Images")
    default_mask = os.path.join(BASE_DIR, "../Annotated_data/GroundTruthMasks")
    default_vector = os.path.join(BASE_DIR, "../Annotated_data/Vectors")

    parser.add_argument("--phase", type=str, default="finetune", choices=["pretrain", "finetune"])
    parser.add_argument("--train_csv", type=str, default=train_csv)
    parser.add_argument("--test_csv", type=str, default=test_csv)
    parser.add_argument("--val_csv", type=str, default=val_csv)
    parser.add_argument("--img_dir", type=str, default=default_img)
    parser.add_argument("--mask_dir", type=str, default=default_mask)
    parser.add_argument("--vector_dir", type=str, default=default_vector)

    parser.add_argument("--backbone_name", type=str, default="resnet50")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument(
        "--no_pretrained",
        action="store_false",
        dest="pretrained",
        help="Disable timm pretrained weights",
    )
    parser.add_argument("--pretrained_phase1_ckpt", type=str, default=None)

    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_pct", type=float, default=1.0)
    parser.add_argument("--crop_size", type=int, default=224)

    parser.add_argument("--lambda_clue", type=float, default=1.0)
    parser.add_argument("--lambda_chaos", type=float, default=1.0)
    parser.add_argument("--lambda_diag", type=float, default=1.0)
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--sampler_clue_weight_scale", type=float, default=1.0)
    parser.add_argument("--sampler_diag_weight_scale", type=float, default=0.5)
    parser.add_argument("--max_pos_weight", type=float, default=50.0)

    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick_test", action="store_true")

    return parser


def _load_rows(csv_path):
    with open(csv_path, newline="") as handle:
        return list(csv.DictReader(handle))



def compute_training_statistics(csv_path, mask_dir, vector_dir, phase, max_pos_weight):
    rows = _load_rows(csv_path)

    clue_vectors = []
    area_positive = None
    area_total_pixels = 0

    for row in rows:
        clue_vector = np.load(os.path.join(vector_dir, row["vector_name"])).astype(np.float32)
        clue_vectors.append(clue_vector)

        clue_mask = np.load(os.path.join(mask_dir, row["mask_name"])).astype(np.float32)
        clue_mask = (clue_mask > 0).astype(np.float32)
        current_positive = clue_mask.sum(axis=(1, 2))
        area_positive = current_positive if area_positive is None else area_positive + current_positive
        area_total_pixels += clue_mask.shape[1] * clue_mask.shape[2]

    clue_vectors = np.stack(clue_vectors)
    clue_positive = clue_vectors.sum(axis=0)
    clue_negative = len(clue_vectors) - clue_positive
    clue_pos_weight = clue_negative / np.clip(clue_positive, a_min=1.0, a_max=None)
    clue_pos_weight = np.clip(clue_pos_weight, a_min=1.0, a_max=max_pos_weight)
    area_negative = area_total_pixels - area_positive
    area_pos_weight = area_negative / np.clip(area_positive, a_min=1.0, a_max=None)
    area_pos_weight = np.clip(area_pos_weight, a_min=1.0, a_max=max_pos_weight)

    return {
        "clue_pos_weight": torch.tensor(clue_pos_weight, dtype=torch.float32),
        "area_pos_weight": torch.tensor(area_pos_weight, dtype=torch.float32),
    }


_FOUR_STAGE_BACKBONES = ("convnext", "swin", "efficientnet", "mobilenet", "densenet", "regnet")


def _get_out_indices(backbone_name: str):
    """ResNet/VGG have 5 stages -> (1,2,3,4). ConvNeXt/Swin/EfficientNet have 4 -> (0,1,2,3)."""
    if any(backbone_name.lower().startswith(p) for p in _FOUR_STAGE_BACKBONES):
        return (0, 1, 2, 3)
    return (1, 2, 3, 4)


def build_model(args):
    stats = compute_training_statistics(
        csv_path=args.train_csv,
        mask_dir=args.mask_dir,
        vector_dir=args.vector_dir,
        phase=args.phase,
        max_pos_weight=args.max_pos_weight,
    )

    out_indices = _get_out_indices(args.backbone_name)

    common_kwargs = {
        "backbone_name": args.backbone_name,
        "pretrained": args.pretrained,
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_clues": len(CLUES_NAMES),
        "num_chaos": 2,
        "lambda_clue": args.lambda_clue,
        "lambda_chaos": args.lambda_chaos,
        "lambda_align": args.lambda_align,
        "clue_pos_weight": stats["clue_pos_weight"],
        "area_pos_weight": stats["area_pos_weight"],
        "out_indices": out_indices,
    }

    if args.phase == "pretrain":
        return PretrainModule(**common_kwargs)

    return FinetuneModule(
        **common_kwargs,
        pretrained_phase1_ckpt=args.pretrained_phase1_ckpt,
        clue_names=CLUES_NAMES,
        lambda_diag=args.lambda_diag,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    seed_everything(args.seed)

    print("=" * 80)
    print(f"ISIC Multi-Task Training: {args.phase.upper()}")
    print("=" * 80)
    print(f"Backbone:        {args.backbone_name}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Max Epochs:      {args.max_epochs}")
    print(f"Learning Rate:   {args.learning_rate}")
    print(f"Crop Size:       {args.crop_size}")
    print(f"Pretrained:      {args.pretrained}")
    print(f"Quick Test:      {args.quick_test}")

    if args.phase == "finetune":
        print(f"Phase-1 Ckpt:    {args.pretrained_phase1_ckpt}")
        print(f"Lambda Diag:     {args.lambda_diag}")

    print(f"Lambda Clue:     {args.lambda_clue}")
    print(f"Lambda Chaos:    {args.lambda_chaos}")
    print(f"Lambda Align:    {args.lambda_align}")

    datamodule = MultiTaskDataModule(
        phase=args.phase,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        vector_dir=args.vector_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_pct=args.data_pct,
        crop_size=args.crop_size,
        use_weighted_sampler=args.use_weighted_sampler,
        sampler_clue_weight_scale=args.sampler_clue_weight_scale,
        sampler_diag_weight_scale=args.sampler_diag_weight_scale,
    )

    model = build_model(args)

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(BASE_DIR, f"checkpoints/{args.phase}/{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoint Dir:  {ckpt_dir}")

    callbacks = [LearningRateMonitor(logging_interval="step")]

    if args.phase == "finetune":
        callbacks.extend(
            [
                ModelCheckpoint(
                    monitor="val_diag_acc",
                    dirpath=ckpt_dir,
                    save_last=True,
                    mode="max",
                    save_top_k=3,
                    filename="multitask-{epoch:02d}-{val_loss:.4f}-{val_diag_acc:.4f}",
                ),
                EarlyStopping(
                    monitor="val_diag_acc",
                    min_delta=0.001,
                    patience=10,
                    verbose=True,
                    mode="max",
                ),
            ]
        )
    else:
        callbacks.extend(
            [
                ModelCheckpoint(
                    monitor="train_loss_epoch",
                    dirpath=ckpt_dir,
                    save_last=False,
                    mode="min",
                    save_top_k=3,
                    filename="multitask-pretrain-{epoch:02d}-{train_loss_epoch:.4f}",
                ),
                EarlyStopping(
                    monitor="train_loss_epoch",
                    min_delta=0.001,
                    patience=10,
                    verbose=True,
                    mode="min",
                ),
            ]
        )

    logger = CSVLogger(
        save_dir=ckpt_dir,
        name="",
        version="",
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
        if args.phase == "finetune":
            trainer_kwargs["limit_val_batches"] = 5
            trainer_kwargs["limit_test_batches"] = 5
        print("Quick test mode enabled with limited batches.")

    trainer = Trainer(**trainer_kwargs)

    print("\nStarting training...")
    trainer.fit(model, datamodule=datamodule)

    if args.phase == "finetune":
        print("\nRunning test with best checkpoint...")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        best_checkpoint = torch.load(best_model_path, map_location="cpu")
        best_pth_path = os.path.join(ckpt_dir, "best.ckpt")
        torch.save(best_checkpoint["state_dict"], best_pth_path)
        print(f"Exported best model weights to: {best_pth_path}")
    else:
        print("No best checkpoint was recorded, skipping best.ckpt export.")

    print("Training completed.")


if __name__ == "__main__":
    main()
