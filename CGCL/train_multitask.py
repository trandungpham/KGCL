"""
================================================================================
Train ISIC Multi-Task Model
================================================================================
This script runs the experiments defined in `models/kgcl/multitask_module.py`.

Supported phases:
- `pretrain`: train the encoder with clue segmentation, clue presence, and chaos
- `finetune`: fine-tune with diagnosis, clue segmentation, clue presence, and chaos

Usage:
    # Quick phase-1 smoke test
    python train_multitask.py --phase pretrain --max_epochs 1 --batch_size 8 --quick_test

    # Standard phase-1 run
    python train_multitask.py --phase pretrain --max_epochs 50 --batch_size 16

    # Phase-2 run initialized from phase-1 checkpoint
    python train_multitask.py --phase finetune --pretrained_phase1_ckpt path/to/phase1.ckpt
================================================================================
"""

import datetime
import os
import sys
from argparse import ArgumentParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

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


class MultiTaskDataModule(LightningDataModule):
    """DataModule for phase-1 pretraining and phase-2 finetuning."""

    def __init__(
        self,
        phase,
        csv_path,
        img_dir,
        mask_dir,
        vector_dir,
        batch_size,
        num_workers,
        data_pct=1.0,
        crop_size=224,
    ):
        super().__init__()
        self.phase = phase
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.vector_dir = vector_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_pct = data_pct
        self.crop_size = crop_size

    def _transform(self, is_train):
        return SpatialClueDataTransforms(
            is_train=is_train,
            crop_size=self.crop_size,
        )

    def train_dataloader(self):
        transform = self._transform(is_train=True)

        if self.phase == "pretrain":
            dataset = PretrainDataset(
                csv_path=self.csv_path,
                img_dir=self.img_dir,
                mask_dir=self.mask_dir,
                vector_dir=self.vector_dir,
                transform=transform,
                data_pct=self.data_pct,
            )
            collate_fn = pretrain_collate_fn
        else:
            dataset = FinetuneDataset(
                csv_path=self.csv_path,
                img_dir=self.img_dir,
                mask_dir=self.mask_dir,
                vector_dir=self.vector_dir,
                split="train",
                transform=transform,
                data_pct=self.data_pct,
            )
            collate_fn = finetune_collate_fn

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.phase == "pretrain":
            return None

        dataset = FinetuneDataset(
            csv_path=self.csv_path,
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
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.phase == "pretrain":
            return None

        dataset = FinetuneDataset(
            csv_path=self.csv_path,
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
    default_img = os.path.join(BASE_DIR, "../Annotated_data/Images")
    default_mask = os.path.join(BASE_DIR, "../Annotated_data/GroundTruthMasks")
    default_vector = os.path.join(BASE_DIR, "../Annotated_data/Vectors")

    parser.add_argument("--phase", type=str, default="finetune", choices=["pretrain", "finetune"])
    parser.add_argument("--csv_path", type=str, default=default_csv)
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

    parser.add_argument("--lambda_seg", type=float, default=1.0)
    parser.add_argument("--lambda_clue", type=float, default=1.0)
    parser.add_argument("--lambda_chaos", type=float, default=1.0)
    parser.add_argument("--lambda_diag", type=float, default=1.0)
    parser.add_argument("--seg_threshold", type=float, default=0.5)

    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick_test", action="store_true")

    return parser


def build_model(args):
    common_kwargs = {
        "backbone_name": args.backbone_name,
        "pretrained": args.pretrained,
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_clues": len(CLUES_NAMES),
        "num_chaos": 2,
        "lambda_seg": args.lambda_seg,
        "lambda_clue": args.lambda_clue,
        "lambda_chaos": args.lambda_chaos,
    }

    if args.phase == "pretrain":
        return PretrainModule(**common_kwargs)

    return FinetuneModule(
        **common_kwargs,
        pretrained_phase1_ckpt=args.pretrained_phase1_ckpt,
        clue_names=CLUES_NAMES,
        seg_threshold=args.seg_threshold,
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

    print(f"Lambda Seg:      {args.lambda_seg}")
    print(f"Lambda Clue:     {args.lambda_clue}")
    print(f"Lambda Chaos:    {args.lambda_chaos}")

    datamodule = MultiTaskDataModule(
        phase=args.phase,
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        vector_dir=args.vector_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_pct=args.data_pct,
        crop_size=args.crop_size,
    )

    model = build_model(args)

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(BASE_DIR, f"checkpoints/multitask/{args.phase}/{timestamp}")
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
        callbacks.append(
            ModelCheckpoint(
                monitor="train_loss_epoch",
                dirpath=ckpt_dir,
                save_last=True,
                mode="min",
                save_top_k=3,
                filename="multitask-pretrain-{epoch:02d}-{train_loss_epoch:.4f}",
            )
        )

    logger = CSVLogger(
        save_dir=os.path.join(BASE_DIR, "logs"),
        name=f"multitask_{args.phase}",
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
