"""
================================================================================
Train Spatial Clue Alignment Model:
Joint Pre-training + Diagnosis + Chaos + Clue Segmentation + Clue Alignment
================================================================================
This script trains the Spatial Clue Alignment model which combines:
- MGCA's 3-level cross-modal pre-training (ITA + CTA + CPA)
- Supervised binary diagnosis classification (NV vs MEL)
- Spatial supervision with segmentation masks
- Clue-specific masked patch-token alignment

The model learns:
1. Visual-semantic alignment from pseudo-generated text descriptions
2. Binary diagnosis prediction (NV=0, MEL=1)
3. Lesion/concept-region localization from segmentation supervision
4. Image-level chaos classification without chaos masks
5. Clue-specific spatial grounding using 9 clue masks and clue-text alignment

Usage:
    # Quick test
    python train_segment.py --max_epochs 1 --batch_size 8 --quick_test

    # Standard training
    python train_segment.py --max_epochs 50 --batch_size 16

    # Train with ViT encoder
    python train_segment.py --max_epochs 50 --batch_size 8 --img_encoder vit_base
================================================================================
"""

import datetime
import os
import sys
from argparse import ArgumentParser

# Add parent directory to path for imports
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

from datasets.spatial_clue import (
    ISICSpatialClueDataset,
    isic_spatial_clue_collate_fn,
)
from datasets.transforms import SpatialClueDataTransforms
from models.kgcl import SpatialClueAlignment

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class ISICSpatialClueDataModule(LightningDataModule):
    """DataModule for ISIC spatial clue-aware MGCA training."""

    def __init__(
        self,
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
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.vector_dir = vector_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_pct = data_pct
        self.crop_size = crop_size

    def train_dataloader(self):
        transform = SpatialClueDataTransforms(
            is_train=True,
            crop_size=self.crop_size,
        )

        dataset = ISICSpatialClueDataset(
            csv_path=self.csv_path,
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            vector_dir=self.vector_dir,
            split="train",
            transform=transform,
            data_pct=self.data_pct,
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=isic_spatial_clue_collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        transform = SpatialClueDataTransforms(
            is_train=False,
            crop_size=self.crop_size,
        )

        dataset = ISICSpatialClueDataset(
            csv_path=self.csv_path,
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            vector_dir=self.vector_dir,
            split="valid",
            transform=transform,
            data_pct=1.0,
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=isic_spatial_clue_collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        transform = SpatialClueDataTransforms(
            is_train=False,
            crop_size=self.crop_size,
        )

        dataset = ISICSpatialClueDataset(
            csv_path=self.csv_path,
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            vector_dir=self.vector_dir,
            split="test",
            transform=transform,
            data_pct=1.0,
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=isic_spatial_clue_collate_fn,
            pin_memory=True,
        )


def main():
    parser = ArgumentParser(description="Train MGCA-ISIC Spatial Clue")

    # -------------------------------------------------------------------------
    # DATA PATHS
    # -------------------------------------------------------------------------
    default_csv = os.path.join(BASE_DIR, "../Annotated_data/annotations_index.csv")
    default_img = os.path.join(BASE_DIR, "../Annotated_data/Images")
    default_mask = os.path.join(BASE_DIR, "../Annotated_data/GroundTruthMasks")
    default_vector = os.path.join(BASE_DIR, "../Annotated_data/Vectors")

    parser.add_argument("--csv_path", type=str, default=default_csv)
    parser.add_argument("--img_dir", type=str, default=default_img)
    parser.add_argument("--mask_dir", type=str, default=default_mask)
    parser.add_argument("--vector_dir", type=str, default=default_vector)

    # -------------------------------------------------------------------------
    # MODEL ARGS
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--img_encoder",
        type=str,
        default="vit_base",
        help="Image encoder: resnet_50 or vit_base",
    )
    parser.add_argument(
        "--freeze_bert",
        action="store_true",
        help="Freeze BERT text encoder",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=128,
        help="Joint embedding dimension",
    )

    # -------------------------------------------------------------------------
    # MGCA LOSS WEIGHTS
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--lambda_1",
        type=float,
        default=0.5,
        help="Instance-wise alignment weight (ITA)",
    )
    parser.add_argument(
        "--lambda_2",
        type=float,
        default=0.3,
        help="Token-wise alignment weight (CTA)",
    )
    parser.add_argument(
        "--lambda_3",
        type=float,
        default=0.2,
        help="Prototype alignment weight (CPA)",
    )

    # -------------------------------------------------------------------------
    # SUPERVISED LOSS WEIGHTS
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--lambda_diagnosis",
        type=float,
        default=2.0,
        help="Diagnosis classification weight (NV vs MEL)",
    )
    parser.add_argument(
        "--lambda_chaos",
        type=float,
        default=1.0,
        help="Chaos classification loss weight",
    )
    parser.add_argument(
        "--lambda_clue_seg",
        type=float,
        default=1.0,
        help="Clue segmentation loss weight",
    )
    parser.add_argument(
        "--lambda_clue_seg_dice",
        type=float,
        default=1.0,
        help="Dice loss weight inside clue segmentation loss",
    )
    parser.add_argument(
        "--lambda_clue_align",
        type=float,
        default=1.0,
        help="Clue-specific masked alignment loss weight",
    )

    # -------------------------------------------------------------------------
    # PROTOTYPE SETTINGS
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--num_prototypes",
        type=int,
        default=100,
        help="Number of disease prototypes",
    )

    # -------------------------------------------------------------------------
    # HEAD SETTINGS
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for classification head",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    parser.add_argument(
        "--seg_mid_dim",
        type=int,
        default=256,
        help="Intermediate channel dimension for segmentation head",
    )

    # -------------------------------------------------------------------------
    # TRAINING ARGS
    # -------------------------------------------------------------------------
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--data_pct",
        type=float,
        default=1.0,
        help="Fraction of training data to use",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=224,
        help="Image crop size",
    )

    # -------------------------------------------------------------------------
    # TRAINER ARGS
    # -------------------------------------------------------------------------
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # -------------------------------------------------------------------------
    # MISC
    # -------------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run quick test with limited batches",
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    print("=" * 80)
    print("MGCA-ISIC Spatial Clue: Joint Training")
    print("=" * 80)

    print("\nModel Components:")
    print("  ├── MGCA Pre-training:")
    print(f"  │   ├── Instance-wise Alignment (ITA)     - λ₁ = {args.lambda_1:.2f}")
    print(f"  │   ├── Token-wise Alignment (CTA)        - λ₂ = {args.lambda_2:.2f}")
    print(f"  │   └── Prototype Alignment (CPA)         - λ₃ = {args.lambda_3:.2f}")
    print("  ├── Supervised Diagnosis:")
    print(f"  │   └── Diagnosis Head (NV vs MEL)        - λ  = {args.lambda_diagnosis:.2f}")
    print("  ├── Chaos Classification:")
    print(f"  │   └── Chaos Head                        - λ  = {args.lambda_chaos:.2f}")
    print("  ├── Clue Segmentation:")
    print(f"  │   └── Clue Segmentation Loss            - λ  = {args.lambda_clue_seg:.2f}")
    print(f"  │       └── Dice component                - λ  = {args.lambda_clue_seg_dice:.2f}")
    print("  └── Clue-specific Alignment:")
    print(f"      └── Clue-region / clue-text align     - λ  = {args.lambda_clue_align:.2f}")

    print("\nConfiguration:")
    print(f"  Image Encoder:   {args.img_encoder}")
    print(f"  Embedding Dim:   {args.emb_dim}")
    print(f"  Batch Size:      {args.batch_size}")
    print(f"  Max Epochs:      {args.max_epochs}")
    print(f"  Learning Rate:   {args.learning_rate}")
    print(f"  Crop Size:       {args.crop_size}")
    print(f"  Quick Test:      {args.quick_test}")

    # -------------------------------------------------------------------------
    # DATA MODULE
    # -------------------------------------------------------------------------
    print("\n📊 Creating data module...")
    datamodule = ISICSpatialClueDataModule(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        vector_dir=args.vector_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_pct=args.data_pct,
        crop_size=args.crop_size,
    )

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    print("🔧 Creating MGCA-ISIC Spatial Clue model...")
    model = SpatialClueAlignment(
        img_encoder=args.img_encoder,
        freeze_bert=args.freeze_bert,
        emb_dim=args.emb_dim,
        num_prototypes=args.num_prototypes,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        lambda_diagnosis=args.lambda_diagnosis,
        lambda_chaos=args.lambda_chaos,
        lambda_clue_seg=args.lambda_clue_seg,
        lambda_clue_seg_dice=args.lambda_clue_seg_dice,
        lambda_clue_align=args.lambda_clue_align,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        seg_mid_dim=args.seg_mid_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # -------------------------------------------------------------------------
    # CHECKPOINT DIR
    # -------------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(BASE_DIR, f"checkpoints/joint_spatial_clue/{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"📁 Checkpoint directory: {ckpt_dir}")

    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val_diagnosis_acc",
            dirpath=ckpt_dir,
            save_last=True,
            mode="max",
            save_top_k=3,
            filename="mgca-spatial-clue-{epoch:02d}-{val_loss:.4f}-{val_diagnosis_acc:.4f}",
        ),
        EarlyStopping(
            monitor="val_diagnosis_acc",
            min_delta=0.001,
            patience=10,
            verbose=True,
            mode="max",
        ),
    ]

    # -------------------------------------------------------------------------
    # LOGGER
    # -------------------------------------------------------------------------
    logger = CSVLogger(
        save_dir=os.path.join(BASE_DIR, "logs"),
        name="joint_spatial_clue_training",
    )

    # -------------------------------------------------------------------------
    # TRAINER
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # TRAINING STEPS
    # -------------------------------------------------------------------------
    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(f"📈 Total training steps: {model.training_steps}")

    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------
    print("\n🚀 Starting training...")
    print("-" * 80)
    trainer.fit(model, datamodule=datamodule)

    print("-" * 80)
    print("\n✅ Training completed!")
    print("\n🧪 Running test with best checkpoint...")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print(f"📁 Best checkpoint saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
