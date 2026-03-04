"""
================================================================================
Train MGCA-ISIC: Joint Pre-training + Classification
================================================================================
This script trains the MGCA-ISIC model which combines:
- MGCA's 3-level contrastive pre-training (ITA + CTA + CPA)
- Supervised classification for chaos and clues

The model learns:
1. Visual-semantic alignment from generated text descriptions
2. Chaos prediction (structure_is_chaotic, colour_is_chaotic)
3. Clues prediction (10 dermoscopic clue indicators)

Usage:
    # Quick test (1 epoch, limited batches)
    python train_joint.py --max_epochs 1 --batch_size 8 --quick_test
    
    # Standard training
    python train_joint.py --max_epochs 50 --batch_size 32
    
    # Full training with ViT encoder
    python train_joint.py --max_epochs 50 --batch_size 16 --img_encoder vit_base
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
from pytorch_lightning import Trainer, seed_everything, LightningDataModule
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import CSVLogger

from datasets.isic_dataset import ISICPretrainingDataset, isic_collate_fn
from datasets.transforms import DataTransforms
from models.kgcl import MGCA_ISIC

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class ISICDataModule(LightningDataModule):
    """DataModule for ISIC-2019 dataset with MGCA-ISIC."""
    
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


def main():
    parser = ArgumentParser(description="Train MGCA-ISIC (Joint Training)")
    
    # Data paths - use relative paths within ISIC-2019 folder
    default_csv = os.path.join(BASE_DIR, "../Data Preprocess/ISIC_2019_annotated_combined.csv")
    default_img = os.path.join(BASE_DIR, "../Images")
    
    parser.add_argument("--csv_path", type=str, default=default_csv)
    parser.add_argument("--img_dir", type=str, default=default_img)
    
    # Model args
    parser.add_argument("--img_encoder", type=str, default="resnet_50",
                        help="Image encoder: resnet_50 or vit_base")
    parser.add_argument("--freeze_bert", action="store_true",
                        help="Freeze BERT text encoder")
    parser.add_argument("--emb_dim", type=int, default=128,
                        help="Joint embedding dimension")
    
    # MGCA loss weights
    parser.add_argument("--lambda_1", type=float, default=1.0,
                        help="Instance-wise alignment weight (ITA)")
    parser.add_argument("--lambda_2", type=float, default=0.7,
                        help="Token-wise alignment weight (CTA)")
    parser.add_argument("--lambda_3", type=float, default=0.5,
                        help="Prototype alignment weight (CPA)")
    
    # Classification loss weights
    parser.add_argument("--lambda_chaos", type=float, default=1.0,
                        help="Chaos classification weight")
    parser.add_argument("--lambda_clues", type=float, default=1.0,
                        help="Clues classification weight")
    
    # Prototype settings
    parser.add_argument("--num_prototypes", type=int, default=100,
                        help="Number of disease prototypes")
    
    # Classification head settings
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for classification heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Training args
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
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
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick_test", action="store_true",
                        help="Run quick test with limited batches")
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    print("="*70)
    print("MGCA-ISIC: Joint Pre-training + Classification")
    print("="*70)
    print("\nModel Components:")
    print("  ├── MGCA Pre-training:")
    print("  │   ├── Instance-wise Alignment (ITA) - λ₁ = {:.1f}".format(args.lambda_1))
    print("  │   ├── Token-wise Alignment (CTA)    - λ₂ = {:.1f}".format(args.lambda_2))
    print("  │   └── Prototype Alignment (CPA)     - λ₃ = {:.1f}".format(args.lambda_3))
    print("  └── Classification Heads:")
    print("      ├── Chaos Head (2 outputs)        - λ = {:.1f}".format(args.lambda_chaos))
    print("      └── Clues Head (10 outputs)       - λ = {:.1f}".format(args.lambda_clues))
    print("\nConfiguration:")
    print(f"  Image Encoder: {args.img_encoder}")
    print(f"  Embedding Dim: {args.emb_dim}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Max Epochs:    {args.max_epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Quick Test:    {args.quick_test}")
    
    # Create data module
    print("\n📊 Creating data module...")
    datamodule = ISICDataModule(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_pct=args.data_pct
    )
    
    # Create model
    print("🔧 Creating MGCA-ISIC model...")
    model = MGCA_ISIC(
        img_encoder=args.img_encoder,
        freeze_bert=args.freeze_bert,
        emb_dim=args.emb_dim,
        num_prototypes=args.num_prototypes,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        lambda_chaos=args.lambda_chaos,
        lambda_clues=args.lambda_clues,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Setup checkpoint directory
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(BASE_DIR, f"checkpoints/joint/{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"📁 Checkpoint directory: {ckpt_dir}")
    
    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=ckpt_dir,
            save_last=True,
            mode="min",
            save_top_k=3,
            filename="mgca-isic-{epoch:02d}-{val_loss:.4f}-{val_chaos_auroc:.4f}"
        ),
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=10,
            verbose=True,
            mode="min"
        )
    ]
    
    # Logger
    logger = CSVLogger(
        save_dir=os.path.join(BASE_DIR, "logs"),
        name="joint_training"
    )
    
    # Trainer
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
    
    # Calculate training steps
    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(f"📈 Total training steps: {model.training_steps}")
    
    # Start training
    print("\n🚀 Starting training...")
    print("-"*70)
    
    trainer.fit(model, datamodule=datamodule)
    
    print("-"*70)
    print("\n✅ Training completed!")
    print(f"📁 Best checkpoint saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
