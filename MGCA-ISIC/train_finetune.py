"""
================================================================================
Fine-tune ISIC Classification with Pre-trained MGCA (Two-Stage Approach)
================================================================================
This script implements the TWO-STAGE training approach from the MGCA paper:

Stage 1 (Pre-training - run train_joint.py first):
    - Train MGCA model with ITA + CTA + CPA losses
    - Learn visual-semantic alignment from image-text pairs
    
Stage 2 (Fine-tuning - THIS SCRIPT):
    - Load pre-trained MGCA backbone
    - Freeze backbone, train classification heads
    - Chaos head (2 outputs): structure_is_chaotic, colour_is_chaotic
    - Clues head (10 outputs): 10 dermoscopic clue indicators

This approach is particularly effective for:
- Limited labeled data scenarios
- Transfer learning to new domains
- Leveraging pre-trained representations

Usage:
    # Quick test
    python train_finetune.py --max_epochs 1 --batch_size 8 --quick_test
    
    # Fine-tune from pre-trained checkpoint
    python train_finetune.py --checkpoint checkpoints/joint/xxx/last.ckpt
    
    # Train with unfrozen backbone (full fine-tuning)
    python train_finetune.py --no_freeze_backbone

Reference:
    Wang et al., "Multi-Granularity Cross-modal Alignment for Generalized 
    Medical Visual Representation Learning", NeurIPS 2022
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
from models.kgcl import MGCA_ISIC, ISICFineTuner
from models.backbones.encoder import ImageEncoder

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class ISICClassificationDataModule(LightningDataModule):
    """DataModule for ISIC classification fine-tuning."""
    
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
    parser = ArgumentParser(description="Fine-tune ISIC Classification (Two-Stage)")
    
    # Data paths - use relative paths within ISIC-2019 folder
    default_csv = os.path.join(BASE_DIR, "../Data Preprocess/ISIC_2019_annotated_combined.csv")
    default_img = os.path.join(BASE_DIR, "../Images")
    
    parser.add_argument("--csv_path", type=str, default=default_csv)
    parser.add_argument("--img_dir", type=str, default=default_img)
    
    # Pre-trained model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pre-trained MGCA checkpoint")
    parser.add_argument("--img_encoder", type=str, default="resnet_50",
                        help="Image encoder architecture (used if no checkpoint)")
    
    # Fine-tuning settings
    parser.add_argument("--no_freeze_backbone", action="store_true",
                        help="Unfreeze backbone for full fine-tuning")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension for classifier heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Loss weights
    parser.add_argument("--chaos_weight", type=float, default=1.0,
                        help="Weight for chaos classification loss")
    parser.add_argument("--clues_weight", type=float, default=1.0,
                        help="Weight for clues classification loss")
    
    # Training args
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_pct", type=float, default=1.0,
                        help="Fraction of training data to use")
    
    # Trainer args
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick_test", action="store_true",
                        help="Run quick test with limited batches")
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    print("="*70)
    print("ISIC Fine-tuning: Two-Stage Classification (MGCA Paper Approach)")
    print("="*70)
    
    # Load backbone
    if args.checkpoint:
        print(f"\n📦 Loading pre-trained MGCA from: {args.checkpoint}")
        mgca_model = MGCA_ISIC.load_from_checkpoint(args.checkpoint, strict=False)
        backbone = mgca_model.img_encoder_q
        model_name = mgca_model.hparams.img_encoder
        
        if "resnet" in model_name:
            in_features = 2048
        else:  # vit
            in_features = 768
    else:
        print(f"\n🔧 Creating new backbone: {args.img_encoder}")
        backbone = ImageEncoder(
            model_name=args.img_encoder,
            output_dim=128,
            pretrained=True
        )
        model_name = args.img_encoder
        
        if "resnet" in model_name:
            in_features = 2048
        else:
            in_features = 768
    
    freeze_backbone = not args.no_freeze_backbone
    
    print(f"\nConfiguration:")
    print(f"  Backbone:        {model_name}")
    print(f"  Freeze Backbone: {freeze_backbone}")
    print(f"  Hidden Dim:      {args.hidden_dim}")
    print(f"  Dropout:         {args.dropout}")
    print(f"  Batch Size:      {args.batch_size}")
    print(f"  Max Epochs:      {args.max_epochs}")
    print(f"  Learning Rate:   {args.learning_rate}")
    print(f"  Data Percentage: {args.data_pct*100:.1f}%")
    
    # Create data module
    print("\n📊 Creating data module...")
    datamodule = ISICClassificationDataModule(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_pct=args.data_pct
    )
    
    # Create fine-tuner
    print("🔧 Creating fine-tuner with dual classification heads...")
    tuner = ISICFineTuner(
        backbone=backbone,
        model_name=model_name,
        in_features=in_features,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_backbone=freeze_backbone,
        chaos_weight=args.chaos_weight,
        clues_weight=args.clues_weight
    )
    
    # Setup checkpoint directory
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(BASE_DIR, f"checkpoints/finetune/{timestamp}")
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
            filename="isic-finetune-{epoch:02d}-{val_loss:.4f}-{val_chaos_auroc:.4f}"
        ),
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=15,
            verbose=True,
            mode="min"
        )
    ]
    
    # Logger
    logger = CSVLogger(
        save_dir=os.path.join(BASE_DIR, "logs"),
        name="finetune"
    )
    
    # Trainer
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "devices": args.gpus,
        "precision": args.precision,
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
    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)
    print(f"📈 Total training steps: {tuner.training_steps}")
    
    # Start training
    print("\n🚀 Starting fine-tuning...")
    print("-"*70)
    
    trainer.fit(tuner, datamodule=datamodule)
    
    print("-"*70)
    print("\n✅ Fine-tuning completed!")
    
    # Test
    if not args.quick_test:
        print("\n📊 Running test evaluation...")
        trainer.test(tuner, datamodule=datamodule, ckpt_path="best")
    
    print(f"📁 Best checkpoint saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
