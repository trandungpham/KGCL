"""
================================================================================
MGCA Data Module: PyTorch Lightning Data Management
================================================================================
This module implements a reusable PyTorch Lightning DataModule for managing
train/validation/test data loaders across different datasets in MGCA.

PyTorch Lightning DataModule Benefits:
- Centralizes all data loading logic in one place
- Ensures reproducibility across different runs
- Handles train/val/test split management automatically
- Integrates seamlessly with Lightning Trainer

Usage:
    from datasets.data_module import DataModule
    from datasets.pretrain_dataset import MultimodalPretrainingDataset, multimodal_collate_fn
    from datasets.transforms import DataTransforms
    
    # Create data module
    datamodule = DataModule(
        dataset=MultimodalPretrainingDataset,
        collate_fn=multimodal_collate_fn,
        transforms=DataTransforms,
        data_pct=1.0,
        batch_size=72,
        num_workers=8
    )
    
    # Use with Lightning Trainer
    trainer.fit(model, datamodule=datamodule)

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DataModule                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ train_dataloader│  │ val_dataloader  │  │ test_dataloader │              │
│  │                 │  │                 │  │                 │              │
│  │ - shuffle=True  │  │ - shuffle=False │  │ - shuffle=False │              │
│  │ - drop_last=True│  │ - drop_last=True│  │ - drop_last=False│             │
│  │ - augmentation  │  │ - no augment    │  │ - no augment    │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
================================================================================
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    """
    Generic Data Module for MGCA framework.
    
    A flexible data module that works with any dataset class that follows
    the MGCA dataset interface. It handles:
    - Creating datasets with appropriate transforms for each split
    - Configuring DataLoaders with optimal settings
    - Managing train/val/test data splits
    
    The module uses dependency injection for maximum flexibility:
    - Dataset class is passed as argument (not instantiated here)
    - Collate function is configurable per dataset
    - Transforms class is passed and instantiated per split
    
    Args:
        dataset (class): Dataset class (not instance) to use.
            Must accept: split, transform, data_pct arguments.
            Examples: MultimodalPretrainingDataset, CheXpertImageDataset
            
        collate_fn (callable): Custom collate function for batching.
            Receives list of samples, returns batched dictionary.
            Example: multimodal_collate_fn
            
        transforms (class): Transform class for data augmentation.
            Called as transforms(is_train, crop_size).
            Example: DataTransforms
            
        data_pct (float): Fraction of training data to use (0.0 to 1.0).
            Useful for few-shot learning experiments.
            Example: 0.01 = use 1% of training data
            
        batch_size (int): Number of samples per batch.
            Larger batches → better gradient estimates but more memory.
            Typical: 64-144 for MGCA pre-training
            
        num_workers (int): Number of parallel data loading processes.
            Rule of thumb: 4 * num_gpus
            Set to 0 for debugging (single-threaded)
            
        crop_size (int): Image crop size after transforms (default: 224).
            Standard for ImageNet-pretrained models.
    
    Example:
        # Pre-training with MIMIC-CXR
        dm = DataModule(
            dataset=MultimodalPretrainingDataset,
            collate_fn=multimodal_collate_fn,
            transforms=DataTransforms,
            data_pct=1.0,
            batch_size=72,
            num_workers=16,
            crop_size=224
        )
        
        # Fine-tuning with 1% of CheXpert
        dm = DataModule(
            dataset=CheXpertImageDataset,
            collate_fn=default_collate,
            transforms=DataTransforms,
            data_pct=0.01,  # Few-shot setting
            batch_size=32,
            num_workers=8
        )
    """
    
    def __init__(self, dataset, collate_fn, transforms, data_pct, batch_size, num_workers, crop_size=224):
        super().__init__()
        
        # Store configuration
        # Note: We store the class, not an instance
        self.dataset = dataset          # Dataset class
        self.collate_fn = collate_fn    # Batching function
        self.transforms = transforms    # Transform class
        self.data_pct = data_pct        # Data fraction (for few-shot)
        self.batch_size = batch_size    # Samples per batch
        self.num_workers = num_workers  # Parallel workers
        self.crop_size = crop_size      # Image size

    def train_dataloader(self):
        """
        Create training DataLoader.
        
        Training-specific settings:
        - shuffle=True: Randomize sample order each epoch
        - drop_last=True: Drop incomplete final batch (ensures consistent batch size)
        - pin_memory=True: Faster GPU transfer (pre-allocate pinned memory)
        - transforms(True, ...): Training augmentations enabled
        
        Returns:
            DataLoader: Training data loader
            
        Training Transforms (typically):
        - Random crop to crop_size
        - Random horizontal flip
        - Color jittering
        - Normalization
        """
        # Create transforms with training augmentation
        if self.transforms:
            transform = self.transforms(True, self.crop_size)  # is_train=True
        else:
            transform = None
        
        # Instantiate dataset for training split
        dataset = self.dataset(
            split="train", 
            transform=transform, 
            data_pct=self.data_pct  # Apply data percentage to training only
        )

        return DataLoader(
            dataset,
            pin_memory=True,      # Faster CPU→GPU transfer
            drop_last=True,       # Ensure consistent batch sizes
            shuffle=True,         # Randomize order each epoch
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        """
        Create validation DataLoader.
        
        Validation-specific settings:
        - shuffle=False: Consistent order for reproducible metrics
        - drop_last=True: Match training behavior for fair comparison
        - transforms(False, ...): No augmentation (deterministic)
        
        Returns:
            DataLoader: Validation data loader
            
        Validation Transforms (typically):
        - Center crop or resize to crop_size
        - Normalization (same as training)
        - NO random augmentations
        """
        # Create transforms without augmentation
        if self.transforms:
            transform = self.transforms(False, self.crop_size)  # is_train=False
        else:
            transform = None
            
        # Instantiate dataset for validation split
        dataset = self.dataset(
            split="valid", 
            transform=transform, 
            data_pct=self.data_pct
        )
        
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,       # Match training for fair comparison
            shuffle=False,        # Deterministic order
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        """
        Create test DataLoader.
        
        Test-specific settings:
        - shuffle=False: Consistent order for final evaluation
        - drop_last=False: Evaluate ALL samples (no dropping)
        - transforms(False, ...): No augmentation
        
        Returns:
            DataLoader: Test data loader
            
        Note: drop_last=False ensures we evaluate every sample,
        unlike training/validation where we drop incomplete batches.
        This is important for accurate test metrics.
        """
        # Create transforms without augmentation
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
            
        # Instantiate dataset for test split
        dataset = self.dataset(
            split="test", 
            transform=transform, 
            data_pct=self.data_pct
        )
        
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,         # Deterministic order
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
            # Note: drop_last=False (default) - evaluate ALL samples
        )
