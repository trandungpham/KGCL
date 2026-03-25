from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import (
    ISICPhase1Dataset,
    ISICPhase2Dataset,
    isic_phase1_collate_fn,
    isic_phase2_collate_fn,
)


class PretrainDataModule(pl.LightningDataModule):
    """
    Phase 1:
    - train only
    - uses only train_csv
    """

    def __init__(
        self,
        train_csv: str,
        img_dir: str,
        mask_dir: str,
        vector_dir: str,
        train_transform=None,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        data_pct: float = 1.0,
    ):
        super().__init__()
        self.train_csv = train_csv
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.vector_dir = vector_dir
        self.train_transform = train_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.data_pct = data_pct

        self.train_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ISICPhase1Dataset(
                csv_path=self.train_csv,
                img_dir=self.img_dir,
                mask_dir=self.mask_dir,
                vector_dir=self.vector_dir,
                transform=self.train_transform,
                data_pct=self.data_pct,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=isic_phase1_collate_fn,
        )

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None


class FinetuneDataModule(pl.LightningDataModule):
    """
    Phase 2:
    - train / val / test
    - each split comes from a separate CSV
    """

    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        img_dir: str,
        mask_dir: str,
        vector_dir: str,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_data_pct: float = 1.0,
    ):
        super().__init__()

        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.vector_dir = vector_dir

        self.train_transform = train_transform
        self.val_transform = val_transform if val_transform is not None else train_transform
        self.test_transform = test_transform if test_transform is not None else self.val_transform

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_data_pct = train_data_pct

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ISICPhase2Dataset(
                csv_path=self.train_csv,
                img_dir=self.img_dir,
                mask_dir=self.mask_dir,
                vector_dir=self.vector_dir,
                transform=self.train_transform,
                data_pct=self.train_data_pct,
            )

            self.val_dataset = ISICPhase2Dataset(
                csv_path=self.val_csv,
                img_dir=self.img_dir,
                mask_dir=self.mask_dir,
                vector_dir=self.vector_dir,
                transform=self.val_transform,
                data_pct=1.0,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ISICPhase2Dataset(
                csv_path=self.test_csv,
                img_dir=self.img_dir,
                mask_dir=self.mask_dir,
                vector_dir=self.vector_dir,
                transform=self.test_transform,
                data_pct=1.0,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=isic_phase2_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=isic_phase2_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=isic_phase2_collate_fn,
        )