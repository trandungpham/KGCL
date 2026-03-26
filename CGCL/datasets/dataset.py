from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import CHAOS_LABELS, CLUES_NAMES


class BaseDataset(Dataset):
    def __init__(
        self,
        csv_path,
        img_dir,
        mask_dir,
        vector_dir,
        transform=None,
        data_pct=1.0,
    ):
        self.df = pd.read_csv(csv_path)

        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.vector_dir = Path(vector_dir)
        self.transform = transform
        self.data_pct = data_pct

        if data_pct < 1.0:
            self.df = self.df.sample(frac=data_pct, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_name: str):
        img_path = self.img_dir / f"{image_name}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_clue_masks_and_vector(self, row):
        mask_path = self.mask_dir / str(row["mask_name"])
        vector_path = self.vector_dir / str(row["vector_name"])

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        if not vector_path.exists():
            raise FileNotFoundError(f"Vector file not found: {vector_path}")

        clue_masks = np.load(mask_path).astype(np.float32)      # [9,H,W]
        clue_present = np.load(vector_path).astype(np.float32)  # [9]

        if clue_masks.ndim != 3:
            raise ValueError(f"Expected clue_masks shape [9,H,W], got {clue_masks.shape}")
        if clue_present.ndim != 1:
            raise ValueError(f"Expected clue_present shape [9], got {clue_present.shape}")

        if clue_masks.shape[0] != len(CLUES_NAMES):
            raise ValueError(
                f"Expected {len(CLUES_NAMES)} clue mask channels, got {clue_masks.shape[0]}"
            )
        if clue_present.shape[0] != len(CLUES_NAMES):
            raise ValueError(
                f"Expected {len(CLUES_NAMES)} clue presence labels, got {clue_present.shape[0]}"
            )

        clue_masks = (clue_masks > 0).astype(np.float32)
        clue_present = (clue_present > 0).astype(np.float32)

        return clue_masks, clue_present

    def _get_chaos_labels(self, row):
        chaos_labels = np.array(
            [float(bool(row[label])) for label in CHAOS_LABELS],
            dtype=np.float32,
        )
        return chaos_labels

    def _get_diagnosis_label(self, row):
        return 1 if str(row["diagnosis"]).upper() == "MEL" else 0

    def _apply_transform(self, image, clue_masks):
        """
        Apply synchronized spatial transforms to image and 9-channel clue masks.
        union_mask is only used to drive transform consistency.
        """
        union_mask = (clue_masks.sum(axis=0) > 0).astype(np.float32)  # [H,W]

        if self.transform is not None:
            # expected transform signature:
            # image, union_mask, clue_masks = transform(image, union_mask, clue_masks)
            image, _, clue_masks = self.transform(image, union_mask, clue_masks)

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if not isinstance(clue_masks, torch.Tensor):
            clue_masks = torch.tensor(clue_masks, dtype=torch.float32)

        return image, clue_masks


class PretrainDataset(BaseDataset):
    """
    Phase 1 dataset: encoder pretraining only.

    Input / supervision returned:
        - imgs
        - clue_masks
        - clue_present
        - chaos_labels

    No diagnosis label.
    Only loads rows with split == "train" when a split column is present.
    """

    def __init__(
        self,
        csv_path,
        img_dir,
        mask_dir,
        vector_dir,
        transform=None,
        data_pct=1.0,
    ):
        super().__init__(
            csv_path=csv_path,
            img_dir=img_dir,
            mask_dir=mask_dir,
            vector_dir=vector_dir,
            transform=transform,
            data_pct=data_pct,
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name = str(row["image"])
        image = self._load_image(image_name)

        clue_masks, clue_present = self._load_clue_masks_and_vector(row)   # [9,H,W], [9]
        chaos_labels = self._get_chaos_labels(row)                         # [2]

        image, clue_masks = self._apply_transform(image, clue_masks)

        sample = {
            "imgs": image,                                                    # [3,H,W]
            "clue_masks": clue_masks.float(),                                 # [9,H,W]
            "clue_present": torch.tensor(clue_present, dtype=torch.float32),  # [9]
            "chaos_labels": torch.tensor(chaos_labels, dtype=torch.float32),  # [2]
            "image_name": image_name,
        }
        return sample


class FinetuneDataset(BaseDataset):
    """
    Phase 2 dataset: image-only fine-tuning.

    Input:
        - imgs

    Targets:
        - diagnosis_labels
        - clue_masks
        - clue_present
        - chaos_labels
    """

    def __init__(
        self,
        csv_path,
        img_dir,
        mask_dir,
        vector_dir,
        split=None,
        transform=None,
        data_pct=1.0,
    ):
        super().__init__(
            csv_path=csv_path,
            img_dir=img_dir,
            mask_dir=mask_dir,
            vector_dir=vector_dir,
            transform=transform,
            data_pct=data_pct,
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name = str(row["image"])
        image = self._load_image(image_name)

        clue_masks, clue_present = self._load_clue_masks_and_vector(row)   # [9,H,W], [9]
        chaos_labels = self._get_chaos_labels(row)                         # [2]
        diagnosis = self._get_diagnosis_label(row)                         # scalar

        image, clue_masks = self._apply_transform(image, clue_masks)

        sample = {
            "imgs": image,                                                     # [3,H,W]
            "diagnosis_labels": torch.tensor(diagnosis, dtype=torch.long),     # scalar
            "clue_masks": clue_masks.float(),                                  # [9,H,W]
            "clue_present": torch.tensor(clue_present, dtype=torch.float32),   # [9]
            "chaos_labels": torch.tensor(chaos_labels, dtype=torch.float32),   # [2]
            "image_name": image_name,
        }
        return sample

    def get_sample_weights(
        self,
        clue_weight_scale: float = 1.0,
        diagnosis_weight_scale: float = 0.5,
    ):
        clue_vectors = []
        diagnosis_labels = []

        for _, row in self.df.iterrows():
            _, clue_present = self._load_clue_masks_and_vector(row)
            clue_vectors.append(clue_present)
            diagnosis_labels.append(self._get_diagnosis_label(row))

        clue_vectors = np.stack(clue_vectors).astype(np.float32)
        diagnosis_labels = np.asarray(diagnosis_labels, dtype=np.int64)

        clue_pos = clue_vectors.sum(axis=0)
        clue_neg = len(clue_vectors) - clue_pos
        clue_pos_weight = clue_neg / np.clip(clue_pos, a_min=1.0, a_max=None)
        clue_scores = (clue_vectors * clue_pos_weight[None, :]).sum(axis=1)

        diagnosis_counts = np.bincount(diagnosis_labels, minlength=2).astype(np.float32)
        diagnosis_class_weight = diagnosis_counts.sum() / np.clip(
            diagnosis_counts,
            a_min=1.0,
            a_max=None,
        )
        diagnosis_scores = diagnosis_class_weight[diagnosis_labels]

        sample_weights = 1.0
        sample_weights += clue_weight_scale * clue_scores
        sample_weights += diagnosis_weight_scale * diagnosis_scores

        return torch.as_tensor(sample_weights, dtype=torch.double)


def pretrain_collate_fn(batch):
    out = {}
    tensor_keys = [
        "imgs",
        "clue_masks",
        "clue_present",
        "chaos_labels",
    ]

    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)

    out["image_name"] = [item["image_name"] for item in batch]
    return out


def finetune_collate_fn(batch):
    out = {}
    tensor_keys = [
        "imgs",
        "diagnosis_labels",
        "clue_masks",
        "clue_present",
        "chaos_labels",
    ]

    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)

    out["image_name"] = [item["image_name"] for item in batch]
    return out

