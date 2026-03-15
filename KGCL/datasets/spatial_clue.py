import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


CONCEPTS = [
    "Structure Chaotic",
    "Colour Chaotic",
    "Eccentric Structureless Area",
    "Thick Lines (Reticular or Branched)",
    "Grey/Blue Structures",
    "Black Dots/Clods, Peripheral",
    "Lines Radial or Pseudopods, Segmental",
    "White Lines",
    "Polymorphous Vessels",
    "Parallel/Ridge Lines or Chaotic Nail Lines",
    "Angulated Lines",
]

CONCEPT_TEXT_PATTERNS = {
    "Structure Chaotic": [
        "chaotic structure",
    ],
    "Colour Chaotic": [
        "chaotic coloring",
    ],
    "Eccentric Structureless Area": [
        "eccentric structureless area",
    ],
    "Thick Lines (Reticular or Branched)": [
        "thick lines (reticular or branched)",
        "thick lines",
    ],
    "Grey/Blue Structures": [
        "grey/blue structures",
    ],
    "Black Dots/Clods, Peripheral": [
        "black dots/clods, peripheral",
        "black dots/clods peripheral",
    ],
    "Lines Radial or Pseudopods, Segmental": [
        "lines radial or pseudopods, segmental",
    ],
    "White Lines": [
        "white lines",
    ],
    "Polymorphous Vessels": [
        "polymorphous vessels",
    ],
    "Parallel/Ridge Lines or Chaotic Nail Lines": [
        "parallel/ridge lines or chaotic nail lines",
    ],
    "Angulated Lines": [
        "angulated lines",
    ],
}


def find_subsequence(sequence, subsequence):
    n = len(sequence)
    m = len(subsequence)
    if m == 0 or m > n:
        return None

    for i in range(n - m + 1):
        if sequence[i:i + m] == subsequence:
            return i, i + m
    return None


class ISICSpatialClueDataset(Dataset):
    def __init__(
        self,
        csv_path,
        img_dir,
        mask_dir,
        vector_dir,
        split="train",
        transform=None,
        tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=128,
        data_pct=1.0,
    ):
        self.df = pd.read_csv(csv_path)

        if "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if split == "train" and data_pct < 1.0:
            self.df = self.df.sample(frac=data_pct, random_state=42).reset_index(drop=True)

        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.vector_dir = Path(vector_dir)
        self.transform = transform
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.df)

    def build_clue_token_masks(self, description, clue_present):
        encoded = self.tokenizer(
            description,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        caption_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        if "token_type_ids" in encoded:
            token_type_ids = encoded["token_type_ids"].squeeze(0)
        else:
            token_type_ids = torch.zeros_like(caption_ids)

        token_ids_list = caption_ids.tolist()
        clue_token_masks = torch.zeros(len(CONCEPTS), self.max_length, dtype=torch.float32)

        for concept_idx, concept_name in enumerate(CONCEPTS):
            if clue_present[concept_idx] <= 0:
                continue

            candidate_phrases = CONCEPT_TEXT_PATTERNS.get(concept_name, [])
            for phrase in candidate_phrases:
                phrase_ids = self.tokenizer(
                    phrase,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]

                match = find_subsequence(token_ids_list, phrase_ids)
                if match is not None:
                    start, end = match
                    clue_token_masks[concept_idx, start:end] = 1.0
                    break

        return caption_ids, attention_mask, token_type_ids, clue_token_masks

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # image path
        image_name = str(row["image"])
        img_path = self.img_dir / f"{image_name}.jpg"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # diagnosis
        diagnosis = 1 if str(row["diagnosis"]).upper() == "MEL" else 0

        # masks and vectors
        mask_path = self.mask_dir / str(row["mask_name"])
        vector_path = self.vector_dir / str(row["vector_name"])

        clue_masks = np.load(mask_path).astype(np.float32)      # [11,H,W]
        clue_present = np.load(vector_path).astype(np.float32)  # [11]

        clue_masks = (clue_masks > 0).astype(np.float32)
        clue_present = (clue_present > 0).astype(np.float32)

        # union mask for segmentation branch
        seg_mask = (clue_masks.sum(axis=0) > 0).astype(np.float32)  # [H,W]

        # transforms must apply same spatial ops to image + masks
        if self.transform is not None:
            image, seg_mask, clue_masks = self.transform(image, seg_mask, clue_masks)

        # to tensor if transform didn't already do it
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if not isinstance(seg_mask, torch.Tensor):
            seg_mask = torch.tensor(seg_mask, dtype=torch.float32)

        if not isinstance(clue_masks, torch.Tensor):
            clue_masks = torch.tensor(clue_masks, dtype=torch.float32)

        if seg_mask.dim() == 2:
            seg_mask = seg_mask.unsqueeze(0)  # [1,H,W]

        description = str(row["description"])

        caption_ids, attention_mask, token_type_ids, clue_token_masks = \
            self.build_clue_token_masks(description, clue_present)

        sample = {
            "imgs": image,  # [3,H,W]
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "diagnosis_labels": torch.tensor(diagnosis, dtype=torch.long),
            "seg_masks": seg_mask.float(),                         # [1,H,W]
            "clue_masks": clue_masks.float(),                     # [11,H,W]
            "clue_token_masks": clue_token_masks.float(),         # [11,L]
            "clue_present": torch.tensor(clue_present, dtype=torch.float32),  # [11]
            "description": description,
            "image_name": image_name,
        }
        return sample


def isic_spatial_clue_collate_fn(batch):
    out = {}

    tensor_keys = [
        "imgs",
        "caption_ids",
        "attention_mask",
        "token_type_ids",
        "diagnosis_labels",
        "seg_masks",
        "clue_masks",
        "clue_token_masks",
        "clue_present",
    ]

    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)

    out["description"] = [item["description"] for item in batch]
    out["image_name"] = [item["image_name"] for item in batch]

    return out