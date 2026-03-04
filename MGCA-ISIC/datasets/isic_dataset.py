"""
================================================================================
ISIC-2019 Dataset for MGCA Pre-training
================================================================================
This module implements a dataset loader for the ISIC-2019 Skin Cancer dataset
to work with the MGCA framework.

Since ISIC doesn't have natural language reports like radiology data, we 
GENERATE text descriptions from the structured annotations, including:
- Diagnosis (MEL, NV, BCC, etc.)
- Dermoscopic clues (grey blue structures, radial lines, etc.)
- Patient metadata (age, sex, anatomical site)
- Structural/color chaos indicators

Example Generated Text:
"This skin lesion is a melanoma located on the lower extremity of a 55 year old 
female patient. The lesion shows chaotic structure and chaotic coloring. 
Dermoscopic examination reveals grey blue structures, lines radial pseudopods, 
and white lines."

This approach allows MGCA to learn meaningful visual-semantic correspondences
between dermoscopic image features and their textual descriptions.
================================================================================
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from transformers import BertTokenizer

# Import shared constants with fallback for standalone vs integrated usage

from .constants import (
    DIAGNOSIS_NAMES,
    SITE_NAMES,
    CHAOS_LABELS,
    CLUE_LABELS,
    CLUE_DESCRIPTIONS,
)


class ISICPretrainingDataset(data.Dataset):
    """
    ISIC-2019 Dataset for MGCA Pre-training.
    
    Generates natural language descriptions from structured annotations
    to create image-text pairs for contrastive learning.
    
    Args:
        csv_path (str): Path to the annotated CSV file
        img_dir (str): Path to the image directory
        split (str): Dataset split ("train", "valid", or "test")
        transform: Image augmentation transforms
        data_pct (float): Fraction of data to use (0.0 to 1.0)
        max_words (int): Maximum token length for text
        seed (int): Random seed for reproducibility
        train_ratio (float): Ratio of training data (default: 0.8)
        val_ratio (float): Ratio of validation data (default: 0.1)
    """
    
    def __init__(self, 
                 csv_path: str = None,
                 img_dir: str = None,
                 split: str = "train", 
                 transform=None, 
                 data_pct: float = 1.0,
                 max_words: int = 112,
                 seed: int = 42,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1):
        super().__init__()
        
        self.transform = transform
        self.max_words = max_words
        self.split = split
        
        # Set default paths if not provided
        if csv_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, "../../data/ISIC-2019/ISIC_2019_annotated_combined.csv")
        if img_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            img_dir = os.path.join(base_dir, "../../data/ISIC-2019/Images")
            
        self.img_dir = img_dir
        
        # Load and preprocess data
        print(f"Loading ISIC dataset from {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # Create train/val/test splits
        np.random.seed(seed)
        n_samples = len(self.df)
        indices = np.random.permutation(n_samples)
        
        train_end = int(train_ratio * n_samples)
        val_end = int((train_ratio + val_ratio) * n_samples)
        
        if split == "train":
            split_indices = indices[:train_end]
        elif split == "valid":
            split_indices = indices[train_end:val_end]
        else:  # test
            split_indices = indices[val_end:]
            
        self.df = self.df.iloc[split_indices].reset_index(drop=True)
        
        # Apply data percentage (for few-shot experiments)
        if data_pct != 1.0 and split == "train":
            n_use = int(len(self.df) * data_pct)
            self.df = self.df.sample(n=n_use, random_state=seed).reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} samples for {split} split")
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
    
    def generate_description(self, row):
        """
        Generate natural language description from structured annotations.
        
        Args:
            row: DataFrame row with annotations
            
        Returns:
            str: Generated text description
        """
        parts = []
        
        # Diagnosis
        diagnosis = row.get('diagnosis')
        diagnosis_name = DIAGNOSIS_NAMES.get(diagnosis)
        
        # Basic description
        if pd.notna(row.get('anatom_site_general')) and row['anatom_site_general']:
            site = SITE_NAMES.get(row['anatom_site_general'], row['anatom_site_general'])
            parts.append(f"This skin lesion is a {diagnosis_name} located on {site}")
        else:
            parts.append(f"This skin lesion is a {diagnosis_name}")
        
        # Patient info
        patient_info = []
        if pd.notna(row.get('age_approx')) and row['age_approx']:
            patient_info.append(f"{int(row['age_approx'])} year old")
        if pd.notna(row.get('sex')) and row['sex']:
            patient_info.append(row['sex'])
        
        if patient_info:
            parts[-1] += f" of a {' '.join(patient_info)} patient"
        
        parts[-1] += "."
        
        # Structure and color chaos
        chaos_parts = []
        if row.get('structure_is_chaotic') == True:
            chaos_parts.append("chaotic structure")
        if row.get('colour_is_chaotic') == True:
            chaos_parts.append("chaotic coloring")
        
        if chaos_parts:
            parts.append(f"The lesion shows {' and '.join(chaos_parts)}.")
        
        # Dermoscopic clues
        clues = []
        for col, description in CLUE_DESCRIPTIONS.items():
            if col in row and row.get(col) == True:
                if col != 'clue_10_no_clues':  # Don't add "no clues" as a positive finding
                    clues.append(description)
        
        if clues:
            if len(clues) == 1:
                parts.append(f"Dermoscopic examination reveals {clues[0]}.")
            elif len(clues) == 2:
                parts.append(f"Dermoscopic examination reveals {clues[0]} and {clues[1]}.")
            else:
                clue_str = ", ".join(clues[:-1]) + f", and {clues[-1]}"
                parts.append(f"Dermoscopic examination reveals {clue_str}.")
        else:
            # No specific clues
            if diagnosis == 'NV':
                parts.append("The nevus appears benign with no concerning dermoscopic features.")
            elif diagnosis == 'MEL':
                parts.append("The melanoma requires further clinical evaluation.")
            else:
                parts.append("No specific dermoscopic clues are observed.")
        
        return " ".join(parts)
    
    def __len__(self):
        return len(self.df)
    
    def get_image(self, img_name):
        """Load and transform image."""
        # Handle different file naming conventions
        if not img_name.endswith('.jpg'):
            img_name = img_name + '.jpg'
            
        img_path = os.path.join(self.img_dir, img_name)
        
        # Try alternate path without _downsampled suffix
        if not os.path.exists(img_path):
            alt_name = img_name.replace('_downsampled', '')
            img_path = os.path.join(self.img_dir, alt_name)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
    def get_caption(self, row):
        """Generate and tokenize caption."""
        text = self.generate_description(row)
        
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        
        # Calculate actual length (non-padding tokens)
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])
        
        return tokens, x_len, text
    
    def get_labels(self, row):
        """Extract chaos and clues labels from row."""
        # Chaos labels (2 binary)
        chaos_labels = torch.tensor([
            1.0 if row.get(col) == True else 0.0
            for col in CHAOS_LABELS
        ], dtype=torch.float32)
        
        # Clues labels (10 binary)
        clues_labels = torch.tensor([
            1.0 if row.get(col) == True else 0.0
            for col in CLUE_LABELS
        ], dtype=torch.float32)
        
        return chaos_labels, clues_labels
    
    def __getitem__(self, index):
        """Get a single sample."""
        row = self.df.iloc[index]
        
        # Get image
        img_name = row['image_name']
        img = self.get_image(img_name)
        
        # Get caption
        caps, cap_len, text = self.get_caption(row)
        
        # Get classification labels
        chaos_labels, clues_labels = self.get_labels(row)
        
        return img, caps, cap_len, img_name, chaos_labels, clues_labels


def isic_collate_fn(batch):
    """
    Custom collate function for ISIC batches.
    Includes chaos and clues labels for MGCA-ISIC.
    """
    imgs, cap_len, ids, tokens, attention = [], [], [], [], []
    path = []
    chaos_labels_list = []
    clues_labels_list = []
    
    for b in batch:
        img, cap, cap_l, p, chaos_labels, clues_labels = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        chaos_labels_list.append(chaos_labels)
        clues_labels_list.append(clues_labels)

    # Stack tensors
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    chaos_labels = torch.stack(chaos_labels_list)
    clues_labels = torch.stack(clues_labels_list)

    # Sort by caption length (descending)
    sorted_cap_lens, sorted_cap_indices = torch.sort(
        torch.tensor(cap_len), 0, True)

    path = np.array(path)

    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices],
        "chaos_labels": chaos_labels[sorted_cap_indices],
        "clues_labels": clues_labels[sorted_cap_indices]
    }
    return return_dict


if __name__ == "__main__":
    # Test the dataset
    from .transforms import DataTransforms
    
    # Update these paths to your actual data location
    csv_path = "/Users/chrispham/Documents/Data Science/Skin Cancer/Skin Cancer Detection/ISIC-2019/ISIC_2019_annotated_combined.csv"
    img_dir = "/Users/chrispham/Documents/Data Science/Skin Cancer/Skin Cancer Detection/ISIC-2019/Images"
    
    transform = DataTransforms(is_train=True, crop_size=224)
    
    dataset = ISICPretrainingDataset(
        csv_path=csv_path,
        img_dir=img_dir,
        split="train",
        transform=transform
    )
    
    # Test a few samples
    print("\n" + "="*60)
    print("Sample Generated Descriptions:")
    print("="*60)
    
    for i in range(min(5, len(dataset))):
        row = dataset.df.iloc[i]
        text = dataset.generate_description(row)
        print(f"\n[{i}] Image: {row['image_name']}")
        print(f"    Diagnosis: {row['diagnosis']}")
        print(f"    Generated Text: {text}")
    
    # Test __getitem__
    print("\n" + "="*60)
    print("Testing DataLoader:")
    print("="*60)
    
    img, caps, cap_len, path, chaos_labels, clues_labels = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Caption IDs shape: {caps['input_ids'].shape}")
    print(f"Caption length: {cap_len}")
    print(f"Chaos labels: {chaos_labels} (shape: {chaos_labels.shape})")
    print(f"Clues labels: {clues_labels} (shape: {clues_labels.shape})")