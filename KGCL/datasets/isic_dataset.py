"""
================================================================================
ISIC-2019 Dataset
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

import ast
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from transformers import BertTokenizer

# Import shared constants
from .constants import (
    DIAGNOSIS_NAMES,
    SITE_NAMES,
    DIAGNOSIS_TO_IDX,
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
            csv_path = os.path.join(base_dir, "../../Dataset/annotated_combined.csv_with_descriptions.csv")
        if img_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            img_dir = os.path.join(base_dir, "../../Dataset/Images")
            
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
    
    def parse_clues(self, clues_str):
        """
        Parse clues from string representation of list.
        
        Args:
            clues_str: String like "['Grey/Blue Structures', 'White Lines']"
            
        Returns:
            list: List of clue strings
        """
        if pd.isna(clues_str) or not clues_str:
            return []
        
        try:
            # Parse the string representation of list
            clues = ast.literal_eval(clues_str)
            # Filter out "No Clues" entries
            return [c for c in clues if c != 'No Clues']
        except (ValueError, SyntaxError):
            return []
    
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
        text = row.get('lesion_description', '')
        
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
        """Extract diagnosis label (binary: NV=0, MEL=1)."""
        diagnosis = row.get('diagnosis', 'NV')
        
        # Binary label: 0 for NV, 1 for MEL
        diagnosis_label = torch.tensor(
            DIAGNOSIS_TO_IDX.get(diagnosis, 0),
            dtype=torch.long
        )
        
        return diagnosis_label
    
    def __getitem__(self, index):
        """Get a single sample."""
        row = self.df.iloc[index]
        
        # Get image
        img_name = row['image_name']
        img = self.get_image(img_name)
        
        # Get caption
        caps, cap_len, text = self.get_caption(row)
        
        # Get diagnosis label
        diagnosis_label = self.get_labels(row)
        
        return img, caps, cap_len, img_name, diagnosis_label


def isic_collate_fn(batch):
    """
    Custom collate function for ISIC batches.
    Includes diagnosis labels for binary classification (NV vs MEL).
    """
    imgs, cap_len, ids, tokens, attention = [], [], [], [], []
    path = []
    diagnosis_labels_list = []
    
    for b in batch:
        img, cap, cap_l, p, diagnosis_label = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        diagnosis_labels_list.append(diagnosis_label)

    # Stack tensors
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze(1)
    tokens = torch.stack(tokens).squeeze(1)
    attention = torch.stack(attention).squeeze(1)
    diagnosis_labels = torch.stack(diagnosis_labels_list)

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
        "diagnosis_labels": diagnosis_labels[sorted_cap_indices],
    }
    return return_dict


if __name__ == "__main__":
    # Test the dataset
    from .transforms import DataTransforms
    
    # Update these paths to your actual data location
    csv_path = "../../Dataset/annotated_combined.csv"
    img_dir = "../../Dataset/Images"
    
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
    
    img, caps, cap_len, path, diagnosis_label = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Caption IDs shape: {caps['input_ids'].shape}")
    print(f"Caption length: {cap_len}")
    print(f"Diagnosis label: {diagnosis_label} (0=NV, 1=MEL)")