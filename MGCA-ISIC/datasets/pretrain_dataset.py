"""
================================================================================
MGCA Pre-training Dataset: MIMIC-CXR Multimodal Dataset
================================================================================
This module implements the dataset loader for MGCA pre-training using the 
MIMIC-CXR dataset - a large publicly available dataset of chest X-rays with 
associated radiology reports.

MIMIC-CXR Dataset:
- ~377,000 chest X-ray images
- ~227,000 radiology reports
- Covers various pathologies: pneumonia, effusion, cardiomegaly, etc.
- Requires PhysioNet credentialed access

Data Structure:
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MIMIC-CXR Dataset                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Image: Chest X-ray (DICOM/JPG)                                              │
│  └── Views: PA (Posterior-Anterior), AP (Anterior-Posterior)                │
│                                                                              │
│  Report: Radiology Report (Text)                                            │
│  ├── Impression: Summary of findings (most important)                       │
│  └── Findings: Detailed observations                                        │
│                                                                              │
│  Metadata: Patient ID, Study ID, View Position, etc.                        │
└─────────────────────────────────────────────────────────────────────────────┘

Pre-processing Pipeline:
1. Load master CSV with image paths and report text
2. Filter to PA/AP views only (frontal views)
3. Parse reports into sentences (impression + findings)
4. Tokenize sentences using BioClinicalBERT tokenizer
5. Create image-text pairs for contrastive learning

Usage:
    from datasets.pretrain_dataset import MultimodalPretrainingDataset
    from datasets.transforms import DataTransforms
    
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)
    
    # Returns: (image_tensor, token_dict, caption_length, image_path)
    img, caps, cap_len, path = dataset[0]
================================================================================
"""

import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from .utils import get_imgs
from transformers import BertTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MultimodalPretrainingDataset(data.Dataset):
    """
    Multimodal Pre-training Dataset for MGCA.
    
    Loads paired chest X-ray images and radiology reports from MIMIC-CXR
    for contrastive pre-training. Each sample provides:
    - Image tensor (augmented during training)
    - Tokenized report text (BERT format)
    - Caption length (for sorting/batching)
    - Image path (for debugging/visualization)
    
    The dataset implements efficient text processing by:
    1. Caching processed reports to disk (captions.pickle)
    2. Creating path→sentences mapping on first run
    3. Filtering low-quality reports (< 3 words)
    
    Args:
        split (str): Dataset split ("train", "valid", or "test")
        transform: Image augmentation transforms (from DataTransforms)
        data_pct (float): Fraction of data to use (1.0 = all data)
            Useful for experiments with limited data
        imsize (int): Target image size (default: 256)
        max_words (int): Maximum token length for reports (default: 112)
            Reports are truncated/padded to this length
        sent_num (int): Number of sentences to use (legacy, not used)
    
    Attributes:
        filenames (list): List of image paths for this split
        path2sent (dict): Mapping from image path to list of sentences
        tokenizer: BioClinicalBERT tokenizer for text processing
    """
    
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=112, sent_num=3):
        super().__init__()
        
        # =====================================================================
        # STEP 1: VALIDATE DATA DIRECTORY EXISTS
        # =====================================================================
        # MIMIC-CXR requires PhysioNet credentials to access
        # Download from: https://physionet.org/content/mimic-cxr-jpg/2.0.0/
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        self.imsize = imsize
        
        # =====================================================================
        # STEP 2: LOAD AND FILTER METADATA
        # =====================================================================
        # master.csv contains: image paths, report text, metadata
        self.df = pd.read_csv(MIMIC_CXR_MASTER_CSV)
        
        # Filter to frontal views only (PA and AP)
        # PA (Posterior-Anterior): Standard standing X-ray, higher quality
        # AP (Anterior-Posterior): Portable/bedside X-ray, lower quality
        # Excluding: Lateral views (side views, different anatomy visible)
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        
        # Fix image paths to absolute paths
        # Original format: "files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
        # Converted to: "/path/to/MIMIC-CXR/files/p10/..."
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))

        # =====================================================================
        # STEP 3: LOAD/CREATE TEXT DATA
        # =====================================================================
        # This loads or creates a mapping from image path → list of sentences
        # The mapping is cached to disk for efficiency
        self.filenames, self.path2sent = self.load_text_data(split)
        
        # =====================================================================
        # STEP 4: FILTER BY SPLIT AND DATA PERCENTAGE
        # =====================================================================
        self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        
        # Optionally use subset of data (for faster experiments)
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)
        
        # =====================================================================
        # STEP 5: INITIALIZE TOKENIZER
        # =====================================================================
        # BioClinicalBERT tokenizer - specialized for medical text
        # Vocabulary includes medical terms like "pneumothorax", "cardiomegaly"
        self.tokenizer = BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

    def load_text_data(self, split):
        """
        Load or create text data mapping.
        
        Creates a mapping from image paths to processed sentences.
        The mapping is cached to disk as captions.pickle for efficiency.
        
        Args:
            split: Dataset split to load
            
        Returns:
            tuple:
                - filenames (list): Image paths for this split
                - path2sent (dict): Mapping from path to sentence list
                
        Cache file structure (captions.pickle):
        {
            "/path/to/image1.jpg": ["no acute cardiopulmonary process", "heart size normal"],
            "/path/to/image2.jpg": ["mild cardiomegaly", "bilateral pleural effusions"],
            ...
        }
        """
        # Check for cached captions file
        filepath = os.path.join(BASE_DIR, "../../data/captions.pickle")
        
        if not os.path.isfile(filepath):
            # First run - create captions from raw reports
            print(f"Caption file {filepath} does not exist. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            
            # Save to disk for future runs
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            # Load cached captions
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # Filter to only include images that have associated text
        # and belong to current split
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            if cur_split == split and path in path2sent:
                filenames.append(path)

        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        """
        Create mapping from image paths to processed sentences.
        
        Processes raw radiology reports into clean sentences:
        1. Concatenate "impression" and "findings" sections
        2. Split into sentences (by period and numbered lists)
        3. Tokenize and clean each sentence
        4. Filter out very short sentences (≤1 token)
        5. Filter out reports with < 3 total tokens
        
        Returns:
            dict: Mapping from image path to list of sentences
            
        Report Structure:
        ├── Impression: "No acute cardiopulmonary abnormality."
        └── Findings: "Heart size is normal. Lungs are clear. 
                       No pleural effusion or pneumothorax."
                       
        Processing Example:
            Input: "1. Heart size normal. 2. No acute process."
            Output: ["heart size normal", "no acute process"]
        """
        sent_lens, num_sents = [], []  # For statistics
        path2sent = {}
        
        # Process each row in the dataframe
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # Concatenate impression and findings
            # Impression: Summary (most important clinical info)
            # Findings: Detailed observations
            captions = ""
            captions += row["impression"]
            captions += " "
            captions += row["findings"]

            # Replace newlines with spaces for consistent parsing
            captions = captions.replace("\n", " ")

            # =========================================================
            # SENTENCE SPLITTING
            # =========================================================
            # Split on numbered lists (e.g., "1. Finding one. 2. Finding two.")
            # Pattern: digit(s) followed by period
            splitter = re.compile(r"[0-9]+\.")
            captions = splitter.split(captions)
            
            # Further split on regular periods
            captions = [point.split(".") for point in captions]
            
            # Flatten nested lists
            captions = [sent for point in captions for sent in point]

            # =========================================================
            # SENTENCE TOKENIZATION AND CLEANING
            # =========================================================
            cnt = 0  # Total token count
            study_sent = []  # Clean sentences for this study
            
            for cap in captions:
                if len(cap) == 0:
                    continue

                # Remove Unicode replacement characters (from PDF extraction)
                cap = cap.replace("\ufffd\ufffd", " ")
                
                # Tokenize: Extract only alphanumeric sequences
                # This removes punctuation, special characters, etc.
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                
                # Skip very short sentences (likely noise)
                # Note: This filters out useful phrases like "no pneumothorax"
                # TODO: Consider lowering threshold to 1
                if len(tokens) <= 1:
                    continue

                # Filter to ASCII-only characters (remove non-English chars)
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            # Only include studies with sufficient text (>= 3 tokens)
            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row[MIMIC_CXR_PATH_COL]] = study_sent

        # Print statistics for sanity checking
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean():.1f},{sent_lens.max()} "
            f"[{np.percentile(sent_lens, 5):.1f}, {np.percentile(sent_lens, 95):.1f}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean():.1f},{num_sents.max()} "
            f"[{np.percentile(num_sents, 5):.1f}, {np.percentile(num_sents, 95):.1f}]"
        )

        return path2sent

    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.filenames)

    def get_caption(self, path):
        """
        Get tokenized caption for an image.
        
        Retrieves pre-processed sentences and tokenizes them using 
        BioClinicalBERT tokenizer into BERT format.
        
        Args:
            path: Image file path
            
        Returns:
            tuple:
                - tokens (dict): BERT token dictionary with:
                    - input_ids: Token IDs [1, max_words]
                    - token_type_ids: Segment IDs [1, max_words]
                    - attention_mask: Padding mask [1, max_words]
                - x_len (int): Actual caption length (before padding)
                
        Token Format (BERT):
            [CLS] word1 word2 ... wordN [SEP] [PAD] [PAD] ...
            
        Example:
            Input sentences: ["heart size normal", "no pneumothorax"]
            Combined: "heart size normal no pneumothorax"
            Tokens: [101, 2550, 2946, 3671, 2053, 16257, 102, 0, 0, ...]
                    [CLS] heart size normal  no  pneumo [SEP][PAD]...
        """
        # Get pre-processed sentences for this image
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # Remove empty strings and combine sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        # Tokenize with BERT tokenizer
        tokens = self.tokenizer(
            sent,
            return_tensors="pt",    # Return PyTorch tensors
            truncation=True,        # Truncate to max_length
            padding="max_length",   # Pad to max_length
            max_length=self.max_words,
        )
        
        # Calculate actual length (non-padding tokens)
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            tuple:
                - imgs: Image tensor [3, H, W] (transformed)
                - caps: Token dictionary for BERT
                - cap_len: Caption length
                - key: Image path (for debugging)
        """
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        return imgs, caps, cap_len, key


def multimodal_collate_fn(batch):
    """
    Custom collate function for multimodal batches.
    
    Collates individual samples into batched tensors and sorts by caption
    length in descending order. Sorting is useful for efficient RNN processing
    (though BERT doesn't require it, it's kept for compatibility).
    
    Args:
        batch: List of tuples (img, caps, cap_len, path)
        
    Returns:
        dict: Batched data with keys:
            - "caption_ids": Token IDs [B, max_words]
            - "token_type_ids": Segment IDs [B, max_words]
            - "attention_mask": Padding masks [B, max_words]
            - "imgs": Image tensors [B, 3, H, W]
            - "cap_lens": Caption lengths [B]
            - "path": Image paths [B]
            
    Sorting by Length:
        Original: [(img1, cap_len=50), (img2, cap_len=112), (img3, cap_len=30)]
        Sorted:   [(img2, cap_len=112), (img1, cap_len=50), (img3, cap_len=30)]
        
        This allows efficient packing for sequence models.
    """
    imgs, cap_len, ids, tokens, attention = [], [], [], [], []
    path = []
    
    # Unpack batch items
    for b in batch:
        img, cap, cap_l, p = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)

    # Stack tensors
    imgs = torch.stack(imgs)              # [B, 3, H, W]
    ids = torch.stack(ids).squeeze()      # [B, max_words]
    tokens = torch.stack(tokens).squeeze()    # [B, max_words]
    attention = torch.stack(attention).squeeze()  # [B, max_words]

    # Sort by caption length (descending)
    sorted_cap_lens, sorted_cap_indices = torch.sort(
        torch.tensor(cap_len), 0, True)  # True = descending

    # Convert paths to numpy array for indexing
    path = np.array(path)

    # Create return dictionary with sorted samples
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices]
    }
    return return_dict


if __name__ == "__main__":
    # Test the dataset - run from package root
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from datasets.transforms import DataTransforms
    
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)
    data = dataset[0]
    print(data)
