"""
================================================================================
MGCA Encoders: Image and Text Encoder Implementations
================================================================================
This module implements the encoder backbones for the MGCA framework:

1. IMAGE ENCODER (ImageEncoder)
   - Supports ResNet-50 and Vision Transformer (ViT) backbones
   - Extracts both GLOBAL features (image-level) and LOCAL features (patch-level)
   - Global features: Used for instance-wise alignment (ITA)
   - Local features: Used for token-wise alignment (CTA)

2. TEXT ENCODER (BertEncoder)  
   - Uses BioClinicalBERT (specialized for medical text)
   - Extracts both GLOBAL features (report-level) and LOCAL features (word-level)
   - Aggregates subword tokens back to word-level features
   - Provides attention weights for importance weighting

Projection Heads:
- GlobalEmbedding: Projects global features to joint embedding space
- LocalEmbedding: Projects local features to joint embedding space
  (Uses 1D convolution to process sequences efficiently)

Architecture:
┌─────────────┐     ┌─────────────────┐
│   Image     │     │ Medical Report  │
│ [224x224]   │     │   [Tokens]      │
└──────┬──────┘     └───────┬─────────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────────┐
│ CNN/ViT      │     │ BioClinicalBERT  │
│ Backbone     │     │ Backbone         │
└──────┬───────┘     └───────┬──────────┘
       │                     │
       ├── Global Pool ──┐   ├── [CLS] Token ──┐
       │                 │   │                  │
       │                 ▼   │                  ▼
       │          ┌───────────┐          ┌───────────┐
       │          │GlobalEmbed│          │GlobalEmbed│
       │          │  (MLP)    │          │  (MLP)    │
       │          └─────┬─────┘          └─────┬─────┘
       │                │                      │
       │                └──── Global Feats ────┘  → ITA Loss
       │                                  
       ├── Patch/Grid ───┐   ├── Word Tokens ──┐
       │                 │   │                  │
       │                 ▼   │                  ▼
       │          ┌───────────┐          ┌───────────┐
       │          │LocalEmbed │          │LocalEmbed │
       │          │ (Conv1D)  │          │ (Conv1D)  │
       │          └─────┬─────┘          └─────┬─────┘
       │                │                      │
       └────────────────┴──── Local Feats ─────┘  → CTA Loss
================================================================================
"""

import os

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoTokenizer, BertConfig, BertTokenizer, logging

# Relative imports for standalone package
from . import cnn_backbones
from .med import BertModel
from .vits import create_vit

# Suppress transformer warnings for cleaner output
logging.set_verbosity_error()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class GlobalEmbedding(nn.Module):
    """
    Global Embedding Projection Head.
    
    Projects global features (image [CLS] or report [CLS]) to the joint 
    embedding space for instance-wise contrastive learning.
    
    Architecture:
        Linear → BatchNorm → ReLU → Linear → BatchNorm (no affine)
    
    The final BatchNorm without affine parameters ensures the output
    has zero mean and unit variance, which is beneficial for contrastive
    learning with L2-normalized features.
    
    Args:
        input_dim: Input feature dimension (from backbone)
        hidden_dim: Hidden layer dimension (2048 default, following SimCLR)
        output_dim: Output embedding dimension (128 default)
    """
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            # First projection layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            # Second projection layer (output)
            nn.Linear(hidden_dim, output_dim),
            # Final BatchNorm without learnable affine parameters
            # This normalizes outputs to unit variance
            nn.BatchNorm1d(output_dim, affine=False)
        )

    def forward(self, x):
        """
        Args:
            x: Global features [batch_size, input_dim]
            
        Returns:
            Projected embeddings [batch_size, output_dim]
        """
        return self.head(x)


class LocalEmbedding(nn.Module):
    """
    Local Embedding Projection Head.
    
    Projects local features (image patches or text tokens) to the joint
    embedding space for token-wise contrastive learning.
    
    Uses 1D convolution (kernel_size=1) instead of Linear layers to 
    efficiently process sequences while maintaining the sequence dimension.
    This is equivalent to applying the same Linear layer to each position.
    
    Architecture:
        Conv1D → BatchNorm1D → ReLU → Conv1D → BatchNorm1D (no affine)
    
    Args:
        input_dim: Input feature dimension (from backbone)
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
    """
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            # 1D convolution acts as position-wise linear transformation
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)
        )

    def forward(self, x):
        """
        Args:
            x: Local features [batch_size, seq_len, input_dim]
            
        Returns:
            Projected embeddings [batch_size, seq_len, output_dim]
        """
        # Conv1d expects [B, C, L] format, so we permute
        x = x.permute(0, 2, 1)  # [B, input_dim, seq_len]
        x = self.head(x)        # [B, output_dim, seq_len]
        return x.permute(0, 2, 1)  # [B, seq_len, output_dim]


class ImageEncoder(nn.Module):
    """
    Image Encoder for MGCA.
    
    Extracts visual features at multiple granularities:
    - GLOBAL features: Single vector representing entire image
    - LOCAL features: Grid of patch/region features
    
    Supports two backbone architectures:
    
    1. ResNet-50 (model_name="resnet_50"):
       - Global: Average pooled features from final layer [B, 2048]
       - Local: Feature map from layer3 [B, 1024, H, W] → [B, H*W, 1024]
       
    2. Vision Transformer (model_name="vit_base"):
       - Global: [CLS] token embedding [B, 768]
       - Local: Patch token embeddings [B, 196, 768] (14x14 patches)
       
    Args:
        model_name: Backbone architecture ("resnet_50" or "vit_base")
        text_feat_dim: Not used (legacy parameter)
        output_dim: Output embedding dimension after projection
        hidden_dim: Hidden dimension in projection heads
        pretrained: Whether to use pretrained weights
    """
    def __init__(self,
                 model_name: str = "resnet_50",
                 text_feat_dim: int = 768,
                 output_dim: int = 768,
                 hidden_dim: int = 2048,
                 pretrained: bool = True
                 ):
        super(ImageEncoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.text_feat_dim = text_feat_dim

        if "vit" in model_name:
            # =========================================================
            # VISION TRANSFORMER BACKBONE
            # =========================================================
            # ViT divides image into patches and processes them with
            # a transformer encoder. Returns:
            # - [CLS] token: aggregated global representation
            # - Patch tokens: local region representations
            # =========================================================
            
            vit_grad_ckpt = False  # Gradient checkpointing (saves memory)
            vit_ckpt_layer = 0     # Checkpoint layer
            image_size = 224       # Standard ViT input size

            # Extract ViT variant name (e.g., "vit_base" → "base")
            vit_name = model_name[4:]
            self.model, vision_width = create_vit(
                vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

            self.feature_dim = vision_width  # 768 for ViT-Base

            # Load pretrained DeiT weights (Data-efficient Image Transformer)
            # DeiT is ViT trained with improved training recipe
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            msg = self.model.load_state_dict(state_dict, strict=False)

            # Global embedding projection: [CLS] token → joint space
            self.global_embed = GlobalEmbedding(
                vision_width, hidden_dim, output_dim
            )

            # Local embedding projection: Patch tokens → joint space
            self.local_embed = LocalEmbedding(
                vision_width, hidden_dim, output_dim
            )

        else:
            # =========================================================
            # CNN (RESNET) BACKBONE
            # =========================================================
            # ResNet extracts hierarchical features through conv layers.
            # We use:
            # - layer4 output (pooled): Global representation
            # - layer3 output: Local/regional features (before final pooling)
            # =========================================================
            
            # Dynamically get the CNN backbone function
            model_function = getattr(cnn_backbones, model_name)
            self.model, self.feature_dim, self.interm_feature_dim = model_function(
                pretrained=pretrained
            )
            # feature_dim: Final layer channels (2048 for ResNet-50)
            # interm_feature_dim: Intermediate layer channels (1024)

            # Global average pooling for global features
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            # Global embedding: pooled features → joint space
            self.global_embed = GlobalEmbedding(
                self.feature_dim, hidden_dim, output_dim
            )

            # Local embedding: intermediate features → joint space
            self.local_embed = LocalEmbedding(
                self.interm_feature_dim, hidden_dim, output_dim
            )

    def resnet_forward(self, x, get_local=True):
        """
        Forward pass through ResNet backbone.
        
        Args:
            x: Input images [batch_size, 3, H, W]
            get_local: Whether to return local features (always True in MGCA)
            
        Returns:
            tuple:
                - global_features: [batch_size, 2048] - pooled final layer
                - local_features: [batch_size, H'*W', 1024] - flattened layer3
        
        ResNet Architecture:
            Input: [B, 3, 224, 224]
            ↓ conv1 + bn + relu + maxpool
            ↓ layer1: [B, 256, 75, 75]
            ↓ layer2: [B, 512, 38, 38]  
            ↓ layer3: [B, 1024, 19, 19] ← LOCAL FEATURES (361 regions)
            ↓ layer4: [B, 2048, 10, 10]
            ↓ avgpool: [B, 2048, 1, 1]
            ↓ flatten: [B, 2048] ← GLOBAL FEATURES
        """
        # Upsample to 299x299 (originally for Inception compatibility)
        x = nn.Upsample(size=(299, 299), mode="bilinear",
                        align_corners=True)(x)
        
        # Early layers
        x = self.model.conv1(x)    # [B, 64, 150, 150]
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # Residual blocks
        x = self.model.layer1(x)   # [B, 256, 75, 75]
        x = self.model.layer2(x)   # [B, 512, 38, 38]
        x = self.model.layer3(x)   # [B, 1024, 19, 19]
        
        # Save intermediate features for local alignment
        local_features = x
        
        x = self.model.layer4(x)   # [B, 2048, 10, 10]

        # Global average pooling
        x = self.pool(x)           # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 2048]

        # Reshape local features: [B, C, H, W] → [B, H*W, C]
        # This creates a sequence of 361 region features
        local_features = rearrange(local_features, "b c w h -> b (w h) c")

        return x, local_features.contiguous()

    def vit_forward(self, x):
        """
        Forward pass through Vision Transformer backbone.
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            
        Returns:
            All tokens [batch_size, 197, 768]
            - First token (index 0): [CLS] token (global)
            - Remaining tokens: Patch tokens (local)
            
        ViT Architecture:
            Input: [B, 3, 224, 224]
            ↓ Patch embedding (16x16 patches): [B, 196, 768]
            ↓ Prepend [CLS] token: [B, 197, 768]
            ↓ Add position embeddings
            ↓ Transformer blocks × 12
            Output: [B, 197, 768]
        """
        # register_blk=11 saves attention map from block 11 (last block)
        # for visualization and importance weighting
        return self.model(x, register_blk=11)

    def forward(self, x, get_local=False):
        """
        Main forward pass.
        
        Args:
            x: Input images [batch_size, 3, H, W]
            get_local: Legacy parameter (always returns local features in MGCA)
            
        Returns:
            tuple:
                - global_features: [batch_size, feature_dim]
                - local_features: [batch_size, num_regions, feature_dim]
        """
        if "resnet" in self.model_name:
            return self.resnet_forward(x, get_local=get_local)
        elif "vit" in self.model_name:
            img_feat = self.vit_forward(x)
            # Split into [CLS] (global) and patch tokens (local)
            return img_feat[:, 0].contiguous(), img_feat[:, 1:].contiguous()


class BertEncoder(nn.Module):
    """
    Text Encoder for MGCA using BioClinicalBERT.
    
    BioClinicalBERT is a domain-specific BERT model pretrained on:
    - PubMed abstracts (biomedical literature)
    - MIMIC-III clinical notes (medical records)
    
    This makes it particularly suited for understanding medical terminology
    and clinical language in radiology reports.
    
    Features:
    - GLOBAL feature: [CLS] token embedding (report-level)
    - LOCAL features: Word embeddings (word-level)
    - Token aggregation: Combines subword tokens back to words
    - Attention weights: For importance weighting in CTA loss
    
    Args:
        tokenizer: Optional custom tokenizer (uses default if None)
        emb_dim: BERT embedding dimension (768)
        output_dim: Output embedding dimension after projection
        hidden_dim: Hidden dimension in projection heads
        freeze_bert: Whether to freeze BERT weights during training
    """
    def __init__(self,
                 tokenizer: BertTokenizer = None,
                 emb_dim: int = 768,
                 output_dim: int = 128,
                 hidden_dim: int = 2048,
                 freeze_bert: bool = True):
        super(BertEncoder, self).__init__()
        
        # Model configuration
        self.bert_type = "emilyalsentzer/Bio_ClinicalBERT"
        self.last_n_layers = 1           # Use only last BERT layer
        self.aggregate_method = "sum"    # How to combine subword tokens
        self.embedding_dim = emb_dim     # BERT hidden size
        self.output_dim = output_dim     # Final embedding size
        self.freeze_bert = freeze_bert   # Whether to freeze BERT
        self.agg_tokens = True           # Aggregate subword tokens to words

        # Load BERT configuration from local file
        self.config = BertConfig.from_json_file(
            os.path.join(BASE_DIR, "../../configs/bert_config.json"))
        
        # Load pretrained BioClinicalBERT
        # Note: We don't use the pooling layer since we extract features manually
        self.model = BertModel.from_pretrained(
            self.bert_type,
            config=self.config,
            add_pooling_layer=False,  # We'll use [CLS] token directly
        )

        # Initialize tokenizer
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)

        # Create reverse vocabulary mapping for decoding
        # idx → word (used for aggregating subword tokens)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        # Optionally freeze BERT parameters
        # This can help with limited data or to preserve pretrained knowledge
        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

        # Projection heads
        # Global: [CLS] token → joint embedding space
        self.global_embed = GlobalEmbedding(
            self.embedding_dim, hidden_dim, self.output_dim)
        
        # Local: Word tokens → joint embedding space
        self.local_embed = LocalEmbedding(
            self.embedding_dim, hidden_dim, self.output_dim)

    def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
        """
        Aggregate subword tokens back to word-level representations.
        
        BERT uses WordPiece tokenization which splits rare words into subwords.
        For example: "pneumothorax" → ["pne", "##um", "##oth", "##or", "##ax"]
        
        This function combines these subword tokens back to word-level:
        - Subwords starting with "##" are merged with previous token
        - Embeddings are summed (not averaged) for merged tokens
        - Attention weights are also summed
        
        Args:
            embeddings: BERT outputs [B, 1, seq_len, hidden_dim]
            caption_ids: Token IDs [B, seq_len]
            last_layer_attn: Attention weights from last layer [B, seq_len-1]
            
        Returns:
            tuple:
                - aggregated_embeddings: [B, 1, seq_len, hidden_dim]
                - sentences: List of decoded words
                - attention_weights: Aggregated attention [B, seq_len]
                
        Example:
            Input tokens: ["[CLS]", "the", "patient", "has", "pne", "##um", "##onia"]
            Output words: ["[CLS]", "the", "patient", "has", "pneumonia"]
        """
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)  # [B, seq_len, layers, dim]
        
        agg_embs_batch = []
        sentences = []
        last_attns = []
        raw_to_agg_batch = []

        # Process each sample in batch
        for embs, caption_id, last_attn in zip(embeddings, caption_ids, last_layer_attn):
            agg_embs = []       # Aggregated embeddings
            token_bank = []     # Buffer for subword tokens
            words = []          # Decoded words
            word_bank = []      # Buffer for subword strings
            attns = []          # Aggregated attention
            attn_bank = []      # Buffer for subword attention
            token_positions = []
            raw_to_agg = torch.full((caption_id.size(0),), -1, dtype=torch.long, device=caption_id.device)

            # Process each token
            for pos, (word_emb, word_id) in enumerate(zip(embs, caption_id)):
                word = self.idxtoword[word_id.item()]
                attn = last_attn[pos - 1] if pos > 0 and (pos - 1) < last_attn.size(0) else last_attn.new_zeros(())

                if word == "[CLS]":
                    agg_embs.append(word_emb)
                    words.append(word)
                    attns.append(attn)
                    raw_to_agg[pos] = len(agg_embs) - 1
                    continue
                
                if word == "[SEP]":
                    # End of sentence - flush buffer and add [SEP]
                    if token_bank:
                        new_emb = torch.stack(token_bank).sum(axis=0)  # Sum subword embeddings
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))
                        attns.append(torch.stack(attn_bank).sum())
                        agg_idx = len(agg_embs) - 1
                        for token_pos in token_positions:
                            raw_to_agg[token_pos] = agg_idx
                    
                    # Add [SEP] token
                    agg_embs.append(word_emb)
                    words.append(word)
                    attns.append(attn)
                    raw_to_agg[pos] = len(agg_embs) - 1
                    break
                    
                # Check if this is a subword (starts with "##")
                if not word.startswith("##"):
                    # New word - flush previous buffer if exists
                    if len(word_bank) == 0:
                        # First token, start buffer
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(attn)
                        token_positions = [pos]
                    else:
                        # Flush previous word
                        new_emb = torch.stack(token_bank).sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))
                        attns.append(torch.stack(attn_bank).sum())
                        agg_idx = len(agg_embs) - 1
                        for token_pos in token_positions:
                            raw_to_agg[token_pos] = agg_idx

                        # Start new word buffer
                        token_bank = [word_emb]
                        word_bank = [word]
                        attn_bank = [attn]
                        token_positions = [pos]
                else:
                    # Subword - add to current buffer (remove "##" prefix)
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])  # Remove "##"
                    attn_bank.append(attn)
                    token_positions.append(pos)

            if token_bank and (len(agg_embs) == 0 or words[-1] != "[SEP]"):
                new_emb = torch.stack(token_bank).sum(axis=0)
                agg_embs.append(new_emb)
                words.append("".join(word_bank))
                attns.append(torch.stack(attn_bank).sum())
                agg_idx = len(agg_embs) - 1
                for token_pos in token_positions:
                    raw_to_agg[token_pos] = agg_idx
                    
            # Stack and pad to original sequence length
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim, device=agg_embs.device)
            paddings = paddings.type_as(agg_embs)
            
            # Pad words and attention
            words = words + ["[PAD]"] * padding_size
            last_attns.append(
                torch.cat(
                    [
                        torch.stack(attns).to(device=agg_embs.device, dtype=agg_embs.dtype),
                        torch.zeros(padding_size, device=agg_embs.device, dtype=agg_embs.dtype),
                    ],
                    dim=0,
                )
            )
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)
            raw_to_agg_batch.append(raw_to_agg)

        # Stack batch and restore original shape
        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)  # [B, layers, seq, dim]
        last_atten_pt = torch.stack(last_attns)
        last_atten_pt = last_atten_pt.type_as(agg_embs_batch)
        raw_to_agg_batch = torch.stack(raw_to_agg_batch)

        return agg_embs_batch, sentences, last_atten_pt, raw_to_agg_batch

    def forward(self, ids, attn_mask, token_type, get_local=False):
        """
        Forward pass through BERT encoder.
        
        Args:
            ids: Token IDs [batch_size, seq_len]
            attn_mask: Attention mask [batch_size, seq_len]
            token_type: Token type IDs [batch_size, seq_len]
            get_local: Legacy parameter
            
        Returns:
            tuple:
                - report_feat: Global features [B, hidden_dim] ([CLS] token)
                - word_feat: Local features [B, seq_len-1, hidden_dim]
                - last_atten_pt: Attention weights [B, seq_len-1]
                - sents: Decoded sentences (list of word lists)
                
        BERT Processing Flow:
            1. Input: ["[CLS]", "the", "patient", "has", ..., "[SEP]", "[PAD]", ...]
            2. BERT encoding: Get hidden states for all tokens
            3. Token aggregation: Merge subwords back to words
            4. Split: [CLS] for global, rest for local
            5. Get attention weights from last layer for importance weighting
        """
        # Run BERT forward pass
        outputs = self.model(ids, attn_mask, token_type,
                             return_dict=True, mode="text")

        # Extract attention from last layer
        # Shape: [B, num_heads, seq_len, seq_len]
        # We get CLS token's attention to other tokens: [B, seq_len-1]
        last_layer_attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1)
        
        # Get last hidden state and add dummy layer dimension
        all_feat = outputs.last_hidden_state.unsqueeze(1)  # [B, 1, seq, hidden]

        # Aggregate subword tokens to word-level
        if self.agg_tokens:
            all_feat, sents, last_atten_pt, raw_to_agg = self.aggregate_tokens(
                all_feat, ids, last_layer_attn)
            # Remove [CLS] token attention (not used)
            last_atten_pt = last_atten_pt[:, 1:].contiguous()
        else:
            # No aggregation - use raw tokens
            sents = [[self.idxtoword[w.item()] for w in sent]
                     for sent in ids]
            raw_to_agg = torch.arange(ids.size(1), device=ids.device).unsqueeze(0).expand(ids.size(0), -1)

        # Remove layer dimension (we only use last layer)
        if self.last_n_layers == 1:
            all_feat = all_feat[:, 0]  # [B, seq, hidden]

        # Split into global and local features
        report_feat = all_feat[:, 0].contiguous()   # [CLS] token → global
        word_feat = all_feat[:, 1:].contiguous()    # Other tokens → local

        return report_feat, word_feat, last_atten_pt, sents, raw_to_agg


if __name__ == "__main__":
    # Test the encoder - run from package root
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from datasets.pretrain_dataset import MultimodalPretrainingDataset
    from datasets.transforms import DataTransforms
    
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)

    for i, data in enumerate(dataset):
        imgs, caps, cap_len, key = data
        # Test on samples with full attention mask (no padding)
        if caps["attention_mask"].sum() == 112:
            model = BertEncoder()
            report_feat, sent_feat, sent_mask, sents = model(
                caps["input_ids"],
                caps["attention_mask"],
                caps["token_type_ids"],
                get_local=True)
