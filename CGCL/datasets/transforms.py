"""
================================================================================
MGCA Data Transforms: Image Augmentation Pipelines
================================================================================
This module implements various image transformation pipelines for MGCA,
including augmentations for pre-training and downstream tasks.

Data Augmentation Overview:
- TRAINING: Random augmentations to improve generalization
- VALIDATION/TEST: Deterministic transforms for fair evaluation

Available Transform Classes:
1. DataTransforms - Basic transforms for general use
2. DetectionDataTransforms - Transforms for object detection tasks
3. Moco2Transform - Strong augmentations following MoCo v2/SimCLR

Key Principles:
- All transforms normalize to [-1, 1] range (mean=0.5, std=0.5)
- Training uses random crops, validation uses center crops
- Stronger augmentation generally improves self-supervised learning

Normalization Choice:
    Using (0.5, 0.5, 0.5) for mean and std normalizes pixels to [-1, 1]:
    - Original range: [0, 1] (after ToTensor)
    - Normalized: (x - 0.5) / 0.5 = 2x - 1 → range [-1, 1]
    
    Alternative: ImageNet stats (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    - Better for transfer from ImageNet pretrained models
    - But medical images have different statistics

Augmentation Pipeline (Training):
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Random Crop │ → │  To Tensor  │ → │  Normalize  │ → │   Output    │
│  (224x224)  │   │  [0,1] RGB  │   │  [-1,1] RGB │   │  [3,224,224]│
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘

Augmentation Pipeline (Validation):
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Center Crop │ → │  To Tensor  │ → │  Normalize  │ → │   Output    │
│  (224x224)  │   │  [0,1] RGB  │   │  [-1,1] RGB │   │  [3,224,224]│
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘

Usage:
    from datasets.transforms import DataTransforms, Moco2Transform
    
    # Basic transforms
    train_transform = DataTransforms(is_train=True, crop_size=224)
    val_transform = DataTransforms(is_train=False, crop_size=224)
    
    # Strong augmentation for self-supervised learning
    ssl_transform = Moco2Transform(is_train=True, crop_size=224)
    
    # Apply to image
    augmented_img = train_transform(pil_image)
================================================================================
"""

import cv2
import numpy as np
import torchvision.transforms as transforms
import random
from PIL import ImageFilter
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class DataTransforms(object):
    """
    Basic Data Transforms.
    
    Provides minimal augmentation suitable for:
    - Fine-tuning pre-trained models
    - Downstream classification tasks
    - Cases where strong augmentation may hurt
    
    Training Augmentations:
    - Random crop: Extracts random 224x224 patch (translation invariance)
    
    Validation Transforms:
    - Center crop: Deterministic 224x224 from center
    
    Both apply:
    - ToTensor: PIL Image → PyTorch tensor [0, 1]
    - Normalize: Scale to [-1, 1] range
    
    Args:
        is_train (bool): Whether to apply training augmentations
        crop_size (int): Output crop size (default: 224 for ImageNet compat)
        
    Example:
        transform = DataTransforms(is_train=True, crop_size=224)
        tensor = transform(pil_image)  # Returns [3, 224, 224] tensor
    """
    
    def __init__(self, is_train: bool = True, crop_size: int = 224):
        if is_train:
            # Training: Random crop for data augmentation
            # This teaches the model translation invariance
            data_transforms = [
                transforms.RandomCrop(crop_size),  # Random 224x224 patch
                transforms.ToTensor(),              # PIL → Tensor [0, 1]
                transforms.Normalize(
                    (0.5, 0.5, 0.5),   # Mean for each RGB channel
                    (0.5, 0.5, 0.5)    # Std for each RGB channel
                )  # Normalizes to [-1, 1] range
            ]
        else:
            # Validation: Center crop for deterministic evaluation
            data_transforms = [
                transforms.CenterCrop(crop_size),  # Center 224x224 patch
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), 
                    (0.5, 0.5, 0.5)
                )
            ]

        # Compose into single transform pipeline
        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        """
        Apply transforms to image.
        
        Args:
            image: PIL Image
            
        Returns:
            torch.Tensor: Transformed image [3, crop_size, crop_size]
        """
        return self.data_transforms(image)


class Moco2Transform(object):
    """
    Strong Augmentation Transforms following MoCo v2 / SimCLR.
    
    These transforms are designed for self-supervised contrastive learning,
    where strong augmentation helps learn robust representations.
    
    The augmentation strategy follows the SimCLR paper:
    "A Simple Framework for Contrastive Learning of Visual Representations"
    
    Training Augmentations (in order):
    1. Random Crop: Random 224x224 patch (translation)
    2. Color Jitter (80% prob): Random brightness/contrast/saturation/hue
    3. Grayscale (20% prob): Convert to grayscale (color invariance)
    4. Gaussian Blur (50% prob): Random blur (texture invariance)
    5. Horizontal Flip (50% prob): Mirror image (reflection invariance)
    6. Normalize: Scale to [-1, 1]
    
    Why Strong Augmentation?
    - Contrastive learning treats augmented versions as positive pairs
    - Stronger augmentation → more diverse positives
    - Model must learn invariances to match augmented versions
    - This leads to more robust, generalizable representations
    
    Args:
        is_train (bool): Whether to apply training augmentations
        crop_size (int): Output crop size (default: 224)
        
    Augmentation Visualization:
    
    Original:          Augmented:
    ┌─────────────┐    ┌─────────────┐
    │   Chest     │    │   Chest     │  ← Same semantic content
    │    X-ray    │ →  │    X-ray    │  ← Different appearance
    │  (normal)   │    │ (darker,    │
    └─────────────┘    │  cropped)   │
                       └─────────────┘
    
    Both should map to similar embeddings!
    """
    
    def __init__(self, is_train: bool = True, crop_size: int = 224) -> None:
        if is_train:
            # Full SimCLR augmentation pipeline
            self.data_transforms = transforms.Compose(
                [
                    # 1. Random crop - spatial augmentation
                    transforms.RandomCrop(crop_size),
                    
                    # 2. Color jitter (80% probability)
                    # Randomly adjusts: brightness, contrast, saturation, hue
                    transforms.RandomApply(
                        [transforms.ColorJitter(
                            brightness=0.4,  # ±40% brightness
                            contrast=0.4,    # ±40% contrast
                            saturation=0.4,  # ±40% saturation
                            hue=0.1          # ±10% hue shift
                        )], 
                        p=0.8
                    ),
                    
                    # 3. Random grayscale (20% probability)
                    # Teaches color invariance - useful since many X-rays are grayscale anyway
                    transforms.RandomGrayscale(p=0.2),
                    
                    # 4. Gaussian blur (50% probability)
                    # Teaches texture/detail invariance
                    # Sigma randomly sampled from [0.1, 2.0]
                    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                    
                    # 5. Random horizontal flip (50% probability)
                    # Note: For chest X-rays, this changes anatomical meaning!
                    # Consider removing for medical imaging tasks
                    transforms.RandomHorizontalFlip(),
                    
                    # 6. Convert to tensor and normalize
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )
        else:
            # Validation: deterministic transforms only
            self.data_transforms = transforms.Compose(
                [
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )

    def __call__(self, img):
        """Apply MoCo v2 transforms."""
        return self.data_transforms(img)


class GaussianBlur:
    """
    Gaussian Blur Augmentation.
    
    Applies Gaussian blur with randomly sampled sigma, as used in SimCLR.
    This teaches the model to be invariant to image sharpness/blur.
    
    Paper Reference:
    "A Simple Framework for Contrastive Learning of Visual Representations"
    https://arxiv.org/abs/2002.05709
    
    Args:
        sigma (tuple): Range for random sigma sampling (min, max)
            Default: (0.1, 2.0) as in SimCLR
            
    How it works:
    1. Random sigma sampled uniformly from [0.1, 2.0]
    2. Gaussian kernel applied with that sigma
    3. Larger sigma = more blur
    
    Blur Examples:
        sigma=0.1: Almost no blur (nearly original)
        sigma=1.0: Moderate blur
        sigma=2.0: Heavy blur (significant detail loss)
    """

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        """
        Apply random Gaussian blur.
        
        Args:
            x: PIL Image
            
        Returns:
            PIL Image: Blurred image
        """
        # Sample random sigma from uniform distribution
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        
        # Apply Gaussian blur filter
        # PIL's GaussianBlur uses 'radius' which is approximately sigma
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class SpatialClueDataTransforms(object):
    """
    Joint transforms for:
    - image:      RGB image
    - seg_mask:   binary segmentation mask [H, W]
    - clue_masks: multi-channel clue masks [C, H, W]

    This class ensures all spatial transforms are applied consistently
    across image and masks, which is required for spatial supervision.

    Training:
    - pad if needed
    - random crop
    - optional horizontal flip

    Validation/Test:
    - center crop

    Image is normalized to [-1, 1]
    Masks remain binary float tensors.
    """

    def __init__(
        self,
        is_train: bool = True,
        crop_size: int = 224,
        hflip_prob: float = 0.5,
    ):
        self.is_train = is_train
        self.crop_size = crop_size
        self.hflip_prob = hflip_prob
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def _ensure_pil_image(self, image):
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            return Image.fromarray(image.astype(np.uint8))
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _ensure_pil_mask(self, mask):
        """
        Convert single-channel mask [H, W] into PIL image.
        """
        if isinstance(mask, Image.Image):
            return mask
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if isinstance(mask, np.ndarray):
            mask = mask.astype(np.uint8)
            return Image.fromarray(mask)
        raise TypeError(f"Unsupported mask type: {type(mask)}")

    def _pad_if_needed(self, pil_obj, fill=0):
        w, h = pil_obj.size
        pad_w = max(0, self.crop_size - w)
        pad_h = max(0, self.crop_size - h)

        if pad_w == 0 and pad_h == 0:
            return pil_obj

        # left, top, right, bottom
        padding = (0, 0, pad_w, pad_h)
        return TF.pad(pil_obj, padding, fill=fill)

    def _get_crop_params(self, image):
        return transforms.RandomCrop.get_params(image, (self.crop_size, self.crop_size))

    def _apply_crop(self, pil_obj, i, j, h, w):
        return TF.crop(pil_obj, i, j, h, w)

    def _apply_hflip(self, pil_obj):
        return TF.hflip(pil_obj)

    def _process_image(self, image):
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        return image

    def _process_mask(self, mask):
        """
        Convert PIL mask to float tensor in {0,1}, shape [H,W]
        """
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 0).astype(np.float32)
        return torch.from_numpy(mask)

    def __call__(self, image, seg_mask, clue_masks):
        """
        Args:
            image: RGB image (numpy array or PIL)
            seg_mask: [H, W]
            clue_masks: [C, H, W]

        Returns:
            image_tensor: [3, crop_size, crop_size]
            seg_mask_tensor: [crop_size, crop_size]
            clue_masks_tensor: [C, crop_size, crop_size]
        """
        image = self._ensure_pil_image(image)
        seg_mask = self._ensure_pil_mask(seg_mask)

        if isinstance(clue_masks, torch.Tensor):
            clue_masks = clue_masks.cpu().numpy()
        clue_masks = np.asarray(clue_masks)

        clue_mask_list = [self._ensure_pil_mask(clue_masks[c]) for c in range(clue_masks.shape[0])]

        # pad if needed
        image = self._pad_if_needed(image, fill=0)
        seg_mask = self._pad_if_needed(seg_mask, fill=0)
        clue_mask_list = [self._pad_if_needed(m, fill=0) for m in clue_mask_list]

        # shared spatial transform
        if self.is_train:
            i, j, h, w = self._get_crop_params(image)

            image = self._apply_crop(image, i, j, h, w)
            seg_mask = self._apply_crop(seg_mask, i, j, h, w)
            clue_mask_list = [self._apply_crop(m, i, j, h, w) for m in clue_mask_list]

            if random.random() < self.hflip_prob:
                image = self._apply_hflip(image)
                seg_mask = self._apply_hflip(seg_mask)
                clue_mask_list = [self._apply_hflip(m) for m in clue_mask_list]
        else:
            image = TF.center_crop(image, [self.crop_size, self.crop_size])
            seg_mask = TF.center_crop(seg_mask, [self.crop_size, self.crop_size])
            clue_mask_list = [
                TF.center_crop(m, [self.crop_size, self.crop_size]) for m in clue_mask_list
            ]

        # to tensors
        image_tensor = self._process_image(image)
        seg_mask_tensor = self._process_mask(seg_mask)
        clue_masks_tensor = torch.stack([self._process_mask(m) for m in clue_mask_list], dim=0)

        return image_tensor, seg_mask_tensor, clue_masks_tensor