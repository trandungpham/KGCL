"""
================================================================================
Train ISIC Multi-Task Model
================================================================================
This script runs the experiments defined in `models/kgcl/multitask_module.py`.

Supported phases:
- `pretrain`: train the encoder with clue presence, clue-area alignment, and chaos
- `finetune`: fine-tune with diagnosis, clue presence, clue-area alignment, and chaos

Task modes for `finetune`:
- `multitask`: diagnosis + clue + chaos + alignment
- `diag_only`: diagnosis classification only

Usage:
    # Quick phase-1 smoke test
    python train_multitask.py --phase pretrain --max_epochs 1 --batch_size 8 --quick_test

    # Standard phase-1 run
    python train_multitask.py --phase pretrain --max_epochs 100 --batch_size 16 --backbone_name convnext_base

    # Phase-2 run initialized from phase-1 checkpoint
    python train_multitask.py --phase finetune --backbone_name convnext_base --lambda_diag 2.0 --lambda_align 0.5 --task_mode multitask --phase1_ckpt checkpoints/pretrain/convnext_base/best.ckpt

    # Phase-2 run diagnosis only (ablation study)
    python train_multitask.py --phase finetune --task_mode diag_only --backbone_name convnext_base
================================================================================
"""

import datetime
import json
import os
import shutil
import sys
from argparse import ArgumentParser
import csv
from copy import deepcopy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger


from datasets.constants import CHAOS_LABELS, CLUES_NAMES
from datasets.dataset import (
    FinetuneDataset,
    PretrainDataset,
    finetune_collate_fn,
    pretrain_collate_fn,
)
from datasets.transforms import SpatialClueDataTransforms
from models.cgcl.multitask_module import AuxiliaryCalibrationModule, FinetuneModule, PretrainModule

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)


class MultiTaskDataModule(LightningDataModule):
    """DataModule for phase-1 pretraining and phase-2 finetuning."""

    def __init__(
        self,
        phase,
        train_csv,
        test_csv,
        val_csv,
        img_dir,
        mask_dir,
        vector_dir,
        batch_size,
        num_workers,
        data_pct=1.0,
        crop_size=224,
        use_weighted_sampler=False,
        sampler_clue_weight_scale=1.0,
        sampler_diag_weight_scale=0.5,
    ):
        super().__init__()
        self.phase = phase
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.val_csv = val_csv
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.vector_dir = vector_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_pct = data_pct
        self.crop_size = crop_size
        self.use_weighted_sampler = use_weighted_sampler
        self.sampler_clue_weight_scale = sampler_clue_weight_scale
        self.sampler_diag_weight_scale = sampler_diag_weight_scale

    def _transform(self, is_train):
        return SpatialClueDataTransforms(
            is_train=is_train,
            crop_size=self.crop_size,
        )

    def train_dataloader(self):
        transform = self._transform(is_train=True)

        if self.phase == "pretrain":
            dataset = PretrainDataset(
                csv_path=self.train_csv,
                img_dir=self.img_dir,
                mask_dir=self.mask_dir,
                vector_dir=self.vector_dir,
                transform=transform,
                data_pct=self.data_pct,
            )
            collate_fn = pretrain_collate_fn
        else:
            dataset = FinetuneDataset(
                csv_path=self.train_csv,
                img_dir=self.img_dir,
                mask_dir=self.mask_dir,
                vector_dir=self.vector_dir,
                split="train",
                transform=transform,
                data_pct=self.data_pct,
            )
            collate_fn = finetune_collate_fn

        sampler = None
        shuffle = True
        if self.phase == "finetune" and self.use_weighted_sampler:
            sample_weights = dataset.get_sample_weights(
                clue_weight_scale=self.sampler_clue_weight_scale,
                diagnosis_weight_scale=self.sampler_diag_weight_scale,
            )
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.phase == "pretrain":
            return None

        dataset = FinetuneDataset(
            csv_path=self.val_csv,
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            vector_dir=self.vector_dir,
            split="valid",
            transform=self._transform(is_train=False),
            data_pct=1.0,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=finetune_collate_fn,
            # drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.phase == "pretrain":
            return None

        dataset = FinetuneDataset(
            csv_path=self.test_csv,
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            vector_dir=self.vector_dir,
            split="test",
            transform=self._transform(is_train=False),
            data_pct=1.0,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=finetune_collate_fn,
            pin_memory=True,
        )


def build_parser():
    parser = ArgumentParser(description="Train multi-task model")

    # ── Phase 1 (pretrain) data — chaos_and_clues dataset ────────────────────
    p1_train_csv = os.path.join(BASE_DIR, "../chaos_and_clues/train.csv")
    p1_img_dir   = os.path.join(BASE_DIR, "../chaos_and_clues/Images")
    p1_mask_dir  = os.path.join(BASE_DIR, "../chaos_and_clues/GroundTruthMasks")
    p1_vec_dir   = os.path.join(BASE_DIR, "../chaos_and_clues/Vectors")

    # ── Phase 2 (finetune) data — Annotated_data with diagnosis labels ─────────
    p2_train_csv = os.path.join(BASE_DIR, "../Annotated_data/train.csv")
    p2_test_csv  = os.path.join(BASE_DIR, "../Annotated_data/test.csv")
    p2_val_csv   = os.path.join(BASE_DIR, "../Annotated_data/val.csv")
    p2_img_dir   = os.path.join(BASE_DIR, "../Annotated_data/Images")
    p2_mask_dir  = os.path.join(BASE_DIR, "../Annotated_data/GroundTruthMasks")
    p2_vec_dir   = os.path.join(BASE_DIR, "../Annotated_data/Vectors")

    parser.add_argument("--phase", type=str, default="all", choices=["pretrain", "finetune", "all", "calibrate"])

    # Phase 1 paths
    parser.add_argument("--phase1_train_csv",  type=str, default=p1_train_csv)
    parser.add_argument("--phase1_img_dir",    type=str, default=p1_img_dir)
    parser.add_argument("--phase1_mask_dir",   type=str, default=p1_mask_dir)
    parser.add_argument("--phase1_vector_dir", type=str, default=p1_vec_dir)

    # Phase 2 paths (also used when --phase pretrain/finetune called directly)
    parser.add_argument("--train_csv",  type=str, default=p2_train_csv)
    parser.add_argument("--test_csv",   type=str, default=p2_test_csv)
    parser.add_argument("--val_csv",    type=str, default=p2_val_csv)
    parser.add_argument("--img_dir",    type=str, default=p2_img_dir)
    parser.add_argument("--mask_dir",   type=str, default=p2_mask_dir)
    parser.add_argument("--vector_dir", type=str, default=p2_vec_dir)

    parser.add_argument("--backbone_name", type=str, default="resnet50")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument(
        "--no_pretrained",
        action="store_false",
        dest="pretrained",
        help="Disable timm pretrained weights",
    )
    parser.add_argument("--phase1_ckpt", type=str, default=None)
    parser.add_argument(
        "--phase2_ckpt",
        type=str,
        default=None,
        help="Path to a trained FinetuneModule checkpoint for Phase 3 confidence calibration.",
    )

    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_pct", type=float, default=1.0)
    parser.add_argument("--crop_size", type=int, default=224)

    parser.add_argument("--lambda_clue", type=float, default=1.0)
    parser.add_argument("--lambda_chaos", type=float, default=1.0)
    parser.add_argument("--lambda_diag", type=float, default=1.0)
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--lambda_confidence", type=float, default=0.1)
    parser.add_argument("--task_mode", type=str, default="multitask", choices=["multitask", "diag_only"])
    parser.add_argument(
        "--auto_finetune_task_mode",
        type=str,
        default="multitask",
        choices=["multitask", "diag_only"],
        help="Finetune task mode used after pretraining when --phase all is selected.",
    )
    parser.add_argument(
        "--use_agentic_aux",
        action="store_true",
        help="Enable label-aware agentic routing for clue/chaos auxiliary heads.",
    )
    parser.add_argument("--num_aux_agents", type=int, default=4)
    parser.add_argument("--aux_agent_hidden_dim", type=int, default=256)
    parser.add_argument(
        "--backbone_lr_scale",
        type=float,
        default=0.1,
        help="Backbone LR multiplier in Phase 2. Use 0.3–1.0 when Phase 1 was trained "
             "on a different dataset (e.g. chaos_and_clues → Annotated_data).",
    )
    parser.add_argument(
        "--no_stop_grad_chaos",
        action="store_true",
        help="Allow diagnosis gradient to flow back through the chaos head. "
             "Recommended when Phase 1 and Phase 2 use different datasets.",
    )
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--sampler_clue_weight_scale", type=float, default=1.0)
    parser.add_argument("--sampler_diag_weight_scale", type=float, default=0.5)
    parser.add_argument("--max_pos_weight", type=float, default=50.0)

    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--logger", type=str, default="wandb", choices=["wandb", "csv"])
    parser.add_argument("--wandb_project", type=str, default="KGCL")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser


def _load_rows(csv_path):
    with open(csv_path, newline="") as handle:
        return list(csv.DictReader(handle))



def compute_training_statistics(csv_path, mask_dir, vector_dir, max_pos_weight, task_mode):
    rows = _load_rows(csv_path)

    clue_vectors = []
    area_positive = None
    area_total_pixels = 0
    diagnosis_labels = []

    for row in rows:
        clue_vector = np.load(os.path.join(vector_dir, row["vector_name"])).astype(np.float32)
        clue_vectors.append(clue_vector)
        diagnosis_labels.append(1 if str(row.get("diagnosis", "NV")).upper() == "MEL" else 0)

        if task_mode != "diag_only":
            clue_mask = np.load(os.path.join(mask_dir, row["mask_name"])).astype(np.float32)
            clue_mask = (clue_mask > 0).astype(np.float32)
            current_positive = clue_mask.sum(axis=(1, 2))
            area_positive = current_positive if area_positive is None else area_positive + current_positive
            area_total_pixels += clue_mask.shape[1] * clue_mask.shape[2]

    clue_vectors = np.stack(clue_vectors)
    clue_positive = clue_vectors.sum(axis=0)
    clue_negative = len(clue_vectors) - clue_positive
    clue_pos_weight = clue_negative / np.clip(clue_positive, a_min=1.0, a_max=None)
    clue_pos_weight = np.clip(clue_pos_weight, a_min=1.0, a_max=max_pos_weight)
    if task_mode == "diag_only":
        area_pos_weight = np.ones_like(clue_pos_weight, dtype=np.float32)
    else:
        area_negative = area_total_pixels - area_positive
        area_pos_weight = area_negative / np.clip(area_positive, a_min=1.0, a_max=None)
        area_pos_weight = np.clip(area_pos_weight, a_min=1.0, a_max=max_pos_weight)

    # Per-class weight for diagnosis cross-entropy: N / (num_classes * count_per_class).
    # Upweights the minority MEL class to counteract the ~3:1 NV/MEL imbalance.
    diag_counts = np.bincount(diagnosis_labels, minlength=2).astype(np.float32)
    diag_class_weight = len(diagnosis_labels) / (2.0 * np.clip(diag_counts, 1.0, None))

    return {
        "clue_pos_weight": torch.tensor(clue_pos_weight, dtype=torch.float32),
        "area_pos_weight": torch.tensor(area_pos_weight, dtype=torch.float32),
        "diag_class_weight": torch.tensor(diag_class_weight, dtype=torch.float32),
    }


_BEST_REGISTRY = os.path.join(BASE_DIR, "checkpoints", "best_per_backbone.json")
_BEST_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "best")


def _maybe_update_best_backbone(backbone: str, val_acc: float, run_id: str, ckpt_path: str) -> bool:
    """
    Maintain a per-backbone best-checkpoint registry across sweep runs.

    If `val_acc` beats the current best for `backbone`, copies `ckpt_path`
    to checkpoints/best/{backbone}/best.ckpt and updates
    checkpoints/best_per_backbone.json.

    Returns True when the registry was updated.
    """
    registry: dict = {}
    if os.path.exists(_BEST_REGISTRY):
        with open(_BEST_REGISTRY) as f:
            registry = json.load(f)

    current_best = registry.get(backbone, {}).get("val_diag_acc", -1.0)
    if val_acc <= current_best:
        print(f"   {backbone}: {val_acc:.4f} ≤ current best {current_best:.4f} — not updated.")
        return False

    dest_dir = os.path.join(_BEST_CKPT_DIR, backbone)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "best.ckpt")
    shutil.copy2(ckpt_path, dest)

    registry[backbone] = {
        "val_diag_acc": round(val_acc, 4),
        "run_id": run_id,
        "ckpt_path": dest,
    }
    os.makedirs(os.path.dirname(_BEST_REGISTRY), exist_ok=True)
    with open(_BEST_REGISTRY, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"★  New best for {backbone}: val_diag_acc={val_acc:.4f}  (run {run_id})")
    print(f"   Saved to: {dest}")
    return True


_FOUR_STAGE_BACKBONES = ("convnext", "swin", "efficientnet", "mobilenet", "densenet", "resnet")

BAD_MODELS = [
    "swin_large_patch4_window7_224_22k",
]


def _get_out_indices(backbone_name: str):
    """ResNet/VGG have 5 stages -> (1,2,3,4). ConvNeXt/Swin/EfficientNet have 4 -> (0,1,2,3)."""
    if any(backbone_name.lower().startswith(p) for p in _FOUR_STAGE_BACKBONES):
        return (0, 1, 2, 3)
    return (1, 2, 3, 4)


def build_model(args, phase=None):
    phase = phase or args.phase
    stats = compute_training_statistics(
        csv_path=args.train_csv,
        mask_dir=args.mask_dir,
        vector_dir=args.vector_dir,
        max_pos_weight=args.max_pos_weight,
        task_mode=args.task_mode,
    )

    out_indices = _get_out_indices(args.backbone_name)

    common_kwargs = {
        "backbone_name": args.backbone_name,
        "pretrained": args.pretrained,
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_clues": len(CLUES_NAMES),
        "num_chaos": 2,
        "lambda_clue": args.lambda_clue,
        "lambda_chaos": args.lambda_chaos,
        "lambda_align": args.lambda_align,
        "lambda_confidence": args.lambda_confidence,
        "clue_pos_weight": stats["clue_pos_weight"],
        "area_pos_weight": stats["area_pos_weight"],
        "out_indices": out_indices,
        "use_agentic_aux": args.use_agentic_aux,
        "num_aux_agents": args.num_aux_agents,
        "aux_agent_hidden_dim": args.aux_agent_hidden_dim,
        "img_size": args.crop_size,
    }

    if phase == "pretrain":
        return PretrainModule(**common_kwargs)

    return FinetuneModule(
        **common_kwargs,
        pretrained_phase1_ckpt=args.phase1_ckpt,
        clue_names=CLUES_NAMES,
        chaos_names=CHAOS_LABELS,
        lambda_diag=args.lambda_diag,
        task_mode=args.task_mode,
        diag_class_weight=stats["diag_class_weight"],
        backbone_lr_scale=args.backbone_lr_scale,
        stop_grad_chaos_for_diag=not args.no_stop_grad_chaos,
    )

def build_datamodule(args, phase):
    return MultiTaskDataModule(
        phase=phase,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        vector_dir=args.vector_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_pct=args.data_pct,
        crop_size=args.crop_size,
        use_weighted_sampler=args.use_weighted_sampler,
        sampler_clue_weight_scale=args.sampler_clue_weight_scale,
        sampler_diag_weight_scale=args.sampler_diag_weight_scale,
    )


def build_checkpoint_dir(args, phase, is_transfer):
    if phase == "pretrain":
        return os.path.join(BASE_DIR, "checkpoints", phase, args.backbone_name)

    task_dir = args.task_mode
    transfer_dir = "transfer" if is_transfer else "no_transfer"
    return os.path.join(BASE_DIR, "checkpoints", phase, task_dir, transfer_dir, args.backbone_name)


def build_logger(args, phase, ckpt_dir, run_suffix):
    if args.logger == "csv":
        return CSVLogger(save_dir=ckpt_dir, name="", version="")

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    run_name = args.wandb_run_name or f"{phase}-{args.backbone_name}-{run_suffix}"
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        save_dir=ckpt_dir,
        log_model=False,
    )
    return logger


def run_phase(args, phase, phase1_ckpt=None):
    run_args = deepcopy(args)
    run_args.phase = phase
    if phase == "finetune":
        run_args.phase1_ckpt = phase1_ckpt or args.phase1_ckpt
    else:
        run_args.phase1_ckpt = None
        # Phase 1 uses skincancer_full (chaos+clues); no val/test needed
        run_args.train_csv  = args.phase1_train_csv
        run_args.img_dir    = args.phase1_img_dir
        run_args.mask_dir   = args.phase1_mask_dir
        run_args.vector_dir = args.phase1_vector_dir

    seed_everything(run_args.seed)

    print("=" * 80)
    print(f"ISIC Multi-Task Training: {phase.upper()}")
    print("=" * 80)
    print(f"Backbone:        {run_args.backbone_name}")
    print(f"Dataset:         {'skincancer_full (chaos+clues)' if phase == 'pretrain' else 'Annotated_data (diagnosis)'}")
    print(f"Batch Size:      {run_args.batch_size}")
    print(f"Max Epochs:      {run_args.max_epochs}")
    print(f"Learning Rate:   {run_args.learning_rate}")
    print(f"Crop Size:       {run_args.crop_size}")
    print(f"Pretrained:      {run_args.pretrained}")
    print(f"Quick Test:      {run_args.quick_test}")
    print(f"Task Mode:       {run_args.task_mode}")
    print(f"Agentic Aux:     {run_args.use_agentic_aux}")
    if run_args.use_agentic_aux:
        print(f"Aux Agents:      {run_args.num_aux_agents}")
        print(f"Aux Hidden Dim:  {run_args.aux_agent_hidden_dim}")
        print(f"Lambda Conf:     {run_args.lambda_confidence}")

    if phase == "finetune":
        print(f"Phase-1 Ckpt:    {run_args.phase1_ckpt}")
        print(f"Lambda Diag:     {run_args.lambda_diag}")

    if phase == "pretrain" or run_args.task_mode != "diag_only":
        print(f"Lambda Clue:     {run_args.lambda_clue}")
        print(f"Lambda Chaos:    {run_args.lambda_chaos}")
        print(f"Lambda Align:    {run_args.lambda_align}")

    datamodule = build_datamodule(run_args, phase)
    model = build_model(run_args, phase=phase)

    is_transfer = phase == "finetune" and bool(run_args.phase1_ckpt)
    run_suffix = "transfer" if is_transfer else "base"
    base_ckpt_dir = build_checkpoint_dir(run_args, phase=phase, is_transfer=is_transfer)

    # ── Create logger first to obtain the W&B run ID ─────────────────────────
    logger = build_logger(run_args, phase=phase, ckpt_dir=base_ckpt_dir, run_suffix=run_suffix)
    logger.log_hyperparams(vars(run_args))

    # Each sweep run gets its own subdirectory so runs never overwrite each other.
    run_id = str(getattr(logger, "version", "local"))
    ckpt_dir = os.path.join(base_ckpt_dir, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoint Dir:  {ckpt_dir}  (run {run_id})")

    callbacks = [LearningRateMonitor(logging_interval="epoch")]

    if phase == "finetune":
        callbacks.extend(
            [
                ModelCheckpoint(
                    monitor="val_diag_acc",
                    dirpath=ckpt_dir,
                    save_last=True,
                    mode="max",
                    save_top_k=1,
                    filename="multitask-{epoch:02d}-{val_loss:.4f}-{val_diag_acc:.4f}",
                ),
                EarlyStopping(
                    monitor="val_diag_acc",
                    min_delta=0.001,
                    patience=10,
                    verbose=True,
                    mode="max",
                ),
            ]
        )
    else:
        callbacks.extend(
            [
                ModelCheckpoint(
                    monitor="pretrain_score",
                    dirpath=ckpt_dir,
                    save_last=False,
                    mode="max",
                    save_top_k=1,
                    filename="multitask-pretrain-{epoch:02d}-{pretrain_score:.4f}",
                ),
                EarlyStopping(
                    monitor="pretrain_score",
                    min_delta=0.001,
                    patience=10,
                    verbose=True,
                    mode="max",
                ),
            ]
        )

    trainer_kwargs = {
        "max_epochs": run_args.max_epochs,
        "accelerator": run_args.accelerator,
        "devices": run_args.gpus,
        "precision": run_args.precision,
        "accumulate_grad_batches": run_args.accumulate_grad_batches,
        "deterministic": True,
        "callbacks": callbacks,
        "logger": logger,
        "enable_progress_bar": True,
    }

    if run_args.quick_test:
        trainer_kwargs["limit_train_batches"] = 10
        if phase == "finetune":
            trainer_kwargs["limit_val_batches"] = 5
            trainer_kwargs["limit_test_batches"] = 5
        print("Quick test mode enabled with limited batches.")

    trainer = Trainer(**trainer_kwargs)

    print("\nStarting training...")
    trainer.fit(model, datamodule=datamodule)

    if phase == "finetune":
        print("\nRunning test with best checkpoint...")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_pth_path = None
    if best_model_path:
        best_checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
        best_pth_path = os.path.join(ckpt_dir, "best.ckpt")
        torch.save(best_checkpoint["state_dict"], best_pth_path)
        print(f"Exported best model weights to: {best_pth_path}")

        # ── Update cross-run best-per-backbone registry (finetune only) ──────
        if phase == "finetune":
            val_acc = trainer.callback_metrics.get("val_diag_acc", 0.0)
            val_acc = float(val_acc.item() if hasattr(val_acc, "item") else val_acc)
            _maybe_update_best_backbone(run_args.backbone_name, val_acc, run_id, best_pth_path)
    else:
        print("No best checkpoint was recorded, skipping best.ckpt export.")

    print("Training completed.")
    return best_pth_path


def run_calibration(args, phase2_ckpt: str = None):
    """
    Phase 3: freeze the trained model and train only the PerConceptAuxiliaryHub.

    `phase2_ckpt` can be supplied directly (when chained from --phase all) or
    read from args.phase2_ckpt (when --phase calibrate is called standalone).

    Example (standalone)
    --------------------
    python train_multitask.py --phase calibrate \\
        --backbone_name convnext_base \\
        --phase2_ckpt checkpoints/best/convnext_base/best.ckpt \\
        --max_epochs 30 --learning_rate 1e-3
    """
    phase2_ckpt = phase2_ckpt or args.phase2_ckpt
    if phase2_ckpt is None:
        raise ValueError("--phase2_ckpt is required for --phase calibrate")

    seed_everything(args.seed)
    out_indices = _get_out_indices(args.backbone_name)

    model = AuxiliaryCalibrationModule(
        backbone_name=args.backbone_name,
        pretrained=False,          # weights come from the Phase 2 checkpoint
        phase2_ckpt=phase2_ckpt,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        out_indices=out_indices,
        num_aux_agents=args.num_aux_agents,
        aux_agent_hidden_dim=args.aux_agent_hidden_dim,
        img_size=args.crop_size,
    )

    # Calibration uses the same Annotated_data split as Phase 2.
    datamodule = build_datamodule(args, phase="finetune")

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", "calibrate", args.backbone_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    run_suffix = "calibrate"
    logger = build_logger(args, phase="calibrate", ckpt_dir=ckpt_dir, run_suffix=run_suffix)
    logger.log_hyperparams(vars(args))

    run_id = str(getattr(logger, "version", "local"))
    ckpt_dir = os.path.join(ckpt_dir, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            monitor="calib_val_loss",
            dirpath=ckpt_dir,
            save_top_k=1,
            mode="min",
            filename="calib-{epoch:02d}-{calib_val_loss:.4f}",
        ),
        EarlyStopping(
            monitor="calib_val_loss",
            min_delta=1e-4,
            patience=10,
            mode="min",
        ),
    ]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.gpus,
        precision=args.precision,
        deterministic=True,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    print("\nStarting confidence calibration (Phase 3)...")
    trainer.fit(model, datamodule=datamodule)

    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        dest = os.path.join(ckpt_dir, "best_calib.ckpt")
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        torch.save(ckpt["state_dict"], dest)
        print(f"Exported best calibration weights to: {dest}")

    print("Calibration completed.")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.backbone_name in BAD_MODELS:
        try:
            import wandb
            wandb.finish(exit_code=1)
        except Exception:
            pass
        raise ValueError(f"Skipping invalid model: {args.backbone_name}")

    if args.phase == "all":
        # ── Phase 1: pretrain ────────────────────────────────────────────────
        pretrain_args = deepcopy(args)
        pretrain_args.task_mode = "multitask"
        pretrain_best = run_phase(pretrain_args, phase="pretrain")

        if pretrain_best is None:
            print(
                "WARNING: pretrain phase saved no best checkpoint "
                "(early stopping may have fired on epoch 0). "
                "Finetune will start from ImageNet weights only."
            )

        # ── Phase 2: finetune ────────────────────────────────────────────────
        finetune_args = deepcopy(args)
        finetune_args.phase1_ckpt = pretrain_best
        finetune_args.task_mode = args.auto_finetune_task_mode
        finetune_best = run_phase(finetune_args, phase="finetune", phase1_ckpt=pretrain_best)

        # ── Phase 3: calibrate auxiliary confidence estimator ────────────────
        if finetune_best is not None:
            run_calibration(finetune_args, phase2_ckpt=finetune_best)
        else:
            print("WARNING: finetune phase saved no best checkpoint — skipping calibration.")
        return

    if args.phase == "calibrate":
        run_calibration(args)
        return

    run_phase(args, phase=args.phase, phase1_ckpt=args.phase1_ckpt)


if __name__ == "__main__":
    main()
