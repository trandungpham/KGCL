"""
================================================================================
Train ISIC Multi-Task Model — End-to-End (Single Phase)
================================================================================
Trains all tasks simultaneously from scratch in one run:
  - Diagnosis (MEL vs NV)
  - Clue presence (9 dermoscopic clues)
  - Clue area alignment (spatial masks)
  - Chaos (structure / colour)

Input:  images (3, 224, 224)
Output: clue_logits (B,9) | clue_area_logits (B,9,H,W) | chaos_logits (B,2) | diagnosis_logits (B,2)

Usage:
    # Smoke test
    python train_end2end.py --max_epochs 1 --batch_size 8 --quick_test --logger csv

    # Full run
    python train_end2end.py --backbone_name resnet50 --max_epochs 100 \
        --lambda_diag 2.0 --lambda_align 0.5 --logger csv
================================================================================
"""

import os
import sys
from argparse import ArgumentParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from datasets.constants import CLUES_NAMES
from models.cgcl.multitask_module import FinetuneModule
from train_multitask import (
    _get_out_indices,
    build_datamodule,
    build_logger,
    compute_training_statistics,
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)


def build_parser():
    parser = ArgumentParser(description="End-to-end multi-task training")

    train_csv = os.path.join(BASE_DIR, "../Annotated_data/train.csv")
    test_csv  = os.path.join(BASE_DIR, "../Annotated_data/test.csv")
    val_csv   = os.path.join(BASE_DIR, "../Annotated_data/val.csv")
    default_img    = os.path.join(BASE_DIR, "../Annotated_data/Images")
    default_mask   = os.path.join(BASE_DIR, "../Annotated_data/GroundTruthMasks")
    default_vector = os.path.join(BASE_DIR, "../Annotated_data/Vectors")

    parser.add_argument("--train_csv",   type=str, default=train_csv)
    parser.add_argument("--test_csv",    type=str, default=test_csv)
    parser.add_argument("--val_csv",     type=str, default=val_csv)
    parser.add_argument("--img_dir",     type=str, default=default_img)
    parser.add_argument("--mask_dir",    type=str, default=default_mask)
    parser.add_argument("--vector_dir",  type=str, default=default_vector)

    parser.add_argument("--backbone_name", type=str, default="resnet50")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained")

    parser.add_argument("--max_epochs",   type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--learning_rate",type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--data_pct",     type=float, default=1.0)
    parser.add_argument("--crop_size",    type=int,   default=224)

    parser.add_argument("--lambda_clue",  type=float, default=1.0)
    parser.add_argument("--lambda_chaos", type=float, default=1.0)
    parser.add_argument("--lambda_diag",  type=float, default=1.0)
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--task_mode",    type=str,   default="multitask",
                        choices=["multitask", "diag_only"])
    parser.add_argument("--max_pos_weight", type=float, default=50.0)

    parser.add_argument("--use_weighted_sampler",       action="store_true")
    parser.add_argument("--sampler_clue_weight_scale",  type=float, default=1.0)
    parser.add_argument("--sampler_diag_weight_scale",  type=float, default=0.5)

    parser.add_argument("--gpus",                    type=int, default=1)
    parser.add_argument("--accelerator",             type=str, default="auto")
    parser.add_argument("--precision",               type=str, default="32")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--quick_test",   action="store_true")
    parser.add_argument("--logger",       type=str,  default="wandb",
                        choices=["wandb", "csv"])
    parser.add_argument("--wandb_project",  type=str, default="KGCL")
    parser.add_argument("--wandb_entity",   type=str, default=None)
    parser.add_argument("--wandb_offline",  action="store_true")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser


def build_model(args):
    stats = compute_training_statistics(
        csv_path=args.train_csv,
        mask_dir=args.mask_dir,
        vector_dir=args.vector_dir,
        phase="finetune",
        max_pos_weight=args.max_pos_weight,
        task_mode=args.task_mode,
    )
    out_indices = _get_out_indices(args.backbone_name)
    return FinetuneModule(
        backbone_name=args.backbone_name,
        pretrained=args.pretrained,
        pretrained_phase1_ckpt=None,
        clue_names=CLUES_NAMES,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        num_clues=len(CLUES_NAMES),
        num_chaos=2,
        lambda_diag=args.lambda_diag,
        lambda_clue=args.lambda_clue,
        lambda_chaos=args.lambda_chaos,
        lambda_align=args.lambda_align,
        task_mode=args.task_mode,
        clue_pos_weight=stats["clue_pos_weight"],
        area_pos_weight=stats["area_pos_weight"],
        out_indices=out_indices,
    )


def build_checkpoint_dir(args):
    return os.path.join(BASE_DIR, "checkpoints", "end2end", args.task_mode, args.backbone_name)


def run(args):
    seed_everything(args.seed)

    print("=" * 80)
    print("ISIC Multi-Task Training: END-TO-END")
    print("=" * 80)
    print(f"Backbone:        {args.backbone_name}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Max Epochs:      {args.max_epochs}")
    print(f"Learning Rate:   {args.learning_rate}")
    print(f"Crop Size:       {args.crop_size}")
    print(f"Task Mode:       {args.task_mode}")
    print(f"Lambda Diag:     {args.lambda_diag}")
    if args.task_mode != "diag_only":
        print(f"Lambda Clue:     {args.lambda_clue}")
        print(f"Lambda Chaos:    {args.lambda_chaos}")
        print(f"Lambda Align:    {args.lambda_align}")

    datamodule = build_datamodule(args, phase="finetune")
    model = build_model(args)

    ckpt_dir = build_checkpoint_dir(args)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoint Dir:  {ckpt_dir}")

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val_diag_acc",
            dirpath=ckpt_dir,
            save_last=True,
            mode="max",
            save_top_k=3,
            filename="end2end-{epoch:02d}-{val_loss:.4f}-{val_diag_acc:.4f}",
        ),
        EarlyStopping(
            monitor="val_diag_acc",
            min_delta=0.001,
            patience=10,
            verbose=True,
            mode="max",
        ),
    ]

    logger = build_logger(
        args,
        phase="end2end",
        ckpt_dir=ckpt_dir,
        run_suffix=args.backbone_name,
    )
    logger.log_hyperparams(vars(args))

    trainer_kwargs = dict(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.gpus,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=True,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    if args.quick_test:
        trainer_kwargs["limit_train_batches"] = 10
        trainer_kwargs["limit_val_batches"] = 5
        trainer_kwargs["limit_test_batches"] = 5
        print("Quick test mode enabled.")

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule)

    print("\nRunning test with best checkpoint...")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        best_ckpt = torch.load(best_model_path, map_location="cpu")
        best_pth_path = os.path.join(ckpt_dir, "best.ckpt")
        torch.save(best_ckpt["state_dict"], best_pth_path)
        print(f"Exported best model weights to: {best_pth_path}")
    else:
        print("No best checkpoint recorded.")

    print("Training completed.")


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
