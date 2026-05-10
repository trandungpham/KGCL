"""
Run all transformer-based backbone experiments for the ISIC multi-task framework.

All runs use --use_agentic_aux (per-concept PerConceptAuxiliaryHub) and
--use_weighted_sampler, matching the sweep.yaml command section.

Transformer backbones covered:
  - swin_base_patch4_window7_224   (from sweep.yaml)
  - swin_small_patch4_window7_224
  - swin_large_patch4_window7_224_22k
  - vit_base_patch16_224
  - deit3_base_patch16_224

Hyperparameter grid is derived from sweep.yaml values.
Each backbone is run with a representative set of configs, not exhaustive grid.

Usage:
    # Dry-run (print commands without executing)
    python run_transformer_experiments.py --dry_run

    # Run all experiments with W&B logging
    python run_transformer_experiments.py

    # Run specific backbones only
    python run_transformer_experiments.py --backbones swin_base_patch4_window7_224 vit_base_patch16_224

    # Use CSV logger instead of W&B
    python run_transformer_experiments.py --logger csv

    # Limit epochs for quick testing
    python run_transformer_experiments.py --max_epochs 5 --quick_test
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Transformer backbones to evaluate
# swin_base_patch4_window7_224 is taken from sweep.yaml; others are common
# timm transformer variants that share the same 4-stage feature pyramid and
# are compatible with the _FOUR_STAGE_BACKBONES routing in train_multitask.py.
# ---------------------------------------------------------------------------
TRANSFORMER_BACKBONES = [
    "swin_base_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "swin_large_patch4_window7_224_22k",
    "vit_base_patch16_224",
    "deit3_base_patch16_224",
]

# ---------------------------------------------------------------------------
# Representative hyperparameter configurations derived from sweep.yaml.
# Covers low/mid/high LR, diag-heavy vs clue-heavy loss weighting, and two
# auxiliary module capacity points (4 agents / 2 agents).
#
# Columns: config_id, lr, wd, bs,
#          λ_diag, λ_clue, λ_chaos, λ_align,
#          clue_scale, diag_scale, max_pos_weight,
#          num_aux_agents, aux_hidden_dim, λ_confidence
# ---------------------------------------------------------------------------
CONFIGS = [
    ("low_lr",    1e-5, 1e-4, 16, 1.0,  1.0, 1.0, 0.1, 1.0,  0.5,  15.0, 4, 128, 0.1),
    ("mid_lr",    3e-5, 1e-4, 16, 1.0,  2.0, 1.0, 0.1, 1.0,  0.5,  15.0, 4, 128, 0.1),
    ("high_lr",   1e-4, 1e-4, 16, 1.0,  2.0, 0.5, 0.3, 1.0,  0.5,  15.0, 4, 256, 0.1),
    ("diag_heavy",3e-5, 5e-4,  8, 1.5,  1.0, 0.5, 0.0, 0.75, 0.25, 10.0, 4, 128, 0.05),
    ("clue_heavy",3e-5, 1e-5, 16, 0.75, 3.0, 1.0, 0.3, 1.25, 0.75, 20.0, 4, 128, 0.2),
    ("aux_light", 3e-5, 1e-4, 16, 1.0,  2.0, 1.0, 0.1, 1.0,  0.5,  15.0, 2, 128, 0.1),
]


def build_command(
    train_script: str,
    backbone: str,
    config: tuple,
    max_epochs: int,
    logger: str,
    quick_test: bool,
    wandb_project: str,
    wandb_entity,
    seed: int,
    num_workers: int,
) -> list:
    (cfg_id, lr, wd, bs,
     l_diag, l_clue, l_chaos, l_align,
     clue_scale, diag_scale, max_pw,
     n_agents, hidden_dim, l_conf) = config

    run_name = f"transformer-{backbone}-{cfg_id}"

    cmd = [
        sys.executable, train_script,
        "--phase",                    "all",
        "--backbone_name",            backbone,
        "--learning_rate",            str(lr),
        "--weight_decay",             str(wd),
        "--batch_size",               str(bs),
        "--max_epochs",               str(max_epochs),
        "--lambda_diag",              str(l_diag),
        "--lambda_clue",              str(l_clue),
        "--lambda_chaos",             str(l_chaos),
        "--lambda_align",             str(l_align),
        "--sampler_clue_weight_scale",str(clue_scale),
        "--sampler_diag_weight_scale",str(diag_scale),
        "--max_pos_weight",           str(max_pw),
        # per-concept auxiliary confidence module (always enabled)
        "--use_agentic_aux",
        "--num_aux_agents",           str(n_agents),
        "--aux_agent_hidden_dim",     str(hidden_dim),
        "--lambda_confidence",        str(l_conf),
        "--use_weighted_sampler",
        "--seed",                     str(seed),
        "--num_workers",              str(num_workers),
        "--auto_finetune_task_mode",  "multitask",
        "--logger",                   logger,
        "--wandb_project",            wandb_project,
        "--wandb_run_name",           run_name,
    ]

    if wandb_entity:
        cmd += ["--wandb_entity", wandb_entity]
    if quick_test:
        cmd.append("--quick_test")

    return cmd


def run_experiments(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "CGCL", "train_multitask.py")

    if not os.path.isfile(train_script):
        sys.exit(f"Training script not found: {train_script}")

    backbones = args.backbones or TRANSFORMER_BACKBONES
    results_path = os.path.join(
        script_dir,
        f"transformer_results_{datetime.datetime.now():%Y%m%d_%H%M%S}.jsonl",
    )

    total = len(backbones) * len(CONFIGS)
    print(f"Transformer backbone sweep: {len(backbones)} backbone(s) × {len(CONFIGS)} config(s) = {total} run(s)")
    print(f"Results log: {results_path}\n")

    run_idx = 0
    for backbone in backbones:
        for config in CONFIGS:
            run_idx += 1
            cfg_id = config[0]
            label = f"[{run_idx}/{total}] {backbone} / {cfg_id}"

            cmd = build_command(
                train_script=train_script,
                backbone=backbone,
                config=config,
                max_epochs=args.max_epochs,
                logger=args.logger,
                quick_test=args.quick_test,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                seed=args.seed,
                num_workers=args.num_workers,
            )

            if args.dry_run:
                print(f"[DRY-RUN] {label}")
                print("  " + " ".join(cmd))
                print()
                continue

            print(f"\n{'='*70}")
            print(f"Starting: {label}")
            print(f"  cmd: {' '.join(cmd)}")
            print(f"{'='*70}")

            start = time.time()
            result = subprocess.run(
                cmd,
                cwd=os.path.join(script_dir, "CGCL"),
            )
            elapsed = time.time() - start

            record = {
                "backbone": backbone,
                "config_id": cfg_id,
                "returncode": result.returncode,
                "elapsed_s": round(elapsed, 1),
                "timestamp": datetime.datetime.now().isoformat(),
            }
            with open(results_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")

            status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
            print(f"\n{label} → {status} in {elapsed/60:.1f} min")

            if result.returncode != 0 and args.stop_on_failure:
                sys.exit(f"Stopping after failure: {label}")

    if not args.dry_run:
        print(f"\n{'='*70}")
        print(f"All {run_idx} run(s) complete. See {results_path} for a summary.")


def main():
    parser = argparse.ArgumentParser(
        description="Run all transformer-based backbone experiments"
    )
    parser.add_argument(
        "--backbones", nargs="+", default=None,
        metavar="BACKBONE",
        help=f"Subset of backbones to run (default: all {len(TRANSFORMER_BACKBONES)})",
    )
    parser.add_argument("--max_epochs",  type=int,   default=100)
    parser.add_argument("--logger",      type=str,   default="wandb", choices=["wandb", "csv"])
    parser.add_argument("--wandb_project", type=str, default="KGCL-transformers")
    parser.add_argument("--wandb_entity",  type=str, default=None)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--quick_test",  action="store_true",
                        help="Pass --quick_test to each run (limited batches)")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--stop_on_failure", action="store_true",
                        help="Abort the sweep if any run exits with a non-zero code")
    args = parser.parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
