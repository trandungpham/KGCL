import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KGCL_ROOT = PROJECT_ROOT / "KGCL"
for path in (PROJECT_ROOT, KGCL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from KGCL.datasets.constants import CHAOS_LABELS, CLUE_NAMES, IDX_TO_DIAGNOSIS
from KGCL.datasets.transforms import DataTransforms
from KGCL.models.kgcl.segmenatation_module import SpatialClueAlignment

"""
This script runs image-only inference for a trained spatial clue model checkpoint. It generates:
- Diagnosis prediction and probabilities
- CHAOS label probabilities
- Clue segmentation masks, heatmaps, and overlays

python3 Inference/run_inference.py \
  --checkpoint_path KGCL/checkpoints/joint_spatial_clue/YOUR_RUN/last.ckpt \
  --image_path Annotated_data/Images/ISIC_0000000.jpg \
  --img_encoder vit_base
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Run image-only inference for the spatial clue model.")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Path to a trained spatial clue checkpoint.")
    parser.add_argument("--image_path", type=Path, required=True, help="Path to the input image.")
    parser.add_argument("--img_encoder", type=str, default="vit_base", help="Image encoder used by the checkpoint.")
    parser.add_argument("--crop_size", type=int, default=224, help="Evaluation crop size.")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "Inference" / "outputs", help="Directory to save masks and overlays.")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Threshold used to binarize clue masks.")
    return parser.parse_args()


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    image = (image * 0.5) + 0.5
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.4) -> np.ndarray:
    overlay = image.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    active = mask > 0
    overlay[active] = (1 - alpha) * overlay[active] + alpha * color_arr
    return overlay.astype(np.uint8)


def load_model(checkpoint_path: Path, img_encoder: str, device: torch.device) -> SpatialClueAlignment:
    model = SpatialClueAlignment.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
        img_encoder=img_encoder,
        freeze_bert=True,
    )
    model.eval()
    model.to(device)
    return model


def run_image_only_inference(model: SpatialClueAlignment, image_tensor: torch.Tensor):
    with torch.no_grad():
        img_feat_q, patch_feat_q = model.img_encoder_q(image_tensor)
        img_feat_raw = model.get_global_image_feature_for_cls(img_feat_q)
        patch_feat_map = model.to_spatial_map(patch_feat_q)

        diagnosis_logits = model.diagnosis_head(img_feat_raw)
        chaos_logits = model.chaos_head(img_feat_raw)
        clue_seg_logits = model.clue_seg_head(patch_feat_map, output_size=image_tensor.shape[-2:])

    diagnosis_probs = F.softmax(diagnosis_logits, dim=1).squeeze(0).cpu().numpy()
    diagnosis_idx = int(diagnosis_probs.argmax())

    chaos_probs = torch.sigmoid(chaos_logits).squeeze(0).cpu().numpy()
    clue_probs = torch.sigmoid(clue_seg_logits).squeeze(0).cpu().numpy()
    return diagnosis_idx, diagnosis_probs, chaos_probs, clue_probs


def main():
    args = parse_args()

    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = DataTransforms(is_train=False, crop_size=args.crop_size)
    model = load_model(args.checkpoint_path, args.img_encoder, device)

    pil_image = Image.open(args.image_path).convert("RGB")
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    base_image = denormalize_image(image_tensor)

    diagnosis_idx, diagnosis_probs, chaos_probs, clue_probs = run_image_only_inference(model, image_tensor)

    image_output_dir = args.output_dir / args.image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    colors = [
        (255, 0, 124),
        (221, 255, 51),
        (52, 209, 183),
        (255, 153, 0),
        (0, 168, 255),
        (255, 255, 255),
        (255, 80, 80),
        (147, 112, 219),
        (0, 204, 102),
    ]

    clue_results = []
    for idx, (clue_key, clue_prob) in enumerate(zip(CLUE_NAMES.keys(), clue_probs)):
        clue_name = CLUE_NAMES[clue_key]
        clue_mask = (clue_prob >= args.mask_threshold).astype(np.uint8)
        overlay = overlay_mask(base_image, clue_mask, colors[idx % len(colors)])

        mask_path = image_output_dir / f"{idx + 1:02d}_{clue_key}_mask.png"
        heatmap_path = image_output_dir / f"{idx + 1:02d}_{clue_key}_heatmap.png"
        overlay_path = image_output_dir / f"{idx + 1:02d}_{clue_key}_overlay.png"

        cv2.imwrite(str(mask_path), clue_mask * 255)
        cv2.imwrite(str(heatmap_path), np.uint8(clue_prob * 255))
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        clue_results.append(
            {
                "clue_key": clue_key,
                "clue_name": clue_name,
                "mean_probability": float(clue_prob.mean()),
                "max_probability": float(clue_prob.max()),
                "foreground_pixels": int(clue_mask.sum()),
                "mask_path": str(mask_path),
                "heatmap_path": str(heatmap_path),
                "overlay_path": str(overlay_path),
            }
        )

    summary = {
        "image_path": str(args.image_path),
        "checkpoint_path": str(args.checkpoint_path),
        "diagnosis_prediction": IDX_TO_DIAGNOSIS[diagnosis_idx],
        "diagnosis_probabilities": {
            IDX_TO_DIAGNOSIS[idx]: float(prob) for idx, prob in enumerate(diagnosis_probs)
        },
        "chaos_probabilities": {
            label: float(prob) for label, prob in zip(CHAOS_LABELS, chaos_probs)
        },
        "clues": clue_results,
    }

    summary_path = image_output_dir / "prediction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Diagnosis: {summary['diagnosis_prediction']}")
    for label, prob in summary["diagnosis_probabilities"].items():
        print(f"  {label}: {prob:.4f}")

    print("Chaos:")
    for label, prob in summary["chaos_probabilities"].items():
        status = "positive" if prob >= args.mask_threshold else "negative"
        print(f"  {label}: {prob:.4f} ({status})")

    print("Clue segmentations saved to:")
    for clue in clue_results:
        print(f"  {clue['clue_name']}: {clue['overlay_path']}")

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
