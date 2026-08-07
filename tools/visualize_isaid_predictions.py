"""Visualize semantic iSAID validation predictions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasam.datasets.industrial import ISAIDSemanticDataset  # noqa: E402
from adasam.models import LabelEfficientSAM, LabelEfficientUNet  # noqa: E402

COLORS = np.array([
    [0, 0, 0], [220, 20, 60], [255, 215, 0], [30, 144, 255],
    [160, 82, 45], [34, 139, 34], [0, 206, 209], [148, 0, 211],
    [255, 140, 0], [255, 105, 180], [0, 191, 255], [128, 0, 0],
    [128, 128, 0], [0, 128, 128], [128, 0, 128], [70, 130, 180],
], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize iSAID semantic segmentation")
    p.add_argument("--data-root", required=True)
    p.add_argument("--models", nargs="+", choices=["unet", "mobilesam", "ours"], required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--mobile-sam-checkpoint", default="weights/mobile_sam.pt")
    p.add_argument("--image-size", type=int, default=800)
    p.add_argument("--sam-image-size", type=int, default=800)
    p.add_argument("--decoder-dim", type=int, default=96)
    p.add_argument("--fusion-version", choices=["hierarchical", "global", "image_conditioned", "scsr"], default="image_conditioned")
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def colorize(mask: np.ndarray) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(COLORS))
    output[valid] = COLORS[mask[valid]]
    return output


def build_model(name: str, checkpoint: Path, args: argparse.Namespace, device: torch.device):
    if name == "unet":
        model = LabelEfficientUNet(ISAIDSemanticDataset.NUM_CLASSES, 32).to(device)
    else:
        model = LabelEfficientSAM.build(
            Path(args.mobile_sam_checkpoint), ISAIDSemanticDataset.NUM_CLASSES,
            img_size=args.sam_image_size, device=device, decoder_dim=args.decoder_dim,
            prompt_version="v2" if name == "ours" else "none", num_prompt=8,
            use_cat_adapter=False, feature_scales="p3_p4_embedding",
            fusion_version=args.fusion_version,
        )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if len(args.models) != len(args.checkpoints):
        raise ValueError("--models and --checkpoints must have the same length")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    for folder in ("panels", "predictions", "errors", "probabilities"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    dataset = ISAIDSemanticDataset(args.data_root, "val", args.image_size)
    indices = random.Random(args.seed).sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)
    models = {n: build_model(n, Path(c), args, device) for n, c in zip(args.models, args.checkpoints)}
    import matplotlib.pyplot as plt
    summary = {"indices": indices, "samples": [], "classes": ISAIDSemanticDataset.CLASS_NAMES}
    for offset, batch in enumerate(loader):
        image = batch["image"].to(device)
        sample_id = batch["id"][0]
        gt = batch["mask"][0].numpy()
        original = (batch["image"][0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        predictions = {}
        for name, model in models.items():
            with torch.no_grad():
                logits = model(image)
                prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = prob.argmax(0).astype(np.uint8)
            predictions[name] = pred
            Image.fromarray(colorize(pred)).save(output / "predictions" / f"{sample_id}_{name}.png")
            Image.fromarray((prob.max(0) * 255).astype(np.uint8)).save(output / "probabilities" / f"{sample_id}_{name}_confidence.png")
            error = np.zeros((*pred.shape, 3), dtype=np.uint8)
            valid = gt != ISAIDSemanticDataset.IGNORE_INDEX
            error[valid & (pred == gt)] = [40, 180, 70]
            error[valid & (pred != gt)] = [220, 50, 50]
            Image.fromarray(error).save(output / "errors" / f"{sample_id}_{name}.png")
        Image.fromarray(original).save(output / "predictions" / f"{sample_id}_image.png")
        Image.fromarray(colorize(gt)).save(output / "predictions" / f"{sample_id}_ground_truth.png")
        visuals = [original / 255.0, colorize(gt) / 255.0] + [colorize(predictions[n]) / 255.0 for n in args.models]
        titles = ["Image", "Ground truth"] + list(args.models)
        fig, axes = plt.subplots(1, len(visuals), figsize=(5 * len(visuals), 5))
        axes = np.atleast_1d(axes)
        for ax, title, visual in zip(axes, titles, visuals):
            ax.imshow(visual); ax.set_title(title); ax.axis("off")
        fig.suptitle(sample_id); fig.tight_layout()
        fig.savefig(output / "panels" / f"{sample_id}.png", dpi=160); plt.close(fig)
        summary["samples"].append({"id": sample_id, "index": indices[offset]})
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
