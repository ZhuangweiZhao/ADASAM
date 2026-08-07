"""Visualize LoveDA validation predictions from controlled fusion checkpoints."""

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

from adasam.datasets.industrial import LoveDASemanticDataset  # noqa: E402
from adasam.models import LabelEfficientSAM  # noqa: E402

COLORS = np.array([
    [0, 0, 0], [220, 20, 60], [255, 215, 0], [30, 144, 255],
    [160, 82, 45], [34, 139, 34], [0, 206, 209],
], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize LoveDA segmentation predictions")
    p.add_argument("--data-root", required=True)
    p.add_argument("--models", nargs="+", choices=["hierarchical", "global", "image_conditioned", "scsr"], required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--mobile-sam-checkpoint", default="weights/mobile_sam.pt")
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--sam-image-size", type=int, default=1024)
    p.add_argument("--decoder-dim", type=int, default=96)
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def colorize(mask: np.ndarray) -> np.ndarray:
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(COLORS))
    result[valid] = COLORS[mask[valid]]
    return result


def boundary_map(mask: np.ndarray) -> np.ndarray:
    valid = mask != LoveDASemanticDataset.IGNORE_INDEX
    edge = np.zeros_like(valid, dtype=bool)
    edge[:, 1:] |= valid[:, 1:] & valid[:, :-1] & (mask[:, 1:] != mask[:, :-1])
    edge[1:, :] |= valid[1:, :] & valid[:-1, :] & (mask[1:, :] != mask[:-1, :])
    return edge


def load_model(name: str, checkpoint: Path, args: argparse.Namespace, device: torch.device):
    model = LabelEfficientSAM.build(
        resolve(args.mobile_sam_checkpoint),
        num_classes=LoveDASemanticDataset.NUM_CLASSES,
        img_size=args.sam_image_size,
        device=device,
        decoder_dim=args.decoder_dim,
        prompt_version="none",
        use_cat_adapter=False,
        feature_scales="p3_p4_embedding",
        fusion_version=name,
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"{name}: checkpoint compatibility missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if len(args.models) != len(args.checkpoints):
        raise ValueError("--models and --checkpoints must have the same length")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output = resolve(args.output_dir)
    for folder in ("panels", "predictions", "errors", "weights"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    dataset = LoveDASemanticDataset(resolve(args.data_root), "val", args.image_size)
    count = min(args.num_samples, len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), count)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)
    models = {name: load_model(name, resolve(path), args, device) for name, path in zip(args.models, args.checkpoints)}
    cmap = ListedColormap(COLORS / 255.0)
    for folder in ("features", "confidence", "class_probabilities", "boundaries"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    confusion = {name: np.zeros((LoveDASemanticDataset.NUM_CLASSES, LoveDASemanticDataset.NUM_CLASSES), dtype=np.int64) for name in args.models}
    summary = {"seed": args.seed, "indices": indices, "samples": []}
    import matplotlib.pyplot as plt

    for batch in loader:
        image = batch["image"].to(device)
        sample_id = batch["id"][0]
        gt = batch["mask"][0].numpy()
        predictions = {}
        routing = {}
        probabilities = {}
        feature_outputs = {}
        with torch.no_grad():
            for name, model in models.items():
                logits = model(image)
                probabilities[name] = torch.softmax(logits, dim=1)[0].cpu().numpy()
                predictions[name] = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                with torch.no_grad():
                    raw_features = model.backbone(model._preprocess(image))
                feature_outputs[name] = raw_features
                last = model.decoder.last_routing
                if last is not None:
                    weights = last["weights"][0].mean(dim=(-2, -1)).cpu().tolist()
                    routing[name] = {"P3": weights[0], "P4": weights[1], "embedding": weights[2]}
        original = (image[0].cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(original).save(output / "predictions" / f"{sample_id}_image.png")
        Image.fromarray(colorize(gt)).save(output / "predictions" / f"{sample_id}_ground_truth.png")
        for name, pred in predictions.items():
            Image.fromarray(colorize(pred)).save(output / "predictions" / f"{sample_id}_{name}.png")
            error = np.zeros((*pred.shape, 3), dtype=np.uint8)
            valid = gt != LoveDASemanticDataset.IGNORE_INDEX
            error[valid & (pred == gt)] = [40, 180, 70]
            error[valid & (pred != gt)] = [220, 50, 50]
            Image.fromarray(error).save(output / "errors" / f"{sample_id}_{name}.png")
            valid = gt != LoveDASemanticDataset.IGNORE_INDEX
            flat_gt, flat_pred = gt[valid], pred[valid]
            for target_class in range(LoveDASemanticDataset.NUM_CLASSES):
                for predicted_class in range(LoveDASemanticDataset.NUM_CLASSES):
                    confusion[name][target_class, predicted_class] += int(((flat_gt == target_class) & (flat_pred == predicted_class)).sum())
            confidence = probabilities[name].max(axis=0)
            uncertainty = 1.0 - confidence
            Image.fromarray((uncertainty.clip(0, 1) * 255).astype(np.uint8)).save(output / "confidence" / f"{sample_id}_{name}_uncertainty.png")
            Image.fromarray((boundary_map(gt) * 255).astype(np.uint8)).save(output / "boundaries" / f"{sample_id}_ground_truth.png")
            Image.fromarray((boundary_map(pred) * 255).astype(np.uint8)).save(output / "boundaries" / f"{sample_id}_{name}.png")
            for class_id in (1, 2, 3):
                Image.fromarray((probabilities[name][class_id].clip(0, 1) * 255).astype(np.uint8)).save(
                    output / "class_probabilities" / f"{sample_id}_{name}_{LoveDASemanticDataset.CLASS_NAMES[class_id]}.png"
                )
        for name, raw_features in feature_outputs.items():
            for feature_name in ("P3", "P4", "embedding"):
                if feature_name not in raw_features:
                    continue
                activation = raw_features[feature_name].abs().mean(1)[0]
                activation = (activation - activation.min()) / (activation.max() - activation.min()).clamp_min(1e-6)
                activation = torch.nn.functional.interpolate(activation[None, None], size=gt.shape, mode="bilinear", align_corners=False)[0, 0]
                Image.fromarray((activation.cpu().numpy() * 255).astype(np.uint8)).save(
                    output / "features" / f"{sample_id}_{name}_{feature_name}.png"
                )
        cols = ["Original", "Ground truth", *args.models, "Image-conditioned error"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        visuals = [original / 255.0, colorize(gt) / 255.0, *[colorize(predictions[n]) / 255.0 for n in args.models], colorize((predictions.get("image_conditioned", predictions[args.models[0]]) != gt).astype(np.uint8)) / 255.0]
        for ax, title, visual in zip(axes.flat, cols, visuals):
            ax.imshow(visual, cmap=cmap if title != "Original" else None)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(sample_id)
        fig.tight_layout()
        fig.savefig(output / "panels" / f"{sample_id}.png", dpi=160)
        plt.close(fig)
        summary["samples"].append({"id": sample_id, "index": indices[len(summary["samples"])], "routing": routing})
    summary["confusion_matrices"] = {name: matrix.tolist() for name, matrix in confusion.items()}
    summary["class_iou"] = {}
    for name, matrix in confusion.items():
        class_scores = {}
        for class_id, class_name in enumerate(LoveDASemanticDataset.CLASS_NAMES):
            denominator = matrix[class_id].sum() + matrix[:, class_id].sum() - matrix[class_id, class_id]
            class_scores[class_name] = float(matrix[class_id, class_id] / denominator) if denominator else 0.0
        summary["class_iou"][name] = class_scores
    (output / "weights" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
