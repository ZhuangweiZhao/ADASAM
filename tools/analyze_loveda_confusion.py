"""Full-set LoveDA confusion analysis and targeted error visualization."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasam.datasets.industrial import LoveDASemanticDataset  # noqa: E402
from adasam.models import LabelEfficientSAM  # noqa: E402

CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren", "Forest", "Agriculture"]
COLORS = np.array([
    [0, 0, 0], [220, 20, 60], [255, 215, 0], [30, 144, 255],
    [160, 82, 45], [34, 139, 34], [0, 206, 209],
], dtype=np.uint8)
PAIRS = [(5, 6), (3, 0), (3, 2), (6, 5), (6, 4), (2, 0)]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", nargs="+", required=True,
                   help="checkpoint paths; labels identify output subfolders")
    p.add_argument("--label", nargs="+", default=None,
                   help="checkpoint labels, same count as --checkpoint")
    p.add_argument("--decoder-version", nargs="+", default=None,
                   choices=["lightweight", "boundary_aux", "boundary"],
                   help="decoder type for each checkpoint")
    p.add_argument("--mobile-sam-checkpoint", default="weights/mobile_sam.pt")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--sam-image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(COLORS))
    out[valid] = COLORS[mask[valid]]
    return out


def load_model(checkpoint: Path, device: torch.device) -> LabelEfficientSAM:
    model = LabelEfficientSAM.build(
        resolve("weights/mobile_sam.pt"),  # replaced below for explicit path
        num_classes=7, img_size=512, device=device, decoder_dim=96,
        prompt_version="none", use_cat_adapter=False,
        feature_scales="p3_p4_embedding", fusion_version="sum",
        decoder_version="lightweight",
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def build_model(checkpoint: Path, mobile_checkpoint: Path, sam_size: int,
                device: torch.device) -> LabelEfficientSAM:
    model = LabelEfficientSAM.build(
        mobile_checkpoint, num_classes=7, img_size=sam_size, device=device,
        decoder_dim=96, prompt_version="none", use_cat_adapter=False,
        feature_scales="p3_p4_embedding", fusion_version="sum",
        decoder_version="lightweight",
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def save_heatmap(matrix: np.ndarray, path: Path, title: str, normalize: bool) -> None:
    values = matrix.astype(np.float64)
    if normalize:
        values = values / np.maximum(values.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(values, cmap="magma", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(7), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(7), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title(title)
    for i in range(7):
        for j in range(7):
            text = f"{values[i, j] * 100:.1f}%" if normalize else f"{int(values[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] > values.max() * 0.45 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_targeted(sample: dict, output: Path, label: str, pair_name: str) -> None:
    gt, pred, image = sample["gt"], sample["pred"], sample["image"]
    pairs = sample["pair_scores"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(image); axes[0, 0].set_title("Image")
    axes[0, 1].imshow(colorize(gt)); axes[0, 1].set_title("Ground truth")
    axes[0, 2].imshow(colorize(pred)); axes[0, 2].set_title("Prediction")
    error = np.zeros((*gt.shape, 3), dtype=np.uint8)
    valid = gt != 255
    error[valid & (gt == pred)] = [40, 180, 70]
    error[valid & (gt != pred)] = [220, 50, 50]
    axes[0, 3].imshow(error); axes[0, 3].set_title("All errors")
    for ax in axes.flat: ax.axis("off")
    for ax, (src, dst) in zip(axes[1], PAIRS):
        mask = (gt == src) & (pred == dst)
        overlay = image.copy()
        overlay[mask] = np.array([255, 0, 255], dtype=np.uint8)
        ax.imshow(overlay)
        ax.set_title(f"{CLASS_NAMES[src]} -> {CLASS_NAMES[dst]}\n{pairs.get((src, dst), 0)} px")
        ax.axis("off")
    fig.suptitle(f"{label} | {sample['id']}")
    fig.tight_layout()
    fig.savefig(output / f"{pair_name}_{sample['id']}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    a = args()
    if a.label is None:
        labels = [Path(x).parent.name for x in a.checkpoint]
    else:
        labels = a.label
    if len(labels) != len(a.checkpoint):
        raise ValueError("--label and --checkpoint must have equal lengths")
    decoder_versions = a.decoder_version or ["lightweight"] * len(a.checkpoint)
    if len(decoder_versions) != len(a.checkpoint):
        raise ValueError("--decoder-version and --checkpoint must have equal lengths")
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    output = resolve(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = LoveDASemanticDataset(resolve(a.data_root), "val", a.image_size)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=device.type == "cuda")
    models = []
    for cp, decoder_version in zip(a.checkpoint, decoder_versions):
        model = LabelEfficientSAM.build(
            resolve(a.mobile_sam_checkpoint), num_classes=7, img_size=a.sam_image_size,
            device=device, decoder_dim=96, prompt_version="none", use_cat_adapter=False,
            feature_scales="p3_p4_embedding", fusion_version="sum",
            decoder_version=decoder_version,
        )
        payload = torch.load(resolve(cp), map_location=device, weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=False)
        model.eval()
        models.append(model)
    results = {}
    for label, model in zip(labels, models):
        matrix = np.zeros((7, 7), dtype=np.int64)
        top = {pair: [] for pair in PAIRS}
        with torch.no_grad():
            offset = 0
            for batch in loader:
                image_tensor = batch["image"].to(device)
                logits = model(image_tensor)
                pred_batch = logits.argmax(1).cpu().numpy()
                gt_batch = batch["mask"].numpy()
                for b, (gt, pred) in enumerate(zip(gt_batch, pred_batch)):
                    valid = gt != 255
                    flat = 7 * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
                    matrix += np.bincount(flat, minlength=49).reshape(7, 7)
                    pair_scores = {(s, d): int(((gt == s) & (pred == d)).sum()) for s, d in PAIRS}
                    sample_id = str(batch["id"][b])
                    image = (batch["image"][b].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
                    for pair, count in pair_scores.items():
                        if count:
                            top[pair].append((count, sample_id, gt.copy(), pred.copy(), image.copy(), pair_scores))
                offset += len(gt_batch)
        folder = output / label
        (folder / "topk").mkdir(parents=True, exist_ok=True)
        save_heatmap(matrix, folder / "confusion_counts.png", f"{label}: pixel counts", False)
        save_heatmap(matrix, folder / "confusion_recall_normalized.png", f"{label}: GT-normalized confusion", True)
        with (folder / "confusion_counts.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(["GT\\Pred", *CLASS_NAMES])
            writer.writerows([[CLASS_NAMES[i], *matrix[i].tolist()] for i in range(7)])
        rows = []
        for (src, dst), items in top.items():
            source_total = int(matrix[src].sum())
            error_total = int(matrix[src, dst])
            rows.append({"gt": CLASS_NAMES[src], "pred": CLASS_NAMES[dst],
                         "pixels": error_total, "gt_pixels": source_total,
                         "recall_error_percent": 100 * error_total / max(source_total, 1)})
            for rank, item in enumerate(sorted(items, reverse=True)[:a.top_k], 1):
                save_targeted({"id": f"{rank:02d}_{item[1]}", "gt": item[2], "pred": item[3],
                               "image": item[4], "pair_scores": item[5]}, folder / "topk", label,
                              f"{CLASS_NAMES[src]}_to_{CLASS_NAMES[dst]}")
        (folder / "targeted_confusions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        results[label] = {"matrix": matrix.tolist(), "targeted": rows}
    (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved={output} samples={len(dataset)}")


if __name__ == "__main__":
    main()
