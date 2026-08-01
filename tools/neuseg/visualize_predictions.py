"""Save representative NEU_Seg predictions and ground-truth comparisons."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "neuseg"))
from adasam.datasets import NEUSegDataset


def overlay(ax, image, mask, title, color):
    ax.imshow(image)
    shown = np.ma.masked_where(mask == 0, mask)
    ax.imshow(shown, cmap=color, alpha=0.55, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_panel(image, target, prediction, class_id, path, name, iou):
    fg_target = target == class_id
    fg_prediction = prediction == class_id
    error = np.zeros((*target.shape, 3), dtype=np.float32)
    error[fg_target & fg_prediction] = (0.1, 0.75, 0.2)
    error[~fg_target & fg_prediction] = (0.9, 0.1, 0.1)
    error[fg_target & ~fg_prediction] = (1.0, 0.7, 0.0)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    overlay(axes[0], image, fg_target, "Ground truth", "Greens")
    overlay(axes[1], image, fg_prediction, "Prediction", "Blues")
    axes[2].imshow(error)
    axes[2].set_title("Green=TP Red=FP Orange=FN", fontsize=9)
    axes[2].axis("off")
    axes[3].imshow(image)
    axes[3].contour(fg_target, levels=[0.5], colors="lime", linewidths=0.8)
    axes[3].contour(fg_prediction, levels=[0.5], colors="red", linewidths=0.8)
    axes[3].set_title("GT lime / Pred red", fontsize=9)
    axes[3].axis("off")
    fig.suptitle(f"{name} | IoU={iou:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def stage2_predictions(args, dataset):
    from train_stage2 import Stage2Trainer
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    cfg.setdefault("data", {})["data_root"] = str(args.data_root)
    cfg.setdefault("train", {})["device"] = args.device
    cfg.setdefault("train", {})["val_samples"] = 10**9
    if checkpoint.get("class_thresholds"):
        cfg.setdefault("eval", {})["class_thresholds"] = checkpoint["class_thresholds"]
    stage1 = args.stage1_ckpt or checkpoint.get("stage1_checkpoint")
    if not stage1:
        raise ValueError("Stage2 requires --stage1-ckpt or an embedded stage1 path")
    trainer_args = argparse.Namespace(stage1_ckpt=stage1, steps=None, epochs=None, episodes=None,
                                      support_shot=None, seed=None, device=args.device,
                                      data_root=str(args.data_root), output_dir=None,
                                      val_samples=None)
    trainer = Stage2Trainer(cfg, trainer_args)
    trainer.model.load_state_dict(checkpoint["model"], strict=False)
    trainer.model.eval()
    supports = trainer.support_cache(use_all_support=True)
    records = {c: [] for c in (1, 2, 3)}
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            features, _ = trainer._embed(sample["image"])
            probs = []
            for class_id in (1, 2, 3):
                output = trainer.model(features, *supports[class_id])
                p = trainer.model.semantic_probability(output)
                p = F.interpolate(p[:, None], sample["image_size"], mode="bilinear", align_corners=False)[0, 0].cpu()
                probs.append(p)
            probs = torch.stack(probs)
            thresholds = torch.tensor(trainer.class_thresholds)[:, None, None]
            adjusted = probs / thresholds
            confidence, cls = adjusted.max(0)
            pred = cls + 1
            pred[confidence < 1] = 0
            target = sample["masks"].squeeze(0).cpu()
            for class_id in (1, 2, 3):
                a, b = pred == class_id, target == class_id
                # An absent GT class has no diagnostic IoU for that class and
                # must not dominate the representative failure selection.
                if b.any():
                    iou = float((a & b).sum() / (a | b).sum().clamp_min(1))
                    records[class_id].append((iou, idx, pred.numpy(), target.numpy(), sample["image"].permute(1, 2, 0).numpy()))
    return records


def unet_predictions(args, dataset):
    sys.path.insert(0, str(ROOT / "tools" / "U-Net"))
    from train_neu_seg import UNet
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = UNet(3, 4, bilinear=True).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    records = {c: [] for c in (1, 2, 3)}
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            pred = model(sample["image"].unsqueeze(0).to(args.device)).argmax(1)[0].cpu()
            target = sample["masks"].squeeze(0).cpu()
            image = sample["image"].permute(1, 2, 0).numpy()
            for class_id in (1, 2, 3):
                a, b = pred == class_id, target == class_id
                if b.any():
                    iou = float((a & b).sum() / (a | b).sum().clamp_min(1))
                    records[class_id].append((iou, idx, pred.numpy(), target.numpy(), image))
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("stage2", "unet"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stage1-ckpt")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"
    dataset = NEUSegDataset(Path(args.data_root), split="test")
    records = stage2_predictions(args, dataset) if args.method == "stage2" else unet_predictions(args, dataset)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for class_id, values in records.items():
        values.sort(key=lambda x: x[0])
        class_name = dataset.CLASS_NAMES[class_id]
        summary[class_name] = []
        if not values:
            continue
        # Show failure, lower-middle, upper-middle, and best performance. This
        # makes the four figures representative rather than four duplicates.
        selected_positions = sorted({round(q * (len(values) - 1)) for q in (0.0, 1 / 3, 2 / 3, 1.0)})
        selected = [values[position] for position in selected_positions]
        for rank, (iou, idx, pred, target, image) in enumerate(selected, 1):
            path = output / f"{class_name.lower()}_{rank}_{dataset.sample_names[idx]}.png"
            save_panel(image, target, pred, class_id, path, dataset.sample_names[idx], iou)
            summary[class_name].append({"rank": rank, "sample_id": dataset.sample_names[idx], "iou": iou, "file": str(path)})
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
