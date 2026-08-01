"""Visualize outputs produced by ``run_label_ratio_benchmark.py``.

The script reads ``summary.json`` plus each ``neu_seg_ratio*_seed*/metrics.json``
directory and writes publication-friendly PNG figures and a CSV table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["background", "Inclusion", "Patch", "Scratch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize label-ratio benchmark results")
    parser.add_argument("--input-dir", default="runs/label_ratio_benchmark_phase1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="png")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def save_figure(fig: plt.Figure, output: Path, fmt: str) -> None:
    if fmt in {"png", "both"}:
        fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    if fmt in {"pdf", "both"}:
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_results(root: Path) -> tuple[dict, list[dict]]:
    summary_path = root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = summary.get("results", [])
    else:
        summary = {"protocol": {}}
        rows = []
    if not rows:
        for metrics_path in sorted(root.glob("neu_seg_ratio*_seed*/metrics.json")):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            match = re.search(r"ratio(\d+)", metrics_path.parent.name)
            if not match:
                continue
            test = metrics["test"]
            rows.append({
                "ratio": int(match.group(1)),
                "labeled_images": metrics["label_pool_samples"],
                "mIoU": test["mIoU"], "mIoU_fg": test["mIoU_fg"],
                "Dice": test["Dice"], "Dice_fg": test["Dice_fg"],
                "train_time_seconds": sum(x["seconds"] for x in metrics["history"]),
                "FPS": test["FPS"], "best_epoch": metrics["best_epoch"],
            })
    return summary, sorted(rows, key=lambda row: row["ratio"])


def load_histories(root: Path, rows: list[dict]) -> dict[int, dict]:
    histories = {}
    for row in rows:
        candidates = sorted(root.glob(f"neu_seg_ratio{row['ratio']}_seed*/metrics.json"))
        if candidates:
            histories[row["ratio"]] = json.loads(candidates[0].read_text(encoding="utf-8"))
    return histories


def plot_loss(histories: dict[int, dict], output: Path, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for ratio, metrics in histories.items():
        history = metrics.get("history", [])
        ax.plot([x["epoch"] for x in history], [x["mean_loss"] for x in history], marker="o", label=f"{ratio}%")
    ax.set(title="Training loss by label ratio", xlabel="Epoch", ylabel="CE + Dice loss")
    ax.grid(alpha=0.25); ax.legend(title="Labels")
    save_figure(fig, output / "loss_curves", fmt)


def plot_validation(histories: dict[int, dict], output: Path, fmt: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ratio, metrics in histories.items():
        history = metrics.get("history", [])
        epochs = [x["epoch"] for x in history]
        axes[0].plot(epochs, [x["validation"]["mIoU_fg"] for x in history], marker="o", label=f"{ratio}%")
        axes[1].plot(epochs, [x["validation"]["Dice_fg"] for x in history], marker="o", label=f"{ratio}%")
    axes[0].set(title="Validation foreground mIoU", xlabel="Epoch", ylabel="mIoU_fg")
    axes[1].set(title="Validation foreground Dice", xlabel="Epoch", ylabel="Dice_fg")
    for ax in axes:
        ax.grid(alpha=0.25); ax.legend(title="Labels")
    fig.tight_layout(); save_figure(fig, output / "validation_curves", fmt)


def plot_ratio_metrics(rows: list[dict], output: Path, fmt: str) -> None:
    ratios = [x["ratio"] for x in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for key, label in [("mIoU", "mIoU"), ("mIoU_fg", "mIoU_fg"), ("Dice_fg", "Dice_fg")]:
        axes[0].plot(ratios, [x[key] for x in rows], marker="o", label=label)
    axes[0].set(title="Accuracy vs annotation ratio", xlabel="Label ratio (%)", ylabel="Score")
    axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(ratios, [x["train_time_seconds"] for x in rows], marker="o", color="tab:orange")
    axes[1].set(title="Training time", xlabel="Label ratio (%)", ylabel="Seconds")
    axes[1].grid(alpha=0.25)
    axes[2].plot(ratios, [x["FPS"] for x in rows], marker="o", color="tab:green")
    axes[2].set(title="Inference throughput", xlabel="Label ratio (%)", ylabel="FPS")
    axes[2].grid(alpha=0.25)
    fig.tight_layout(); save_figure(fig, output / "ratio_metrics", fmt)


def plot_class_heatmaps(histories: dict[int, dict], rows: list[dict], output: Path, fmt: str) -> None:
    ratios = [x["ratio"] for x in rows]
    iou = np.array([histories[r]["test"]["per_class_IoU"] for r in ratios], dtype=float)
    dice = np.array([histories[r]["test"]["per_class_Dice"] for r in ratios], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, values, title in [(axes[0], iou, "Test IoU by class"), (axes[1], dice, "Test Dice by class")]:
        image = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set(title=title, xlabel="Class", ylabel="Label ratio (%)", xticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticks=range(len(ratios)), yticklabels=[f"{r}%" for r in ratios])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="white" if values[i, j] < 0.55 else "black", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); save_figure(fig, output / "class_heatmaps", fmt)


def write_tables(rows: list[dict], output: Path) -> None:
    fields = ["ratio", "labeled_images", "mIoU", "mIoU_fg", "Dice", "Dice_fg", "train_time_seconds", "FPS", "best_epoch"]
    with (output / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (output / "benchmark_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = resolve(args.input_dir)
    output = resolve(args.output_dir) if args.output_dir else root / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    _, rows = load_results(root)
    if not rows:
        raise FileNotFoundError(f"No benchmark results found under {root}")
    histories = load_histories(root, rows)
    write_tables(rows, output)
    plot_loss(histories, output, args.format)
    plot_validation(histories, output, args.format)
    plot_ratio_metrics(rows, output, args.format)
    if all(row["ratio"] in histories for row in rows):
        plot_class_heatmaps(histories, rows, output, args.format)
    print(f"visualizations={output}")
    print("generated=loss_curves, validation_curves, ratio_metrics, class_heatmaps, benchmark_summary.csv")


if __name__ == "__main__":
    main()
