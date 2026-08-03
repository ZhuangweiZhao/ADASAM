"""Run the complete model/augmentation/label-ratio/seed experiment matrix."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("mIoU", "mIoU_fg", "Dice", "Dice_fg", "pixel_accuracy", "FPS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the label-efficient multi-seed matrix")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--ratios", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100])
    parser.add_argument("--models", nargs="+", choices=["unet", "mobilesam", "dapg"], default=["unet", "mobilesam", "dapg"])
    parser.add_argument("--augmentations", nargs="+", choices=["none", "basic"], default=["none", "basic"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--data-root", default="data/NEU_Seg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="runs/multiseed_matrix")
    parser.add_argument("--split-protocol", choices=["legacy", "fixed"], default="fixed")
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def experiment_dir(root: Path, model: str, augmentation: str, ratio: int, seed: int) -> Path:
    return root / f"{model}_{augmentation}" / f"neu_seg_ratio{ratio}_seed{seed}"


def is_complete(path: Path, epochs: int, split_protocol: str, validation_seed: int) -> bool:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return (
            len(metrics.get("history", [])) == epochs
            and "test" in metrics
            and metrics.get("split_protocol", "legacy") == split_protocol
            and int(metrics.get("validation_seed", 42)) == validation_seed
        )
    except (OSError, json.JSONDecodeError):
        return False


def training_command(args: argparse.Namespace, model: str, augmentation: str, ratio: int, seed: int, output_root: Path) -> list[str]:
    common = [
        "--label_ratio", str(ratio),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--data-root", args.data_root,
        "--augmentation", augmentation,
        "--device", args.device,
        "--seed", str(seed),
        "--output-dir", str(output_root / f"{model}_{augmentation}"),
        "--split-protocol", args.split_protocol,
        "--validation-seed", str(args.validation_seed),
    ]
    if model == "unet":
        return [sys.executable, str(ROOT / "tools" / "train_unet.py"), *common, "--base-channels", str(args.base_channels)]
    command = [sys.executable, str(ROOT / "tools" / "train_segmentation.py"), *common, "--adapter", "cat"]
    command.extend(["--prompt-version", "v2", "--prompt-fusion-mode", "both"] if model == "dapg" else ["--prompt-version", "none"])
    return command


def collect(output_root: Path, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rows = []
    for ratio in args.ratios:
        for seed in args.seeds:
            for augmentation in args.augmentations:
                for model in args.models:
                    path = experiment_dir(output_root, model, augmentation, ratio, seed) / "metrics.json"
                    if not path.exists():
                        continue
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    test = saved["test"]
                    rows.append({
                        "model": model,
                        "augmentation": augmentation,
                        "ratio": ratio,
                        "seed": seed,
                        "labeled_images": saved["label_pool_samples"],
                        "training_pool_images": saved.get("training_pool_samples"),
                        "validation_images": saved.get("validation_samples"),
                        **{metric: test[metric] for metric in METRICS},
                        "train_time_seconds": sum(epoch["seconds"] for epoch in saved["history"]),
                        "best_epoch": saved["best_epoch"],
                    })
    aggregates = []
    for model in args.models:
        for augmentation in args.augmentations:
            for ratio in args.ratios:
                group = [row for row in rows if row["model"] == model and row["augmentation"] == augmentation and row["ratio"] == ratio]
                if not group:
                    continue
                item = {"model": model, "augmentation": augmentation, "ratio": ratio, "runs": len(group)}
                for metric in (*METRICS, "train_time_seconds"):
                    values = [float(row[metric]) for row in group]
                    item[f"{metric}_mean"] = statistics.mean(values)
                    item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                aggregates.append(item)
    return rows, aggregates


def write_summary(output_root: Path, args: argparse.Namespace) -> None:
    rows, aggregates = collect(output_root, args)
    summary = {
        "protocol": {
            "dataset": "NEU_Seg",
            "seeds": args.seeds,
            "ratios": args.ratios,
            "models": args.models,
            "augmentations": args.augmentations,
            "epochs": args.epochs,
            "validation_fraction": 0.2,
            "validation_augmentation": "none",
            "test_augmentation": "none",
            "split_protocol": args.split_protocol,
            "validation_seed": args.validation_seed,
            "planned_runs": len(args.seeds) * len(args.ratios) * len(args.models) * len(args.augmentations),
            "completed_runs": len(rows),
        },
        "results": rows,
        "aggregates": aggregates,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "multiseed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    total = len(args.seeds) * len(args.ratios) * len(args.models) * len(args.augmentations)
    current = 0
    for model in args.models:
        for augmentation in args.augmentations:
            for ratio in args.ratios:
                for seed in args.seeds:
                    current += 1
                    path = experiment_dir(output_root, model, augmentation, ratio, seed)
                    if not args.rerun_completed and is_complete(
                        path, args.epochs, args.split_protocol, args.validation_seed
                    ):
                        print(f"[{current}/{total}] skip completed: {model} aug={augmentation} ratio={ratio} seed={seed}")
                        continue
                    print(f"[{current}/{total}] run: {model} aug={augmentation} ratio={ratio} seed={seed}")
                    subprocess.run(
                        training_command(args, model, augmentation, ratio, seed, output_root),
                        cwd=ROOT,
                        check=True,
                    )
                    write_summary(output_root, args)
    write_summary(output_root, args)
    print(f"saved={output_root / 'multiseed_summary.json'}")


if __name__ == "__main__":
    main()
