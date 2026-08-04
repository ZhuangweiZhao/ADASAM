"""Run a resumable LoveDA matrix for U-Net, MobileSAM, and the proposed model."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoveDA label-efficient benchmark")
    parser.add_argument("--models", nargs="+", choices=["unet", "mobilesam", "ours"], default=["unet", "mobilesam", "ours"])
    parser.add_argument("--ratios", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--augmentations", nargs="+", choices=["none", "basic"], default=["basic"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--sam-image-size", type=int, default=224)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--adapters", nargs="+", choices=["cat", "none"], default=["cat"])
    parser.add_argument("--feature-scales", nargs="+", choices=["embedding", "p4_embedding", "p3_p4_embedding"], default=["p3_p4_embedding"])
    parser.add_argument("--decoder-versions", nargs="+", choices=["lightweight", "boundary_aux", "boundary"], default=["lightweight"])
    parser.add_argument("--boundary-loss-weight", type=float, default=0.1)
    parser.add_argument("--data-root", default="data/LoveDA")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs/loveda_benchmark")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    rows = []
    total = sum(
        1
        for _ratio in args.ratios
        for _seed in args.seeds
        for _augmentation in args.augmentations
        for model in args.models
        for adapter in args.adapters
        for feature_scales in args.feature_scales
        for decoder in args.decoder_versions
        if (model != "unet" or (decoder == "lightweight" and adapter == "cat" and feature_scales == "p3_p4_embedding"))
    )
    current = 0
    for ratio in args.ratios:
        for seed in args.seeds:
            for augmentation in args.augmentations:
                for model in args.models:
                  for adapter in args.adapters:
                   for feature_scales in args.feature_scales:
                    for decoder_version in args.decoder_versions:
                     if model == "unet" and (decoder_version != "lightweight" or adapter != "cat" or feature_scales != "p3_p4_embedding"):
                        continue
                    current += 1
                    variant_name = f"{model}_{augmentation}"
                    if decoder_version != "lightweight":
                        variant_name += f"_{decoder_version}"
                    if adapter != "cat":
                        variant_name += f"_{adapter}_adapter"
                    if feature_scales != "p3_p4_embedding":
                        variant_name += f"_{feature_scales}"
                    variant_root = output_root / variant_name
                    run_dir = variant_root / f"loveda_ratio{ratio}_seed{seed}"
                    metrics_path = run_dir / "metrics.json"
                    complete = False
                    if metrics_path.exists() and not args.rerun_completed:
                        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
                        complete = len(saved.get("history", [])) == args.epochs
                    if complete:
                        print(f"[{current}/{total}] skip completed: {model} aug={augmentation} ratio={ratio} seed={seed}")
                    else:
                        print(f"[{current}/{total}] run: {model} aug={augmentation} ratio={ratio} seed={seed}")
                        command = [
                            sys.executable, str(ROOT / "tools" / "train_loveda.py"),
                            "--model", model,
                            "--label-ratio", str(ratio),
                            "--epochs", str(args.epochs),
                            "--batch-size", str(args.batch_size),
                            "--num-workers", str(args.num_workers),
                            "--image-size", str(args.image_size),
                            "--sam-image-size", str(args.sam_image_size),
                            "--base-channels", str(args.base_channels),
                            "--augmentation", augmentation,
                            "--decoder-version", decoder_version,
                            "--boundary-loss-weight", str(args.boundary_loss_weight),
                            "--adapter", adapter,
                            "--feature-scales", feature_scales,
                            "--data-root", args.data_root,
                            "--device", args.device,
                            "--validation-seed", str(args.validation_seed),
                            "--seed", str(seed),
                            "--output-dir", str(variant_root),
                        ]
                        subprocess.run(command, cwd=ROOT, check=True)
                    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
                    test = saved["test"]
                    rows.append({
                        "model": model,
                        "augmentation": augmentation,
                        "decoder_version": decoder_version,
                        "adapter": adapter,
                        "feature_scales": feature_scales,
                        "ratio": ratio,
                        "seed": seed,
                        "labeled_images": saved["label_pool_samples"],
                        "mIoU": test["mIoU"],
                        "mIoU_fg": test["mIoU_fg"],
                        "Dice": test["Dice"],
                        "Dice_fg": test["Dice_fg"],
                        "pixel_accuracy": test["pixel_accuracy"],
                        "Boundary_F1": test.get("Boundary_F1"),
                        "building_IoU": test["per_class_IoU"][1],
                        "road_IoU": test["per_class_IoU"][3],
                        "FPS": test["FPS"],
                        "train_time_seconds": sum(epoch["seconds"] for epoch in saved["history"]),
                        "best_epoch": saved["best_epoch"],
                    })
                    output_root.mkdir(parents=True, exist_ok=True)
                    (output_root / "loveda_results_partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    aggregates = []
    for model in args.models:
        for augmentation in args.augmentations:
            for decoder_version in args.decoder_versions:
             for adapter in args.adapters:
              for feature_scales in args.feature_scales:
               for ratio in args.ratios:
                    group = [row for row in rows if row["model"] == model and row["augmentation"] == augmentation and row["decoder_version"] == decoder_version and row["adapter"] == adapter and row["feature_scales"] == feature_scales and row["ratio"] == ratio]
                    if not group:
                        continue
                    item = {"model": model, "augmentation": augmentation, "decoder_version": decoder_version, "adapter": adapter, "feature_scales": feature_scales, "ratio": ratio, "runs": len(group)}
                    for metric in ("mIoU", "mIoU_fg", "Dice", "Dice_fg", "pixel_accuracy", "Boundary_F1", "building_IoU", "road_IoU", "FPS", "train_time_seconds"):
                        values = [float(row[metric]) for row in group if row.get(metric) is not None]
                        if not values:
                            continue
                        item[f"{metric}_mean"] = statistics.mean(values)
                        item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                    aggregates.append(item)
    summary = {
        "protocol": {
            "dataset": "LoveDA",
            "train_images": 2522,
            "official_validation_images": 1669,
            "models": args.models,
            "ratios": args.ratios,
            "seeds": args.seeds,
            "augmentations": args.augmentations,
            "epochs": args.epochs,
            "image_size": args.image_size,
            "sam_image_size": args.sam_image_size,
            "validation_seed": args.validation_seed,
            "decoder_versions": args.decoder_versions,
            "boundary_loss_weight": args.boundary_loss_weight,
            "adapters": args.adapters,
            "feature_scales": args.feature_scales,
            "official_val_used_as_test": True,
        },
        "results": rows,
        "aggregates": aggregates,
    }
    (output_root / "loveda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={output_root / 'loveda_summary.json'}")


if __name__ == "__main__":
    main()
