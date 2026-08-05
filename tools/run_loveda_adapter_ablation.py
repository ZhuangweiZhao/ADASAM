"""Run the four controlled LoveDA adapter/feature-scale ablations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    ("decoder_only", "none", "embedding", "pre_fusion"),
    ("multiscale_decoder", "none", "p3_p4_embedding", "pre_fusion"),
    ("cat_embedding", "cat", "embedding", "pre_fusion"),
    ("full_baseline", "cat", "p3_p4_embedding", "pre_fusion"),
    ("post_fusion_adapter", "cat", "p3_p4_embedding", "post_fusion"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoveDA CATAdapter and feature-scale ablation")
    parser.add_argument("--ratios", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--sam-image-size", type=int, default=1024)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--augmentation", choices=["none", "basic"], default="basic")
    parser.add_argument("--data-root", default="data/LoveDA")
    parser.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="runs/loveda_adapter_ablation")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    rows = []
    total = len(VARIANTS) * len(args.ratios)
    current = 0
    for ratio in args.ratios:
        for variant, adapter, feature_scales, adapter_placement in VARIANTS:
            current += 1
            variant_root = output_root / variant
            run_dir = variant_root / f"loveda_ratio{ratio}_seed{args.seed}"
            metrics_path = run_dir / "metrics.json"
            complete = False
            if metrics_path.exists() and not args.rerun_completed:
                saved = json.loads(metrics_path.read_text(encoding="utf-8"))
                complete = len(saved.get("history", [])) == args.epochs
            if complete:
                print(f"[{current}/{total}] skip: {variant} ratio={ratio}")
            else:
                print(f"[{current}/{total}] run: {variant} ratio={ratio}")
                command = [
                    sys.executable, str(ROOT / "tools" / "train_loveda.py"),
                    "--model", "mobilesam",
                    "--label-ratio", str(ratio),
                    "--adapter", adapter,
                    "--feature-scales", feature_scales,
                    "--adapter-placement", adapter_placement,
                    "--augmentation", args.augmentation,
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--image-size", str(args.image_size),
                    "--sam-image-size", str(args.sam_image_size),
                    "--validation-seed", str(args.validation_seed),
                    "--seed", str(args.seed),
                    "--checkpoint", args.checkpoint,
                    "--data-root", args.data_root,
                    "--device", args.device,
                    "--output-dir", str(variant_root),
                ]
                subprocess.run(command, cwd=ROOT, check=True)
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            test = saved["test"]
            rows.append({
                "variant": variant,
                "adapter": adapter,
                "feature_scales": feature_scales,
                "adapter_placement": adapter_placement,
                "ratio": ratio,
                "seed": args.seed,
                "labeled_images": saved["label_pool_samples"],
                "total_params": saved["parameters"]["total"],
                "trainable_params": saved["parameters"]["trainable"],
                "mIoU": test["mIoU"],
                "Dice": test["Dice"],
                "pixel_accuracy": test["pixel_accuracy"],
                "Boundary_F1": test.get("Boundary_F1"),
                "FPS": test["FPS"],
                "train_time_seconds": sum(epoch["seconds"] for epoch in saved["history"]),
                "best_epoch": saved["best_epoch"],
            })
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "adapter_ablation_partial.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )
    summary = {
        "protocol": {
            "dataset": "LoveDA",
            "ratios": args.ratios,
            "seed": args.seed,
            "epochs": args.epochs,
            "augmentation": args.augmentation,
            "image_size": args.image_size,
            "sam_image_size": args.sam_image_size,
            "validation_seed": args.validation_seed,
            "variants": [variant for variant, _, _, _ in VARIANTS],
        },
        "results": rows,
    }
    path = output_root / "adapter_ablation_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={path}")


if __name__ == "__main__":
    main()
