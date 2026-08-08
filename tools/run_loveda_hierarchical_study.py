"""Run controlled LoveDA hierarchical feature and fusion studies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBES = (
    ("p3_only", "p3", "hierarchical"),
    ("p4_only", "p4", "hierarchical"),
    ("embedding_only", "embedding", "hierarchical"),
    ("p3_p4", "p3_p4", "hierarchical"),
    ("p4_embedding", "p4_embedding", "hierarchical"),
    ("p3_p4_embedding", "p3_p4_embedding", "hierarchical"),
)
FUSIONS = tuple(
    (name, "p3_p4_embedding", name)
    for name in ("hierarchical", "concat", "global", "image_conditioned", "scsr")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoveDA frozen-feature hierarchy study")
    parser.add_argument("--study", choices=["probe", "fusion", "budget", "all"], default="all")
    parser.add_argument("--ratios", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
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
    parser.add_argument("--output-dir", default="runs/loveda_hierarchical_study")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--budgets", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = []
    if args.study in {"probe", "all"}:
        variants.extend((f"probe_{name}", scales, fusion) for name, scales, fusion in PROBES)
    if args.study in {"fusion", "all"}:
        variants.extend((f"fusion_{name}", scales, fusion) for name, scales, fusion in FUSIONS)
    if args.study in {"budget", "all"}:
        variants.extend((f"budget_{budget}", "p3_p4_embedding", "semantic_budget", budget) for budget in args.budgets)
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    rows = []
    total = len(variants) * len(args.ratios) * len(args.seeds)
    current = 0
    for seed in args.seeds:
        for ratio in args.ratios:
            for item in variants:
                variant, scales, fusion = item[:3]
                budget = item[3] if len(item) > 3 else 3
                current += 1
                variant_root = output_root / variant
                metrics_path = variant_root / f"loveda_ratio{ratio}_seed{seed}" / "metrics.json"
                complete = False
                if metrics_path.exists() and not args.rerun_completed:
                    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
                    complete = len(saved.get("history", [])) == args.epochs
                if complete:
                    print(f"[{current}/{total}] skip {variant} ratio={ratio} seed={seed}")
                else:
                    print(f"[{current}/{total}] run {variant} ratio={ratio} seed={seed}")
                    command = [
                        sys.executable, str(ROOT / "tools" / "train_loveda.py"),
                        "--model", "mobilesam", "--adapter", "none",
                        "--decoder-version", "lightweight",
                        "--feature-scales", scales, "--fusion-version", fusion,
                        "--representation-budget", str(budget),
                        "--label-ratio", str(ratio), "--seed", str(seed),
                        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                        "--num-workers", str(args.num_workers),
                        "--image-size", str(args.image_size),
                        "--sam-image-size", str(args.sam_image_size),
                        "--validation-seed", str(args.validation_seed),
                        "--augmentation", args.augmentation,
                        "--data-root", args.data_root, "--checkpoint", args.checkpoint,
                        "--device", args.device, "--output-dir", str(variant_root),
                    ]
                    subprocess.run(command, cwd=ROOT, check=True)
                saved = json.loads(metrics_path.read_text(encoding="utf-8"))
                test = saved["test"]
                rows.append({
                    "variant": variant, "feature_scales": scales, "fusion_version": fusion,
                    "representation_budget": budget,
                    "ratio": ratio, "seed": seed,
                    "labeled_images": saved["label_pool_samples"],
                    "trainable_params": saved["parameters"]["trainable"],
                    "mIoU": test["mIoU"], "Dice": test["Dice"],
                    "pixel_accuracy": test["pixel_accuracy"], "FPS": test["FPS"],
                    "train_time_seconds": sum(epoch["seconds"] for epoch in saved["history"]),
                    "best_epoch": saved["best_epoch"],
                    "routing_statistics": saved.get("routing_statistics"),
                })
                output_root.mkdir(parents=True, exist_ok=True)
                (output_root / f"hierarchical_{args.study}_partial.json").write_text(
                    json.dumps(rows, indent=2), encoding="utf-8"
                )
    summary = {"protocol": vars(args), "results": rows}
    path = output_root / f"hierarchical_{args.study}_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={path}")


if __name__ == "__main__":
    main()
