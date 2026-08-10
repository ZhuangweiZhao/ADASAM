"""Run the five controlled LoveDA studies required by the semantic-budget claim."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled LoveDA semantic-budget study")
    parser.add_argument(
        "--study", choices=["layer", "calibration", "compression", "budget", "low_label", "all"],
        default="all",
    )
    parser.add_argument("--data-root", default="data/LoveDA")
    parser.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    parser.add_argument("--output-dir", default="runs/loveda_budget_study")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--sam-image-size", type=int, default=1024)
    parser.add_argument("--augmentation", choices=["none", "basic"], default="basic")
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--screening-ratio", type=int, default=10)
    parser.add_argument("--screening-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--final-seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--low-label-ratios", nargs="+", type=int, default=[5, 10, 20, 100])
    parser.add_argument("--retention-ratios", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run(command: list[str]) -> None:
    print("COMMAND " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def train_variant(args: argparse.Namespace, variant: dict, ratio: int, seed: int) -> tuple[Path, dict]:
    root = resolve(args.output_dir) / variant["group"] / variant["name"]
    metrics_path = root / f"loveda_ratio{ratio}_seed{seed}" / "metrics.json"
    complete = False
    if metrics_path.exists() and not args.rerun_completed:
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        complete = len(saved.get("history", [])) == args.epochs
    if not complete:
        command = [
            sys.executable, str(ROOT / "tools" / "train_loveda.py"),
            "--model", variant.get("model", "mobilesam"), "--adapter", "none",
            "--label-ratio", str(ratio), "--seed", str(seed),
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--image-size", str(args.image_size), "--sam-image-size", str(args.sam_image_size),
            "--validation-seed", str(args.validation_seed), "--augmentation", args.augmentation,
            "--data-root", str(resolve(args.data_root)), "--checkpoint", str(resolve(args.checkpoint)),
            "--device", args.device, "--output-dir", str(root),
            "--feature-scales", variant.get("feature_scales", "p3_p4_embedding"),
            "--fusion-version", variant.get("fusion", "hierarchical"),
            "--representation-budget", str(variant.get("level_budget", 3)),
            "--spatial-policy", variant.get("policy", "adaptive"),
            "--feature-retention-ratio", str(variant.get("retention", 1.0)),
        ]
        if variant.get("static_map"):
            command.extend(["--static-importance-map", str(variant["static_map"])])
        run(command)
    return metrics_path, json.loads(metrics_path.read_text(encoding="utf-8"))


def export_static_map(
    args: argparse.Namespace, adaptive_metrics: Path, output: Path, ratio: int, seed: int
) -> None:
    if output.exists() and not args.rerun_completed:
        return
    checkpoint = adaptive_metrics.parent / "best_model.pt"
    run([
        sys.executable, str(ROOT / "tools" / "export_static_importance.py"),
        "--data-root", str(resolve(args.data_root)), "--checkpoint", str(resolve(args.checkpoint)),
        "--model-checkpoint", str(checkpoint), "--output", str(output),
        "--label-ratio", str(ratio), "--seed", str(seed),
        "--validation-seed", str(args.validation_seed),
        "--image-size", str(args.image_size), "--sam-image-size", str(args.sam_image_size),
        "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
        "--device", args.device,
    ])


def profile_variant(args: argparse.Namespace, variant: dict, metrics_path: Path) -> None:
    if args.skip_profile:
        return
    output = metrics_path.parent / "efficiency.csv"
    if output.exists() and not args.rerun_completed:
        return
    command = [
        sys.executable, str(ROOT / "tools" / "profile_efficiency.py"),
        "--dataset", "loveda", "--models", "mobilesam",
        "--data-root", str(resolve(args.data_root)), "--checkpoint", str(resolve(args.checkpoint)),
        "--model-checkpoints", str(metrics_path.parent / "best_model.pt"),
        "--image-size", str(args.image_size), "--sam-image-size", str(args.sam_image_size),
        "--adapter", "none", "--feature-scales", variant.get("feature_scales", "p3_p4_embedding"),
        "--fusion-version", variant.get("fusion", "hierarchical"),
        "--representation-budget", str(variant.get("level_budget", 3)),
        "--spatial-policy", variant.get("policy", "adaptive"),
        "--feature-retention-ratio", str(variant.get("retention", 1.0)),
        "--device", args.device, "--output-csv", str(output),
    ]
    if variant.get("static_map"):
        command.extend(["--static-importance-map", str(variant["static_map"])])
    run(command)


def variants_for(args: argparse.Namespace, study: str) -> list[dict]:
    if study == "layer":
        return [
            {"group": study, "name": name, "feature_scales": scales, "fusion": "hierarchical"}
            for name, scales in (
                ("p3", "p3"), ("p4", "p4"), ("embedding", "embedding"),
                ("p3_p4", "p3_p4"), ("p4_embedding", "p4_embedding"),
                ("all_levels", "p3_p4_embedding"),
            )
        ]
    if study == "calibration":
        return [
            {"group": study, "name": name, "fusion": fusion}
            for name, fusion in (
                ("concat", "concat"), ("sum", "sum"), ("static_weight", "global"),
                ("image_conditioned", "image_conditioned"), ("local_dynamic", "scsr_v2"),
            )
        ]
    if study == "compression":
        return [
            {"group": study, "name": f"{policy}_r{retention:g}", "fusion": "semantic_budget", "policy": policy, "retention": retention}
            for retention in args.retention_ratios for policy in ("random", "magnitude", "adaptive")
        ]
    if study == "budget":
        return [
            {"group": study, "name": f"adaptive_r{retention:g}", "fusion": "semantic_budget", "policy": "adaptive", "retention": retention}
            for retention in args.retention_ratios
        ]
    return [
        {"group": study, "name": "unet", "model": "unet", "fusion": "hierarchical"},
        {"group": study, "name": "frozen_dense", "fusion": "hierarchical"},
        {"group": study, "name": "adaptive_r0.5", "fusion": "semantic_budget", "policy": "adaptive", "retention": 0.5},
    ]


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["group"], row["variant"], row["label_ratio"]), []).append(row)
    summary = []
    for (group, variant, ratio), values in sorted(groups.items()):
        item = {"group": group, "variant": variant, "label_ratio": ratio, "seeds": [v["seed"] for v in values]}
        for metric in ("mIoU", "Boundary_F1", "FPS"):
            numbers = [v[metric] for v in values]
            item[f"{metric}_mean"] = statistics.fmean(numbers)
            item[f"{metric}_std"] = statistics.stdev(numbers) if len(numbers) > 1 else None
        summary.append(item)
    return summary


def main() -> None:
    args = parse_args()
    selected = [args.study] if args.study != "all" else ["layer", "calibration", "compression", "budget", "low_label"]
    rows = []
    for study in selected:
        ratios = args.low_label_ratios if study == "low_label" else [args.screening_ratio]
        seeds = args.final_seeds if study in {"layer", "compression", "low_label"} else args.screening_seeds
        for ratio in ratios:
            for seed in seeds:
                for variant in variants_for(args, study):
                    metrics_path, metrics = train_variant(args, variant, ratio, seed)
                    test = metrics["test"]
                    rows.append({
                        "group": study, "variant": variant["name"], "label_ratio": ratio,
                        "seed": seed, "labeled_images": metrics["label_pool_samples"],
                        "trainable_params": metrics["parameters"]["trainable"],
                        "mIoU": test["mIoU"], "Boundary_F1": test["Boundary_F1"],
                        "FPS": test["FPS"], "size_conditioned_region_IoU": test.get("size_conditioned_region_IoU"),
                        "routing_statistics": metrics.get("routing_statistics"),
                        "metrics_path": str(metrics_path),
                    })
                    if study == "budget":
                        profile_variant(args, variant, metrics_path)
                    if study == "compression" and variant["policy"] == "adaptive":
                        static_map = metrics_path.parent / "average_static_importance.pt"
                        export_static_map(args, metrics_path, static_map, ratio, seed)
                        static_variant = {
                            "group": study, "name": f"average_static_r{variant['retention']:g}",
                            "fusion": "semantic_budget", "policy": "static",
                            "retention": variant["retention"], "static_map": static_map,
                        }
                        static_path, static_metrics = train_variant(args, static_variant, ratio, seed)
                        static_test = static_metrics["test"]
                        rows.append({
                            "group": study, "variant": static_variant["name"], "label_ratio": ratio,
                            "seed": seed, "labeled_images": static_metrics["label_pool_samples"],
                            "trainable_params": static_metrics["parameters"]["trainable"],
                            "mIoU": static_test["mIoU"], "Boundary_F1": static_test["Boundary_F1"],
                            "FPS": static_test["FPS"],
                            "size_conditioned_region_IoU": static_test.get("size_conditioned_region_IoU"),
                            "routing_statistics": static_metrics.get("routing_statistics"),
                            "metrics_path": str(static_path),
                        })
                output = resolve(args.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                (output / "study_partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    result = {"protocol": vars(args), "results": rows, "aggregate": aggregate(rows)}
    output = resolve(args.output_dir) / "study_summary.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved={output} rows={len(rows)}")


if __name__ == "__main__":
    main()
