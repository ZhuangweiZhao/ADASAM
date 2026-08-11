"""Run and judge the single 25%-budget Magnitude-teacher survival experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoveDA Magnitude-teacher survival test")
    parser.add_argument("--data-root", default="/root/autodl-tmp/LoveDA")
    parser.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    parser.add_argument("--output-dir", default="runs/loveda_magnitude_teacher_survival")
    parser.add_argument(
        "--reference-root",
        default="runs/loveda_budget_study/compression",
        help="Directory containing adaptive/random/magnitude 25%% seed42 results.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--sam-image-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run(command: list[str]) -> None:
    print("COMMAND " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_reference(root: Path, variant: str) -> dict | None:
    path = root / variant / "loveda_ratio10_seed42" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    args = parse_args()
    output_root = resolve(args.output_dir)
    run_root = output_root / "distilled_magnitude_r0.25"
    metrics_path = run_root / "loveda_ratio10_seed42" / "metrics.json"
    complete = False
    if metrics_path.exists() and not args.rerun:
        current = json.loads(metrics_path.read_text(encoding="utf-8"))
        complete = len(current.get("history", [])) == args.epochs
    if not complete:
        run([
            sys.executable, str(ROOT / "tools" / "train_loveda.py"),
            "--model", "mobilesam", "--adapter", "none",
            "--fusion-version", "semantic_budget",
            "--representation-budget", "3",
            "--spatial-policy", "distilled_magnitude",
            "--feature-retention-ratio", "0.25",
            "--magnitude-distill-weight", "1.0",
            "--label-ratio", "10", "--seed", "42", "--validation-seed", "42",
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--image-size", str(args.image_size),
            "--sam-image-size", str(args.sam_image_size),
            "--augmentation", "basic", "--data-root", str(resolve(args.data_root)),
            "--checkpoint", str(resolve(args.checkpoint)), "--device", args.device,
            "--output-dir", str(run_root),
        ])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint_path = metrics_path.parent / "best_model.pt"
    efficiency_path = metrics_path.parent / "efficiency.csv"
    if not args.skip_profile and (args.rerun or not efficiency_path.exists()):
        run([
            sys.executable, str(ROOT / "tools" / "profile_efficiency.py"),
            "--dataset", "loveda", "--models", "mobilesam",
            "--data-root", str(resolve(args.data_root)),
            "--checkpoint", str(resolve(args.checkpoint)),
            "--model-checkpoints", str(checkpoint_path),
            "--image-size", str(args.image_size),
            "--sam-image-size", str(args.sam_image_size),
            "--adapter", "none", "--fusion-version", "semantic_budget",
            "--representation-budget", "3",
            "--spatial-policy", "distilled_magnitude",
            "--feature-retention-ratio", "0.25",
            "--device", args.device, "--output-csv", str(efficiency_path),
        ])
    visualization_dir = metrics_path.parent / "visualizations"
    if not args.skip_visualization and (args.rerun or not visualization_dir.exists()):
        run([
            sys.executable, str(ROOT / "tools" / "visualize_budget_allocation.py"),
            "--data-root", str(resolve(args.data_root)),
            "--checkpoint", str(resolve(args.checkpoint)),
            "--model-checkpoint", str(checkpoint_path), "--metrics", str(metrics_path),
            "--output-dir", str(visualization_dir),
            "--image-size", str(args.image_size),
            "--sam-image-size", str(args.sam_image_size),
            "--adapter", "none", "--representation-budget", "3",
            "--spatial-policy", "distilled_magnitude",
            "--feature-retention-ratio", "0.25", "--device", args.device,
        ])

    references = {
        name: read_reference(resolve(args.reference_root), f"{name}_r0.25")
        for name in ("adaptive", "random", "magnitude")
    }
    best_epoch = int(metrics["best_epoch"])
    best_record = next(record for record in metrics["history"] if record["epoch"] == best_epoch)
    result = {
        "experiment": {
            "mIoU": metrics["test"]["mIoU"],
            "Boundary_F1": metrics["test"]["Boundary_F1"],
            "FPS_from_test": metrics["test"]["FPS"],
            "best_epoch": best_epoch,
            "teacher_student_mask_iou_at_best_epoch": best_record.get(
                "mean_teacher_student_mask_iou"
            ),
            "size_conditioned_region_IoU": metrics["test"].get(
                "size_conditioned_region_IoU"
            ),
            "routing_statistics": metrics.get("routing_statistics"),
        },
        "references": {
            name: (
                {
                    "mIoU": value["test"]["mIoU"],
                    "Boundary_F1": value["test"]["Boundary_F1"],
                    "FPS": value["test"]["FPS"],
                }
                if value else None
            )
            for name, value in references.items()
        },
    }
    teacher_iou = result["experiment"]["teacher_student_mask_iou_at_best_epoch"]
    gates = {
        "beats_old_adaptive_by_1_point": (
            references["adaptive"] is not None
            and metrics["test"]["mIoU"] >= references["adaptive"]["test"]["mIoU"] + 0.01
        ),
        "matches_magnitude_within_2_points": (
            references["magnitude"] is not None
            and metrics["test"]["mIoU"] >= references["magnitude"]["test"]["mIoU"] - 0.02
        ),
        "teacher_mask_iou_at_least_0_50": teacher_iou is not None and teacher_iou >= 0.50,
        "beats_random": (
            references["random"] is not None
            and metrics["test"]["mIoU"] > references["random"]["test"]["mIoU"]
        ),
    }
    result["gates"] = gates
    result["verdict"] = "SURVIVE" if all(gates.values()) else "STOP_OR_REDESIGN"
    output_root.mkdir(parents=True, exist_ok=True)
    verdict_path = output_root / "survival_verdict.json"
    verdict_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": gates}, indent=2))
    print(f"saved={verdict_path}")


if __name__ == "__main__":
    main()
