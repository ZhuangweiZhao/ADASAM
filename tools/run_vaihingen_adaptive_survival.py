"""Run the minimal Vaihingen 25% spatial-routing survival experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("random", "magnitude", "adaptive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    parser.add_argument("--output-dir", default="runs/vaihingen_adaptive_survival")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--sam-image-size", type=int, default=512)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--retention-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_policy(args: argparse.Namespace, policy: str) -> Path:
    output_root = resolve(args.output_dir)
    run_name = f"development_{policy}_r{args.retention_ratio:g}_lora_r4_seed{args.seed}"
    metrics_path = output_root / run_name / "metrics.json"
    if metrics_path.exists() and not args.rerun_completed:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if len(metrics.get("history", [])) == args.epochs and metrics.get("test") is None:
            print(f"SKIP completed {metrics_path}", flush=True)
            return metrics_path

    command = [
        sys.executable, str(ROOT / "tools" / "train_vaihingen_lora.py"),
        "--data-root", str(resolve(args.data_root)),
        "--checkpoint", str(resolve(args.checkpoint)),
        "--output-dir", str(output_root), "--run-name", run_name,
        "--protocol", "development", "--no-evaluate-test", "--conditioned-validation",
        "--fusion-version", "semantic_budget", "--feature-scales", "p3_p4_embedding",
        "--representation-budget", "3", "--spatial-policy", policy,
        "--feature-retention-ratio", str(args.retention_ratio),
        "--label-ratio", "100", "--epochs", str(args.epochs),
        "--image-size", str(args.image_size), "--sam-image-size", str(args.sam_image_size),
        "--decoder-dim", str(args.decoder_dim), "--batch-size", str(args.batch_size),
        "--gradient-accumulation", str(args.gradient_accumulation),
        "--class-balanced-ce", "--augmentation", "basic", "--lr", "0.0005",
        "--weight-decay", "0.0001", "--num-workers", str(args.num_workers),
        "--lora-rank", "4", "--lora-alpha", "8", "--lora-targets", "qkv", "proj",
        "--validation-seed", str(args.validation_seed), "--seed", str(args.seed),
        "--amp", "--device", args.device,
    ]
    print("COMMAND " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return metrics_path


def best_validation(metrics_path: Path, policy: str) -> dict:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    record = next(item for item in metrics["history"] if item["epoch"] == metrics["best_epoch"])
    validation = record["validation"]
    size = validation["size_conditioned_region_IoU"]
    return {
        "policy": policy,
        "best_epoch": metrics["best_epoch"],
        "mIoU_5": validation["mIoU_5"],
        "mean_F1_5": validation["mean_F1_5"],
        "Boundary_F1": validation["Boundary_F1"],
        "Small_IoU": size["small"],
        "Medium_IoU": size["medium"],
        "Large_IoU": size["large"],
        "metrics_path": str(metrics_path),
    }


def main() -> None:
    args = parse_args()
    rows = [best_validation(run_policy(args, policy), policy) for policy in POLICIES]
    by_policy = {row["policy"]: row for row in rows}
    adaptive = by_policy["adaptive"]
    random = by_policy["random"]
    magnitude = by_policy["magnitude"]
    gates = {
        "beats_random_by_1_mIoU_point": adaptive["mIoU_5"] >= random["mIoU_5"] + 0.01,
        "matches_or_beats_magnitude": adaptive["mIoU_5"] >= magnitude["mIoU_5"],
        "beats_magnitude_on_boundary_or_small": (
            adaptive["Boundary_F1"] > magnitude["Boundary_F1"]
            or adaptive["Small_IoU"] > magnitude["Small_IoU"]
        ),
    }
    verdict = "GO" if all(gates.values()) else "REDESIGN"
    result = {"protocol": vars(args), "results": rows, "gates": gates, "verdict": verdict}
    output = resolve(args.output_dir) / "survival_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
