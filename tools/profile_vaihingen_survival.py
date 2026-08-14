"""Profile and aggregate Vaihingen adaptive-survival checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasam.datasets.industrial import VaihingenSemanticDataset  # noqa: E402
from tools.profile_efficiency import measure, measure_flops  # noqa: E402
from tools.train_vaihingen import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-roots", nargs="+", required=True)
    parser.add_argument("--mobile-sam-checkpoint", default=None)
    parser.add_argument("--profile-images", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir", default="runs/vaihingen_adaptive_survival_efficiency"
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def discover(run_roots: list[str]) -> list[Path]:
    checkpoints = []
    for value in run_roots:
        root = resolve(value)
        checkpoints.extend(sorted(root.glob("development_*_r0.25_lora_r4_seed*/best_model.pt")))
    if not checkpoints:
        raise FileNotFoundError("No adaptive-survival best_model.pt checkpoints found")
    return checkpoints


def sample_sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def aggregate(rows: list[dict]) -> list[dict]:
    result = []
    for policy in ("random", "magnitude", "adaptive"):
        group = [row for row in rows if row["policy"] == policy]
        if not group:
            continue
        item = {"policy": policy, "seeds": [row["seed"] for row in group]}
        for key in (
            "mIoU_5", "Small_IoU", "FPS", "FLOPs",
            "peak_memory_allocated_MB", "executed_detail_projection_FLOPs_per_image",
            "P3_projected_positions", "P4_projected_positions",
        ):
            values = [float(row[key]) for row in group if row.get(key) is not None]
            item[f"{key}_mean"] = statistics.fmean(values) if values else None
            item[f"{key}_std"] = sample_sd(values) if values else None
        result.append(item)
    return result


def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device if cli.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoints = discover(cli.run_roots)
    first_payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    image_size = int(first_payload["args"]["image_size"])
    dataset = VaihingenSemanticDataset(resolve(cli.data_root), "test", image_size)
    count = min(len(dataset), max(1, cli.profile_images))
    loader = DataLoader(
        Subset(dataset, range(count)), batch_size=1, shuffle=False,
        num_workers=cli.num_workers, pin_memory=device.type == "cuda",
    )

    rows = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        train_args = argparse.Namespace(**payload["args"])
        if cli.mobile_sam_checkpoint:
            train_args.checkpoint = str(resolve(cli.mobile_sam_checkpoint))
        model = build_model(train_args, device)
        model.load_state_dict(payload["model"])
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        flops = measure_flops(model, dummy)
        timing = measure(model, loader, device, dummy)
        if flops is not None and timing.get("sparse_lateral_projection_observed"):
            flops += timing.get("executed_detail_projection_FLOPs_per_image", 0.0)

        metrics_path = checkpoint.with_name("metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        best = next(item for item in metrics["history"] if item["epoch"] == metrics["best_epoch"])
        validation = best["validation"]
        projected = timing.get("mean_projected_positions", {})
        counts = model.parameter_counts()
        row = {
            "seed": int(train_args.seed),
            "policy": train_args.spatial_policy,
            "retention_ratio": float(train_args.feature_retention_ratio),
            "mIoU_5": validation["mIoU_5"],
            "Boundary_F1": validation["Boundary_F1"],
            "Small_IoU": validation["size_conditioned_region_IoU"]["small"],
            "total_params": counts["total"],
            "trainable_params": counts["trainable"],
            "FLOPs": flops,
            "FPS": timing["FPS"],
            "peak_memory_allocated_MB": timing["peak_memory_allocated_MB"],
            "peak_memory_reserved_MB": timing["peak_memory_reserved_MB"],
            "executed_detail_projection_FLOPs_per_image": timing.get(
                "executed_detail_projection_FLOPs_per_image"
            ),
            "P3_projected_positions": projected.get("P3"),
            "P4_projected_positions": projected.get("P4"),
            "sparse_lateral_projection_observed": timing.get(
                "sparse_lateral_projection_observed", False
            ),
            "profile_images": timing["images"],
            "checkpoint": str(checkpoint),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_dir = resolve(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "efficiency.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": {
            "dataset": "ISPRS Vaihingen test images",
            "image_size": image_size,
            "profile_images": count,
            "device": str(device),
            "timing": "batch1, eval, no_grad, warmup10, forward_only",
            "flops": "THOP executed graph plus explicit sparse 1x1 projections",
        },
        "results": rows,
        "aggregate": aggregate(rows),
    }
    summary_path = output_dir / "efficiency_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={csv_path}")
    print(f"saved={summary_path}")


if __name__ == "__main__":
    main()
