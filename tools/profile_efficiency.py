"""Measure model size, FLOPs, inference FPS, and peak CUDA memory.

The timing protocol deliberately excludes data loading, metric computation, and
prediction serialization. It includes the complete model forward pass.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile segmentation inference efficiency")
    p.add_argument("--dataset", choices=["loveda", "neu_seg", "isaid"], required=True)
    p.add_argument("--models", nargs="+", choices=["unet", "mobilesam", "ours", "deeplabv3plus", "segformer"], default=["mobilesam"])
    p.add_argument("--baseline-encoder", choices=["resnet50", "resnet101", "mobilenet_v2"], default="resnet50")
    p.add_argument("--segformer-variant", choices=["b0", "b1", "b2"], default="b0")
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    p.add_argument("--model-checkpoints", nargs="*", default=[])
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--sam-image-size", type=int, default=None)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--decoder-dim", type=int, default=96)
    p.add_argument("--adapter", choices=["cat", "none"], default="cat")
    p.add_argument("--feature-scales", choices=["p3", "p4", "embedding", "p3_p4", "p3_embedding", "p4_embedding", "p3_p4_embedding"], default="p3_p4_embedding")
    p.add_argument("--fusion-version", choices=["hierarchical", "concat", "sum", "global", "image_conditioned", "scsr", "scsr_v2", "scsr_task", "semantic_budget", "semantic_progressive", "semantic_progressive_v2", "semantic_progressive_v3", "regional_semantic"], default="hierarchical")
    p.add_argument("--representation-budget", type=int, choices=[1, 2, 3], default=3)
    p.add_argument("--spatial-policy", choices=["adaptive", "static", "magnitude", "distilled_magnitude", "random"], default="adaptive")
    p.add_argument("--feature-retention-ratio", type=float, default=1.0)
    p.add_argument("--spatial-budget-temperature", type=float, default=1.0)
    p.add_argument("--static-importance-map", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-csv", default="runs/efficiency_results.csv")
    p.add_argument("--merge-csv", default=None)
    return p.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_model(args: argparse.Namespace, name: str, checkpoint: Path, device: torch.device):
    from adasam.models import LabelEfficientSAM, LabelEfficientUNet
    from adasam.utils import load_static_importance_map

    if args.dataset == "loveda":
        classes = 7
    elif args.dataset == "neu_seg":
        classes = 4
    else:
        classes = 16
    if name == "unet":
        return LabelEfficientUNet(classes, args.base_channels).to(device)
    if name in {"deeplabv3plus", "segformer"}:
        from adasam.models import build_baseline

        return build_baseline(
            name,
            num_classes=classes,
            pretrained=False,
            encoder_name=args.baseline_encoder,
            segformer_variant=args.segformer_variant,
            weights_root=ROOT / "weights",
            device=device,
        )
    return LabelEfficientSAM.build(
        checkpoint,
        num_classes=classes,
        img_size=args.sam_image_size,
        decoder_dim=args.decoder_dim,
        prompt_version="v2" if name == "ours" else "none",
        num_prompt=8,
        prompt_fusion_mode="both",
        use_cat_adapter=args.adapter == "cat",
        feature_scales=args.feature_scales,
        fusion_version=args.fusion_version,
        representation_budget=args.representation_budget,
        spatial_policy=args.spatial_policy,
        feature_retention_ratio=args.feature_retention_ratio,
        spatial_budget_temperature=args.spatial_budget_temperature,
        static_importance_map=load_static_importance_map(
            resolve(args.static_importance_map) if args.static_importance_map else None
        ),
        device=device,
    )


def load_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"warning: missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"warning: unexpected checkpoint keys: {len(unexpected)}")


def measure(model: torch.nn.Module, loader: DataLoader, device: torch.device, dummy: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    total_images = 0
    elapsed = 0.0
    projected_positions = torch.zeros(2, dtype=torch.float64)
    sparse_projection_observed = False
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            model(image)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            total_images += image.shape[0]
            routing = getattr(getattr(model, "decoder", None), "last_routing", None)
            if routing is not None and "lateral_projected_positions" in routing:
                projected_positions += routing["lateral_projected_positions"].sum(0).cpu().double()
                sparse_projection_observed |= bool(routing.get("sparse_lateral_projection", False))
    allocated = reserved = 0.0
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    result = {
        "images": total_images,
        "inference_seconds": elapsed,
        "FPS": total_images / max(elapsed, 1e-12),
        "peak_memory_allocated_MB": allocated,
        "peak_memory_reserved_MB": reserved,
    }
    if projected_positions.sum() > 0 and hasattr(model, "decoder"):
        names = ("P3", "P4")
        projection_flops = 0.0
        for index, name in enumerate(names):
            layer = model.decoder.lateral[name]
            projection_flops += float(
                projected_positions[index] * 2 * layer.in_channels * layer.out_channels
            )
        result["executed_detail_projection_FLOPs_per_image"] = projection_flops / max(total_images, 1)
        result["mean_projected_positions"] = {
            name: float(projected_positions[index] / max(total_images, 1))
            for index, name in enumerate(names)
        }
        result["sparse_lateral_projection_observed"] = sparse_projection_observed
    return result


def measure_flops(model: torch.nn.Module, dummy: torch.Tensor) -> float | None:
    try:
        from thop import profile
    except ImportError:
        print("warning: thop is not installed; FLOPs will be blank")
        return None
    model.eval()
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    return float(2.0 * macs)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    image_size = args.image_size or (1024 if args.dataset == "loveda" else 224)
    sam_size = args.sam_image_size or (1024 if args.dataset == "loveda" else 224)
    args.image_size = image_size
    args.sam_image_size = sam_size
    if args.dataset == "loveda":
        from adasam.datasets.industrial import LoveDASemanticDataset
        dataset = LoveDASemanticDataset(resolve(args.data_root), "val", image_size)
    elif args.dataset == "neu_seg":
        from adasam.datasets.industrial import NEUSegSemanticDataset
        dataset = NEUSegSemanticDataset(resolve(args.data_root), "test")
    else:
        from adasam.datasets.industrial import ISAIDSemanticDataset
        dataset = ISAIDSemanticDataset(resolve(args.data_root), "val", image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    rows = []
    checkpoints = {name: resolve(path) for name, path in zip(args.models, args.model_checkpoints)}
    for name in args.models:
        model = build_model(args, name, resolve(args.checkpoint), device)
        if name in checkpoints:
            load_weights(model, checkpoints[name], device)
        counts = model.parameter_counts()
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        flops = measure_flops(model, dummy)
        timing = measure(model, loader, device, dummy)
        adjusted_flops = flops
        if (
            adjusted_flops is not None
            and timing.get("sparse_lateral_projection_observed")
        ):
            adjusted_flops += timing.get("executed_detail_projection_FLOPs_per_image", 0.0)
        row = {
            "dataset": args.dataset,
            "model": name,
            "image_size": f"{image_size}x{image_size}",
            "sam_image_size": sam_size,
            "total_params": counts["total"],
            "trainable_params": counts["trainable"],
            "model_variant": (
                args.baseline_encoder if name == "deeplabv3plus"
                else args.segformer_variant if name == "segformer" else "n/a"
            ),
            "adapter": args.adapter if name in {"mobilesam", "ours"} else "n/a",
            "feature_scales": args.feature_scales if name in {"mobilesam", "ours"} else "n/a",
            "fusion_version": args.fusion_version if name in {"mobilesam", "ours"} else "n/a",
            "spatial_policy": args.spatial_policy if name in {"mobilesam", "ours"} else "n/a",
            "feature_retention_ratio": args.feature_retention_ratio if name in {"mobilesam", "ours"} else "n/a",
            "active_detail_representation_fraction": (
                ((args.representation_budget - 1) / 2.0) * args.feature_retention_ratio
                if name in {"mobilesam", "ours"} and args.fusion_version == "semantic_budget" else 1.0
            ),
            "FLOPs": adjusted_flops,
            "FLOPs_protocol": "THOP executed module graph plus explicit sparse 1x1 detail projections",
            **timing,
            "device": str(device),
            "protocol": "batch1,eval,no_grad,warmup10,forward_only",
        }
        rows.append(row)
        print(row)
    output = resolve(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if args.merge_csv and resolve(args.merge_csv).exists():
        with resolve(args.merge_csv).open(newline="", encoding="utf-8") as handle:
            old = list(csv.DictReader(handle))
    merged = {
        (r.get("dataset"), r.get("model"), r.get("model_variant", "n/a"), r.get("image_size")): r
        for r in old
    }
    merged.update({
        (r["dataset"], r["model"], r["model_variant"], r["image_size"]): r
        for r in rows
    })
    all_rows = list(merged.values())
    fields = sorted({key for row in all_rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
