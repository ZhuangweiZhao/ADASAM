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
    p.add_argument("--dataset", choices=["loveda", "neu_seg"], required=True)
    p.add_argument("--models", nargs="+", choices=["unet", "mobilesam", "ours"], default=["mobilesam"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    p.add_argument("--model-checkpoints", nargs="*", default=[])
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--sam-image-size", type=int, default=None)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--decoder-dim", type=int, default=96)
    p.add_argument("--adapter", choices=["cat", "none"], default="cat")
    p.add_argument("--feature-scales", choices=["p3", "p4", "embedding", "p3_p4", "p3_embedding", "p4_embedding", "p3_p4_embedding"], default="p3_p4_embedding")
    p.add_argument("--fusion-version", choices=["hierarchical", "concat", "global", "image_conditioned", "scsr"], default="hierarchical")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-csv", default="runs/efficiency_results.csv")
    p.add_argument("--merge-csv", default=None)
    return p.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_model(args: argparse.Namespace, name: str, checkpoint: Path, device: torch.device):
    from adasam.models import LabelEfficientSAM, LabelEfficientUNet

    if args.dataset == "loveda":
        classes = 7
    else:
        classes = 4
    if name == "unet":
        return LabelEfficientUNet(classes, args.base_channels).to(device)
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
    allocated = reserved = 0.0
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    return {
        "images": total_images,
        "inference_seconds": elapsed,
        "FPS": total_images / max(elapsed, 1e-12),
        "peak_memory_allocated_MB": allocated,
        "peak_memory_reserved_MB": reserved,
    }


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
    else:
        from adasam.datasets.industrial import NEUSegSemanticDataset
        dataset = NEUSegSemanticDataset(resolve(args.data_root), "test")
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
        row = {
            "dataset": args.dataset,
            "model": name,
            "image_size": f"{image_size}x{image_size}",
            "sam_image_size": sam_size,
            "total_params": counts["total"],
            "trainable_params": counts["trainable"],
            "adapter": args.adapter if name != "unet" else "n/a",
            "feature_scales": args.feature_scales if name != "unet" else "n/a",
            "fusion_version": args.fusion_version if name != "unet" else "n/a",
            "FLOPs": flops,
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
    merged = {(r.get("dataset"), r.get("model"), r.get("image_size")): r for r in old}
    merged.update({(r["dataset"], r["model"], r["image_size"]): r for r in rows})
    all_rows = list(merged.values())
    fields = sorted({key for row in all_rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
