"""Go/no-go analysis for semantic-detail disagreement as a boundary-error signal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasam.adapters import inject_tinyvit_lora  # noqa: E402
from adasam.datasets.industrial import LoveDASemanticDataset  # noqa: E402
from adasam.metrics.boundary_difficulty import (  # noqa: E402
    HistogramBinaryMetrics,
    boundary_band,
)
from adasam.models import LabelEfficientSAM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--mobile-sam-checkpoint", default="weights/mobile_sam.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--sam-image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decoder-dim", type=int, default=96)
    parser.add_argument("--decoder-version", default="lightweight",
                        choices=["lightweight", "boundary_aux", "boundary"])
    parser.add_argument("--feature-scales", default="p3_p4_embedding")
    parser.add_argument("--fusion-version", default="sum")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-targets", nargs="+", default=["qkv", "proj"])
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--boundary-radii", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--histogram-bins", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-examples", type=int, default=8)
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def project(feature: torch.Tensor, output_dim: int, size: tuple[int, int]) -> torch.Tensor:
    """Align channels without learned parameters by contiguous channel-group averaging."""
    batch, channels, height, width = feature.shape
    flattened = feature.permute(0, 2, 3, 1).reshape(-1, 1, channels)
    projected = F.adaptive_avg_pool1d(flattened, output_dim)
    projected = projected.reshape(batch, height, width, output_dim).permute(0, 3, 1, 2)
    projected = F.interpolate(projected, size=size, mode="bilinear", align_corners=False)
    return F.normalize(projected.float(), dim=1, eps=1e-6)


def robust_unit_interval(value: torch.Tensor) -> torch.Tensor:
    """Normalize each image by its 99th percentile to limit isolated outliers."""
    flat = value.flatten(1)
    scale = torch.quantile(flat, 0.99, dim=1).clamp_min(1e-6)
    return (value / scale[:, None, None]).clamp(0, 1)


def prediction_gradient(probability: torch.Tensor) -> torch.Tensor:
    dx = F.pad((probability[:, :, :, 1:] - probability[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((probability[:, :, 1:, :] - probability[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return robust_unit_interval(torch.sqrt(dx.square() + dy.square()).amax(1))


def make_signals(features: dict[str, torch.Tensor], probability: torch.Tensor,
                 output_size: tuple[int, int], projection_dim: int) -> dict[str, torch.Tensor]:
    aligned = {
        name: project(value, projection_dim, output_size)
        for name, value in features.items() if name in {"P3", "P4", "embedding"}
    }
    p3_e = (1 - (aligned["P3"] * aligned["embedding"]).sum(1)) * 0.5
    p4_e = (1 - (aligned["P4"] * aligned["embedding"]).sum(1)) * 0.5
    p3_p4 = (1 - (aligned["P3"] * aligned["P4"]).sum(1)) * 0.5
    p3_norm = F.interpolate(features["P3"].float().square().mean(1, keepdim=True).sqrt(),
                            output_size, mode="bilinear", align_corners=False)[:, 0]
    embedding_norm = F.interpolate(
        features["embedding"].float().square().mean(1, keepdim=True).sqrt(),
        output_size, mode="bilinear", align_corners=False,
    )[:, 0]
    return {
        "uncertainty": 1 - probability.amax(1),
        "probability_gradient": prediction_gradient(probability),
        "feature_norm_difference": robust_unit_interval((p3_norm - embedding_norm).abs()),
        "p3_embedding_disagreement": p3_e.clamp(0, 1),
        "p4_embedding_disagreement": p4_e.clamp(0, 1),
        "p3_p4_disagreement": p3_p4.clamp(0, 1),
        "p3_p4_combined": ((p3_e + p4_e) * 0.5).clamp(0, 1),
    }


def new_metric_tree(signal_names: list[str], radii: list[int], classes: int,
                    bins: int) -> dict:
    return {
        signal: {
            radius: {
                "boundary": HistogramBinaryMetrics(bins),
                "all_error": HistogramBinaryMetrics(bins),
                "boundary_error_global": HistogramBinaryMetrics(bins),
                "boundary_error_conditional": HistogramBinaryMetrics(bins),
                "per_class_conditional": [HistogramBinaryMetrics(bins) for _ in range(classes)],
            }
            for radius in radii
        }
        for signal in signal_names
    }


def save_example(output: Path, sample_id: str, image: torch.Tensor, gt: torch.Tensor,
                 pred: torch.Tensor, signals: dict[str, torch.Tensor]) -> None:
    folder = output / "examples" / sample_id
    folder.mkdir(parents=True, exist_ok=True)
    rgb = (image.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(folder / "image.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(folder / "gt.png"), gt.byte().cpu().numpy())
    cv2.imwrite(str(folder / "prediction.png"), pred.byte().cpu().numpy())
    for name, value in signals.items():
        heat = (value.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(folder / f"{name}.png"), cv2.applyColorMap(heat, cv2.COLORMAP_TURBO))


def main() -> None:
    args = parse_args()
    if any(radius < 0 for radius in args.boundary_radii):
        raise ValueError("boundary radii must be non-negative")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = LoveDASemanticDataset(resolve(args.data_root), args.split, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = LabelEfficientSAM.build(
        resolve(args.mobile_sam_checkpoint), num_classes=dataset.NUM_CLASSES,
        img_size=args.sam_image_size, device=device, decoder_dim=args.decoder_dim,
        prompt_version="none", use_cat_adapter=False, decoder_version=args.decoder_version,
        feature_scales=args.feature_scales, fusion_version=args.fusion_version,
    )
    if args.lora_rank > 0:
        model.lora_modules = inject_tinyvit_lora(
            model.backbone.image_encoder, rank=args.lora_rank, alpha=args.lora_alpha,
            targets=tuple(args.lora_targets),
        )
        model.enable_encoder_peft(True)
    payload = torch.load(resolve(args.model_checkpoint), map_location=device, weights_only=False)
    state = payload.get("model", payload)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model configuration mismatch. "
            f"Missing={incompatible.missing_keys[:10]}, unexpected={incompatible.unexpected_keys[:10]}"
        )
    model.eval()
    signal_names = [
        "uncertainty", "probability_gradient", "feature_norm_difference",
        "p3_embedding_disagreement", "p4_embedding_disagreement",
        "p3_p4_disagreement", "p3_p4_combined",
    ]
    raw_metrics = new_metric_tree(signal_names, args.boundary_radii,
                                  dataset.NUM_CLASSES, args.histogram_bins)
    adapted_metrics = new_metric_tree(signal_names, args.boundary_radii,
                                      dataset.NUM_CLASSES, args.histogram_bins)
    processed = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="boundary diagnosis"):
            if args.max_samples is not None and processed >= args.max_samples:
                break
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            if args.max_samples is not None:
                keep = min(len(image), args.max_samples - processed)
                image, target = image[:keep], target[:keep]
            logits, raw_features, adapted_features = model.forward_for_diagnostics(image)
            probability = logits.softmax(1)
            prediction = probability.argmax(1)
            valid = target != dataset.IGNORE_INDEX
            error = prediction != target
            signal_groups = {
                "raw": make_signals(raw_features, probability, target.shape[-2:],
                                    args.projection_dim),
                "adapted": make_signals(adapted_features, probability, target.shape[-2:],
                                        args.projection_dim),
            }
            for group_name, signals in signal_groups.items():
                tree = raw_metrics if group_name == "raw" else adapted_metrics
                for radius in args.boundary_radii:
                    band = boundary_band(target, radius, dataset.IGNORE_INDEX) & valid
                    boundary_error = band & error
                    for signal_name, score in signals.items():
                        metrics = tree[signal_name][radius]
                        metrics["boundary"].update(score, band, valid)
                        metrics["all_error"].update(score, error, valid)
                        metrics["boundary_error_global"].update(score, boundary_error, valid)
                        metrics["boundary_error_conditional"].update(score, error, band)
                        for class_index in range(dataset.NUM_CLASSES):
                            class_band = band & (target == class_index)
                            metrics["per_class_conditional"][class_index].update(
                                score, error, class_band
                            )
            for index in range(len(image)):
                if processed + index >= args.save_examples:
                    break
                save_example(output, str(batch["id"][index]), image[index], target[index],
                             prediction[index], {
                                 key: value[index]
                                 for key, value in signal_groups["raw"].items()
                             } | {
                                 f"adapted_{key}": value[index]
                                 for key, value in signal_groups["adapted"].items()
                             })
            processed += len(image)

    def serialize(tree: dict) -> dict:
        result = {}
        for signal, radius_map in tree.items():
            result[signal] = {}
            for radius, metrics in radius_map.items():
                result[signal][str(radius)] = {
                    key: value.compute() for key, value in metrics.items()
                    if key != "per_class_conditional"
                }
                result[signal][str(radius)]["per_class_conditional"] = {
                    dataset.CLASS_NAMES[index]: metric.compute()
                    for index, metric in enumerate(metrics["per_class_conditional"])
                }
        return result

    report = {
        "protocol": {
            "dataset": "LoveDA", "split": args.split, "samples": processed,
            "image_size": args.image_size, "sam_image_size": args.sam_image_size,
            "model_checkpoint": str(resolve(args.model_checkpoint)),
            "projection": "fixed_contiguous_channel_group_average",
            "projection_dim": args.projection_dim,
            "boundary_radii": args.boundary_radii, "histogram_bins": args.histogram_bins,
        },
        "raw_encoder_features": serialize(raw_metrics),
        "decoder_input_features": serialize(adapted_metrics),
    }
    path = output / "boundary_disagreement_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={path} samples={processed}")


if __name__ == "__main__":
    main()
