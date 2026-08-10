"""Create the three evidence visualizations for semantic budget allocation on LoveDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasam.datasets.industrial import LoveDASemanticDataset  # noqa: E402
from adasam.models import LabelEfficientSAM  # noqa: E402
from adasam.utils import load_static_importance_map  # noqa: E402


COLORS = np.asarray([
    [255, 255, 255], [220, 20, 60], [255, 215, 0], [30, 144, 255],
    [210, 180, 140], [34, 139, 34], [154, 205, 50], [0, 0, 0],
], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize LoveDA hierarchy and budget allocation")
    parser.add_argument("--data-root", default="data/LoveDA")
    parser.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--sam-image-size", type=int, default=1024)
    parser.add_argument("--decoder-dim", type=int, default=96)
    parser.add_argument("--adapter", choices=["cat", "none"], default="none")
    parser.add_argument("--representation-budget", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--spatial-policy", choices=["adaptive", "static", "magnitude", "random"], default="adaptive")
    parser.add_argument("--feature-retention-ratio", type=float, default=0.5)
    parser.add_argument("--static-importance-map", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def color_mask(mask: np.ndarray) -> np.ndarray:
    safe = mask.copy()
    safe[safe == 255] = 7
    return COLORS[np.clip(safe, 0, len(COLORS) - 1)]


def response_map(feature: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    response = feature.square().mean(1, keepdim=True).sqrt()
    response = F.interpolate(response, size=size, mode="bilinear", align_corners=False)[0, 0]
    response = response - response.min()
    response = response / response.max().clamp_min(1e-6)
    return response.cpu().numpy()


def selection_score(mask: np.ndarray) -> float:
    background = float((mask == 0).mean())
    building = (mask == 1).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(building, 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.asarray([])
    small = int((areas <= mask.size * 0.001).sum())
    large = int((areas > mask.size * 0.01).sum())
    return background + min(small, 20) / 20.0 + min(large, 3) / 3.0


def select_indices(dataset: LoveDASemanticDataset, count: int) -> list[int]:
    scored = []
    for index in range(len(dataset)):
        sample = dataset[index]
        scored.append((selection_score(sample["mask"].numpy()), index))
    return [index for _, index in sorted(scored, reverse=True)[:count]]


def save_panel(path: Path, titles: list[str], images: list[np.ndarray], cmaps: list[str | None]) -> None:
    figure, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), constrained_layout=True)
    for axis, title, image, cmap in zip(axes, titles, images, cmaps):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_condition_statistics(metrics_path: Path, output: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    routing = metrics.get("routing_statistics") or {}
    spatial = routing.get("spatial_budget")
    if not spatial:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "size_retention_ratio", "Retention by target size"),
        (axes[1], "region_retention_ratio", "Retention by spatial region"),
    ):
        groups = list(spatial[key])
        x = np.arange(len(groups))
        axis.bar(x - 0.18, [spatial[key][g]["P3"] for g in groups], 0.36, label="P3")
        axis.bar(x + 0.18, [spatial[key][g]["P4"] for g in groups], 0.36, label="P4")
        axis.set_xticks(x, groups, rotation=15)
        axis.set_ylim(0, 1)
        axis.set_ylabel("retained feature ratio")
        axis.set_title(title)
        axis.legend()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = LoveDASemanticDataset(resolve(args.data_root), args.split, args.image_size)
    indices = args.indices or select_indices(dataset, args.num_samples)
    model = LabelEfficientSAM.build(
        resolve(args.checkpoint), LoveDASemanticDataset.NUM_CLASSES,
        img_size=args.sam_image_size, decoder_dim=args.decoder_dim,
        use_cat_adapter=args.adapter == "cat", fusion_version="semantic_budget",
        representation_budget=args.representation_budget,
        spatial_policy=args.spatial_policy,
        feature_retention_ratio=args.feature_retention_ratio,
        static_importance_map=load_static_importance_map(
            resolve(args.static_importance_map) if args.static_importance_map else None
        ),
        device=device,
    )
    payload = torch.load(resolve(args.model_checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model", payload), strict=True)
    model.eval()
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.no_grad():
        for index in indices:
            sample = dataset[index]
            image = sample["image"].unsqueeze(0).to(device)
            target = sample["mask"].numpy()
            features = model.backbone(model._preprocess(image))
            logits = model(image)
            prediction = logits.argmax(1)[0].cpu().numpy()
            routing = model.decoder.last_routing
            original = image[0].permute(1, 2, 0).cpu().numpy()
            size = tuple(image.shape[-2:])
            layer_maps = [response_map(features[name], size) for name in ("P3", "P4", "embedding")]
            save_panel(
                output / f"{sample['id']}_layer_response.png",
                ["Original", "GT", "P3 response", "P4 response", "Embedding response"],
                [original, color_mask(target), *layer_maps],
                [None, None, "magma", "magma", "magma"],
            )
            importance = routing["importance_maps"].mean(1, keepdim=True)
            retained = routing["retained_masks"].amax(1, keepdim=True)
            importance = F.interpolate(importance, size=size, mode="bilinear", align_corners=False)[0, 0]
            importance = importance - importance.min()
            importance = importance / importance.max().clamp_min(1e-6)
            retained = F.interpolate(retained, size=size, mode="nearest")[0, 0]
            save_panel(
                output / f"{sample['id']}_allocation.png",
                ["Original", "GT", "Importance", "Retained features", "Prediction"],
                [original, color_mask(target), importance.cpu().numpy(), retained.cpu().numpy(), color_mask(prediction)],
                [None, None, "inferno", "gray", None],
            )
            records.append({
                "index": index, "id": sample["id"],
                "policy": routing["spatial_policy"],
                "target_retention_ratio": routing["target_retention_ratio"],
                "actual_retention_ratio": routing["retention_ratio"].mean(0).cpu().tolist(),
                "selected_detail_scales": routing["budget_mask"][0].cpu().tolist(),
            })
    (output / "visualization_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    metrics_path = resolve(args.metrics) if args.metrics else resolve(args.model_checkpoint).parent / "metrics.json"
    if metrics_path.exists():
        plot_condition_statistics(metrics_path, output / "conditioned_retention.png")
    print(f"saved={output} samples={len(records)}")


if __name__ == "__main__":
    main()
