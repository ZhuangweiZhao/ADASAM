"""Select auditable 5% LoveDA subsets from frozen MobileSAM representations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasam.backbone import LabelEfficientMobileSAMBackbone  # noqa: E402
from adasam.datasets.industrial import (  # noqa: E402
    LoveDASemanticDataset,
    fixed_validation_split_indices,
)
from adasam.datasets.selection import sample_id_fingerprint  # noqa: E402
from adasam.utils.transforms import PIXEL_MEAN, PIXEL_STD  # noqa: E402


METHODS = ("random", "embedding_kcenter", "hrcs")
LEVELS = ("P3", "P4", "embedding")


class UnlabeledLoveDAPool(Dataset):
    """Read only RGB images; masks are deliberately never opened."""

    def __init__(
        self,
        samples: list[tuple[Path, Path | None, str]],
        indices: list[int],
        image_size: int,
    ) -> None:
        self.samples = samples
        self.indices = indices
        self.image_size = (image_size, image_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, object]:
        dataset_index = self.indices[position]
        image_path, _, sample_id = self.samples[dataset_index]
        image_array = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_array is None:
            raise ValueError(f"cannot read LoveDA image: {image_path}")
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = torch.from_numpy(image_array).permute(2, 0, 1)
        image = F.interpolate(
            image.unsqueeze(0), self.image_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        return {"image": image, "index": dataset_index, "id": sample_id}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "weights" / "mobile_sam.pt"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-ratio", type=int, default=5, choices=[1, 5, 10, 20, 25, 50])
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--sam-image-size", type=int, default=512)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    return parser.parse_args()


def pca_normalize(features: torch.Tensor, output_dim: int) -> torch.Tensor:
    """Fit deterministic full-SVD PCA and return L2-normalized descriptors."""
    if features.ndim != 2:
        raise ValueError("features must have shape [samples, channels]")
    centered = features.float() - features.float().mean(0, keepdim=True)
    rank = min(output_dim, centered.shape[0] - 1, centered.shape[1])
    if rank < 1:
        raise ValueError("PCA requires at least two samples and one feature")
    _, _, right = torch.linalg.svd(centered, full_matrices=False)
    projected = centered @ right[:rank].T
    return F.normalize(projected, dim=1, eps=1e-8)


def distance_to_point(
    descriptors: dict[str, torch.Tensor], point: int, levels: tuple[str, ...]
) -> torch.Tensor:
    distances = [1.0 - descriptors[level] @ descriptors[level][point] for level in levels]
    return torch.stack(distances).mean(0).clamp_min(0.0)


def deterministic_kcenter(
    descriptors: dict[str, torch.Tensor], count: int, levels: tuple[str, ...]
) -> list[int]:
    """Greedy farthest-first traversal with a deterministic central first point."""
    size = next(iter(descriptors.values())).shape[0]
    if not 0 < count <= size:
        raise ValueError("selection count must be between one and pool size")
    centrality = torch.stack(
        [descriptors[level] @ F.normalize(descriptors[level].mean(0), dim=0) for level in levels]
    ).mean(0)
    first = int(centrality.argmax())
    selected = [first]
    minimum = distance_to_point(descriptors, first, levels)
    minimum[first] = -1.0
    while len(selected) < count:
        point = int(minimum.argmax())
        selected.append(point)
        minimum = torch.minimum(minimum, distance_to_point(descriptors, point, levels))
        minimum[selected] = -1.0
    return selected


def coverage_statistics(
    descriptors: dict[str, torch.Tensor], selected: list[int], levels: tuple[str, ...]
) -> dict[str, float]:
    minimum = torch.full((next(iter(descriptors.values())).shape[0],), float("inf"))
    for point in selected:
        minimum = torch.minimum(minimum, distance_to_point(descriptors, point, levels))
    return {
        "mean_nearest_distance": float(minimum.mean()),
        "max_nearest_distance": float(minimum.max()),
    }


@torch.no_grad()
def extract_descriptors(
    backbone: LabelEfficientMobileSAMBackbone,
    loader: DataLoader,
    device: torch.device,
    pca_dim: int,
) -> tuple[dict[str, torch.Tensor], list[int], list[str]]:
    raw = {level: [] for level in LEVELS}
    indices: list[int] = []
    sample_ids: list[str] = []
    mean = torch.tensor(PIXEL_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(PIXEL_STD, device=device).view(1, 3, 1, 1)
    backbone.eval()
    for batch in tqdm(loader, desc="extract frozen MobileSAM features"):
        image = batch["image"].to(device, non_blocking=True)
        features = backbone((image - mean) / std)
        for level in LEVELS:
            raw[level].append(features[level].mean(dim=(-2, -1)).cpu())
        indices.extend(int(index) for index in batch["index"])
        sample_ids.extend(str(sample_id) for sample_id in batch["id"])
    descriptors = {
        level: pca_normalize(torch.cat(raw[level]), pca_dim) for level in LEVELS
    }
    return descriptors, indices, sample_ids


def main() -> None:
    args = parse_args()
    if args.pca_dim <= 0:
        raise ValueError("--pca-dim must be positive")
    data_root = Path(args.data_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LoveDASemanticDataset(data_root, "train", args.sam_image_size)
    random_selected, validation_indices, training_pool = fixed_validation_split_indices(
        len(dataset), args.label_ratio, args.selection_seed,
        args.val_fraction, args.validation_seed,
    )
    selected_count = len(random_selected)
    pool_dataset = UnlabeledLoveDAPool(dataset.samples, training_pool, args.sam_image_size)
    loader = DataLoader(
        pool_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    backbone = LabelEfficientMobileSAMBackbone.build(
        checkpoint, device=device, img_size=args.sam_image_size
    )
    descriptors, ordered_indices, ordered_ids = extract_descriptors(
        backbone, loader, device, args.pca_dim
    )
    if ordered_indices != training_pool:
        raise RuntimeError("feature extraction changed the fixed training-pool order")

    random_positions = [training_pool.index(index) for index in random_selected]
    selections: dict[str, list[int]] = {"random": random_positions}
    if "embedding_kcenter" in args.methods:
        selections["embedding_kcenter"] = deterministic_kcenter(
            descriptors, selected_count, ("embedding",)
        )
    if "hrcs" in args.methods:
        selections["hrcs"] = deterministic_kcenter(descriptors, selected_count, LEVELS)

    dataset_ids = [sample[2] for sample in dataset.samples]
    common = {
        "schema_version": 1,
        "dataset": "LoveDA",
        "dataset_size": len(dataset),
        "dataset_fingerprint": sample_id_fingerprint(dataset_ids),
        "training_pool_size": len(training_pool),
        "training_pool_fingerprint": sample_id_fingerprint(ordered_ids),
        "validation_samples": len(validation_indices),
        "validation_seed": args.validation_seed,
        "validation_fraction": args.val_fraction,
        "label_ratio": args.label_ratio,
        "selection_seed": args.selection_seed,
        "pca_dim": args.pca_dim,
        "sam_image_size": args.sam_image_size,
        "uses_ground_truth": False,
    }
    summary: dict[str, object] = {"protocol": common, "methods": {}}
    for method in args.methods:
        positions = selections[method]
        selected_indices = [training_pool[position] for position in positions]
        coverage = {
            "embedding": coverage_statistics(descriptors, positions, ("embedding",)),
            "hierarchical": coverage_statistics(descriptors, positions, LEVELS),
        }
        manifest = {
            **common,
            "strategy": method,
            "selected_indices": selected_indices,
            "selected_sample_ids": [dataset_ids[index] for index in selected_indices],
            "coverage": coverage,
        }
        path = output_dir / f"{method}_ratio{args.label_ratio}.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        summary["methods"][method] = {
            "manifest": str(path), "coverage": coverage,
            "selected_samples": len(selected_indices),
        }
        print(f"saved={path}")
    (output_dir / "selection_statistics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
