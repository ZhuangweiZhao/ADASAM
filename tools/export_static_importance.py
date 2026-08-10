"""Export the train-set average spatial importance for the strict static-mask control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasam.datasets.industrial import (  # noqa: E402
    LoveDASemanticDataset,
    fixed_validation_split_indices,
)
from adasam.models import LabelEfficientSAM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export average adaptive importance on LoveDA train")
    parser.add_argument("--data-root", default="data/LoveDA")
    parser.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--sam-image-size", type=int, default=1024)
    parser.add_argument("--decoder-dim", type=int, default=96)
    parser.add_argument("--label-ratio", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    full_dataset = LoveDASemanticDataset(resolve(args.data_root), "train", args.image_size)
    train_indices, _, _ = fixed_validation_split_indices(
        len(full_dataset), args.label_ratio, args.seed,
        args.val_fraction, args.validation_seed,
    )
    dataset = Subset(full_dataset, train_indices)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = LabelEfficientSAM.build(
        resolve(args.checkpoint), LoveDASemanticDataset.NUM_CLASSES,
        img_size=args.sam_image_size, decoder_dim=args.decoder_dim,
        use_cat_adapter=False, fusion_version="semantic_budget",
        representation_budget=3, spatial_policy="adaptive", device=device,
    )
    payload = torch.load(resolve(args.model_checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model", payload), strict=True)
    model.eval()
    importance_sum = None
    samples = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="average train importance"):
            image = batch["image"].to(device, non_blocking=True)
            model(image)
            importance = model.decoder.last_routing["importance_maps"].double()
            batch_sum = importance.sum(dim=0).cpu()
            importance_sum = batch_sum if importance_sum is None else importance_sum + batch_sum
            samples += image.shape[0]
    if not samples or importance_sum is None:
        raise RuntimeError("training dataset produced no importance maps")
    mean_importance = (importance_sum / samples).float()
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "mean_importance": mean_importance,
        "samples": samples,
        "source_split": "the exact labeled LoveDA/Train subset used by the paired adaptive run",
        "label_ratio": args.label_ratio,
        "experiment_seed": args.seed,
        "validation_seed": args.validation_seed,
        "sample_indices": train_indices,
        "source_checkpoint": str(resolve(args.model_checkpoint)),
        "protocol": "per-pixel arithmetic mean before top-ratio selection",
        "args": vars(args),
    }
    torch.save(artifact, output)
    metadata = {key: value for key, value in artifact.items() if key != "mean_importance"}
    metadata["shape"] = list(mean_importance.shape)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved={output} shape={tuple(mean_importance.shape)} samples={samples}")


if __name__ == "__main__":
    main()
