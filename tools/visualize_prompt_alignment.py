"""Visualize prompt alignment heatmaps from a trained label-efficient SAM checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize prompt alignment outputs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/NEU_Seg")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-dir", default="runs/prompt_visualizations")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [128, 128, 128],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        dtype=np.uint8,
    )
    return palette[mask]


def normalize_map(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().float().cpu().numpy()
    value = value - value.min()
    value = value / (value.max() + 1e-9)
    return value


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata_path = checkpoint_path.parent / "last_model.pt"
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False) if metadata_path.exists() else {}
    ckpt_args = metadata.get("args", checkpoint.get("args", {}))
    model_args = {
        "num_classes": 4,
        "img_size": ckpt_args.get("img_size", 224),
        "decoder_dim": ckpt_args.get("decoder_dim", 96),
        "adapter_ratio": ckpt_args.get("adapter_ratio", 0.25),
        "use_dapg": ckpt_args.get("use_dapg", False),
        "num_prompt": ckpt_args.get("num_prompt", 8),
        "prompt_version": ckpt_args.get("prompt_version"),
        "prompt_fusion_mode": ckpt_args.get("prompt_fusion_mode", "both"),
    }

    from adasam.datasets.industrial import NEUSegSemanticDataset
    from adasam.models import LabelEfficientSAM

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = NEUSegSemanticDataset(resolve(args.data_root), split=args.split)
    backbone_checkpoint = resolve(ckpt_args.get("checkpoint", ckpt_args.get("backbone_checkpoint", "weights/mobile_sam.pt")))
    model = LabelEfficientSAM.build(
        checkpoint=backbone_checkpoint,
        num_classes=model_args["num_classes"],
        img_size=model_args["img_size"],
        device=device,
        decoder_dim=model_args["decoder_dim"],
        adapter_ratio=model_args["adapter_ratio"],
        use_dapg=model_args["use_dapg"],
        num_prompt=model_args["num_prompt"],
        prompt_version=model_args["prompt_version"],
        prompt_fusion_mode=model_args["prompt_fusion_mode"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    summary = []
    for index in args.indices:
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].numpy()
        with torch.no_grad():
            logits, prompts = model.forward_with_prompts(image)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        row = {
            "index": index,
            "id": sample["id"],
            "pred": pred,
            "mask": mask,
            "image": sample["image"].permute(1, 2, 0).numpy(),
        }
        if isinstance(prompts, dict) and "dense_prompt" in prompts:
            row["prompt_map"] = normalize_map(prompts["dense_prompt"].abs().mean(dim=1).squeeze(0))
        summary.append(row)

    fig, axes = plt.subplots(len(summary), 4, figsize=(14, 4 * len(summary)))
    if len(summary) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, row in enumerate(summary):
        image = np.clip(row["image"], 0.0, 1.0)
        mask_rgb = colorize_mask(row["mask"])
        pred_rgb = colorize_mask(row["pred"])
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(f"Image {row['id']}")
        axes[row_idx, 1].imshow(mask_rgb)
        axes[row_idx, 1].set_title("Ground Truth")
        axes[row_idx, 2].imshow(pred_rgb)
        axes[row_idx, 2].set_title("Prediction")
        if "prompt_map" in row:
            axes[row_idx, 3].imshow(image)
            axes[row_idx, 3].imshow(row["prompt_map"], cmap="magma", alpha=0.55)
            axes[row_idx, 3].set_title("Prompt Heatmap")
        else:
            axes[row_idx, 3].axis("off")
        for col in range(4):
            axes[row_idx, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "prompt_alignment_grid.png", dpi=220, bbox_inches="tight")
    (output_dir / "prompt_alignment_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "data_root": str(resolve(args.data_root)),
                "split": args.split,
                "indices": args.indices,
                "output": "prompt_alignment_grid.png",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={output_dir / 'prompt_alignment_grid.png'}")


if __name__ == "__main__":
    main()
