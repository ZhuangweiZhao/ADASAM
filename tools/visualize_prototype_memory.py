"""Visualize DAPG and Defect Prototype Memory responses from a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize prototype similarity and response")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/NEU_Seg")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--class-id", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--output-dir", default="runs/prototype_visualizations")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _REPO_ROOT / path


def normalize_map(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().float().cpu().numpy()
    low, high = np.percentile(value, (2.0, 98.0))
    value = np.clip(value, low, high)
    return (value - value.min()) / (value.max() - value.min() + 1e-9)


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata_path = checkpoint_path.parent / "last_model.pt"
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False) if metadata_path.exists() else {}
    saved_args = metadata.get("args", checkpoint.get("args", {}))

    from adasam.datasets.industrial import NEUSegSemanticDataset
    from adasam.models import LabelEfficientSAM

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    prompt_version = saved_args.get("prompt_version") or ("v1" if saved_args.get("use_dapg") else "none")
    num_prompt = saved_args.get("num_prompt") or (8 if prompt_version in {"v2", "v3"} else 16)
    model = LabelEfficientSAM.build(
        checkpoint=resolve(saved_args.get("checkpoint", "weights/mobile_sam.pt")),
        num_classes=4,
        img_size=saved_args.get("img_size", 224),
        device=device,
        decoder_dim=saved_args.get("decoder_dim", 96),
        use_dapg=saved_args.get("use_dapg", False),
        num_prompt=num_prompt,
        prompt_version=prompt_version,
        prompt_fusion_mode=saved_args.get("prompt_fusion_mode", "both"),
        use_cat_adapter=saved_args.get("adapter", "cat") == "cat",
        prototype_version=metadata.get("prototype_version", saved_args.get("prototype_version")) or "dpm",
        prototype_momentum=metadata.get("prototype_momentum", saved_args.get("prototype_momentum", 0.9)),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = NEUSegSemanticDataset(resolve(args.data_root), split=args.split)
    rows = []
    for index in args.indices:
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits, prompts, auxiliary = model.forward_with_auxiliary(image)
        if auxiliary is None:
            raise RuntimeError("checkpoint/model does not contain enabled Defect Prototype Memory")
        response = auxiliary["prototype_response"].abs().mean(dim=1).squeeze(0)
        rows.append(
            (
                sample["image"].permute(1, 2, 0).numpy(),
                sample["mask"].numpy(),
                logits.argmax(1).squeeze(0).cpu().numpy(),
                normalize_map(auxiliary["similarity"][0, args.class_id]),
                normalize_map(response),
                normalize_map(prompts["dense_prompt"].abs().mean(1).squeeze(0))
                if isinstance(prompts, dict) and "dense_prompt" in prompts else None,
                sample["id"],
            )
        )
    columns = 6
    figure, axes = plt.subplots(len(rows), columns, figsize=(19, 3.4 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, 0)
    titles = ["Image", "Ground Truth", "Prediction", f"Class {args.class_id} Similarity", "Prototype Response", "DAPG Prompt"]
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row[:6]):
            if value is None:
                axes[row_index, column].axis("off")
                continue
            axes[row_index, column].imshow(value, cmap=None if column == 0 else ("tab10" if column in {1, 2} else "inferno"))
            axes[row_index, column].set_title(titles[column])
            axes[row_index, column].axis("off")
    figure.tight_layout()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"prototype_class{args.class_id}_grid.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
