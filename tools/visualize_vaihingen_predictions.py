"""Save Vaihingen image, ground-truth, prediction, and error panels."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasam.datasets.industrial import VaihingenSemanticDataset  # noqa: E402
from tools.train_vaihingen import build_model, resolve  # noqa: E402

COLORS = np.array([[255,255,255], [0,0,255], [0,255,255], [0,255,0], [255,255,0], [255,0,0]], np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    return COLORS[np.clip(mask, 0, 5)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--model-checkpoint", required=True)
    p.add_argument("--mobile-sam-checkpoint", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    cli = p.parse_args()
    device = torch.device(cli.device if cli.device != "cuda" or torch.cuda.is_available() else "cpu")
    payload = torch.load(resolve(cli.model_checkpoint), map_location=device, weights_only=False)
    args = argparse.Namespace(**payload["args"])
    if cli.mobile_sam_checkpoint:
        args.checkpoint = cli.mobile_sam_checkpoint
    model = build_model(args, device)
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = VaihingenSemanticDataset(resolve(cli.data_root), "test", args.image_size)
    indices = random.Random(cli.seed).sample(range(len(dataset)), min(cli.num_samples, len(dataset)))
    output = resolve(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    manifest = []
    for index in indices:
        sample = dataset[index]
        image = sample["image"]
        with torch.inference_mode():
            prediction = model(image[None].to(device)).argmax(1)[0].cpu().numpy()
        gt = sample["mask"].numpy()
        original = (image.permute(1,2,0).numpy().clip(0,1) * 255).astype(np.uint8)
        error = np.zeros_like(original); error[prediction == gt] = [40,180,70]; error[prediction != gt] = [220,50,50]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, title, visual in zip(axes, ["IRRG input", "Ground truth", "Prediction", "Error"], [original, colorize(gt), colorize(prediction), error]):
            ax.imshow(visual); ax.set_title(title); ax.axis("off")
        fig.tight_layout()
        path = output / f"{sample['id']}.png"
        fig.savefig(path, dpi=140); plt.close(fig)
        manifest.append({"id": sample["id"], "area_id": sample["area_id"], "path": str(path)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Image.fromarray(COLORS.reshape(1, 6, 3)).resize((600, 100), Image.Resampling.NEAREST).save(output / "palette.png")
    print(f"saved={output} samples={len(manifest)}")


if __name__ == "__main__":
    main()
