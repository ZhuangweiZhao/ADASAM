"""Evaluate a Vaihingen checkpoint on the held-out test areas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasam.datasets.industrial import VaihingenSemanticDataset  # noqa: E402
from tools.train_segmentation import evaluate  # noqa: E402
from tools.train_vaihingen import add_official_metrics, build_model, resolve  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--model-checkpoint", required=True)
    p.add_argument("--mobile-sam-checkpoint", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    cli = p.parse_args()
    device = torch.device(cli.device if cli.device != "cuda" or torch.cuda.is_available() else "cpu")
    payload = torch.load(resolve(cli.model_checkpoint), map_location=device, weights_only=False)
    args = argparse.Namespace(**payload["args"])
    if cli.mobile_sam_checkpoint:
        args.checkpoint = cli.mobile_sam_checkpoint
    model = build_model(args, device)
    model.load_state_dict(payload["model"])
    dataset = VaihingenSemanticDataset(resolve(cli.data_root), "test", args.image_size)
    loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False, num_workers=cli.num_workers, pin_memory=device.type == "cuda")
    result = add_official_metrics(evaluate(model, loader, device, 6, 255, conditioned=True))
    result["per_class"] = dict(zip(VaihingenSemanticDataset.CLASS_NAMES, result["per_class_IoU"]))
    output = resolve(cli.output) if cli.output else resolve(cli.model_checkpoint).with_name("evaluation.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
