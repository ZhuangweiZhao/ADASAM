"""Train U-Net or frozen MobileSAM on semantic iSAID tiles."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasam.datasets.augmentation import build_augmentation  # noqa: E402
from adasam.datasets.industrial import ISAIDSemanticDataset, fixed_validation_split_indices  # noqa: E402
from adasam.losses import LabelEfficientSegmentationLoss  # noqa: E402
from adasam.models import LabelEfficientSAM, LabelEfficientUNet  # noqa: E402
from adasam.utils import set_seed  # noqa: E402
from tools.train_segmentation import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label-efficient iSAID semantic segmentation")
    p.add_argument("--model", choices=["unet", "mobilesam", "ours"], required=True)
    p.add_argument("--label-ratio", type=int, choices=[1, 5, 10, 20, 50, 100], required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=800)
    p.add_argument("--sam-image-size", type=int, default=800)
    p.add_argument("--decoder-dim", type=int, default=96)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--adapter", choices=["cat", "none"], default="none")
    p.add_argument("--feature-scales", choices=["embedding", "p4_embedding", "p3_p4_embedding"], default="p3_p4_embedding")
    p.add_argument("--fusion-version", choices=["hierarchical", "global", "image_conditioned", "scsr", "semantic_budget"], default="hierarchical")
    p.add_argument("--representation-budget", type=int, choices=[1, 2, 3], default=3)
    p.add_argument("--augmentation", choices=["none", "basic"], default="basic")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--validation-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    root = resolve(args.data_root)
    selection = ISAIDSemanticDataset(root, "train", args.image_size)
    train_idx, val_idx, pool = fixed_validation_split_indices(
        len(selection), args.label_ratio, args.seed, args.val_fraction, args.validation_seed
    )
    train_base = ISAIDSemanticDataset(root, "train", args.image_size, transforms=build_augmentation(args.augmentation))
    val_base = ISAIDSemanticDataset(root, "train", args.image_size)
    test_dataset = ISAIDSemanticDataset(root, "val", args.image_size)
    options = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(Subset(train_base, train_idx), shuffle=True, **options)
    val_loader = DataLoader(Subset(val_base, val_idx), shuffle=False, **options)
    test_loader = DataLoader(test_dataset, shuffle=False, **options)
    if args.model == "unet":
        model = LabelEfficientUNet(ISAIDSemanticDataset.NUM_CLASSES, args.base_channels).to(device)
    else:
        model = LabelEfficientSAM.build(
            resolve(args.checkpoint), ISAIDSemanticDataset.NUM_CLASSES,
            img_size=args.sam_image_size, device=device, decoder_dim=args.decoder_dim,
            prompt_version="v2" if args.model == "ours" else "none", num_prompt=8,
            use_cat_adapter=args.adapter == "cat", feature_scales=args.feature_scales,
            fusion_version=args.fusion_version,
            representation_budget=args.representation_budget,
        )
    criterion = LabelEfficientSegmentationLoss(ignore_index=ISAIDSemanticDataset.IGNORE_INDEX)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    out = resolve(args.output_dir) / f"isaid_ratio{args.label_ratio}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    best_path, best_score, history = out / "best_model.pt", -1.0, []
    print(f"model={args.model} ratio={args.label_ratio}% train={len(train_idx)} val={len(val_idx)} test={len(test_dataset)}")
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []; started = time.perf_counter()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            image, target = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image)
            loss = criterion(prediction, target)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        metrics = evaluate(model, val_loader, device, ISAIDSemanticDataset.NUM_CLASSES, ISAIDSemanticDataset.IGNORE_INDEX)
        record = {"epoch": epoch, "mean_loss": sum(losses) / len(losses), "seconds": time.perf_counter() - started, "validation": metrics}
        history.append(record); print(json.dumps(record))
        if metrics["mIoU"] > best_score:
            best_score = metrics["mIoU"]; torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)
    best = torch.load(best_path, map_location=device, weights_only=False); model.load_state_dict(best["model"])
    test = evaluate(model, test_loader, device, ISAIDSemanticDataset.NUM_CLASSES, ISAIDSemanticDataset.IGNORE_INDEX)
    from tools.train_loveda import collect_routing_statistics
    routing_statistics = collect_routing_statistics(
        model, test_loader, device, ISAIDSemanticDataset.NUM_CLASSES,
        ISAIDSemanticDataset.IGNORE_INDEX,
    )
    metrics = {"dataset": "iSAID", "model": args.model, "label_ratio": args.label_ratio, "seed": args.seed,
               "train_samples": len(train_idx), "validation_samples": len(val_idx), "test_samples": len(test_dataset),
               "parameters": model.parameter_counts(), "history": history, "best_epoch": best["epoch"],
               "test": test, "args": vars(args), "routing_statistics": routing_statistics}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "metrics": metrics}, out / "last_model.pt")
    print(f"test={json.dumps(test)}\nsaved={out}")


if __name__ == "__main__":
    main()
