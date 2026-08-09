"""Train U-Net, Frozen MobileSAM, or DAPG-v2 on LoveDA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasam.datasets.augmentation import build_augmentation  # noqa: E402
from adasam.datasets.industrial import LoveDASemanticDataset, fixed_validation_split_indices  # noqa: E402
from adasam.losses import BoundaryLoss, LabelEfficientSegmentationLoss  # noqa: E402
from adasam.models import LabelEfficientSAM, LabelEfficientUNet  # noqa: E402
from adasam.utils import set_seed  # noqa: E402
from tools.train_segmentation import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label-efficient LoveDA segmentation")
    parser.add_argument("--model", choices=["unet", "mobilesam", "ours"], required=True)
    parser.add_argument("--label-ratio", type=int, choices=[1, 5, 10, 20, 25, 50, 100], required=True)
    parser.add_argument("--data-root", default="data/LoveDA")
    parser.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--sam-image-size", type=int, default=224)
    parser.add_argument("--decoder-dim", type=int, default=96)
    parser.add_argument("--adapter", choices=["cat", "none"], default="cat")
    parser.add_argument("--adapter-placement", choices=["pre_fusion", "post_fusion"], default="pre_fusion")
    parser.add_argument("--feature-scales", choices=["p3", "p4", "embedding", "p3_p4", "p3_embedding", "p4_embedding", "p3_p4_embedding"], default="p3_p4_embedding")
    parser.add_argument("--fusion-version", choices=["hierarchical", "concat", "global", "image_conditioned", "scsr", "scsr_v2", "semantic_budget"], default="hierarchical")
    parser.add_argument("--representation-budget", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--decoder-version", choices=["lightweight", "boundary_aux", "boundary"], default="lightweight")
    parser.add_argument("--boundary-loss-weight", type=float, default=0.1)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--augmentation", choices=["none", "basic"], default="basic")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="runs/loveda")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@torch.no_grad()
def collect_routing_statistics(
    model, loader, device, num_classes: int, ignore_index: int | None
):
    if getattr(model.decoder, "fusion_version", None) not in {"scsr", "scsr_v2", "semantic_budget"}:
        return None
    weight_sum = torch.zeros(3, dtype=torch.float64)
    dominant = torch.zeros(3, dtype=torch.float64)
    class_sum = torch.zeros(num_classes, 3, dtype=torch.float64)
    class_pixels = torch.zeros(num_classes, dtype=torch.float64)
    entropy_sum = 0.0
    selected_sum = torch.zeros(2, dtype=torch.float64)
    selected_samples = 0
    pixels = 0
    model.eval()
    for batch in loader:
        target = batch["mask"].to(device, non_blocking=True)
        model(batch["image"].to(device, non_blocking=True))
        routing = model.decoder.last_routing
        if "budget_mask" in routing:
            selected_sum += routing["budget_mask"].sum(0).cpu().double()
            selected_samples += routing["budget_mask"].shape[0]
        weights = routing["weights"]
        routed_target = F.interpolate(target[:, None].float(), weights.shape[-2:], mode="nearest")[:, 0].long()
        valid = (
            torch.ones_like(routed_target, dtype=torch.bool)
            if ignore_index is None
            else routed_target != ignore_index
        )
        count = int(valid.sum())
        if not count:
            continue
        weight_sum += weights.permute(1, 0, 2, 3)[:, valid].sum(1).cpu().double()
        dominant += torch.stack([((weights.argmax(1) == i) & valid).sum() for i in range(3)]).cpu().double()
        entropy_sum += float(routing["entropy"][valid].sum().cpu())
        pixels += count
        for class_id in range(num_classes):
            selected = valid & (routed_target == class_id)
            class_count = int(selected.sum())
            if class_count:
                class_sum[class_id] += weights.permute(1, 0, 2, 3)[:, selected].sum(1).cpu().double()
                class_pixels[class_id] += class_count
    names = ("P3", "P4", "embedding")
    return {
        "scale_names": names,
        "mean_weights": {name: float(weight_sum[i] / pixels) for i, name in enumerate(names)},
        "mean_entropy": entropy_sum / pixels,
        "dominant_pixel_fraction": {name: float(dominant[i] / pixels) for i, name in enumerate(names)},
        "class_mean_weights": {
            str(c): {name: float(class_sum[c, i] / class_pixels[c]) for i, name in enumerate(names)}
            for c in range(num_classes) if class_pixels[c] > 0
        },
        "pixels": pixels,
        "representation_budget": getattr(model.decoder, "representation_budget", None),
        "selected_scale_fraction": (
            {name: float(selected_sum[i] / max(1, selected_samples)) for i, name in enumerate(("P3", "P4"))}
            if selected_samples else None
        ),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = resolve(args.data_root)
    selection_dataset = LoveDASemanticDataset(data_root, "train", args.image_size)
    train_indices, validation_indices, training_pool = fixed_validation_split_indices(
        len(selection_dataset), args.label_ratio, args.seed, args.val_fraction, args.validation_seed
    )
    train_base = LoveDASemanticDataset(
        data_root, "train", args.image_size, transforms=build_augmentation(args.augmentation)
    )
    validation_base = LoveDASemanticDataset(data_root, "train", args.image_size)
    official_validation = LoveDASemanticDataset(data_root, "val", args.image_size)
    train_dataset = Subset(train_base, train_indices)
    validation_dataset = Subset(validation_base, validation_indices)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(official_validation, shuffle=False, **loader_options)

    if args.model == "unet":
        model = LabelEfficientUNet(
            LoveDASemanticDataset.NUM_CLASSES, args.base_channels
        ).to(device)
    else:
        model = LabelEfficientSAM.build(
            resolve(args.checkpoint),
            num_classes=LoveDASemanticDataset.NUM_CLASSES,
            img_size=args.sam_image_size,
            device=device,
            decoder_dim=args.decoder_dim,
            prompt_version="v2" if args.model == "ours" else "none",
            num_prompt=8,
            prompt_fusion_mode="both",
            use_cat_adapter=args.adapter == "cat",
            decoder_version=args.decoder_version,
            feature_scales=args.feature_scales,
            adapter_placement=args.adapter_placement,
            fusion_version=args.fusion_version,
            representation_budget=args.representation_budget,
        )
    if args.model == "unet" and args.decoder_version != "lightweight":
        raise ValueError("boundary decoder variants are only available for MobileSAM models")
    criterion = LabelEfficientSegmentationLoss(ignore_index=LoveDASemanticDataset.IGNORE_INDEX)
    boundary_criterion = BoundaryLoss(ignore_index=LoveDASemanticDataset.IGNORE_INDEX)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    counts = model.parameter_counts()
    output_dir = resolve(args.output_dir) / f"loveda_ratio{args.label_ratio}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    best_score = -1.0
    history = []
    print(
        f"model={args.model} ratio={args.label_ratio}% train={len(train_dataset)} "
        f"validation={len(validation_dataset)} official_val={len(official_validation)} "
        f"augmentation={args.augmentation} image_size={args.image_size}"
    )
    print(
        f"parameters total={counts['total']:,} trainable={counts['trainable']:,} "
        f"frozen={counts['frozen']:,}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.decoder_version == "lightweight":
                prediction = model(image)
                boundary_logits = None
            else:
                prediction, _, auxiliary = model.forward_with_auxiliary(image, target)
                boundary_logits = auxiliary["boundary_logits"] if auxiliary is not None else None
            loss = criterion(prediction, target)
            if boundary_logits is not None and args.boundary_loss_weight > 0.0:
                loss = loss + args.boundary_loss_weight * boundary_criterion(boundary_logits, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        elapsed = time.perf_counter() - started
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            LoveDASemanticDataset.NUM_CLASSES,
            ignore_index=LoveDASemanticDataset.IGNORE_INDEX,
        )
        record = {
            "epoch": epoch,
            "mean_loss": sum(losses) / len(losses),
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "seconds": elapsed,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record))
        if validation_metrics["mIoU"] > best_score:
            best_score = validation_metrics["mIoU"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        LoveDASemanticDataset.NUM_CLASSES,
        ignore_index=LoveDASemanticDataset.IGNORE_INDEX,
    )
    routing_statistics = collect_routing_statistics(
        model, test_loader, device, LoveDASemanticDataset.NUM_CLASSES,
        LoveDASemanticDataset.IGNORE_INDEX,
    )
    metrics = {
        "dataset": "LoveDA",
        "model": args.model,
        "augmentation": args.augmentation,
        "label_ratio": args.label_ratio,
        "label_pool_samples": len(train_dataset),
        "training_pool_samples": len(training_pool),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(official_validation),
        "split_protocol": "fixed_train_validation_official_val_test",
        "validation_seed": args.validation_seed,
        "parameters": counts,
        "history": history,
        "best_epoch": best["epoch"],
        "test": test_metrics,
        "args": vars(args),
        "adapter": args.adapter,
        "adapter_placement": args.adapter_placement,
        "feature_scales": args.feature_scales,
        "fusion_version": args.fusion_version,
        "routing_statistics": routing_statistics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "metrics": metrics}, output_dir / "last_model.pt")
    print(f"test={json.dumps(test_metrics)}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
