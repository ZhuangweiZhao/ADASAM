"""Train the independent label-efficient semantic segmentation baseline."""

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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.datasets.industrial import (  # noqa: E402
    LabelRatioSubset,
    NEUSegSemanticDataset,
    fixed_validation_split_indices,
)
from adasam.datasets.augmentation import build_augmentation  # noqa: E402
from adasam.losses import (  # noqa: E402
    DefectPromptAlignmentLoss,
    LabelEfficientSegmentationLoss,
    PrototypeCompactnessLoss,
)
from adasam.models import LabelEfficientSAM  # noqa: E402
from adasam.utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label-efficient semantic segmentation")
    parser.add_argument("--dataset", choices=["neu_seg"], default="neu_seg")
    parser.add_argument("--label_ratio", type=int, choices=[1, 5, 10, 20, 25, 50, 100], required=True)
    parser.add_argument("--data-root", default="data/NEU_Seg")
    parser.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--decoder-dim", type=int, default=96)
    parser.add_argument("--adapter", choices=["cat", "none"], default="cat")
    parser.add_argument("--use-dapg", action="store_true")
    parser.add_argument("--prompt-version", choices=["none", "v1", "v2", "v3"], default=None)
    parser.add_argument("--prompt-fusion-mode", choices=["both", "dense", "token"], default="both")
    parser.add_argument("--num-prompt", type=int, default=None)
    parser.add_argument("--prompt-align-weight", type=float, default=0.0)
    parser.add_argument("--use-prototype", action="store_true")
    parser.add_argument("--prototype-version", choices=["none", "dpm"], default=None)
    parser.add_argument("--prototype-momentum", type=float, default=0.9)
    parser.add_argument("--prototype-loss-weight", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="runs/label_efficient")
    parser.add_argument("--augmentation", choices=["none", "basic", "defect"], default="none")
    parser.add_argument("--split-protocol", choices=["legacy", "fixed"], default="legacy")
    parser.add_argument("--validation-seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _REPO_ROOT / path


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(
    model: LabelEfficientSAM,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict:
    model.eval()
    intersection = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)
    pred_area = torch.zeros(num_classes, dtype=torch.float64)
    target_area = torch.zeros(num_classes, dtype=torch.float64)
    correct = 0
    pixels = 0
    samples = 0
    synchronize(device)
    started = time.perf_counter()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        prediction = model(image).argmax(dim=1)
        samples += image.shape[0]
        valid = torch.ones_like(target, dtype=torch.bool)
        if ignore_index is not None:
            valid = target != ignore_index
        correct += int(((prediction == target) & valid).sum())
        pixels += int(valid.sum())
        for class_id in range(num_classes):
            pred_class = (prediction == class_id) & valid
            target_class = target == class_id
            inter = (pred_class & target_class).sum().cpu()
            intersection[class_id] += inter
            union[class_id] += (pred_class | target_class).sum().cpu()
            pred_area[class_id] += pred_class.sum().cpu()
            target_area[class_id] += target_class.sum().cpu()
    synchronize(device)
    elapsed = time.perf_counter() - started
    iou = intersection / union.clamp_min(1.0)
    dice = 2.0 * intersection / (pred_area + target_area).clamp_min(1.0)
    return {
        "mIoU": float(iou.mean()),
        "mIoU_fg": float(iou[1:].mean()),
        "Dice": float(dice.mean()),
        "Dice_fg": float(dice[1:].mean()),
        "pixel_accuracy": correct / max(pixels, 1),
        "per_class_IoU": [float(value) for value in iou],
        "per_class_Dice": [float(value) for value in dice],
        "samples": samples,
        "seconds": elapsed,
        "FPS": samples / max(elapsed, 1e-9),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base_dataset = NEUSegSemanticDataset(resolve_path(args.data_root), split="train")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be between 0 and 1")
    train_dataset = NEUSegSemanticDataset(
        resolve_path(args.data_root), split="train", transforms=build_augmentation(args.augmentation)
    )
    validation_dataset = NEUSegSemanticDataset(resolve_path(args.data_root), split="train")
    if args.split_protocol == "fixed":
        train_indices, validation_indices, training_pool = fixed_validation_split_indices(
            len(base_dataset), args.label_ratio, args.seed, args.val_fraction, args.validation_seed
        )
        label_pool_samples = len(train_indices)
        dataset = Subset(train_dataset, train_indices)
        validation = Subset(validation_dataset, validation_indices)
        training_pool_samples = len(training_pool)
    else:
        label_pool = LabelRatioSubset(base_dataset, args.label_ratio, seed=args.seed)
        val_count = max(1, round(len(label_pool) * args.val_fraction))
        if val_count >= len(label_pool):
            raise ValueError("label pool is too small for a non-empty train/validation split")
        validation = Subset(validation_dataset, label_pool.indices[:val_count])
        dataset = Subset(train_dataset, label_pool.indices[val_count:])
        label_pool_samples = len(label_pool)
        training_pool_samples = len(base_dataset)
    test_dataset = NEUSegSemanticDataset(resolve_path(args.data_root), split="test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    prompt_version = args.prompt_version or ("v1" if args.use_dapg else "none")
    prototype_version = args.prototype_version or ("dpm" if args.use_prototype else "none")
    num_prompt = args.num_prompt if args.num_prompt is not None else (8 if prompt_version in {"v2", "v3"} else 16)
    model = LabelEfficientSAM.build(
        resolve_path(args.checkpoint),
        num_classes=base_dataset.NUM_CLASSES,
        img_size=args.img_size,
        device=device,
        decoder_dim=args.decoder_dim,
        use_dapg=args.use_dapg, num_prompt=num_prompt, prompt_version=prompt_version,
        prompt_fusion_mode=args.prompt_fusion_mode,
        use_cat_adapter=args.adapter == "cat",
        prototype_version=prototype_version,
        prototype_momentum=args.prototype_momentum,
    )
    criterion = LabelEfficientSegmentationLoss()
    prompt_criterion = DefectPromptAlignmentLoss()
    prototype_criterion = PrototypeCompactnessLoss()
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    counts = model.parameter_counts()
    print(
        f"parameters total={counts['total']:,} trainable={counts['trainable']:,} "
        f"frozen={counts['frozen']:,} ratio={counts['trainable'] / counts['total']:.2%}"
    )
    print(
        f"dataset={args.dataset} label_ratio={args.label_ratio}% "
        f"label_pool={label_pool_samples} train={len(dataset)} validation={len(validation)} "
        f"split_protocol={args.split_protocol} validation_seed={args.validation_seed}"
    )

    adapter_name = "cat" if args.adapter == "cat" else "no_cat"
    variant = f"dapg_{prompt_version}" if prompt_version != "none" else "baseline"
    if prototype_version == "dpm":
        variant += "+dpm"
    output_dir = resolve_path(args.output_dir) / f"neu_seg_ratio{args.label_ratio}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"variant={variant} adapter={adapter_name} "
        f"num_prompt={num_prompt if prompt_version != 'none' else 0} "
        f"prototype={prototype_version} augmentation={args.augmentation}"
    )
    history = []
    best_score = -1.0
    best_path = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        losses = []
        prompt_losses = []
        prototype_losses = []
        for batch in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}"):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, prompts, prototype_aux = model.forward_with_auxiliary(image, target)
            loss = criterion(prediction, target)
            if args.prompt_align_weight > 0.0 and prompts is not None and "dense_prompt" in prompts:
                align_loss = prompt_criterion(prompts["dense_prompt"], target)
                loss = loss + args.prompt_align_weight * align_loss
                prompt_losses.append(float(align_loss.detach()))
            if prototype_aux is not None and args.prototype_loss_weight > 0.0:
                prototype_loss = prototype_criterion(
                    prototype_aux["source_feature"],
                    target,
                    prototype_aux["prototypes"],
                    prototype_aux["initialized"],
                )
                loss = loss + args.prototype_loss_weight * prototype_loss
                prototype_losses.append(float(prototype_loss.detach()))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        elapsed = time.perf_counter() - started
        record = {
            "epoch": epoch,
            "mean_loss": sum(losses) / len(losses),
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "seconds": elapsed,
        }
        if prompt_losses:
            record["mean_prompt_align_loss"] = sum(prompt_losses) / len(prompt_losses)
        if prototype_losses:
            record["mean_prototype_loss"] = sum(prototype_losses) / len(prototype_losses)
        record["validation"] = evaluate(
            model, validation_loader, device, base_dataset.NUM_CLASSES
        )
        history.append(record)
        print(json.dumps(record))

        if record["validation"]["mIoU_fg"] > best_score:
            best_score = record["validation"]["mIoU_fg"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(model, test_loader, device, base_dataset.NUM_CLASSES)
    checkpoint = {
        "model": model.state_dict(),
        "args": vars(args),
        "parameters": counts,
        "history": history,
        "best_epoch": best["epoch"],
        "test": test_metrics,
        "prompt_align_weight": args.prompt_align_weight,
        "prompt_fusion_mode": args.prompt_fusion_mode,
        "adapter": args.adapter,
        "prototype_version": prototype_version,
        "prototype_momentum": args.prototype_momentum,
        "prototype_loss_weight": args.prototype_loss_weight,
        "augmentation": args.augmentation,
    }
    torch.save(checkpoint, output_dir / "last_model.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "parameters": counts,
                "label_pool_samples": label_pool_samples,
                "train_samples": len(dataset),
                "validation_samples": len(validation),
                "training_pool_samples": training_pool_samples,
                "split_protocol": args.split_protocol,
                "validation_seed": args.validation_seed,
                "history": history,
                "best_epoch": best["epoch"],
                "test": test_metrics,
                "adapter": args.adapter,
                "prototype_version": prototype_version,
                "prototype_momentum": args.prototype_momentum,
                "prototype_loss_weight": args.prototype_loss_weight,
                "augmentation": args.augmentation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"test={json.dumps(test_metrics)}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
