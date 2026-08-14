"""Train frozen MobileSAM semantic segmentation on pre-cut Vaihingen patches."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasam.datasets.augmentation import build_augmentation  # noqa: E402
from adasam.adapters import inject_tinyvit_lora  # noqa: E402
from adasam.datasets.industrial import VaihingenSemanticDataset  # noqa: E402
from adasam.losses import LabelEfficientSegmentationLoss  # noqa: E402
from adasam.models import LabelEfficientSAM  # noqa: E402
from adasam.utils import set_seed  # noqa: E402
from tools.train_segmentation import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train frozen MobileSAM on Vaihingen")
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    p.add_argument("--output-dir", default="runs/vaihingen")
    p.add_argument("--run-name", default=None, help="Explicit leaf directory; prevents configuration collisions")
    p.add_argument(
        "--protocol", choices=["development", "official_full_train"],
        default="development",
        help="official_full_train uses all 16 training areas and a fixed final epoch",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--sam-image-size", type=int, default=512)
    p.add_argument("--decoder-dim", type=int, default=64)
    p.add_argument("--feature-scales", default="p3_p4_embedding")
    p.add_argument("--fusion-version", default="semantic_budget")
    p.add_argument("--representation-budget", type=int, default=3)
    p.add_argument("--spatial-policy", choices=["adaptive", "static", "magnitude", "random"], default="magnitude")
    p.add_argument("--feature-retention-ratio", type=float, default=0.25)
    p.add_argument("--label-ratio", type=int, choices=[10, 25, 50, 100], default=100)
    p.add_argument("--val-area-fraction", type=float, default=0.25)
    p.add_argument("--validation-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--class-balanced-ce", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--input-adapter", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lora-rank", type=int, default=0, help="0 disables TinyViT attention LoRA")
    p.add_argument("--lora-alpha", type=float, default=8.0)
    p.add_argument("--lora-targets", nargs="+", choices=["qkv", "proj"], default=["qkv", "proj"])
    p.add_argument("--augmentation", choices=["none", "basic"], default="basic")
    p.add_argument(
        "--evaluate-test", action=argparse.BooleanOptionalAction, default=True,
        help="Evaluate the held-out test areas after training",
    )
    p.add_argument(
        "--conditioned-validation", action=argparse.BooleanOptionalAction, default=False,
        help="Compute small/medium/large component IoU on the development validation split",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def area_split(dataset: VaihingenSemanticDataset, val_fraction: float, val_seed: int, label_ratio: int, seed: int):
    areas = sorted({area for _, _, area in dataset.samples})
    shuffled = areas[:]
    random.Random(val_seed).shuffle(shuffled)
    count = max(1, round(len(areas) * val_fraction))
    validation_areas = set(shuffled[:count])
    train_pool = [i for i, (_, _, area) in enumerate(dataset.samples) if area not in validation_areas]
    validation = [i for i, (_, _, area) in enumerate(dataset.samples) if area in validation_areas]
    random.Random(seed).shuffle(train_pool)
    selected_count = max(1, round(len(train_pool) * label_ratio / 100))
    return sorted(train_pool[:selected_count]), validation, sorted(validation_areas), len(train_pool)


def build_model(args: argparse.Namespace, device: torch.device) -> LabelEfficientSAM:
    model = LabelEfficientSAM.build(
        resolve(args.checkpoint), num_classes=VaihingenSemanticDataset.NUM_CLASSES,
        img_size=args.sam_image_size, device=device, decoder_dim=args.decoder_dim,
        prompt_version="none", use_cat_adapter=False,
        feature_scales=args.feature_scales, fusion_version=args.fusion_version,
        representation_budget=args.representation_budget, spatial_policy=args.spatial_policy,
        feature_retention_ratio=args.feature_retention_ratio,
        use_input_adapter=args.input_adapter,
    )
    if getattr(args, "lora_rank", 0) > 0:
        model.lora_modules = inject_tinyvit_lora(
            model.backbone.image_encoder,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            targets=tuple(args.lora_targets),
        )
        model.encoder_requires_grad = True
    return model.to(device)


def class_weights(dataset: VaihingenSemanticDataset, indices: list[int]) -> torch.Tensor:
    """Compute inverse-square-root frequency weights from the selected training masks."""
    counts = torch.zeros(dataset.NUM_CLASSES, dtype=torch.float64)
    for index in indices:
        mask_path = dataset.samples[index][1]
        import cv2
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read Vaihingen mask: {mask_path}")
        values = torch.from_numpy(mask.astype("int64"))
        valid = (values >= 0) & (values < dataset.NUM_CLASSES)
        counts += torch.bincount(values[valid], minlength=dataset.NUM_CLASSES)
    weights = counts.sum().div(counts.clamp_min(1)).sqrt()
    return (weights / weights.mean()).float()


def add_official_metrics(result: dict) -> dict:
    """Add the five foreground-class metrics used by common Vaihingen papers."""
    result = dict(result)
    result["mIoU_5"] = sum(result["per_class_IoU"][:5]) / 5
    result["mean_F1_5"] = sum(result["per_class_Dice"][:5]) / 5
    result["official_classes"] = VaihingenSemanticDataset.CLASS_NAMES[:5]
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    root = resolve(args.data_root)
    selection = VaihingenSemanticDataset(root, "train", args.image_size)
    if args.protocol == "official_full_train":
        if args.label_ratio != 100:
            raise ValueError("official_full_train requires --label-ratio 100")
        train_indices = list(range(len(selection)))
        val_indices, val_areas, pool_size = [], [], len(selection)
    else:
        train_indices, val_indices, val_areas, pool_size = area_split(
            selection, args.val_area_fraction, args.validation_seed, args.label_ratio, args.seed
        )
    train_base = VaihingenSemanticDataset(root, "train", args.image_size, build_augmentation(args.augmentation))
    test_set = (
        VaihingenSemanticDataset(root, "test", args.image_size)
        if args.evaluate_test else None
    )
    options = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if args.num_workers:
        options["persistent_workers"] = True
    train_loader = DataLoader(Subset(train_base, train_indices), shuffle=True, **options)
    val_loader = None
    if val_indices:
        val_base = VaihingenSemanticDataset(root, "train", args.image_size)
        val_loader = DataLoader(Subset(val_base, val_indices), shuffle=False, **options)
    test_loader = (
        DataLoader(test_set, shuffle=False, **options) if test_set is not None else None
    )

    model = build_model(args, device)
    ce_weights = class_weights(selection, train_indices).to(device) if args.class_balanced_ce else None
    criterion = LabelEfficientSegmentationLoss(
        ignore_index=VaihingenSemanticDataset.IGNORE_INDEX, class_weights=ce_weights
    )
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    default_name = (
        f"{args.fusion_version}_dense_ratio{args.label_ratio}_seed{args.seed}"
        if args.fusion_version != "semantic_budget"
        else f"semantic_budget_{args.spatial_policy}_r{args.feature_retention_ratio:g}_ratio{args.label_ratio}_seed{args.seed}"
    )
    if args.lora_rank > 0:
        default_name = f"{default_name}_lora_r{args.lora_rank}"
    output = resolve(args.output_dir) / (args.run_name or default_name)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best_model.pt"
    best_score, history = -1.0, []
    test_samples = len(test_set) if test_set is not None else 0
    print(f"device={device} amp={scaler.is_enabled()} train={len(train_indices)}/{pool_size} val={len(val_indices)} areas={val_areas} test={test_samples}")
    print(f"parameters={model.parameter_counts()}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, started = [], time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"), 1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                raw_loss = criterion(model(image), target)
                loss = raw_loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(raw_loss.detach()))
        scheduler.step()
        validation = (
            add_official_metrics(evaluate(
                model, val_loader, device, 6, 255,
                conditioned=args.conditioned_validation,
            ))
            if val_loader is not None else None
        )
        record = {"epoch": epoch, "loss": sum(losses) / len(losses), "lr": scheduler.get_last_lr()[0], "seconds": time.perf_counter() - started, "validation": validation}
        history.append(record)
        print(json.dumps(record))
        if validation is not None and validation["mIoU_5"] > best_score:
            best_score = validation["mIoU_5"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args), "validation_areas": val_areas}, best_path)

    if args.protocol == "official_full_train":
        torch.save(
            {"model": model.state_dict(), "epoch": args.epochs, "args": vars(args), "validation_areas": []},
            best_path,
        )

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    test = (
        add_official_metrics(evaluate(model, test_loader, device, 6, 255, conditioned=True))
        if test_loader is not None else None
    )
    metrics = {"dataset": "ISPRS Vaihingen", "classes": VaihingenSemanticDataset.CLASS_NAMES,
               "best_epoch": payload["epoch"], "validation_areas": val_areas,
               "train_samples": len(train_indices), "validation_samples": len(val_indices),
               "test_samples": test_samples, "split_protocol": args.protocol,
               "train_areas": sorted({selection.samples[i][2] for i in train_indices}),
               "checkpoint_selection": "fixed_final_epoch" if args.protocol == "official_full_train" else "validation_mIoU_5",
               "history": history, "test": test, "args": vars(args)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"test={json.dumps(test)}\nsaved={output}")


if __name__ == "__main__":
    main()
