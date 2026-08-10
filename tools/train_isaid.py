"""Train U-Net or frozen MobileSAM on semantic iSAID tiles."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import islice
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
from adasam.utils import load_static_importance_map, set_seed  # noqa: E402
from tools.train_segmentation import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label-efficient iSAID semantic segmentation")
    p.add_argument("--model", choices=["unet", "mobilesam", "ours"], required=True)
    p.add_argument("--label-ratio", type=int, choices=[1, 5, 10, 20, 50, 100], required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Stop after this many optimizer updates; overrides --epochs when set.",
    )
    p.add_argument(
        "--eval-interval",
        type=int,
        default=2000,
        help="Validation interval in optimizer updates when --max-iterations is set.",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=800)
    p.add_argument("--sam-image-size", type=int, default=800)
    p.add_argument("--decoder-dim", type=int, default=96)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--adapter", choices=["cat", "none"], default="none")
    p.add_argument("--feature-scales", choices=["embedding", "p4_embedding", "p3_p4_embedding"], default="p3_p4_embedding")
    p.add_argument("--fusion-version", choices=["hierarchical", "sum", "global", "image_conditioned", "scsr", "semantic_budget"], default="hierarchical")
    p.add_argument("--representation-budget", type=int, choices=[1, 2, 3], default=3)
    p.add_argument("--spatial-policy", choices=["adaptive", "static", "magnitude", "random"], default="adaptive")
    p.add_argument("--feature-retention-ratio", type=float, default=1.0)
    p.add_argument("--spatial-budget-temperature", type=float, default=1.0)
    p.add_argument("--static-importance-map", default=None)
    p.add_argument("--augmentation", choices=["none", "basic"], default="basic")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--validation-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    if args.epochs <= 0:
        p.error("--epochs must be positive")
    if args.max_iterations is not None and args.max_iterations <= 0:
        p.error("--max-iterations must be positive")
    if args.eval_interval <= 0:
        p.error("--eval-interval must be positive")
    return args


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def training_epoch_limit(num_batches: int, epochs: int, max_iterations: int | None) -> int:
    """Return enough data-loader passes to satisfy the selected stopping rule."""
    if num_batches <= 0:
        raise ValueError("training loader must contain at least one batch")
    return epochs if max_iterations is None else math.ceil(max_iterations / num_batches)


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
            spatial_policy=args.spatial_policy,
            feature_retention_ratio=args.feature_retention_ratio,
            spatial_budget_temperature=args.spatial_budget_temperature,
            static_importance_map=load_static_importance_map(
                resolve(args.static_importance_map) if args.static_importance_map else None
            ),
        )
    criterion = LabelEfficientSegmentationLoss(ignore_index=ISAIDSemanticDataset.IGNORE_INDEX)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    out = resolve(args.output_dir) / f"isaid_ratio{args.label_ratio}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    best_path, best_score, history = out / "best_model.pt", -1.0, []
    max_epochs = training_epoch_limit(len(train_loader), args.epochs, args.max_iterations)
    target_iterations = args.max_iterations or (args.epochs * len(train_loader))
    training_mode = "iterations" if args.max_iterations is not None else "epochs"
    print(
        f"model={args.model} ratio={args.label_ratio}% train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_dataset)} mode={training_mode} "
        f"target_iterations={target_iterations}"
    )
    global_step = 0
    best_iteration = None
    interval_losses: list[float] = []
    interval_started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        remaining = target_iterations - global_step
        steps_this_epoch = min(len(train_loader), remaining)
        batches = islice(train_loader, steps_this_epoch)
        description = (
            f"iterations {global_step}/{target_iterations}"
            if args.max_iterations is not None
            else f"epoch {epoch}/{max_epochs}"
        )
        for batch in tqdm(batches, total=steps_this_epoch, desc=description):
            image, target = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
            epoch_losses.append(loss_value)
            interval_losses.append(loss_value)
            global_step += 1

            validate_by_iteration = args.max_iterations is not None and (
                global_step % args.eval_interval == 0 or global_step == target_iterations
            )
            if validate_by_iteration:
                metrics = evaluate(
                    model, val_loader, device,
                    ISAIDSemanticDataset.NUM_CLASSES, ISAIDSemanticDataset.IGNORE_INDEX,
                )
                record = {
                    "epoch": epoch,
                    "iteration": global_step,
                    "mean_loss": sum(interval_losses) / len(interval_losses),
                    "seconds": time.perf_counter() - interval_started,
                    "validation": metrics,
                }
                history.append(record)
                print(json.dumps(record))
                if metrics["mIoU"] > best_score:
                    best_score = metrics["mIoU"]
                    best_iteration = global_step
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch, "iteration": global_step},
                        best_path,
                    )
                interval_losses = []
                interval_started = time.perf_counter()
                model.train()

        if args.max_iterations is None:
            metrics = evaluate(
                model, val_loader, device,
                ISAIDSemanticDataset.NUM_CLASSES, ISAIDSemanticDataset.IGNORE_INDEX,
            )
            record = {
                "epoch": epoch,
                "iteration": global_step,
                "mean_loss": sum(epoch_losses) / len(epoch_losses),
                "seconds": time.perf_counter() - interval_started,
                "validation": metrics,
            }
            history.append(record)
            print(json.dumps(record))
            if metrics["mIoU"] > best_score:
                best_score = metrics["mIoU"]
                best_iteration = global_step
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "iteration": global_step},
                    best_path,
                )
            interval_losses = []
            interval_started = time.perf_counter()

        if global_step >= target_iterations:
            break
    best = torch.load(best_path, map_location=device, weights_only=False); model.load_state_dict(best["model"])
    test = evaluate(
        model, test_loader, device, ISAIDSemanticDataset.NUM_CLASSES,
        ISAIDSemanticDataset.IGNORE_INDEX, conditioned=True,
    )
    from tools.train_loveda import collect_routing_statistics
    routing_statistics = collect_routing_statistics(
        model, test_loader, device, ISAIDSemanticDataset.NUM_CLASSES,
        ISAIDSemanticDataset.IGNORE_INDEX,
    )
    metrics = {"dataset": "iSAID", "model": args.model, "label_ratio": args.label_ratio, "seed": args.seed,
               "train_samples": len(train_idx), "validation_samples": len(val_idx), "test_samples": len(test_dataset),
               "parameters": model.parameter_counts(), "history": history, "best_epoch": best["epoch"],
               "best_iteration": best.get("iteration", best_iteration), "trained_iterations": global_step,
               "target_iterations": target_iterations,
               "test": test, "args": vars(args), "routing_statistics": routing_statistics}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "metrics": metrics}, out / "last_model.pt")
    print(f"test={json.dumps(test)}\nsaved={out}")


if __name__ == "__main__":
    main()
