"""Train the independent label-efficient semantic segmentation baseline."""

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
    BoundaryLoss,
    boundary_f1_counts,
    DefectPromptAlignmentLoss,
    LabelEfficientSegmentationLoss,
    PrototypeCompactnessLoss,
)
from adasam.models import LabelEfficientSAM, LabelEfficientUNet, build_baseline  # noqa: E402
from adasam.metrics import component_iou_sums, summarize_component_iou  # noqa: E402
from adasam.utils import load_static_importance_map, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label-efficient semantic segmentation")
    parser.add_argument("--dataset", choices=["neu_seg"], default="neu_seg")
    parser.add_argument("--model", choices=["mobilesam", "dapg", "unet", "deeplabv3plus", "segformer"], default="mobilesam")
    parser.add_argument("--label_ratio", type=int, choices=[1, 5, 10, 20, 25, 50, 100], required=True)
    parser.add_argument("--baseline-encoder", choices=["resnet50", "resnet101", "mobilenet_v2"], default="resnet50")
    parser.add_argument("--segformer-variant", choices=["b0", "b1", "b2"], default="b0")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="use ImageNet-1k pretrained backbones for the baseline models")
    parser.add_argument("--data-root", default="data/NEU_Seg")
    parser.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--decoder-dim", type=int, default=96)
    parser.add_argument(
        "--feature-scales",
        choices=["p3", "p4", "embedding", "p3_p4", "p3_embedding", "p4_embedding", "p3_p4_embedding"],
        default="p3_p4_embedding",
    )
    parser.add_argument(
        "--fusion-version",
        choices=["hierarchical", "concat", "sum", "global", "image_conditioned", "scsr", "scsr_v2", "scsr_task", "semantic_budget", "semantic_progressive", "semantic_progressive_v2", "semantic_progressive_v3", "regional_semantic"],
        default="hierarchical",
    )
    parser.add_argument("--representation-budget", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--spatial-policy", choices=["adaptive", "static", "magnitude", "random"], default="adaptive")
    parser.add_argument("--feature-retention-ratio", type=float, default=1.0)
    parser.add_argument("--spatial-budget-temperature", type=float, default=1.0)
    parser.add_argument("--static-importance-map", default=None)
    parser.add_argument("--routing-loss-weight", type=float, default=0.1)
    parser.add_argument("--routing-aux-weight", type=float, default=0.05)
    parser.add_argument("--routing-target-temperature", type=float, default=0.25)
    parser.add_argument("--routing-warmup-epochs", type=int, default=10)
    parser.add_argument("--routing-hard", action="store_true")
    parser.add_argument("--decoder-version", choices=["lightweight", "boundary_aux", "boundary"], default="lightweight")
    parser.add_argument("--boundary-loss-weight", type=float, default=0.1)
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


def task_utility_routing_loss(
    model: LabelEfficientSAM,
    target: torch.Tensor,
    ignore_index: int | None = None,
    target_temperature: float = 0.5,
) -> dict[str, torch.Tensor] | None:
    """Supervise routing with detached per-scale pixel-wise task utility."""
    if getattr(getattr(model, "decoder", None), "fusion_version", None) != "scsr_task":
        return None
    routing = model.decoder.last_routing
    if routing is None or "scale_logits" not in routing or "route_weights" not in routing:
        raise RuntimeError("scsr_task requires routing outputs from the decoder")
    if target_temperature <= 0.0:
        raise ValueError("target_temperature must be positive")
    weights = routing["route_weights"]
    target_small = F.interpolate(
        target[:, None].float(), size=weights.shape[-2:], mode="nearest"
    )[:, 0].long()
    ignore_value = ignore_index if ignore_index is not None else -100
    pixel_losses = torch.stack(
        [
            F.cross_entropy(
                logits, target_small, reduction="none", ignore_index=ignore_value
            )
            for logits in routing["scale_logits"]
        ],
        dim=1,
    )
    valid = target_small != ignore_value
    if not bool(valid.any()):
        zero = weights.sum() * 0.0
        return {"route_loss": zero, "aux_loss": zero, "target_entropy": zero}
    utility_target = torch.softmax(
        -pixel_losses.detach() / target_temperature, dim=1
    )
    route_loss_map = -(
        utility_target * weights.clamp_min(1e-8).log()
    ).sum(dim=1)
    route_loss = route_loss_map[valid].mean()
    aux_loss = pixel_losses.permute(0, 2, 3, 1)[valid].mean()
    target_entropy = (
        -utility_target.clamp_min(1e-8)
        * utility_target.clamp_min(1e-8).log()
    ).sum(dim=1)[valid].mean()
    return {
        "route_loss": route_loss,
        "aux_loss": aux_loss,
        "target_entropy": target_entropy.detach(),
    }


@torch.no_grad()
def evaluate(
    model: LabelEfficientSAM,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int | None = None,
    conditioned: bool = False,
) -> dict:
    model.eval()
    intersection = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)
    pred_area = torch.zeros(num_classes, dtype=torch.float64)
    target_area = torch.zeros(num_classes, dtype=torch.float64)
    correct = 0
    pixels = 0
    samples = 0
    boundary_matched_pred = boundary_predicted = 0
    boundary_matched_target = boundary_target = 0
    boundary_metric_seconds = 0.0
    conditioned_metric_seconds = 0.0
    component_sums = torch.zeros(3, dtype=torch.float64)
    component_counts = torch.zeros(3, dtype=torch.int64)
    synchronize(device)
    started = time.perf_counter()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        prediction = model(image).argmax(dim=1)
        synchronize(device)
        if conditioned:
            conditioned_started = time.perf_counter()
            batch_sums, batch_counts = component_iou_sums(
                prediction, target, num_classes, ignore_index
            )
            component_sums += batch_sums
            component_counts += batch_counts
            conditioned_metric_seconds += time.perf_counter() - conditioned_started
        boundary_started = time.perf_counter()
        tolerance = max(1, round(2 * max(target.shape[-2:]) / 1024))
        mp, npred, mt, ntarget = boundary_f1_counts(
            prediction, target, ignore_index=ignore_index, tolerance=tolerance
        )
        boundary_matched_pred += mp
        boundary_predicted += npred
        boundary_matched_target += mt
        boundary_target += ntarget
        synchronize(device)
        boundary_metric_seconds += time.perf_counter() - boundary_started
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
    elapsed = (
        time.perf_counter() - started - boundary_metric_seconds - conditioned_metric_seconds
    )
    iou = intersection / union.clamp_min(1.0)
    dice = 2.0 * intersection / (pred_area + target_area).clamp_min(1.0)
    boundary_precision = boundary_matched_pred / max(boundary_predicted, 1)
    boundary_recall = boundary_matched_target / max(boundary_target, 1)
    boundary_f1 = (
        2.0 * boundary_precision * boundary_recall
        / max(boundary_precision + boundary_recall, 1e-12)
    )
    result = {
        "mIoU": float(iou.mean()),
        "mIoU_fg": float(iou[1:].mean()),
        "Dice": float(dice.mean()),
        "Dice_fg": float(dice[1:].mean()),
        "pixel_accuracy": correct / max(pixels, 1),
        "per_class_IoU": [float(value) for value in iou],
        "per_class_Dice": [float(value) for value in dice],
        "Boundary_F1": boundary_f1,
        "Boundary_precision": boundary_precision,
        "Boundary_recall": boundary_recall,
        "samples": samples,
        "seconds": elapsed,
        "FPS": samples / max(elapsed, 1e-9),
    }
    if conditioned:
        result["size_conditioned_region_IoU"] = summarize_component_iou(
            component_sums, component_counts
        )
        result["size_conditioned_protocol"] = {
            "unit": "8-connected foreground GT component",
            "small_max_image_fraction": 0.001,
            "medium_max_image_fraction": 0.01,
            "evaluation_window_padding_pixels": 2,
        }
    return result


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
    if args.model == "unet":
        model = LabelEfficientUNet(base_dataset.NUM_CLASSES, base_channels=32).to(device)
    elif args.model in {"deeplabv3plus", "segformer"}:
        model = build_baseline(
            args.model,
            num_classes=base_dataset.NUM_CLASSES,
            pretrained=args.pretrained,
            encoder_name=args.baseline_encoder,
            segformer_variant=args.segformer_variant,
            weights_root=_REPO_ROOT / "weights",
            device=device,
        )
    else:
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
            decoder_version=args.decoder_version,
            feature_scales=args.feature_scales,
            fusion_version=args.fusion_version,
            representation_budget=args.representation_budget,
            spatial_policy=args.spatial_policy,
            feature_retention_ratio=args.feature_retention_ratio,
            spatial_budget_temperature=args.spatial_budget_temperature,
            static_importance_map=load_static_importance_map(
                resolve_path(args.static_importance_map) if args.static_importance_map else None
            ),
        )
    criterion = LabelEfficientSegmentationLoss()
    prompt_criterion = DefectPromptAlignmentLoss()
    prototype_criterion = PrototypeCompactnessLoss()
    boundary_criterion = BoundaryLoss()
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
    if args.model == "deeplabv3plus":
        run_name = f"deeplabv3plus_{args.baseline_encoder}_ratio{args.label_ratio}_seed{args.seed}"
    elif args.model == "segformer":
        run_name = f"segformer_{args.segformer_variant}_ratio{args.label_ratio}_seed{args.seed}"
    else:
        run_name = f"neu_seg_ratio{args.label_ratio}_seed{args.seed}"
    output_dir = resolve_path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"variant={variant} adapter={adapter_name} "
        f"num_prompt={num_prompt if prompt_version != 'none' else 0} "
        f"prototype={prototype_version} augmentation={args.augmentation} "
        f"features={args.feature_scales} fusion={args.fusion_version} "
        f"budget={args.representation_budget}"
    )
    history = []
    best_score = -1.0
    best_path = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        if hasattr(model, "decoder"):
            model.decoder.task_routing_hard = (
                args.routing_hard and epoch > args.routing_warmup_epochs
            )
        started = time.perf_counter()
        losses = []
        prompt_losses = []
        prototype_losses = []
        routing_losses = []
        routing_aux_losses = []
        routing_target_entropies = []
        for batch in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}"):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, prompts, prototype_aux = model.forward_with_auxiliary(image, target)
            loss = criterion(prediction, target)
            routing = task_utility_routing_loss(
                model,
                target,
                target_temperature=args.routing_target_temperature,
            )
            if routing is not None:
                loss = (
                    loss
                    + (
                        args.routing_loss_weight * routing["route_loss"]
                        if epoch > args.routing_warmup_epochs
                        else 0.0 * routing["route_loss"]
                    )
                    + args.routing_aux_weight * routing["aux_loss"]
                )
                routing_losses.append(float(routing["route_loss"].detach()))
                routing_aux_losses.append(float(routing["aux_loss"].detach()))
                routing_target_entropies.append(float(routing["target_entropy"]))
            if args.prompt_align_weight > 0.0 and prompts is not None and "dense_prompt" in prompts:
                align_loss = prompt_criterion(prompts["dense_prompt"], target)
                loss = loss + args.prompt_align_weight * align_loss
                prompt_losses.append(float(align_loss.detach()))
            if (
                prototype_aux is not None
                and "source_feature" in prototype_aux
                and args.prototype_loss_weight > 0.0
            ):
                prototype_loss = prototype_criterion(
                    prototype_aux["source_feature"],
                    target,
                    prototype_aux["prototypes"],
                    prototype_aux["initialized"],
                )
                loss = loss + args.prototype_loss_weight * prototype_loss
                prototype_losses.append(float(prototype_loss.detach()))
            if (
                args.decoder_version != "lightweight"
                and prototype_aux is not None
                and "boundary_logits" in prototype_aux
                and args.boundary_loss_weight > 0.0
            ):
                boundary_loss = boundary_criterion(prototype_aux["boundary_logits"], target)
                loss = loss + args.boundary_loss_weight * boundary_loss
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
        if routing_losses:
            record["mean_routing_loss"] = sum(routing_losses) / len(routing_losses)
            record["mean_routing_aux_loss"] = sum(routing_aux_losses) / len(routing_aux_losses)
            record["mean_routing_target_entropy"] = sum(routing_target_entropies) / len(routing_target_entropies)
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
    if hasattr(model, "decoder"):
        model.decoder.task_routing_hard = (
            args.routing_hard and args.epochs > args.routing_warmup_epochs
        )
    test_metrics = evaluate(
        model, test_loader, device, base_dataset.NUM_CLASSES, conditioned=True
    )
    from tools.train_loveda import collect_routing_statistics

    routing_statistics = collect_routing_statistics(
        model,
        test_loader,
        device,
        base_dataset.NUM_CLASSES,
        ignore_index=None,
        target_temperature=args.routing_target_temperature,
    )
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
        "decoder_version": args.decoder_version,
        "boundary_loss_weight": args.boundary_loss_weight,
        "routing_statistics": routing_statistics,
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
                "decoder_version": args.decoder_version,
                "boundary_loss_weight": args.boundary_loss_weight,
                "args": vars(args),
                "routing_statistics": routing_statistics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"test={json.dumps(test_metrics)}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
