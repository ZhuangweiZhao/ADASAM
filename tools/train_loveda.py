"""Train U-Net, Frozen MobileSAM, or DAPG-v2 on LoveDA."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasam.datasets.augmentation import build_augmentation  # noqa: E402
from adasam.datasets.industrial import LoveDASemanticDataset, fixed_validation_split_indices  # noqa: E402
from adasam.losses import (  # noqa: E402
    BoundaryLoss,
    LabelEfficientSegmentationLoss,
    MagnitudeTeacherDistillationLoss,
    semantic_boundary_target,
)
from adasam.metrics import SIZE_NAMES, component_size_map  # noqa: E402
from adasam.models import LabelEfficientSAM, LabelEfficientUNet, build_baseline  # noqa: E402
from adasam.utils import load_static_importance_map, set_seed  # noqa: E402
from tools.train_segmentation import evaluate, task_utility_routing_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label-efficient LoveDA segmentation")
    parser.add_argument(
        "--model",
        choices=["unet", "mobilesam", "mobilesam_finetune", "ours", "deeplabv3plus", "segformer"],
        required=True,
    )
    parser.add_argument("--label-ratio", type=int, choices=[1, 5, 10, 20, 25, 50, 100], required=True)
    parser.add_argument("--baseline-encoder", choices=["resnet50", "resnet101", "mobilenet_v2"], default="resnet50")
    parser.add_argument("--segformer-variant", choices=["b0", "b1", "b2"], default="b0")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="use ImageNet-1k pretrained backbones for the baseline models")
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
    parser.add_argument("--fusion-version", choices=["hierarchical", "concat", "sum", "global", "image_conditioned", "scsr", "scsr_v2", "scsr_task", "semantic_budget", "semantic_progressive"], default="hierarchical")
    parser.add_argument("--representation-budget", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--spatial-policy", choices=["adaptive", "static", "magnitude", "distilled_magnitude", "random"], default="adaptive")
    parser.add_argument("--feature-retention-ratio", type=float, default=1.0)
    parser.add_argument("--spatial-budget-temperature", type=float, default=1.0)
    parser.add_argument("--static-importance-map", default=None)
    parser.add_argument("--magnitude-distill-weight", type=float, default=1.0)
    parser.add_argument("--routing-loss-weight", type=float, default=0.1)
    parser.add_argument("--routing-aux-weight", type=float, default=0.05)
    parser.add_argument("--routing-target-temperature", type=float, default=0.25)
    parser.add_argument("--routing-warmup-epochs", type=int, default=10)
    parser.add_argument("--routing-hard", action="store_true")
    parser.add_argument(
        "--progressive-aux-weight", type=float, default=0.2,
        help="coarse semantic supervision weight for semantic_progressive fusion",
    )
    parser.add_argument("--decoder-version", choices=["lightweight", "boundary_aux", "boundary"], default="lightweight")
    parser.add_argument("--boundary-loss-weight", type=float, default=0.1)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--augmentation", choices=["none", "basic", "remote_strong"], default="basic")
    parser.add_argument(
        "--rural-sampling-multiplier", type=float, default=1.0,
        help="relative sampling weight for Rural training tiles; 1.0 preserves uniform sampling",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--class-balanced-ce", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument(
        "--train-batch-norm", action="store_true",
        help="update DeepLabV3+ BatchNorm statistics (requires physical batch size > 1)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="runs/loveda")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def estimate_class_weights(
    dataset: LoveDASemanticDataset,
    indices: list[int],
) -> torch.Tensor:
    """Estimate stable inverse-sqrt pixel-frequency weights on the labeled subset."""
    counts = torch.zeros(dataset.NUM_CLASSES, dtype=torch.float64)
    for index in indices:
        target = dataset[index]["mask"]
        valid = target != dataset.IGNORE_INDEX
        counts += torch.bincount(
            target[valid], minlength=dataset.NUM_CLASSES
        ).to(torch.float64)
    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"labeled subset contains no pixels for classes {missing}")
    weights = counts.sum().sqrt() / counts.sqrt()
    weights = (weights / weights.mean()).clamp(max=5.0)
    return weights.to(torch.float32)


@torch.no_grad()
def collect_routing_statistics(
    model,
    loader,
    device,
    num_classes: int,
    ignore_index: int | None,
    target_temperature: float = 0.5,
):
    if getattr(getattr(model, "decoder", None), "fusion_version", None) not in {"scsr", "scsr_v2", "scsr_task", "semantic_budget", "semantic_progressive"}:
        return None
    weight_sum = torch.zeros(3, dtype=torch.float64)
    dominant = torch.zeros(3, dtype=torch.float64)
    class_sum = torch.zeros(num_classes, 3, dtype=torch.float64)
    class_pixels = torch.zeros(num_classes, dtype=torch.float64)
    entropy_sum = 0.0
    selected_sum = torch.zeros(2, dtype=torch.float64)
    selected_samples = 0
    oracle_agreement = 0
    oracle_pixels = 0
    oracle_scale_sum = torch.zeros(3, dtype=torch.float64)
    oracle_entropy_sum = 0.0
    region_names = ("background", "foreground_interior", "boundary")
    region_retained_sum = {name: torch.zeros(2, dtype=torch.float64) for name in region_names}
    region_importance_sum = {name: torch.zeros(2, dtype=torch.float64) for name in region_names}
    region_pixels = {name: 0 for name in region_names}
    size_retained_sum = {name: torch.zeros(2, dtype=torch.float64) for name in SIZE_NAMES}
    size_importance_sum = {name: torch.zeros(2, dtype=torch.float64) for name in SIZE_NAMES}
    size_pixels = {name: 0 for name in SIZE_NAMES}
    domain_weight_sum: dict[str, torch.Tensor] = {}
    domain_pixels: dict[str, int] = {}
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
        if getattr(model.decoder, "fusion_version", None) == "scsr_task":
            ignore_value = ignore_index if ignore_index is not None else -100
            scale_losses = torch.stack(
                [
                    F.cross_entropy(
                        logits,
                        routed_target,
                        reduction="none",
                        ignore_index=ignore_value,
                    )
                    for logits in routing["scale_logits"]
                ],
                dim=1,
            )
            utility_target = torch.softmax(
                -scale_losses / target_temperature, dim=1
            )
            oracle = utility_target.argmax(1)
            predicted = weights.argmax(1)
            oracle_agreement += int(((predicted == oracle) & valid).sum())
            oracle_pixels += count
            oracle_scale_sum += torch.stack(
                [(oracle == index)[valid].sum() for index in range(3)]
            ).cpu().double()
            oracle_entropy_sum += float(
                (
                    -utility_target.clamp_min(1e-8)
                    * utility_target.clamp_min(1e-8).log()
                ).sum(1)[valid].sum().cpu()
            )
        weight_sum += weights.permute(1, 0, 2, 3)[:, valid].sum(1).cpu().double()
        dominant += torch.stack([((weights.argmax(1) == i) & valid).sum() for i in range(3)]).cpu().double()
        entropy_sum += float(routing["entropy"][valid].sum().cpu())
        pixels += count
        sample_ids = batch.get("id", [""] * target.shape[0])
        for sample_index, sample_id in enumerate(sample_ids):
            domain = str(sample_id).split("_", 1)[0].lower()
            sample_valid = valid[sample_index]
            sample_count = int(sample_valid.sum())
            if sample_count:
                domain_weight_sum.setdefault(domain, torch.zeros(3, dtype=torch.float64))
                domain_pixels[domain] = domain_pixels.get(domain, 0) + sample_count
                domain_weight_sum[domain] += (
                    weights[sample_index, :, sample_valid].sum(1).cpu().double()
                )
        if "retained_masks" in routing:
            retained = routing["retained_masks"]
            importance = routing["importance_maps"]
            boundary, _ = semantic_boundary_target(routed_target, ignore_index)
            foreground = valid & (routed_target != 0)
            region_masks = {
                "background": valid & (routed_target == 0),
                "foreground_interior": foreground & ~boundary,
                "boundary": boundary & valid,
            }
            strata = component_size_map(routed_target, num_classes, ignore_index).to(retained.device)
            for name, selected_region in region_masks.items():
                selected_count = int(selected_region.sum())
                if selected_count:
                    region_pixels[name] += selected_count
                    region_retained_sum[name] += retained.permute(1, 0, 2, 3)[:, selected_region].sum(1).cpu().double()
                    region_importance_sum[name] += importance.permute(1, 0, 2, 3)[:, selected_region].sum(1).cpu().double()
            for band, name in enumerate(SIZE_NAMES):
                selected_region = (strata == band) & valid
                selected_count = int(selected_region.sum())
                if selected_count:
                    size_pixels[name] += selected_count
                    size_retained_sum[name] += retained.permute(1, 0, 2, 3)[:, selected_region].sum(1).cpu().double()
                    size_importance_sum[name] += importance.permute(1, 0, 2, 3)[:, selected_region].sum(1).cpu().double()
        for class_id in range(num_classes):
            selected = valid & (routed_target == class_id)
            class_count = int(selected.sum())
            if class_count:
                class_sum[class_id] += weights.permute(1, 0, 2, 3)[:, selected].sum(1).cpu().double()
                class_pixels[class_id] += class_count
    names = ("P3", "P4", "embedding")
    result = {
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
        "domain_mean_weights": {
            domain: {name: float(domain_weight_sum[domain][i] / domain_pixels[domain]) for i, name in enumerate(names)}
            for domain in sorted(domain_pixels)
        },
    }
    if any(region_pixels.values()):
        detail_names = ("P3", "P4")
        result["spatial_budget"] = {
            "policy": getattr(model.decoder, "spatial_policy", None),
            "target_retention_ratio": getattr(model.decoder, "feature_retention_ratio", None),
            "region_retention_ratio": {
                region: {name: float(region_retained_sum[region][i] / max(region_pixels[region], 1)) for i, name in enumerate(detail_names)}
                for region in region_names
            },
            "region_mean_importance": {
                region: {name: float(region_importance_sum[region][i] / max(region_pixels[region], 1)) for i, name in enumerate(detail_names)}
                for region in region_names
            },
            "size_retention_ratio": {
                band: {name: float(size_retained_sum[band][i] / max(size_pixels[band], 1)) for i, name in enumerate(detail_names)}
                for band in SIZE_NAMES
            },
            "size_mean_importance": {
                band: {name: float(size_importance_sum[band][i] / max(size_pixels[band], 1)) for i, name in enumerate(detail_names)}
                for band in SIZE_NAMES
            },
            "region_pixels": region_pixels,
            "size_pixels": size_pixels,
            "size_definition": {
                "small_max_image_fraction": 0.001,
                "medium_max_image_fraction": 0.01,
                "connectivity": 8,
            },
        }
    if oracle_pixels:
        result.update(
            {
                "oracle_route_agreement": oracle_agreement / oracle_pixels,
                "oracle_scale_fraction": {
                    name: float(oracle_scale_sum[i] / oracle_pixels)
                    for i, name in enumerate(names)
                },
                "oracle_target_entropy": oracle_entropy_sum / oracle_pixels,
            }
        )
    return result


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
    if args.rural_sampling_multiplier <= 0:
        raise ValueError("--rural-sampling-multiplier must be positive")
    train_sampler = None
    if args.rural_sampling_multiplier != 1.0:
        weights = [
            args.rural_sampling_multiplier
            if train_base.samples[index][2].startswith("Rural_") else 1.0
            for index in train_indices
        ]
        generator = torch.Generator().manual_seed(args.seed)
        train_sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True, generator=generator
        )
    train_loader = DataLoader(
        train_dataset, shuffle=train_sampler is None, sampler=train_sampler, **loader_options
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(official_validation, shuffle=False, **loader_options)

    if args.model == "unet":
        model = LabelEfficientUNet(
            LoveDASemanticDataset.NUM_CLASSES, args.base_channels
        ).to(device)
    elif args.model in {"deeplabv3plus", "segformer"}:
        model = build_baseline(
            args.model,
            num_classes=LoveDASemanticDataset.NUM_CLASSES,
            pretrained=args.pretrained,
            encoder_name=args.baseline_encoder,
            segformer_variant=args.segformer_variant,
            weights_root=ROOT / "weights",
            freeze_batch_norm=not args.train_batch_norm,
            device=device,
        )
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
            spatial_policy=args.spatial_policy,
            feature_retention_ratio=args.feature_retention_ratio,
            spatial_budget_temperature=args.spatial_budget_temperature,
            static_importance_map=load_static_importance_map(
                resolve(args.static_importance_map) if args.static_importance_map else None
            ),
        )
        if args.model == "mobilesam_finetune":
            model.set_encoder_trainable(True)
    if args.model == "unet" and args.decoder_version != "lightweight":
        raise ValueError("boundary decoder variants are only available for MobileSAM models")
    if args.model in {"deeplabv3plus", "segformer"} and args.decoder_version != "lightweight":
        raise ValueError("boundary decoder variants are only available for MobileSAM models")
    if args.train_batch_norm and args.model != "deeplabv3plus":
        raise ValueError("--train-batch-norm is only valid for DeepLabV3+")
    if args.train_batch_norm and args.batch_size < 2:
        raise ValueError("--train-batch-norm requires --batch-size of at least 2")
    class_weights = None
    if args.class_balanced_ce:
        class_weights = estimate_class_weights(selection_dataset, train_indices).to(device)
        print(f"class_weights={class_weights.detach().cpu().tolist()}")
    criterion = LabelEfficientSegmentationLoss(
        ignore_index=LoveDASemanticDataset.IGNORE_INDEX,
        class_weights=class_weights,
    )
    boundary_criterion = BoundaryLoss(ignore_index=LoveDASemanticDataset.IGNORE_INDEX)
    magnitude_distillation_criterion = MagnitudeTeacherDistillationLoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    backbone = None
    if args.model == "deeplabv3plus":
        backbone = model.model.encoder
    elif args.model == "segformer":
        backbone = model.backbone
    elif args.model == "mobilesam_finetune":
        backbone = model.backbone
    if backbone is not None and args.backbone_lr_multiplier != 1.0:
        backbone_ids = {id(parameter) for parameter in backbone.parameters() if parameter.requires_grad}
        parameter_groups = [
            {
                "params": [parameter for parameter in trainable_parameters if id(parameter) in backbone_ids],
                "lr": args.lr * args.backbone_lr_multiplier,
                "group_name": "backbone",
            },
            {
                "params": [parameter for parameter in trainable_parameters if id(parameter) not in backbone_ids],
                "lr": args.lr,
                "group_name": "head",
            },
        ]
    else:
        parameter_groups = trainable_parameters
    optimizer = AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        LambdaLR(
            optimizer,
            lr_lambda=lambda completed_epochs: 0.01 + 0.99 * 0.5 * (
                1.0 + math.cos(math.pi * completed_epochs / args.epochs)
            ),
        )
        if args.lr_scheduler == "cosine"
        else None
    )
    counts = model.parameter_counts()
    if args.model == "deeplabv3plus":
        run_name = f"deeplabv3plus_{args.baseline_encoder}_ratio{args.label_ratio}_seed{args.seed}"
    elif args.model == "segformer":
        run_name = f"segformer_{args.segformer_variant}_ratio{args.label_ratio}_seed{args.seed}"
    elif args.model == "mobilesam_finetune":
        run_name = f"mobilesam_finetune_ratio{args.label_ratio}_seed{args.seed}"
    else:
        run_name = f"loveda_ratio{args.label_ratio}_seed{args.seed}"
    output_dir = resolve(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    best_score = -1.0
    history = []
    print(
        f"model={args.model} ratio={args.label_ratio}% train={len(train_dataset)} "
        f"validation={len(validation_dataset)} official_val={len(official_validation)} "
        f"augmentation={args.augmentation} image_size={args.image_size} "
        f"rural_sampling_multiplier={args.rural_sampling_multiplier}"
    )
    print(
        f"parameters total={counts['total']:,} trainable={counts['trainable']:,} "
        f"frozen={counts['frozen']:,}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        if hasattr(model, "decoder"):
            model.decoder.task_routing_hard = (
                args.routing_hard and epoch > args.routing_warmup_epochs
            )
        started = time.perf_counter()
        losses = []
        routing_losses = []
        routing_aux_losses = []
        routing_target_entropies = []
        magnitude_distillation_losses = []
        magnitude_teacher_agreements = []
        magnitude_teacher_mask_ious = []
        progressive_aux_losses = []
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
            budget_routing = getattr(getattr(model, "decoder", None), "last_routing", None)
            if budget_routing is not None and "coarse_logits" in budget_routing:
                if args.progressive_aux_weight < 0.0:
                    raise ValueError("--progressive-aux-weight must be non-negative")
                coarse_logits = F.interpolate(
                    budget_routing["coarse_logits"], target.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
                coarse_loss = F.cross_entropy(
                    coarse_logits, target,
                    weight=class_weights,
                    ignore_index=LoveDASemanticDataset.IGNORE_INDEX,
                )
                loss = loss + args.progressive_aux_weight * coarse_loss
                progressive_aux_losses.append(float(coarse_loss.detach()))
            if budget_routing is not None and "teacher_masks" in budget_routing:
                distillation_loss = magnitude_distillation_criterion(
                    budget_routing["student_logits"], budget_routing["teacher_masks"]
                )
                loss = loss + args.magnitude_distill_weight * distillation_loss
                magnitude_distillation_losses.append(float(distillation_loss.detach()))
                magnitude_teacher_agreements.append(
                    float(budget_routing["teacher_student_agreement"])
                )
                magnitude_teacher_mask_ious.append(
                    float(budget_routing["teacher_student_mask_iou"])
                )
            routing = task_utility_routing_loss(
                model,
                target,
                ignore_index=LoveDASemanticDataset.IGNORE_INDEX,
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
            if boundary_logits is not None and args.boundary_loss_weight > 0.0:
                loss = loss + args.boundary_loss_weight * boundary_criterion(boundary_logits, target)
            loss.backward()
            if args.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip_norm)
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
            "learning_rates": {
                group.get("group_name", f"group_{index}"): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            "validation": validation_metrics,
        }
        if routing_losses:
            record["mean_routing_loss"] = sum(routing_losses) / len(routing_losses)
            record["mean_routing_aux_loss"] = sum(routing_aux_losses) / len(routing_aux_losses)
            record["mean_routing_target_entropy"] = sum(routing_target_entropies) / len(routing_target_entropies)
        if magnitude_distillation_losses:
            record["mean_magnitude_distillation_loss"] = sum(
                magnitude_distillation_losses
            ) / len(magnitude_distillation_losses)
            record["mean_teacher_student_mask_agreement"] = sum(
                magnitude_teacher_agreements
            ) / len(magnitude_teacher_agreements)
            record["mean_teacher_student_mask_iou"] = sum(
                magnitude_teacher_mask_ious
            ) / len(magnitude_teacher_mask_ious)
        if progressive_aux_losses:
            record["mean_progressive_aux_loss"] = sum(progressive_aux_losses) / len(
                progressive_aux_losses
            )
        history.append(record)
        print(json.dumps(record))
        if validation_metrics["mIoU"] > best_score:
            best_score = validation_metrics["mIoU"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)
        if scheduler is not None:
            scheduler.step()

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    if hasattr(model, "decoder"):
        model.decoder.task_routing_hard = (
            args.routing_hard and args.epochs > args.routing_warmup_epochs
        )
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        LoveDASemanticDataset.NUM_CLASSES,
        ignore_index=LoveDASemanticDataset.IGNORE_INDEX,
        conditioned=True,
    )
    routing_statistics = collect_routing_statistics(
        model, test_loader, device, LoveDASemanticDataset.NUM_CLASSES,
        LoveDASemanticDataset.IGNORE_INDEX,
        target_temperature=args.routing_target_temperature,
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
