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
from adasam.losses import (  # noqa: E402
    BoundaryLoss,
    LabelEfficientSegmentationLoss,
    semantic_boundary_target,
)
from adasam.metrics import SIZE_NAMES, component_size_map  # noqa: E402
from adasam.models import LabelEfficientSAM, LabelEfficientUNet  # noqa: E402
from adasam.utils import load_static_importance_map, set_seed  # noqa: E402
from tools.train_segmentation import evaluate, task_utility_routing_loss  # noqa: E402


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
    parser.add_argument("--fusion-version", choices=["hierarchical", "concat", "sum", "global", "image_conditioned", "scsr", "scsr_v2", "scsr_task", "semantic_budget"], default="hierarchical")
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
    model,
    loader,
    device,
    num_classes: int,
    ignore_index: int | None,
    target_temperature: float = 0.5,
):
    if getattr(model.decoder, "fusion_version", None) not in {"scsr", "scsr_v2", "scsr_task", "semantic_budget"}:
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
            spatial_policy=args.spatial_policy,
            feature_retention_ratio=args.feature_retention_ratio,
            spatial_budget_temperature=args.spatial_budget_temperature,
            static_importance_map=load_static_importance_map(
                resolve(args.static_importance_map) if args.static_importance_map else None
            ),
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
        if hasattr(model, "decoder"):
            model.decoder.task_routing_hard = (
                args.routing_hard and epoch > args.routing_warmup_epochs
            )
        started = time.perf_counter()
        losses = []
        routing_losses = []
        routing_aux_losses = []
        routing_target_entropies = []
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
        if routing_losses:
            record["mean_routing_loss"] = sum(routing_losses) / len(routing_losses)
            record["mean_routing_aux_loss"] = sum(routing_aux_losses) / len(routing_aux_losses)
            record["mean_routing_target_entropy"] = sum(routing_target_entropies) / len(routing_target_entropies)
        history.append(record)
        print(json.dumps(record))
        if validation_metrics["mIoU"] > best_score:
            best_score = validation_metrics["mIoU"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

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
