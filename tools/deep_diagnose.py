#!/usr/bin/env python3
"""
深度诊断: 对比 fix_baseline vs fix_samrsp checkpoint
=====================================================
回答 5 个核心问题:
  1. SAM Decoder 权重有没有真正更新?
  2. RSP map 是否聚焦前景?
  3. Query 激活模式 (修复后多 query 是否真的在工作)?
  4. 每类 IoU 分解 (瓶颈在哪些类)?
  5. Mask 质量 (预测面积 vs GT 面积)?

用法:
  python tools/deep_diagnose.py
  python tools/deep_diagnose.py --ckpt-baseline runs/fix_baseline/.../best_model.pt
  python tools/deep_diagnose.py --ckpt-samrsp runs/fix_samrsp/.../best_model.pt --single
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5I_CATEGORIES, ISAID5I_FOLDS
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image, resize_mask


ISAID5I_CAT = ISAID5I_CATEGORIES


def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def build_model(ckpt: dict, device: str = "cuda") -> tuple:
    """从 checkpoint 重建模型和 backbone."""
    cfg = ckpt["config"]
    ckpt_path = cfg["backbone"]["checkpoint"]
    if not Path(ckpt_path).exists():
        ckpt_path = str(_REPO_ROOT / ckpt_path)

    sam = build_mobile_sam(ckpt_path, cfg["backbone"].get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(cfg)).to(device)

    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    backbone.eval()

    return model, backbone, sam, cfg


def build_support_memory(
    model, backbone, dataset, class_id: int, k_shot: int, seed: int
) -> tuple:
    """固定 seed 构建 support memory."""
    rng = random.Random(seed + class_id * 1000)
    tiles = dataset.class_to_tiles(class_id)
    if len(tiles) < 1:
        return None, None

    # Scene-disjoint grouping
    scenes: dict[str, list[int]] = defaultdict(list)
    for idx in tiles:
        tid = dataset.tile_ids[idx]
        src = dataset._source_images.get(tid, str(idx))
        scenes[src].append(idx)

    chosen = []
    scene_list = list(scenes.keys())
    rng.shuffle(scene_list)
    for src in scene_list:
        if len(chosen) >= k_shot:
            break
        idx = rng.choice(scenes[src])
        chosen.append(idx)

    if len(chosen) < 1:
        return None, None

    # Build support features + masks
    features, masks = [], []
    image_size = backbone.img_size
    for idx in chosen:
        sample = dataset[idx]
        img = sample["image"]
        inst_masks = [m["mask"] for m in sample["instances"] if m["category_id"] == class_id]
        if not inst_masks:
            continue
        cls_mask = inst_masks[0]
        for m in inst_masks[1:]:
            cls_mask = cls_mask | m

        proc, _ = preprocess_image(img, image_size)
        proc = proc.unsqueeze(0)
        with torch.no_grad():
            emb = backbone(proc.to(next(model.parameters()).device))
        feat = emb["image_embedding"]
        mask_low = resize_mask(cls_mask.unsqueeze(0), (64, 64)).float()

        features.append(feat.squeeze(0))
        masks.append(mask_low)

    if not features:
        return None, None

    sup_feat = torch.stack(features, dim=0)
    sup_mask = torch.stack(masks, dim=0)
    return sup_feat, sup_mask


def print_sep(title: str, char: str = "=") -> None:
    print(f"\n{' ' + title + ' ':{char}^{78}}")


# ═══════════════════════════════════════════════════════════════════
# Question 1: SAM Decoder 权重更新
# ═══════════════════════════════════════════════════════════════════
def q1_decoder_weight_analysis(ckpt: dict, label: str) -> dict:
    """检查 SAM mask_decoder 参数是否有意义的变化."""
    print_sep(f"Q1: {label} — SAM Decoder 权重分析")
    state = ckpt["model"]

    # 找出所有 mask_decoder 参数
    dec_params = {k: v for k, v in state.items() if "mask_decoder" in k.lower()}
    other_params = {k: v for k, v in state.items() if "mask_decoder" not in k.lower()}

    stats = {}
    for group_name, params in [("Decoder", dec_params), ("Other", other_params)]:
        norms = []
        for k, v in params.items():
            if v.dim() > 0:
                norms.append(float(v.norm().item()))
        if norms:
            print(f"  {group_name}: {len(params)} params, "
                  f"norm mean={np.mean(norms):.4f}, max={np.max(norms):.4f}, min={np.min(norms):.4f}")
            stats[group_name] = {"n": len(params), "norm_mean": np.mean(norms)}

    # Check specific key params
    key_params = [
        "sam.mask_decoder.output_upscaling",
        "sam.mask_decoder.output_hypernetworks_mlps",
        "sam.mask_decoder.iou_prediction_head",
        "sam.mask_decoder.transformer",
    ]
    for key in key_params:
        matching = [k for k in state if key in k.lower()]
        if matching:
            sample_k = matching[0]
            print(f"  {key}: {len(matching)} tensors, e.g. {sample_k} shape={list(state[sample_k].shape)}")

    return stats


# ═══════════════════════════════════════════════════════════════════
# Question 2: RSP map 诊断
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def q2_rsp_analysis(model, backbone, dataset, num_samples: int = 50, device: str = "cuda") -> dict:
    """检查 RSP map 是否聚焦前景区域."""
    print_sep("Q2: RSP Map 诊断")

    if model.coarse_prior is None:
        print("  CoarsePrior = None, RSP 未启用。跳过。")
        return {"enabled": False}

    image_size = backbone.img_size
    fg_activations = []
    bg_activations = []
    rsp_entropies = []
    rsp_max_vals = []

    classes = dataset.visible_classes()
    samples_per_class = max(1, num_samples // len(classes))

    for cls in classes:
        tiles = dataset.class_to_tiles(cls)
        if len(tiles) < 2:
            continue

        # Build support
        sup_feat, sup_mask = build_support_memory(
            model, backbone, dataset, cls, k_shot=1, seed=42
        )
        if sup_feat is None:
            continue

        support_features = sup_feat.to(device)
        support_masks = sup_mask.to(device)

        # Encode support
        sup_emb = model.support_encoder(support_features, support_masks)

        # Sample query tiles
        rng = random.Random(42 + cls)
        sampled = rng.sample(tiles, min(samples_per_class, len(tiles)))
        for idx in sampled:
            sample = dataset[idx]
            img = sample["image"]
            proc, _ = preprocess_image(img, image_size)
            proc = proc.unsqueeze(0).to(device)
            emb = backbone(proc)
            query_features = emb["image_embedding"]

            # Forward through CoarsePrior
            enriched, rsp_map = model.coarse_prior(query_features, sup_emb)
            rsp_np = rsp_map.squeeze().cpu().numpy()  # [64, 64]

            # GT mask at 64x64
            gt_mask = torch.zeros(1, 64, 64, device=device)
            for inst in sample["instances"]:
                if inst["category_id"] == cls:
                    m = resize_mask(inst["mask"].unsqueeze(0), (64, 64)).float().to(device)
                    gt_mask = torch.maximum(gt_mask, m)
            gt_np = gt_mask.squeeze().cpu().numpy().astype(bool)

            # Statistics
            rsp_max_vals.append(float(rsp_np.max()))
            # Entropy: treat RSP as distribution, compute entropy
            rsp_flat = rsp_np.flatten()
            rsp_flat = np.clip(rsp_flat, 1e-7, 1.0)
            rsp_flat = rsp_flat / rsp_flat.sum()
            entropy = -np.sum(rsp_flat * np.log(rsp_flat))
            rsp_entropies.append(entropy)

            if gt_np.sum() > 0:
                fg_act = rsp_np[gt_np].mean()
                bg_act = rsp_np[~gt_np].mean()
                fg_activations.append(fg_act)
                bg_activations.append(bg_act)

    print(f"  Samples: {len(rsp_entropies)}")
    print(f"  RSP max:     mean={np.mean(rsp_max_vals):.4f}, std={np.std(rsp_max_vals):.4f}")
    print(f"  RSP entropy: mean={np.mean(rsp_entropies):.4f}, std={np.std(rsp_entropies):.4f}")
    if fg_activations:
        print(f"  FG activation: mean={np.mean(fg_activations):.4f}")
        print(f"  BG activation: mean={np.mean(bg_activations):.4f}")
        fg_bg_ratio = np.mean(fg_activations) / (np.mean(bg_activations) + 1e-7)
        print(f"  FG/BG ratio:   {fg_bg_ratio:.2f}x {'✓' if fg_bg_ratio > 2.0 else '✗ 前景背景区分度不足!'}")
    else:
        print(f"  (no FG pixels found)")

    return {
        "enabled": True,
        "rsp_max_mean": np.mean(rsp_max_vals),
        "rsp_entropy_mean": np.mean(rsp_entropies),
        "fg_act_mean": np.mean(fg_activations) if fg_activations else 0,
        "bg_act_mean": np.mean(bg_activations) if bg_activations else 0,
    }


# ═══════════════════════════════════════════════════════════════════
# Question 3: Per-class IoU breakdown
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def q3_per_class_analysis(
    model, backbone, dataset, num_samples: int = 100, device: str = "cuda"
) -> dict:
    """逐类 IoU 分解."""
    print_sep("Q3: 逐类 IoU 分解")

    image_size = backbone.img_size
    per_class_ious = defaultdict(list)
    per_class_counts = defaultdict(int)

    classes = dataset.visible_classes()
    for cls in classes:
        sup_feat, sup_mask = build_support_memory(
            model, backbone, dataset, cls, k_shot=1, seed=42
        )
        if sup_feat is None:
            print(f"  class {cls}: no support available, skip")
            continue

        sup_emb = model.support_encoder(
            sup_feat.to(device), sup_mask.to(device)
        )

        tiles = dataset.class_to_tiles(cls)
        rng = random.Random(42 + cls)
        sampled = rng.sample(tiles, min(num_samples, len(tiles)))

        class_ious = []
        for idx in sampled:
            sample = dataset[idx]
            img = sample["image"]
            proc, _ = preprocess_image(img, image_size)
            proc = proc.unsqueeze(0).to(device)
            emb = backbone(proc)

            # Predict
            pred_masks, pred_scores = model.predict(
                emb["image_embedding"], sup_emb,
                emb.get("dense_pe", torch.zeros(1, 256, 64, 64, device=device)),
                (image_size, image_size), (256, 256),
                score_thr=0.3,
            )

            # GT mask
            gt_mask = torch.zeros(256, 256, dtype=torch.bool, device=device)
            for inst in sample["instances"]:
                if inst["category_id"] == cls:
                    gt_mask = gt_mask | inst["mask"].to(device)

            if len(pred_masks) > 0:
                pred_union = pred_masks.any(dim=0)
                inter = (pred_union & gt_mask).sum().float()
                union = (pred_union | gt_mask).sum().float()
                iou = (inter / (union + 1e-7)).item()
            else:
                iou = 0.0

            class_ious.append(iou)

        mean_iou = np.mean(class_ious) if class_ious else 0.0
        name = ISAID5I_CAT.get(cls, f"class_{cls}")
        print(f"  class {cls:2d} ({name:20s}): mIoU={mean_iou:.4f}, n={len(class_ious)}")
        per_class_ious[cls] = class_ious
        per_class_counts[cls] = len(class_ious)

    all_ious = [v for vs in per_class_ious.values() for v in vs]
    overall = np.mean(all_ious) if all_ious else 0.0
    print(f"  ──────────────────────────────")
    print(f"  Overall mIoU: {overall:.4f}")

    return {"per_class_ious": dict(per_class_ious), "overall_mIoU": overall}


# ═══════════════════════════════════════════════════════════════════
# Question 4: Query activation pattern
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def q4_query_activation(model, backbone, dataset, num_samples: int = 100, device: str = "cuda") -> dict:
    """检查 DPG 的每个 query 是否被激活."""
    print_sep("Q4: Query 激活模式")

    image_size = backbone.img_size
    query_objectness = defaultdict(list)  # query_idx -> [objectness values]
    n_matched_per_ep = []

    classes = dataset.visible_classes()
    for cls in classes:
        sup_feat, sup_mask = build_support_memory(
            model, backbone, dataset, cls, k_shot=1, seed=42
        )
        if sup_feat is None:
            continue

        sup_emb = model.support_encoder(sup_feat.to(device), sup_mask.to(device))

        tiles = dataset.class_to_tiles(cls)
        rng = random.Random(42 + cls)
        n = min(num_samples // len(classes), len(tiles))
        sampled = rng.sample(tiles, n)

        for idx in sampled:
            sample = dataset[idx]
            img = sample["image"]
            proc, _ = preprocess_image(img, image_size)
            proc = proc.unsqueeze(0).to(device)
            emb = backbone(proc)

            # Forward through DPG
            dpg_out = model.dpg(
                emb["image_embedding"],
                sup_emb,
                emb.get("dense_pe", torch.zeros(1, 256, 64, 64, device=device)),
            )
            obj = dpg_out.objectness_logits.sigmoid().cpu()  # [N]

            for qi in range(len(obj)):
                query_objectness[qi].append(float(obj[qi]))

            # Count active queries (obj > 0.3)
            active = int((obj > 0.3).sum().item())
            n_matched_per_ep.append(active)

    # Summary
    print(f"  Episodes analyzed: {len(n_matched_per_ep)}")
    print(f"  Active queries (obj>0.3): mean={np.mean(n_matched_per_ep):.2f}, "
          f"median={np.median(n_matched_per_ep):.0f}, max={max(n_matched_per_ep)}")
    print(f"\n  Per-query objectness (sorted by mean):")
    print(f"  {'Query':>6s}  {'Mean Obj':>9s}  {'Std':>7s}  {'Active%':>7s}  {'Status'}")
    print(f"  {'-'*45}")
    sorted_q = sorted(query_objectness.items(), key=lambda x: -np.mean(x[1]))
    n_active_queries = 0
    for qi, vals in sorted_q[:16]:
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        active_pct = 100 * sum(1 for v in vals if v > 0.3) / len(vals)
        status = "ACTIVE" if mean_v > 0.3 else "semi" if mean_v > 0.1 else "dead"
        if mean_v > 0.1:
            n_active_queries += 1
        print(f"  Q{qi:5d}  {mean_v:9.4f}  {std_v:7.4f}  {active_pct:6.1f}%  {status}")

    print(f"\n  Queries with mean_obj > 0.1: {n_active_queries}/16")

    return {
        "active_queries_mean": np.mean(n_matched_per_ep),
        "n_significant_queries": n_active_queries,
    }


# ═══════════════════════════════════════════════════════════════════
# Question 5: Mask quality (area ratio)
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def q5_mask_quality(model, backbone, dataset, num_samples: int = 100, device: str = "cuda") -> dict:
    """对比预测 mask 面积与 GT mask 面积."""
    print_sep("Q5: Mask 质量 (面积比)")

    image_size = backbone.img_size
    area_ratios = []
    ious_with_best = []

    classes = dataset.visible_classes()
    for cls in classes:
        sup_feat, sup_mask = build_support_memory(
            model, backbone, dataset, cls, k_shot=1, seed=42
        )
        if sup_feat is None:
            continue

        sup_emb = model.support_encoder(sup_feat.to(device), sup_mask.to(device))

        tiles = dataset.class_to_tiles(cls)
        rng = random.Random(42 + cls)
        n = min(num_samples // len(classes), len(tiles))
        sampled = rng.sample(tiles, n)

        for idx in sampled:
            sample = dataset[idx]
            img = sample["image"]
            proc, _ = preprocess_image(img, image_size)
            proc = proc.unsqueeze(0).to(device)
            emb = backbone(proc)

            pred_masks, pred_scores = model.predict(
                emb["image_embedding"], sup_emb,
                emb.get("dense_pe", torch.zeros(1, 256, 64, 64, device=device)),
                (image_size, image_size), (256, 256),
                score_thr=0.3,
            )

            # GT: union of all instances of this class
            gt_area = 0.0
            gt_mask = torch.zeros(256, 256, dtype=torch.bool)
            for inst in sample["instances"]:
                if inst["category_id"] == cls:
                    gt_mask = gt_mask | inst["mask"]
                    gt_area += float(inst["mask"].sum())

            if len(pred_masks) > 0:
                pred_union = pred_masks.any(dim=0)
                pred_area = float(pred_union.sum())

                if gt_area > 0:
                    ratio = pred_area / gt_area
                    area_ratios.append(ratio)

                inter = (pred_union & gt_mask).sum().float()
                union = (pred_union | gt_mask).sum().float()
                iou = (inter / (union + 1e-7)).item()
                ious_with_best.append(iou)

    print(f"  Samples with predictions: {len(area_ratios)}")
    if area_ratios:
        print(f"  Pred/GT area ratio: mean={np.mean(area_ratios):.3f}, "
              f"median={np.median(area_ratios):.3f}, std={np.std(area_ratios):.3f}")
        underseg = sum(1 for r in area_ratios if r < 0.5)
        overseg = sum(1 for r in area_ratios if r > 2.0)
        good = len(area_ratios) - underseg - overseg
        print(f"  Underseg (ratio<0.5): {underseg}/{len(area_ratios)} ({100*underseg/len(area_ratios):.1f}%)")
        print(f"  Overseg  (ratio>2.0): {overseg}/{len(area_ratios)} ({100*overseg/len(area_ratios):.1f}%)")
        print(f"  Good     (0.5-2.0): {good}/{len(area_ratios)} ({100*good/len(area_ratios):.1f}%)")
    if ious_with_best:
        print(f"  IoU (union pred vs union GT): mean={np.mean(ious_with_best):.4f}")

    return {
        "area_ratio_mean": np.mean(area_ratios) if area_ratios else 0.0,
        "iou_mean": np.mean(ious_with_best) if ious_with_best else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════
# Question 6: DPG dense prompt impact
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def q6_dense_prompt_ablation(model, backbone, dataset, num_samples: int = 30, device: str = "cuda") -> dict:
    """Normal vs Zero support 对 DPG 输出的影响."""
    print_sep("Q6: Support → DPG 影响 (Normal vs Zero)")

    image_size = backbone.img_size
    normal_objs = []
    zero_objs = []
    normal_obj_gaps = []

    classes = dataset.visible_classes()
    for cls in classes:
        sup_feat, sup_mask = build_support_memory(
            model, backbone, dataset, cls, k_shot=1, seed=42
        )
        if sup_feat is None:
            continue

        sup_emb = model.support_encoder(sup_feat.to(device), sup_mask.to(device))
        zero_emb = torch.zeros_like(sup_emb)

        tiles = dataset.class_to_tiles(cls)
        rng = random.Random(42 + cls)
        n = min(num_samples // len(classes), len(tiles))
        sampled = rng.sample(tiles, n)

        for idx in sampled:
            sample = dataset[idx]
            img = sample["image"]
            proc, _ = preprocess_image(img, image_size)
            proc = proc.unsqueeze(0).to(device)
            emb = backbone(proc)
            qf = emb["image_embedding"]
            pe = emb.get("dense_pe", torch.zeros(1, 256, 64, 64, device=device))

            dpg_normal = model.dpg(qf, sup_emb, pe)
            dpg_zero = model.dpg(qf, zero_emb, pe)

            obj_n = dpg_normal.objectness_logits.sigmoid().cpu()
            obj_z = dpg_zero.objectness_logits.sigmoid().cpu()

            normal_objs.append(float(obj_n.max()))
            zero_objs.append(float(obj_z.max()))
            normal_obj_gaps.append(float(obj_n.max() - obj_n.min()))

    print(f"  Samples: {len(normal_objs)}")
    print(f"  Normal support: max_obj mean={np.mean(normal_objs):.4f}, "
          f"gap mean={np.mean(normal_obj_gaps):.4f}")
    print(f"  Zero support:   max_obj mean={np.mean(zero_objs):.4f}")
    delta = np.mean(normal_objs) - np.mean(zero_objs)
    print(f"  Δ(Normal - Zero): {delta:.4f} {'✓ support 有影响' if abs(delta) > 0.05 else '✗ support 几乎无影响!'}")

    return {"delta": delta}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="深度诊断 fix_baseline vs fix_samrsp")
    parser.add_argument("--ckpt-baseline", default=None)
    parser.add_argument("--ckpt-samrsp", default=None)
    parser.add_argument("--single", action="store_true", help="只分析一个 checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--skip-q3", action="store_true", help="跳过逐类 IoU (慢)")
    parser.add_argument("--skip-q5", action="store_true", help="跳过 Mask 质量 (较慢)")
    args = parser.parse_args()

    # Auto-detect checkpoints
    run_dir = _REPO_ROOT / "runs"
    baseline_dir = run_dir / "fix_baseline" / "isaid5i_fold0_k1_novel_seed42"
    samrsp_dir = run_dir / "fix_samrsp" / "isaid5i_fold0_k1_novel_seed42"

    ckpt_baseline = args.ckpt_baseline or str(baseline_dir / "best_model.pt")
    ckpt_samrsp = args.ckpt_samrsp or str(samrsp_dir / "best_model.pt")

    if args.single:
        checkpoints = [(ckpt_samrsp, "SAM-RSP")]
    else:
        checkpoints = [(ckpt_baseline, "Baseline"), (ckpt_samrsp, "SAM-RSP")]

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    set_seed(42)

    for ckpt_path, label in checkpoints:
        if not Path(ckpt_path).exists():
            print(f"\nSKIP {label}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'#' * 78}")
        print(f"#  Analyzing: {label}")
        print(f"#  Checkpoint: {ckpt_path}")
        print(f"{'#' * 78}")

        ckpt = load_checkpoint(ckpt_path)
        model, backbone, sam, cfg = build_model(ckpt, device)

        fold = cfg["data"].get("fold", 0)
        data_root = cfg["data"].get("data_root", "data/iSAID-5i")
        ds = ISAID5iDataset(root=data_root, fold=fold, split="val", mode="novel")
        print(f"Val dataset: {len(ds)} tiles, classes={ds.visible_classes()}")

        # Q1: Weight analysis
        q1_decoder_weight_analysis(ckpt, label)

        # Q2: RSP
        q2_rsp_analysis(model, backbone, ds, num_samples=args.num_samples, device=device)

        # Q3: Per-class IoU (slow)
        if not args.skip_q3:
            q3_per_class_analysis(model, backbone, ds, num_samples=min(args.num_samples, 50), device=device)

        # Q4: Query activation
        q4_query_activation(model, backbone, ds, num_samples=args.num_samples, device=device)

        # Q5: Mask quality
        if not args.skip_q5:
            q5_mask_quality(model, backbone, ds, num_samples=args.num_samples, device=device)

        # Q6: Support impact
        q6_dense_prompt_ablation(model, backbone, ds, num_samples=min(args.num_samples, 30), device=device)

        del model, backbone, sam
        torch.cuda.empty_cache()

    print_sep("诊断完成", "=")
    print("\n建议下一步:")
    print("  1. 如果 Q6 Δ ≈ 0 → support 信息到达不了 DPG")
    print("  2. 如果 Q2 FG/BG < 2x → RSP 定位失败")
    print("  3. 如果 Q4 active_queries ≈ 1 → Objectness collapse 仍然存在")
    print("  4. 如果 Q5 underseg > 50% → 模型系统性地 under-predict")


if __name__ == "__main__":
    main()
