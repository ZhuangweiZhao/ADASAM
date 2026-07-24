#!/usr/bin/env python3
"""
[DEPRECATED] Prompt-to-Mask 信息链诊断 | Prompt-to-Mask Information Chain Diagnosis
=====================================================================================
⚠️ 此工具引用旧 DPG API (dpg_out.fg_queries, dpg_out.fg_logits, model.coarse_prior,
   model.dpg.feedback_conv), 与新 SPG 架构不兼容, 运行会崩溃。需更新后才能使用。

4 个精确诊断测试, 解剖 SAM MaskDecoder 的 prompt 信息链是否断裂。

Test 1: Dense Prompt 激活检查 — 是否死亡 (std < 0.01)?
Test 2: Dense Prompt 扰动实验 — 真实 vs Zero prompt, 输出差多少?
Test 3: Query Embedding 相似度 — 是否 collapse (cos_sim > 0.95)?
Test 4: low_res_masks 分布 — 是否饱和/全正/死亡?

用法:
  python tools/diag_prompt_chain.py --ckpt runs/fix_samrsp/.../best_model.pt
  python tools/diag_prompt_chain.py --ckpt-baseline ... --ckpt-samrsp ...  # 对比
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5I_CATEGORIES
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils.transforms import preprocess_image, resize_mask


CAT = ISAID5I_CATEGORIES


def sep(title: str, char: str = "═") -> None:
    print(f"\n{' ' + title + ' ':{char}^72}")


def load_and_build(ckpt_path: str, device: str = "cuda"):
    """Load checkpoint, build model and backbone."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    sam_ckpt = cfg["backbone"]["checkpoint"]
    if not Path(sam_ckpt).exists():
        sam_ckpt = str(_REPO_ROOT / sam_ckpt)

    sam = build_mobile_sam(sam_ckpt, cfg["backbone"].get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(cfg)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    backbone.eval()

    return model, backbone, cfg, ckpt


def build_support(model, backbone, dataset, class_id: int, k_shot: int, device: str):
    """Build raw support features + masks for a class."""
    rng = random.Random(42 + class_id * 1000)
    tiles = dataset.class_to_tiles(class_id)
    if len(tiles) < 1:
        return None, None

    scenes: dict[str, list[int]] = defaultdict(list)
    for idx in tiles:
        tid = dataset.tile_ids[idx]
        src = dataset._source_images.get(tid, str(idx))
        scenes[src].append(idx)

    chosen = []
    for src in rng.sample(list(scenes.keys()), len(scenes)):
        if len(chosen) >= k_shot:
            break
        chosen.append(rng.choice(scenes[src]))
    if not chosen:
        return None, None

    img_size = backbone.img_size
    features, masks = [], []
    for idx in chosen:
        sample = dataset[idx]
        img = sample["image"]
        cls_mask = None
        for inst in sample["regions"]:
            if inst["category_id"] == class_id:
                cls_mask = inst["mask"] if cls_mask is None else cls_mask | inst["mask"]
        if cls_mask is None:
            continue
        proc, _ = preprocess_image(img, img_size)
        proc = proc.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = backbone(proc)
        feat = emb["image_embedding"].squeeze(0)
        mask_low = resize_mask(cls_mask.float(), (64, 64))
        features.append(feat)
        masks.append(mask_low)

    if not features:
        return None, None
    return torch.stack(features, dim=0).to(device), torch.stack(masks, dim=0).to(device)


# ═══════════════════════════════════════════════════════════════════
# Test 1: Dense Prompt 激活检查
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def test1_dense_prompt(model, backbone, dataset, device: str):
    """检查 dense_prompt 是否死亡."""
    sep("Test 1: Dense Prompt 激活检查")
    if model.coarse_prior is None:
        print("  SKIP: CoarsePrior=None, no RSP module")
        return {}

    all_stats = []
    classes = dataset.visible_classes()

    for cls in classes[:3]:  # test on first 3 classes
        sup_feat, sup_mask = build_support(model, backbone, dataset, cls, k_shot=1, device=device)
        if sup_feat is None:
            continue

        tiles = dataset.class_to_tiles(cls)
        rng = random.Random(42)
        sampled = rng.sample(tiles, min(5, len(tiles)))

        for idx in sampled:
            sample = dataset[idx]
            proc, _ = preprocess_image(sample["image"], backbone.img_size)
            proc = proc.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = backbone(proc)

            # Use forward_train to get dpg_out
            dpg_out, _, _ = model.forward_train(
                emb["image_embedding"], sup_feat, sup_mask
            )
            if dpg_out.dense_prompt is not None:
                dp = dpg_out.dense_prompt  # [1, C, 1, 1]
                all_stats.append({
                    "mean": float(dp.mean()),
                    "std": float(dp.std()),
                    "abs_mean": float(dp.abs().mean()),
                    "abs_max": float(dp.abs().max()),
                    "norm": float(dp.norm()),
                })

    if not all_stats:
        print("  No dense_prompt found (all None)")
        return {}

    means = [s["mean"] for s in all_stats]
    stds = [s["std"] for s in all_stats]
    abs_means = [s["abs_mean"] for s in all_stats]
    abs_maxs = [s["abs_max"] for s in all_stats]
    norms = [s["norm"] for s in all_stats]

    print(f"  Samples: {len(all_stats)}")
    print(f"  dense_prompt.mean:      {np.mean(means):.6f} ± {np.std(means):.6f}")
    print(f"  dense_prompt.std:       {np.mean(stds):.6f} ± {np.std(stds):.6f}")
    print(f"  dense_prompt.abs_mean:  {np.mean(abs_means):.6f} ± {np.std(abs_means):.6f}")
    print(f"  dense_prompt.abs_max:   {np.mean(abs_maxs):.6f} ± {np.std(abs_maxs):.6f}")
    print(f"  dense_prompt.norm:      {np.mean(norms):.6f} ± {np.std(norms):.6f}")

    avg_std = np.mean(stds)
    if avg_std < 0.01:
        verdict = "DEAD - dense prompt has negligible variation"
    elif avg_std < 0.1:
        verdict = "WEAK - dense prompt is very weak"
    else:
        verdict = "ALIVE - dense prompt has meaningful signal"
    print(f"  Verdict: {verdict}")

    return {"avg_std": avg_std, "avg_abs_mean": np.mean(abs_means)}


# ═══════════════════════════════════════════════════════════════════
# Test 2: Dense Prompt 扰动实验
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def test2_prompt_ablation(model, backbone, dataset, device: str):
    """真实 dense_prompt vs Zero dense_prompt → IoU 差异."""
    sep("Test 2: Dense Prompt 扰动实验")

    classes = dataset.visible_classes()
    ious_normal = []
    ious_zero = []
    ious_nomask = []  # SAM default no_mask_embed

    for cls in classes[:3]:
        sup_feat, sup_mask = build_support(model, backbone, dataset, cls, k_shot=1, device=device)
        if sup_feat is None:
            continue

        tiles = dataset.class_to_tiles(cls)
        rng = random.Random(42)
        sampled = rng.sample(tiles, min(15, len(tiles)))

        # Get SAM's default no_mask_embed
        no_mask = model.sam_decoder.prompt_encoder.no_mask_embed.weight.view(1, -1, 1, 1)  # [1, C, 1, 1]

        for idx in sampled:
            sample = dataset[idx]
            proc, _ = preprocess_image(sample["image"], backbone.img_size)
            proc = proc.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = backbone(proc)
            qf = emb["image_embedding"]

            # GT
            gt = torch.zeros(256, 256, dtype=torch.bool, device=device)
            for inst in sample["regions"]:
                if inst["category_id"] == cls:
                    gt = gt | inst["mask"].to(device)

            # Path A: Normal (with support-conditioned dense prompt)
            dpg_out, low_res, iou_pred = model.forward_train(qf, sup_feat, sup_mask)
            masks_n, scores_n = model.predict(
                qf, sup_feat, sup_mask,
                (backbone.img_size, backbone.img_size), (256, 256), score_thr=0.3,
            )
            iou_n = _compute_iou(masks_n, gt)

            # Path B: Zero dense prompt (dense_prompt_override = zeros, spatial)
            zero_dense = torch.zeros(1, 256, 64, 64, device=device)
            z_mask = model.sam_decoder(
                qf, dpg_out.fg_queries, zero_dense
            )
            z_logits = z_mask[0]  # [N, 1, 256, 256]
            z_scores = dpg_out.fg_logits.sigmoid() * z_mask[1][:, 0].clamp(0, 1)
            keep = z_scores >= 0.3
            if keep.any():
                z_up = model.sam_decoder.upscale_logits(
                    z_logits[keep], (backbone.img_size, backbone.img_size), (256, 256)
                )
                masks_z = z_up > 0.0
            else:
                masks_z = torch.zeros(0, 256, 256, dtype=torch.bool, device=device)
            iou_z = _compute_iou(masks_z, gt)

            # Path C: SAM default no_mask_embed (no support influence)
            nm_dense = no_mask.expand(dpg_out.fg_queries.shape[0], -1, 64, 64)
            nm_mask_out = model.sam_decoder.mask_decoder(
                image_embeddings=qf,
                image_pe=model.sam_decoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=dpg_out.fg_queries.unsqueeze(1),
                dense_prompt_embeddings=nm_dense,
                multimask_output=False,
            )
            nm_scores = dpg_out.fg_logits.sigmoid() * nm_mask_out[1][:, 0].clamp(0, 1)
            keep_nm = nm_scores >= 0.3
            if keep_nm.any():
                nm_up = model.sam_decoder.upscale_logits(
                    nm_mask_out[0][keep_nm], (backbone.img_size, backbone.img_size), (256, 256)
                )
                masks_nm = nm_up > 0.0
            else:
                masks_nm = torch.zeros(0, 256, 256, dtype=torch.bool, device=device)
            iou_nm = _compute_iou(masks_nm, gt)

            ious_normal.append(iou_n)
            ious_zero.append(iou_z)
            ious_nomask.append(iou_nm)

    print(f"  Samples: {len(ious_normal)}")
    print(f"  IoU (Support dense prompt):     {np.mean(ious_normal):.4f} ± {np.std(ious_normal):.4f}")
    print(f"  IoU (Zero dense prompt):        {np.mean(ious_zero):.4f} ± {np.std(ious_zero):.4f}")
    print(f"  IoU (SAM default no_mask_embed): {np.mean(ious_nomask):.4f} ± {np.std(ious_nomask):.4f}")

    delta_zn = np.mean(ious_normal) - np.mean(ious_zero)
    delta_nm = np.mean(ious_normal) - np.mean(ious_nomask)
    print(f"  Δ(Normal - Zero):  {delta_zn:+.4f}")
    print(f"  Δ(Normal - SAM):   {delta_nm:+.4f}")

    if abs(delta_zn) < 0.01 and abs(delta_nm) < 0.01:
        verdict = "NO EFFECT - Dense prompt does not affect mask output"
    elif abs(delta_zn) < 0.05:
        verdict = "MARGINAL - Dense prompt has marginal effect"
    else:
        verdict = "EFFECTIVE - Dense prompt meaningfully affects output"
    print(f"  Verdict: {verdict}")

    return {"delta_zero": delta_zn, "delta_nomask": delta_nm}


def _compute_iou(masks: torch.Tensor, gt: torch.Tensor) -> float:
    if len(masks) == 0:
        return 0.0
    pred = masks.any(dim=0)
    inter = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    return float(inter / (union + 1e-7))


# ═══════════════════════════════════════════════════════════════════
# Test 3: Query Embedding 相似度
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def test3_query_similarity(model, backbone, dataset, device: str):
    """检查 DPG fg_queries 是否 collapse."""
    sep("Test 3: Query Embedding 相似度")

    all_cos = []
    all_obj = []

    classes = dataset.visible_classes()
    for cls in classes[:3]:
        sup_feat, sup_mask = build_support(model, backbone, dataset, cls, k_shot=1, device=device)
        if sup_feat is None:
            continue

        tiles = dataset.class_to_tiles(cls)
        rng = random.Random(42)
        sampled = rng.sample(tiles, min(5, len(tiles)))

        for idx in sampled:
            sample = dataset[idx]
            proc, _ = preprocess_image(sample["image"], backbone.img_size)
            proc = proc.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = backbone(proc)

            dpg_out, _, _ = model.forward_train(
                emb["image_embedding"], sup_feat, sup_mask
            )
            q = dpg_out.fg_queries  # [N, C]
            qn = F.normalize(q, dim=1)    # [N, C]
            cos = torch.mm(qn, qn.t())    # [N, N]

            # Exclude diagonal
            mask = ~torch.eye(q.shape[0], dtype=torch.bool, device=device)
            off_diag = cos[mask]
            all_cos.append(off_diag.cpu())

            obj = dpg_out.fg_logits.sigmoid().cpu()
            all_obj.append(obj)

    if not all_cos:
        return {}

    all_cos_cat = torch.cat(all_cos)  # all off-diagonal cos similarities
    all_obj_stack = torch.stack(all_obj)  # [samples, N]

    print(f"  Query pairs analyzed: {len(all_cos_cat)}")
    print(f"  Cosine similarity (off-diagonal):")
    print(f"    mean={all_cos_cat.mean():.4f}, std={all_cos_cat.std():.4f}")
    print(f"    min={all_cos_cat.min():.4f}, max={all_cos_cat.max():.4f}")
    print(f"    P10={torch.quantile(all_cos_cat, 0.1):.4f}, "
          f"P50={torch.quantile(all_cos_cat, 0.5):.4f}, "
          f"P90={torch.quantile(all_cos_cat, 0.9):.4f}")

    # Per-query objectness
    print(f"\n  Per-query objectness (sorted):")
    mean_obj = all_obj_stack.mean(dim=0)
    sorted_idx = mean_obj.argsort(descending=True)
    for rank, qi in enumerate(sorted_idx[:8]):
        print(f"    Q{qi:2d}: obj_mean={mean_obj[qi]:.4f}")
    n_active = int((mean_obj > 0.3).sum())
    print(f"  Active queries (obj>0.3): {n_active}/16")

    avg_cos = float(all_cos_cat.mean())
    if avg_cos > 0.9:
        verdict = "COLLAPSED - queries are nearly identical"
    elif avg_cos > 0.7:
        verdict = "HIGH similarity - queries lack diversity"
    elif avg_cos > 0.3:
        verdict = "MODERATE diversity"
    else:
        verdict = "GOOD diversity"
    print(f"  Verdict: {verdict}")

    return {"avg_cos": avg_cos, "n_active": n_active}


# ═══════════════════════════════════════════════════════════════════
# Test 4: low_res_masks 分布
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def test4_mask_logits(model, backbone, dataset, device: str):
    """检查 SAM decoder 输出的 mask logits 分布."""
    sep("Test 4: low_res_masks 分布")

    all_min = []
    all_max = []
    all_mean = []
    all_std = []
    all_pos_ratio = []  # fraction of positive logits
    all_entropy = []

    classes = dataset.visible_classes()
    for cls in classes[:3]:
        sup_feat, sup_mask = build_support(model, backbone, dataset, cls, k_shot=1, device=device)
        if sup_feat is None:
            continue

        tiles = dataset.class_to_tiles(cls)
        rng = random.Random(42)
        sampled = rng.sample(tiles, min(5, len(tiles)))

        for idx in sampled:
            sample = dataset[idx]
            proc, _ = preprocess_image(sample["image"], backbone.img_size)
            proc = proc.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = backbone(proc)

            _, low_res, _ = model.forward_train(
                emb["image_embedding"], sup_feat, sup_mask
            )
            # low_res: [N, 1, 256, 256]
            for qi in range(low_res.shape[0]):
                m = low_res[qi]  # [1, 256, 256]
                all_min.append(float(m.min()))
                all_max.append(float(m.max()))
                all_mean.append(float(m.mean()))
                all_std.append(float(m.std()))
                all_pos_ratio.append(float((m > 0).float().mean()))
                # Entropy-like measure: sigmoid → binary entropy
                p = m.sigmoid()
                h = -(p * (p + 1e-7).log() + (1 - p) * (1 - p + 1e-7).log()).mean()
                all_entropy.append(float(h))

    print(f"  Mask samples: {len(all_min)}")
    print(f"  logits min:        {np.mean(all_min):.4f} ± {np.std(all_min):.4f}  (range: {min(all_min):.4f} ~ {max(all_min):.4f})")
    print(f"  logits max:        {np.mean(all_max):.4f} ± {np.std(all_max):.4f}  (range: {min(all_max):.4f} ~ {max(all_max):.4f})")
    print(f"  logits mean:       {np.mean(all_mean):.4f} ± {np.std(all_mean):.4f}")
    print(f"  logits std:        {np.mean(all_std):.4f} ± {np.std(all_std):.4f}")
    print(f"  pos ratio (>0):    {np.mean(all_pos_ratio):.4f} ± {np.std(all_pos_ratio):.4f}")
    print(f"  sigmoid entropy:   {np.mean(all_entropy):.4f} ± {np.std(all_entropy):.4f}")

    avg_min = np.mean(all_min)
    avg_max = np.mean(all_max)
    avg_pos = np.mean(all_pos_ratio)

    issues = []
    if avg_min >= 0:
        issues.append("ALL POSITIVE — no negative boundary signal")
    if avg_max < 1.0:
        issues.append(f"LOW ACTIVATION — max={avg_max:.2f}, masks are weak")
    if avg_max > 15:
        issues.append(f"SATURATED — max={avg_max:.2f}, logits too extreme")
    if avg_pos > 0.9:
        issues.append("TOO POSITIVE — >90% of pixels are positive")
    if avg_pos < 0.05:
        issues.append("ALL NEGATIVE — masks are empty")

    if not issues:
        verdict = "OK NORMAL - mask logits look healthy"
    else:
        verdict = "ISSUES: " + "; ".join(issues)
    print(f"  Verdict: {verdict}")

    return {"min_mean": avg_min, "max_mean": avg_max, "pos_ratio": avg_pos, "issues": issues}


# ═══════════════════════════════════════════════════════════════════
# Bonus: Gradient flow check
# ═══════════════════════════════════════════════════════════════════
def test5_gradient_check(model, backbone, dataset, device: str):
    """检查 SAM decoder 参数是否真的在更新."""
    sep("Test 5: SAM Decoder 梯度流检查")

    # Get one sample
    cls = dataset.visible_classes()[0]
    sup_feat, sup_mask = build_support(model, backbone, dataset, cls, k_shot=1, device=device)
    if sup_feat is None:
        print("  Cannot build support")
        return {}

    tiles = dataset.class_to_tiles(cls)
    sample = dataset[random.Random(42).choice(tiles)]
    proc, _ = preprocess_image(sample["image"], backbone.img_size)
    proc = proc.unsqueeze(0).to(device)

    # Build GT
    gt_list = [inst["mask"] for inst in sample["regions"] if inst["category_id"] == cls]
    if not gt_list:
        print("  No GT for this class")
        return {}
    if len(gt_list) > model.num_queries:
        gt_list = sorted(gt_list, key=lambda m: m.sum(), reverse=True)[:model.num_queries]
    gt_masks = torch.stack([m.float() for m in gt_list], dim=0).to(device)

    # Enable grad for decoder only
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.sam_decoder.mask_decoder.parameters():
        p.requires_grad_(True)

    with torch.no_grad():
        emb = backbone(proc)

    # Forward
    dpg_out, low_res, iou_pred = model.forward_train(
        emb["image_embedding"], sup_feat, sup_mask
    )

    # Simple dice loss on predictions
    losses = []
    for qi in range(min(dpg_out.fg_logits.shape[0], gt_masks.shape[0])):
        pred = low_res[qi, 0].sigmoid()  # [256, 256]
        gt_256 = F.interpolate(
            gt_masks[qi].unsqueeze(0).unsqueeze(0).float(),
            (256, 256), mode="nearest"
        ).squeeze()
        inter = (pred * gt_256).sum()
        union = pred.sum() + gt_256.sum()
        dice = 1 - 2 * inter / (union + 1e-7)
        losses.append(dice)

    if losses:
        loss = torch.stack(losses).mean()
        loss.backward()

    # Check gradients
    dec_grad_norms = {}
    for name, p in model.sam_decoder.mask_decoder.named_parameters():
        if p.grad is not None:
            gn = float(p.grad.norm().item())
            dec_grad_norms[name] = gn

    if dec_grad_norms:
        vals = list(dec_grad_norms.values())
        print(f"  Decoder params with grad: {len(vals)}")
        print(f"  Gradient norm: mean={np.mean(vals):.6f}, max={np.max(vals):.6f}, min={np.min(vals):.6f}")
        nonzero = sum(1 for v in vals if v > 1e-8)
        print(f"  Non-zero grads: {nonzero}/{len(vals)}")

        if np.mean(vals) < 1e-6:
            verdict = "NO GRADIENT - decoder is not learning"
        elif np.mean(vals) < 1e-4:
            verdict = "WEAK GRADIENT - decoder barely learning"
        else:
            verdict = "GRADIENT FLOWING"
        print(f"  Verdict: {verdict}")

        # Show top params by grad norm
        sorted_params = sorted(dec_grad_norms.items(), key=lambda x: -x[1])[:5]
        print(f"  Top-5 params by grad norm:")
        for name, gn in sorted_params:
            print(f"    {name}: {gn:.6f}")
    else:
        print("  NO GRADIENTS AT ALL")
        verdict = "NO GRADIENT"

    model.eval()
    return {"grad_mean": np.mean(vals) if vals else 0}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("[DEPRECATED] diag_prompt_chain.py 引用旧 DPG API, 与新 SPG 架构不兼容。")
    print("请使用 tools/eval_isaid_5i.py --diagnostics 替代。")
    print("=" * 60)
    import sys; sys.exit(1)

    p = argparse.ArgumentParser(description="Prompt-to-Mask 信息链诊断")
    p.add_argument("--ckpt", default=None, help="Single checkpoint path")
    p.add_argument("--ckpt-baseline", default=None)
    p.add_argument("--ckpt-samrsp", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-grad", action="store_true")
    p.add_argument("--data-root", default="data/iSAID-5i",
                   help="iSAID dataset root directory (parent of iSAID/)")
    args = p.parse_args()

    # Auto-detect
    run_dir = _REPO_ROOT / "runs"
    ckpt_baseline = args.ckpt_baseline or str(
        run_dir / "fix_baseline" / "isaid5i_fold0_k1_novel_seed42" / "best_model.pt"
    )
    ckpt_samrsp = args.ckpt_samrsp or str(
        run_dir / "fix_samrsp" / "isaid5i_fold0_k1_novel_seed42" / "best_model.pt"
    )

    if args.ckpt:
        checkpoints = [(args.ckpt, Path(args.ckpt).stem)]
    else:
        checkpoints = []
        if Path(ckpt_baseline).exists():
            checkpoints.append((ckpt_baseline, "Baseline"))
        if Path(ckpt_samrsp).exists():
            checkpoints.append((ckpt_samrsp, "SAM-RSP"))

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoints: {[c[1] for c in checkpoints]}")

    # Read fold/mode from checkpoint config if available, else defaults
    fold, mode = 0, "novel"
    if checkpoints:
        ckpt_data = torch.load(checkpoints[0][0], map_location="cpu")
        if "config" in ckpt_data:
            fold = int(ckpt_data["config"].get("data", {}).get("fold", 0))
            mode = ckpt_data["config"].get("fewshot", {}).get("train_mode", "novel")
            if "fold" in ckpt_data:  # checkpoint-level override
                fold = int(ckpt_data["fold"])

    dataset = ISAID5iDataset(root=args.data_root, fold=fold, split="val", mode=mode)
    print(f"Val tiles: {len(dataset)}, classes: {dataset.visible_classes()}")

    results = {}
    for ckpt_path, label in checkpoints:
        print(f"\n{'#' * 72}")
        print(f"#  {label}: {Path(ckpt_path).name}")
        print(f"{'#' * 72}")

        model, backbone, cfg, _ = load_and_build(ckpt_path, device)
        print(f"  coarse_prior={'ON' if model.coarse_prior else 'OFF'}  "
              f"feedback={'ON' if model.dpg.feedback_conv else 'OFF'}")

        r = {}

        r["t1_dense"] = test1_dense_prompt(model, backbone, dataset, device)
        r["t2_ablation"] = test2_prompt_ablation(model, backbone, dataset, device)
        r["t3_query"] = test3_query_similarity(model, backbone, dataset, device)
        r["t4_masks"] = test4_mask_logits(model, backbone, dataset, device)
        if not args.skip_grad:
            r["t5_grad"] = test5_gradient_check(model, backbone, dataset, device)

        results[label] = r
        del model, backbone
        torch.cuda.empty_cache()

    # Summary
    sep("SUMMARY")
    for label, r in results.items():
        print(f"\n  {label}:")
        t1 = r.get("t1_dense", {})
        t2 = r.get("t2_ablation", {})
        t3 = r.get("t3_query", {})
        t4 = r.get("t4_masks", {})
        t5 = r.get("t5_grad", {})

        if t1:
            print(f"    T1 DensePrompt std:  {t1.get('avg_std', 0):.6f}  {'DEAD' if t1.get('avg_std', 0) < 0.01 else 'ALIVE'}")
        if t2:
            print(f"    T2 Δ(Normal-Zero):   {t2.get('delta_zero', 0):+.4f}  {'NO EFFECT' if abs(t2.get('delta_zero', 0)) < 0.01 else 'HAS EFFECT'}")
        if t3:
            print(f"    T3 Query cos_sim:    {t3.get('avg_cos', 0):.4f}  {'COLLAPSED' if t3.get('avg_cos', 0) > 0.9 else 'OK'}")
            print(f"    T3 Active queries:   {t3.get('n_active', 0)}/16")
        if t4:
            issues = t4.get("issues", [])
            print(f"    T4 Mask issues:      {issues if issues else 'NONE'}")
        if t5:
            print(f"    T5 Decoder grad:     {t5.get('grad_mean', 0):.6f}")

    print("\nDone. Run with --ckpt to analyze a single checkpoint.")


if __name__ == "__main__":
    main()
