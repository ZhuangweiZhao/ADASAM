#!/usr/bin/env python3
"""
[DEPRECATED] 训练后一键检查 | Post-Training One-Click Check
============================================================
⚠️ 此工具引用旧 DPG API (dpg.spatial_prompt_scale, dpg.spatial_prompt_proj,
   dpg.prompt_mask_head, dpg_out.fg_queries, dpg_out.fg_logits, model.coarse_prior,
   model.dpg.feedback_conv), 与新 SPG 架构不兼容, 运行会崩溃。需更新后才能使用。

训练完成后运行此脚本, 自动完成:
  1. checkpoint 关键参数检查 (spatial_prompt_scale, prompt_mask_head, etc.)
  2. Dense Prompt 诊断 (diag_prompt_chain 全部 5 项测试)
  3. 正式评估 (eval_isaid_5i)

用法:
  python tools/post_train_check.py --ckpt runs/fix_v3_v2/.../best_model.pt --data-root /root/autodl-tmp
  python tools/post_train_check.py --ckpt runs/fix_v3_v2/.../best_model.pt --k-shot 1 --skip-eval  # 仅诊断
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


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Checkpoint 参数检查
# ═══════════════════════════════════════════════════════════════════════════

def check_checkpoint_params(ckpt_path: str):
    """检查 checkpoint 中 V3 关键参数的值."""
    print("=" * 72)
    print("  Phase 0: Checkpoint 参数检查 | Parameter Inspection")
    print("=" * 72)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]

    # ---- spatial_prompt_scale ----
    print("\n── V3.1 Spatial Dense Prompt ──")
    scale_key = "dpg.spatial_prompt_scale"
    if scale_key in state:
        scale_val = state[scale_key].item()
        print(f"  spatial_prompt_scale = {scale_val:.6f}")
        if scale_val < 0.001:
            print(f"  ⚠ WARNING: scale 接近 0, 空间 prompt 被训练压死!")
            print(f"    建议: 增大 prompt_weight (0.5→2.0) 或固定 scale=1.0 再训")
        elif scale_val < 0.05:
            print(f"  ⚡ scale 偏小 (init=0.1), 训练压低了空间信号幅度")
            print(f"    T1 std 会显示 DEAD 但 T2 可能仍然有效")
        elif scale_val < 0.5:
            print(f"  ✅ scale 健康增长中 ({0.1 if 'init' in 'scale_val' else 'init=0.1 → ' + str(scale_val)})")
        else:
            print(f"  ✅ scale 充分激活")
    else:
        print(f"  ⚠ 未找到 spatial_prompt_scale — 这是 V2 模型?")

    # ---- spatial_prompt_proj ----
    for k in ["dpg.spatial_prompt_proj.0.weight", "dpg.spatial_prompt_proj.2.weight"]:
        if k in state:
            w = state[k].float()
            print(f"  {k}: mean={w.mean():.6f}, std={w.std():.6f}, "
                  f"norm={w.norm():.4f}")

    # ---- prompt_mask_head ----
    print("\n── V3.2 Prompt BCE Head ──")
    pmh_key = "dpg.prompt_mask_head.weight"
    if pmh_key in state:
        w = state[pmh_key].float()
        b = state["dpg.prompt_mask_head.bias"].float()
        print(f"  prompt_mask_head.weight: mean={w.mean():.6f}, std={w.std():.6f}, norm={w.norm():.4f}")
        print(f"  prompt_mask_head.bias:   {b.item():.6f}")

    # ---- Legacy dense prompt (对比) ----
    print("\n── Legacy Global Dense Prompt (对比) ──")
    for k in ["dpg.dense_pool_attn.weight", "dpg.dense_prompt_gen.0.weight", "dpg.dense_prompt_gen.2.weight"]:
        if k in state:
            w = state[k].float()
            print(f"  {k}: mean={w.mean():.6f}, std={w.std():.6f}, norm={w.norm():.4f}")

    # ---- Config ----
    cfg = ckpt.get("config", {})
    print("\n── Config ──")
    pw = cfg.get("loss", {}).get("prompt_weight", "N/A")
    aw = cfg.get("loss", {}).get("aux_weight", "N/A")
    kshot = cfg.get("fewshot", {}).get("k_shot", "N/A")
    print(f"  prompt_weight: {pw}")
    print(f"  aux_weight:    {aw}")
    print(f"  k_shot:        {kshot}")

    # ---- Training metrics (if available) ----
    epoch = ckpt.get("epoch", "N/A")
    print(f"\n  Epoch: {epoch}")

    del ckpt
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1-5: 复用 diag_prompt_chain 的测试逻辑
# ═══════════════════════════════════════════════════════════════════════════

def load_and_build(ckpt_path: str, device: str = "cuda"):
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


# ── T1: Dense Prompt 激活检查 (含空间诊断) ──

@torch.no_grad()
def test1_dense_prompt(model, backbone, dataset, device: str):
    """增强版 T1: 除全局 std 外, 增加空间相关性诊断."""
    print("\n" + "─" * 72)
    print("  T1: Dense Prompt 激活检查 (含空间诊断)")
    print("─" * 72)

    all_stats = []
    all_spatial_corr = []  # 空间位置间余弦相似度
    all_chan_std = []      # 每通道 std

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
            if dpg_out.dense_prompt is None:
                continue

            dp = dpg_out.dense_prompt  # [1, C, H, W]

            # 基础统计
            all_stats.append({
                "mean": float(dp.mean()),
                "std": float(dp.std()),
                "abs_mean": float(dp.abs().mean()),
                "abs_max": float(dp.abs().max()),
                "norm": float(dp.norm(p="fro")),
            })

            # 空间相关性: 随机采样位置对, 计算特征余弦相似度
            # 完全平坦 → cos≈1; 空间多样 → cos 分布在 [0,1]
            C, H, W = dp.shape[1], dp.shape[2], dp.shape[3]
            dp_flat = dp[0].reshape(C, H * W).permute(1, 0)  # [H*W, C]
            # 采样 500 对位置
            n_pairs = min(500, H * W * (H * W - 1) // 2)
            idx1 = torch.randint(0, H * W, (n_pairs,), device=device)
            idx2 = torch.randint(0, H * W, (n_pairs,), device=device)
            # 去除相同位置
            same_mask = idx1 == idx2
            idx2[same_mask] = (idx2[same_mask] + 1) % (H * W)

            v1 = F.normalize(dp_flat[idx1], dim=1)
            v2 = F.normalize(dp_flat[idx2], dim=1)
            cos_pairs = (v1 * v2).sum(dim=1)  # [n_pairs]
            all_spatial_corr.append(float(cos_pairs.mean()))

            # 每通道 std (通道内 spatial variation)
            chan_std = dp[0].std(dim=[1, 2])  # [C] — 每个 channel 的空间变化
            all_chan_std.append(float(chan_std.mean()))

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

    # 空间诊断
    print(f"\n  ── Spatial Diagnostics ──")
    avg_spatial_corr = np.mean(all_spatial_corr)
    avg_chan_std = np.mean(all_chan_std)
    print(f"  spatial_pos_corr (随机位置对 cos_sim): {avg_spatial_corr:.4f}")
    print(f"    → 1.0=完全平坦(广播), <0.5=空间丰富, <0=反相关")
    print(f"  per-channel spatial std (通道内空间变化): {avg_chan_std:.6f}")
    print(f"    → 越大说明每个通道有独立的空间模式")

    # 综合判断
    avg_std = np.mean(stds)
    print(f"\n  ── Verdict ──")

    # 空间 prompt 需要不同的判断标准:
    # 全局 std 低但空间相关性 <0.95 → 有空间结构, 只是幅度小
    # 全局 std 低且空间相关性 >0.95 → 真的接近平坦
    if avg_std < 0.01:
        if avg_spatial_corr > 0.95:
            verdict = "DEAD (FLAT) — 空间结构和幅度都接近零, dense prompt 未学习"
        elif avg_spatial_corr > 0.80:
            verdict = "WEAK SPATIAL — 幅度极小但略有空间差异, 可能被 scale 压制"
        else:
            verdict = "TINY BUT STRUCTURED — 幅度极小 (scale↓) 但空间结构丰富, DECODER 仍可利用"
    elif avg_std < 0.05:
        if avg_spatial_corr > 0.90:
            verdict = "WEAK — 有微弱信号但空间差异不足"
        else:
            verdict = "SPATIALLY ALIVE — 幅度偏小但空间结构良好"
    elif avg_std < 0.1:
        verdict = "WEAK — dense prompt has weak signal"
    else:
        verdict = "ALIVE — dense prompt has meaningful signal"
    print(f"  {verdict}")

    return {
        "avg_std": avg_std,
        "avg_abs_mean": np.mean(abs_means),
        "spatial_corr": avg_spatial_corr,
        "chan_std": avg_chan_std,
        "verdict": verdict,
    }


# ── T2: Dense Prompt 扰动实验 ──

def _compute_iou(masks: torch.Tensor, gt: torch.Tensor) -> float:
    if len(masks) == 0:
        return 0.0
    pred = masks.any(dim=0)
    inter = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    return float(inter / (union + 1e-7))


@torch.no_grad()
def test2_prompt_ablation(model, backbone, dataset, device: str):
    """真实 dense_prompt vs Zero vs SAM default."""
    print("\n" + "─" * 72)
    print("  T2: Dense Prompt 扰动实验")
    print("─" * 72)

    classes = dataset.visible_classes()
    ious_normal = []
    ious_zero = []
    ious_nomask = []

    for cls in classes[:3]:
        sup_feat, sup_mask = build_support(model, backbone, dataset, cls, k_shot=1, device=device)
        if sup_feat is None:
            continue

        tiles = dataset.class_to_tiles(cls)
        rng = random.Random(42)
        sampled = rng.sample(tiles, min(15, len(tiles)))

        no_mask = model.sam_decoder.prompt_encoder.no_mask_embed.weight.view(1, -1, 1, 1)

        for idx in sampled:
            sample = dataset[idx]
            proc, _ = preprocess_image(sample["image"], backbone.img_size)
            proc = proc.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = backbone(proc)
            qf = emb["image_embedding"]

            gt = torch.zeros(256, 256, dtype=torch.bool, device=device)
            for inst in sample["regions"]:
                if inst["category_id"] == cls:
                    gt = gt | inst["mask"].to(device)

            # Path A: Normal (support-conditioned dense prompt)
            dpg_out, low_res, iou_pred = model.forward_train(qf, sup_feat, sup_mask)
            masks_n, scores_n = model.predict(
                qf, sup_feat, sup_mask,
                (backbone.img_size, backbone.img_size), (256, 256), score_thr=0.3,
            )
            iou_n = _compute_iou(masks_n, gt)

            # Path B: Zero dense prompt
            zero_dense = torch.zeros(1, 256, 64, 64, device=device)
            z_mask = model.sam_decoder(qf, dpg_out.fg_queries, zero_dense)
            z_logits = z_mask[0]
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

            # Path C: SAM default no_mask_embed
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

    if abs(delta_nm) < 0.01:
        verdict = "NO EFFECT — dense prompt 对 mask 输出无影响"
    elif abs(delta_nm) < 0.05:
        verdict = "MARGINAL — dense prompt 效果微弱"
    elif delta_nm > 0.10:
        verdict = f"STRONG (+{delta_nm:.3f}) — support dense prompt 大幅优于 SAM default"
    else:
        verdict = f"EFFECTIVE (+{delta_nm:.3f}) — dense prompt 有效改善输出"
    print(f"  Verdict: {verdict}")

    return {"delta_zero": delta_zn, "delta_nomask": delta_nm, "verdict": verdict}


# ── T3: Query Embedding 相似度 ──

@torch.no_grad()
def test3_query_similarity(model, backbone, dataset, device: str):
    print("\n" + "─" * 72)
    print("  T3: Query Embedding 相似度")
    print("─" * 72)

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
            q = dpg_out.fg_queries
            qn = F.normalize(q, dim=1)
            cos = torch.mm(qn, qn.t())

            mask = ~torch.eye(q.shape[0], dtype=torch.bool, device=device)
            off_diag = cos[mask]
            all_cos.append(off_diag.cpu())

            obj = dpg_out.fg_logits.sigmoid().cpu()
            all_obj.append(obj)

    if not all_cos:
        return {}

    all_cos_cat = torch.cat(all_cos)
    all_obj_stack = torch.stack(all_obj)

    print(f"  Query pairs analyzed: {len(all_cos_cat)}")
    print(f"  Cosine similarity (off-diagonal):")
    print(f"    mean={all_cos_cat.mean():.4f}, std={all_cos_cat.std():.4f}")
    print(f"    min={all_cos_cat.min():.4f}, max={all_cos_cat.max():.4f}")
    print(f"    P10={torch.quantile(all_cos_cat, 0.1):.4f}, "
          f"P50={torch.quantile(all_cos_cat, 0.5):.4f}, "
          f"P90={torch.quantile(all_cos_cat, 0.9):.4f}")

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


# ── T4: low_res_masks 分布 ──

@torch.no_grad()
def test4_mask_logits(model, backbone, dataset, device: str):
    print("\n" + "─" * 72)
    print("  T4: low_res_masks 分布")
    print("─" * 72)

    all_min, all_max, all_mean, all_std, all_pos_ratio, all_entropy = [], [], [], [], [], []

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
            for qi in range(low_res.shape[0]):
                m = low_res[qi]
                all_min.append(float(m.min()))
                all_max.append(float(m.max()))
                all_mean.append(float(m.mean()))
                all_std.append(float(m.std()))
                all_pos_ratio.append(float((m > 0).float().mean()))
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

    avg_min, avg_max, avg_pos = np.mean(all_min), np.mean(all_max), np.mean(all_pos_ratio)
    issues = []
    if avg_min >= 0:
        issues.append("ALL POSITIVE — no negative boundary signal")
    if avg_max < 1.0:
        issues.append(f"LOW ACTIVATION — max={avg_max:.2f}")
    if avg_max > 15:
        issues.append(f"SATURATED — max={avg_max:.2f}")
    if avg_pos > 0.9:
        issues.append("TOO POSITIVE")
    if avg_pos < 0.05:
        issues.append("ALL NEGATIVE")

    if not issues:
        verdict = "OK NORMAL"
    else:
        verdict = "ISSUES: " + "; ".join(issues)
    print(f"  Verdict: {verdict}")

    return {"min_mean": avg_min, "max_mean": avg_max, "pos_ratio": avg_pos, "issues": issues}


# ── T5: 梯度流检查 ──

def test5_gradient_check(model, backbone, dataset, device: str):
    print("\n" + "─" * 72)
    print("  T5: SAM Decoder 梯度流检查")
    print("─" * 72)

    cls = dataset.visible_classes()[0]
    sup_feat, sup_mask = build_support(model, backbone, dataset, cls, k_shot=1, device=device)
    if sup_feat is None:
        print("  Cannot build support")
        return {}

    tiles = dataset.class_to_tiles(cls)
    sample = dataset[random.Random(42).choice(tiles)]
    proc, _ = preprocess_image(sample["image"], backbone.img_size)
    proc = proc.unsqueeze(0).to(device)

    gt_list = [inst["mask"] for inst in sample["regions"] if inst["category_id"] == cls]
    if not gt_list:
        print("  No GT for this class")
        return {}
    if len(gt_list) > model.num_queries:
        gt_list = sorted(gt_list, key=lambda m: m.sum(), reverse=True)[:model.num_queries]
    gt_masks = torch.stack([m.float() for m in gt_list], dim=0).to(device)

    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.sam_decoder.mask_decoder.parameters():
        p.requires_grad_(True)

    with torch.no_grad():
        emb = backbone(proc)

    dpg_out, low_res, iou_pred = model.forward_train(
        emb["image_embedding"], sup_feat, sup_mask
    )

    losses = []
    for qi in range(min(dpg_out.fg_logits.shape[0], gt_masks.shape[0])):
        pred = low_res[qi, 0].sigmoid()
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
        print(f"  Top-5 params by grad norm:")
        sorted_params = sorted(dec_grad_norms.items(), key=lambda x: -x[1])[:5]
        for name, gn in sorted_params:
            print(f"    {name}: {gn:.6f}")

        if np.mean(vals) < 1e-6:
            verdict = "NO GRADIENT"
        elif np.mean(vals) < 1e-4:
            verdict = "WEAK GRADIENT"
        else:
            verdict = "GRADIENT FLOWING"
    else:
        verdict = "NO GRADIENT"
        vals = []

    print(f"  Verdict: {verdict}")
    model.eval()
    return {"grad_mean": np.mean(vals) if vals else 0}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: 正式评估
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(ckpt_path: str, data_root: str, k_shot: int, device: str):
    """调用 eval_isaid_5i.py 进行正式评估."""
    print("\n" + "=" * 72)
    print("  Phase 6: 正式评估 | Evaluation")
    print("=" * 72)

    import subprocess
    eval_script = str(_REPO_ROOT / "tools" / "eval_isaid_5i.py")
    cmd = [
        sys.executable, eval_script,
        "--checkpoint", ckpt_path,
        "--k-shot", str(k_shot),
        "--data-root", data_root,
    ]
    print(f"  Running: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    return result.returncode == 0


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("[DEPRECATED] post_train_check.py 引用旧 DPG API, 与新 SPG 架构不兼容。")
    print("请使用 tools/eval_isaid_5i.py --diagnostics 替代。")
    print("=" * 60)
    import sys; sys.exit(1)

    p = argparse.ArgumentParser(
        description="训练后一键检查 | Post-Training One-Click Check"
    )
    p.add_argument("--ckpt", required=True, help="Checkpoint path")
    p.add_argument("--data-root", default="data/iSAID-5i",
                   help="iSAID dataset root directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--k-shot", type=int, default=None,
                   help="K-shot (auto-detect from checkpoint if omitted)")
    p.add_argument("--skip-t5", action="store_true", help="Skip gradient check")
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ═══════════════════ Phase 0: 参数检查 ═══════════════════
    state = check_checkpoint_params(args.ckpt)

    # 从 checkpoint 读取配置
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    fold = cfg.get("data", {}).get("fold", 0)
    mode = cfg.get("fewshot", {}).get("train_mode", "novel")
    k_shot = args.k_shot or cfg.get("fewshot", {}).get("k_shot", 1)
    del ckpt

    print(f"\n  Auto-detected: fold={fold}, mode={mode}, k_shot={k_shot}")

    # ═══════════════════ Phase 1-5: 诊断 ═══════════════════
    print("\n" + "=" * 72)
    print("  Phase 1-5: 信息链诊断 | Prompt Chain Diagnosis")
    print("=" * 72)

    dataset = ISAID5iDataset(root=args.data_root, fold=fold, split="val", mode=mode)
    print(f"Val tiles: {len(dataset)}, classes: {dataset.visible_classes()}")

    model, backbone, cfg, _ = load_and_build(args.ckpt, device)
    print(f"coarse_prior={'ON' if model.coarse_prior else 'OFF'}  "
          f"feedback={'ON' if model.dpg.feedback_conv else 'OFF'}")

    results = {}

    results["t1"] = test1_dense_prompt(model, backbone, dataset, device)
    results["t2"] = test2_prompt_ablation(model, backbone, dataset, device)
    results["t3"] = test3_query_similarity(model, backbone, dataset, device)
    results["t4"] = test4_mask_logits(model, backbone, dataset, device)

    if not args.skip_t5:
        results["t5"] = test5_gradient_check(model, backbone, dataset, device)

    del model, backbone
    torch.cuda.empty_cache()

    # ═══════════════════ Summary ═══════════════════
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    t1 = results.get("t1", {})
    t2 = results.get("t2", {})
    t3 = results.get("t3", {})
    t4 = results.get("t4", {})
    t5 = results.get("t5", {})

    print(f"\n  {'Metric':<30} {'Value':<20} {'Judgment'}")
    print(f"  {'-'*30} {'-'*20} {'-'*20}")

    if t1:
        print(f"  {'T1 DensePrompt std':<30} {t1.get('avg_std', 0):.6f}           {'DEAD' if t1.get('avg_std', 0) < 0.01 else 'ALIVE'}")
        print(f"  {'T1 Spatial corr':<30} {t1.get('spatial_corr', 0):.4f}           {'FLAT' if t1.get('spatial_corr', 0) > 0.95 else 'STRUCTURED' if t1.get('spatial_corr', 0) < 0.80 else 'WEAK STRUCTURE'}")
        print(f"  {'T1 Verdict':<30} {t1.get('verdict', 'N/A')}")
    if t2:
        print(f"  {'T2 Δ(Normal-SAM)':<30} {t2.get('delta_nomask', 0):+.4f}           {t2.get('verdict', 'N/A')}")
    if t3:
        print(f"  {'T3 Query cos_sim':<30} {t3.get('avg_cos', 0):.4f}           {'COLLAPSED' if t3.get('avg_cos', 0) > 0.9 else 'OK'}")
        print(f"  {'T3 Active queries':<30} {t3.get('n_active', 0)}/16")
    if t4:
        issues = t4.get("issues", [])
        print(f"  {'T4 Mask issues':<30} {str(issues) if issues else 'NONE'}")
    if t5:
        print(f"  {'T5 Decoder grad':<30} {t5.get('grad_mean', 0):.6f}")

    # 最终判断
    print(f"\n  {'─'*60}")
    t1_ok = t1.get('spatial_corr', 1.0) < 0.95 if t1 else False
    t2_ok = t2.get('delta_nomask', 0) > 0.02 if t2 else False
    t3_ok = t3.get('avg_cos', 1.0) < 0.9 if t3 else False

    if t2_ok:
        print(f"  ✅ V3 spatial prompt IS WORKING (T2 confirms +{t2.get('delta_nomask', 0):.3f} gain)")
        if t1_ok:
            print(f"  ✅ Spatial structure is present (corr={t1.get('spatial_corr', 0):.3f})")
        else:
            print(f"  ⚠  But spatial_prompt_scale may be too small → T1 shows low std")
            print(f"     This is NOT a bug — decoder can still use tiny but structured signal")
    else:
        print(f"  ❌ V3 spatial prompt has NO significant effect over SAM default")
        print(f"     Possible causes: scale collapsed, wrong normalization, or architecture issue")

    # ═══════════════════ Phase 6: 评估 ═══════════════════
    if not args.skip_eval:
        ok = run_evaluation(args.ckpt, args.data_root, k_shot, device)
        if not ok:
            print("\n  ⚠ Evaluation failed (see above)")

    print("\nDone.")


if __name__ == "__main__":
    main()
