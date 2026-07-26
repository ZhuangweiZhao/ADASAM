"""
Decoder Prompt Sensitivity — 量化 Decoder 对 Prompt 的依赖程度
==============================================================

核心问题: Decoder 把 DP Top20 IoU=0.034 变成 Pred IoU=0.352，这个 10x 增益
到底是因为 decoder 在解码隐式编码，还是 decoder 自己在做 heavy lifting？

通过系统性扰动 dense_prompt，观察 prediction 变化，直接量化因果依赖。

扰动类型:
  1. 幅值缩放: prompt × [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
  2. 加性噪声: prompt + N(0, σ·std)  σ ∈ [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
  3. 通道打乱: 随机排列 256 通道
  4. 空间打乱: 随机排列空间位置 (破坏空间结构)
  5. 通道 dropout: 随机置零 p% 通道  p ∈ [0.1, 0.25, 0.5, 0.75, 0.9]
  6. 仅保留 top-k 通道 (按与 GT 相关性)

输出:
  - 敏感性曲线: 每种扰动 vs Pred IoU / Pearson r
  - 鲁棒性评分: decoder 在多大扰动范围内仍能保持预测质量
  - 每 tile 详细数据

用法:
    python tools/debug/decoder_prompt_sensitivity.py \
        --checkpoint <ckpt> --mode novel --k-shot 5 --num-tiles 30 --save-vis
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5I_CATEGORIES
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils.transforms import preprocess_image, resize_mask


# ═══════════════════════════════════════════════════════════════════
# Perturbation functions
# ═══════════════════════════════════════════════════════════════════

def perturb_scale(dp: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale dense_prompt by factor."""
    return dp * factor


def perturb_noise(dp: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise scaled by channel std."""
    std = dp.std()
    noise = torch.randn_like(dp) * sigma * std
    return dp + noise


def perturb_channel_shuffle(dp: torch.Tensor, seed: int) -> torch.Tensor:
    """Randomly permute channel order."""
    C = dp.shape[1]
    rng = torch.Generator(device=dp.device).manual_seed(seed)
    perm = torch.randperm(C, generator=rng, device=dp.device)
    return dp[:, perm, :, :]


def perturb_spatial_shuffle(dp: torch.Tensor, seed: int) -> torch.Tensor:
    """Randomly permute spatial positions (per channel, fully shuffle)."""
    B, C, H, W = dp.shape
    rng = torch.Generator(device=dp.device).manual_seed(seed)
    flat = dp.reshape(B, C, H * W)
    perm = torch.randperm(H * W, generator=rng, device=dp.device)
    return flat[:, :, perm].reshape(B, C, H, W)


def perturb_channel_dropout(dp: torch.Tensor, drop_pct: float, seed: int) -> torch.Tensor:
    """Randomly zero out drop_pct fraction of channels."""
    B, C, H, W = dp.shape
    rng = torch.Generator(device=dp.device).manual_seed(seed)
    mask = (torch.rand(B, C, 1, 1, generator=rng, device=dp.device) > drop_pct).float()
    return dp * mask


def perturb_topk_channels(dp: torch.Tensor, gt_main: np.ndarray, keep_k: int) -> torch.Tensor:
    """Keep only the top-k channels most correlated with GT (others zeroed)."""
    B, C, H, W = dp.shape
    dp_cpu = dp[0].detach().cpu().float()

    gt_t = torch.from_numpy(gt_main.astype(np.float32))
    gt_rsz = F.interpolate(gt_t.unsqueeze(0).unsqueeze(0), (H, W), mode="area")[0, 0]
    gt_f = gt_rsz.numpy().ravel()

    cors = []
    for c in range(C):
        ch_f = dp_cpu[c].numpy().ravel()
        s_c, s_g = ch_f.std(), gt_f.std()
        cors.append(np.corrcoef(ch_f, gt_f)[0, 1] if s_c > 1e-8 and s_g > 1e-8 else 0.0)
    cors = np.array(cors)
    # Use absolute correlation (both positive and negative can be useful)
    top_k = np.argsort(-np.abs(cors))[:keep_k]

    out = torch.zeros_like(dp)
    out[:, top_k, :, :] = dp[:, top_k, :, :]
    return out


def perturb_keep_positive_channels(dp: torch.Tensor, gt_main: np.ndarray) -> torch.Tensor:
    """Keep only channels with positive correlation with GT."""
    B, C, H, W = dp.shape
    dp_cpu = dp[0].detach().cpu().float()

    gt_t = torch.from_numpy(gt_main.astype(np.float32))
    gt_rsz = F.interpolate(gt_t.unsqueeze(0).unsqueeze(0), (H, W), mode="area")[0, 0]
    gt_f = gt_rsz.numpy().ravel()

    keep = []
    for c in range(C):
        ch_f = dp_cpu[c].numpy().ravel()
        s_c, s_g = ch_f.std(), gt_f.std()
        r = np.corrcoef(ch_f, gt_f)[0, 1] if s_c > 1e-8 and s_g > 1e-8 else 0.0
        if r > 0:
            keep.append(c)

    out = torch.zeros_like(dp)
    if keep:
        out[:, keep, :, :] = dp[:, keep, :, :]
    return out


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _decode_and_eval(dense_prompt: torch.Tensor, model, query_emb,
                     sup_feat, sup_mask, gt_main: np.ndarray,
                     H_orig: int, W_orig: int) -> dict:
    """Decode dense_prompt and compute metrics against GT."""
    with torch.no_grad():
        if model.bypass_head is not None:
            low_res = model.bypass_head(dense_prompt)
        else:
            support_proto = model._compute_support_prototype(sup_feat, sup_mask)
            sparse_token = dense_prompt.mean(dim=(2, 3))
            low_res, _ = model.sam_decoder(query_emb, sparse_token, dense_prompt,
                                           support_prototype=support_proto)

        pred_logits = F.interpolate(low_res.float(), size=(H_orig, W_orig),
                                    mode="bilinear", align_corners=False)[0, 0]
        vals = pred_logits.cpu()
        if vals.min() >= 0 and vals.max() <= 1:
            pred_np = (vals > 0.5).numpy()
            prob_np = vals.numpy()
        else:
            prob_np = vals.sigmoid().numpy()
            pred_np = prob_np > 0.5

    # IoU
    inter = (pred_np & gt_main).sum()
    union = (pred_np | gt_main).sum()
    iou = float(inter / union) if union > 0 else 0.0

    # Pearson r (continuous prob vs GT)
    gt_f = gt_main.astype(np.float32).ravel()
    pr_f = prob_np.ravel()
    s_p, s_g = pr_f.std(), gt_f.std()
    pearson = float(np.corrcoef(pr_f, gt_f)[0, 1]) if s_p > 1e-8 and s_g > 1e-8 else 0.0

    # Dice
    dice = float(2 * inter / (pred_np.sum() + gt_main.sum())) if (pred_np.sum() + gt_main.sum()) > 0 else 0.0

    # Precision / Recall
    tp = inter
    fp = pred_np.sum() - tp
    fn = gt_main.sum() - tp
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "iou": iou, "dice": dice, "pearson": pearson,
        "precision": precision, "recall": recall,
        "prob_mean": float(prob_np.mean()), "prob_std": float(prob_np.std()),
        "pred_area": int(pred_np.sum()),
    }


def _channel_stats(dp: torch.Tensor) -> dict:
    """Basic statistics of dense_prompt."""
    x = dp.detach().cpu().float()[0]  # [C, H, W]
    return {
        "dp_mean": float(x.mean()),
        "dp_std": float(x.std()),
        "dp_l2_norm": float(x.pow(2).mean().sqrt()),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Decoder Prompt Sensitivity Test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/iSAID-5i")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--mode", default="novel")
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tiles", type=int, default=30)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-vis", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    fold = args.fold if args.fold is not None else ckpt.get("fold", 0)
    mode = args.mode if args.mode is not None else ckpt.get("mode", "novel")
    k_shot = args.k_shot if args.k_shot is not None else ckpt.get("k_shot", 5)

    # ── Load model ──
    bb_cfg = cfg.get("backbone", {})
    bb_path = Path(bb_cfg.get("checkpoint", "weights/mobile_sam.pt"))
    bb_path = bb_path if bb_path.is_absolute() else _REPO_ROOT / bb_path
    sam = build_mobile_sam(str(bb_path), bb_cfg.get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    embed_dim = int(cfg.get("support_encoder", {}).get("embed_dim", 256))
    model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(cfg)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    cat_adapter = None
    if ckpt.get("cat_adapter") is not None:
        tcfg = cfg.get("train", {})
        cat_adapter = CATAdapter(
            dim=embed_dim,
            bottleneck=int(tcfg.get("cat_adapter", {}).get("bottleneck", 64)),
        ).to(device)
        cat_adapter.load_state_dict(ckpt["cat_adapter"]); cat_adapter.eval()

    has_pf = model.prompt_fusion is not None
    has_gp = model.geometric_prior is not None
    use_bypass = model.bypass_head is not None
    print(f"[model] bypass_head={use_bypass}, prompt_fusion={has_pf}, geometric_prior={has_gp}")

    data_root = Path(args.data_root)
    if not data_root.is_absolute(): data_root = _REPO_ROOT / data_root
    val_ds = ISAID5iDataset(root=str(data_root), fold=fold, split="val", mode=mode)
    visible_classes = val_ds.visible_classes()

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "decoder_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir = out_dir / "figures"
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Build support cache ──
    print(f"[setup] building support cache ({len(visible_classes)} classes)...")
    support_cache: dict = {}
    for cls in visible_classes:
        ds = ISAID5iDataset(root=str(data_root), fold=fold, split="train", mode=mode)
        tiles = ds.class_to_tiles(cls)
        if not tiles: continue
        scenes = defaultdict(list)
        for idx in tiles:
            src = ds._source_images.get(ds.tile_ids[idx], ds.tile_ids[idx])
            scenes[src].append(idx)
        rng = random.Random(args.seed)
        keys = list(scenes)
        k = min(k_shot, len(keys))
        chosen = rng.sample(keys, k)
        images, masks = [], []
        for sid in chosen:
            idx = rng.choice(scenes[sid])
            s = ds[idx]
            fg = ds.get_class_mask(idx, cls)
            if fg is None or fg.sum() < 1: continue
            xx, _ = preprocess_image(s["image"])
            images.append(xx.to(device)); masks.append(fg)
        if not images: continue
        feats = backbone(torch.stack(images, dim=0))["image_embedding"]
        if cat_adapter is not None: feats = cat_adapter(feats)
        mg = torch.stack([resize_mask(m, (feats.shape[2], feats.shape[3])).to(device)
                          for m in masks], dim=0)
        support_cache[cls] = (feats, mg)
    print(f"[setup] support cache: {len(support_cache)}/{len(visible_classes)}")

    # ── Find tiles ──
    candidates = []
    for idx in range(len(val_ds)):
        present = [c for c in visible_classes
                   if val_ds.get_class_mask(idx, c) is not None
                   and val_ds.get_class_mask(idx, c).sum() > 50]
        if len(present) >= 1: candidates.append((idx, present))
    random.Random(args.seed + 9999).shuffle(candidates)
    selected = candidates[:args.num_tiles]
    print(f"[select] {len(candidates)} candidates, using {len(selected)}")

    # ── Define perturbation grid ──
    scale_factors = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
    noise_sigmas = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    channel_drop_pcts = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    topk_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # Accumulators: perturbation_name → list of per-tile metric dicts
    all_results: dict[str, list[dict]] = defaultdict(list)

    for tile_idx, present_classes in tqdm(selected, desc="sensitivity"):
        sample = val_ds[tile_idx]
        H, W = sample["image"].shape[1], sample["image"].shape[2]
        img_np = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        xx, meta = preprocess_image(sample["image"])
        query_emb = backbone(xx.unsqueeze(0).to(device))["image_embedding"]
        if cat_adapter is not None: query_emb = cat_adapter(query_emb)

        main_cls = max(present_classes, key=lambda c: val_ds.get_class_mask(tile_idx, c).sum())
        sup_data = support_cache.get(main_cls)
        if sup_data is None: continue
        sup_feat, sup_mask = sup_data
        gt_main = val_ds.get_class_mask(tile_idx, main_cls).numpy().astype(bool)

        # ── Build the normal dense_prompt (baseline) ──
        support_memory = model.support_encoder(sup_feat, sup_mask)

        geometric_prior = None
        if model.geometric_prior is not None:
            geometric_prior = model.geometric_prior(query_emb, support_memory)

        dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
        spg_out = model.spg(query_emb, support_memory, dense_pe)

        if model.prompt_fusion is not None and geometric_prior is not None:
            dp_baseline, _ = model.prompt_fusion(geometric_prior, spg_out.semantic_prior)
        else:
            dp_baseline = spg_out.semantic_prior

        dp_stats = _channel_stats(dp_baseline)

        # ── Evaluate baseline ──
        baseline_metrics = _decode_and_eval(dp_baseline, model, query_emb,
                                            sup_feat, sup_mask, gt_main, H, W)
        all_results["baseline"].append({**baseline_metrics, **dp_stats,
                                        "tile_idx": tile_idx, "class_id": main_cls})

        tile_seed_base = args.seed * 10000 + tile_idx * 100 + main_cls

        # ── 1. Scale perturbation ──
        for factor in scale_factors:
            if abs(factor - 1.0) < 1e-6:
                continue  # baseline already covers this
            dp_p = perturb_scale(dp_baseline, factor)
            m = _decode_and_eval(dp_p, model, query_emb, sup_feat, sup_mask, gt_main, H, W)
            m["perturb_value"] = factor
            m["tile_idx"] = tile_idx; m["class_id"] = main_cls
            all_results[f"scale_{factor}"].append(m)

        # ── 2. Noise perturbation ──
        for sigma in noise_sigmas:
            dp_p = perturb_noise(dp_baseline, sigma)
            m = _decode_and_eval(dp_p, model, query_emb, sup_feat, sup_mask, gt_main, H, W)
            m["perturb_value"] = sigma
            m["tile_idx"] = tile_idx; m["class_id"] = main_cls
            all_results[f"noise_{sigma}"].append(m)

        # ── 3. Channel shuffle ──
        for rep in range(3):  # 3 random shuffles for variance
            seed = tile_seed_base + rep
            dp_p = perturb_channel_shuffle(dp_baseline, seed)
            m = _decode_and_eval(dp_p, model, query_emb, sup_feat, sup_mask, gt_main, H, W)
            m["perturb_value"] = rep
            m["tile_idx"] = tile_idx; m["class_id"] = main_cls
            all_results["channel_shuffle"].append(m)

        # ── 4. Spatial shuffle ──
        for rep in range(3):
            seed = tile_seed_base + 100 + rep
            dp_p = perturb_spatial_shuffle(dp_baseline, seed)
            m = _decode_and_eval(dp_p, model, query_emb, sup_feat, sup_mask, gt_main, H, W)
            m["perturb_value"] = rep
            m["tile_idx"] = tile_idx; m["class_id"] = main_cls
            all_results["spatial_shuffle"].append(m)

        # ── 5. Channel dropout ──
        for drop_pct in channel_drop_pcts:
            seed = tile_seed_base + 200
            dp_p = perturb_channel_dropout(dp_baseline, drop_pct, seed)
            m = _decode_and_eval(dp_p, model, query_emb, sup_feat, sup_mask, gt_main, H, W)
            m["perturb_value"] = drop_pct
            m["tile_idx"] = tile_idx; m["class_id"] = main_cls
            all_results[f"ch_drop_{drop_pct}"].append(m)

        # ── 6. Top-K channels ──
        for keep_k in topk_values:
            dp_p = perturb_topk_channels(dp_baseline, gt_main, keep_k)
            m = _decode_and_eval(dp_p, model, query_emb, sup_feat, sup_mask, gt_main, H, W)
            m["perturb_value"] = keep_k
            m["tile_idx"] = tile_idx; m["class_id"] = main_cls
            all_results[f"topk_{keep_k}"].append(m)

        # ── 7. Only positive-correlation channels ──
        dp_p = perturb_keep_positive_channels(dp_baseline, gt_main)
        m = _decode_and_eval(dp_p, model, query_emb, sup_feat, sup_mask, gt_main, H, W)
        m["perturb_value"] = 0
        m["tile_idx"] = tile_idx; m["class_id"] = main_cls
        all_results["pos_channels_only"].append(m)

    # ── Aggregate ──
    summary = _aggregate(all_results)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Sensitivity scores ──
    scores = _compute_sensitivity_scores(summary, all_results)
    scores_path = out_dir / "sensitivity_scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    # ── Print ──
    _print_sensitivity_report(summary, scores)

    if args.save_vis:
        _save_sensitivity_charts(summary, scores, all_results, vis_dir, scale_factors,
                                noise_sigmas, channel_drop_pcts, topk_values)

    print(f"\n[Summary] {summary_path}")
    print(f"[Scores] {scores_path}")
    print("[Done]")


# ═══════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════

def _agg_vals(vals: list) -> dict | None:
    clean = [v for v in vals if v is not None and v == v]
    if not clean: return None
    return {
        "mean": round(float(np.mean(clean)), 4),
        "median": round(float(np.median(clean)), 4),
        "std": round(float(np.std(clean)), 4),
        "n": len(clean),
    }


def _aggregate(all_results: dict) -> dict:
    summary = {}
    for pname, items in all_results.items():
        if not items: continue
        sm = {}
        for key in items[0]:
            if key in ("tile_idx", "class_id", "perturb_value"): continue
            agg = _agg_vals([d[key] for d in items])
            if agg: sm[key] = agg
        summary[pname] = {"n_episodes": len(items), "metrics": sm}
    return summary


def _compute_sensitivity_scores(summary: dict, all_results: dict) -> dict:
    """Compute robustness scores from perturbation curves.

    Sensitivity = how fast IoU drops as perturbation increases.
    Lower sensitivity = decoder is more robust to that perturbation type.
    """
    base_iou = summary.get("baseline", {}).get("metrics", {}).get("iou", {}).get("mean", 0)

    def _half_life(results_dict: dict, prefix: str, values: list[float],
                   value_key: str = "perturb_value", invert: bool = False) -> dict | None:
        """Find the perturbation value where IoU drops to 50% of baseline.

        For topk, we invert (higher keep_k = less perturbation).
        """
        ious = []
        vals = []
        for v in values:
            key = f"{prefix}_{v}"
            if key in summary:
                m = summary[key]["metrics"].get("iou", {})
                if m:
                    ious.append(m["mean"])
                    vals.append(v)

        if not ious:
            return None

        # Find half-life
        half_iou = base_iou / 2
        half_val = None
        for i in range(len(ious) - 1):
            if (ious[i] >= half_iou and ious[i+1] <= half_iou) or \
               (ious[i] <= half_iou and ious[i+1] >= half_iou):
                # Linear interpolation
                frac = (half_iou - ious[i]) / (ious[i+1] - ious[i] + 1e-8)
                half_val = vals[i] + frac * (vals[i+1] - vals[i])
                break

        return {
            "values": vals,
            "ious": ious,
            "half_life": round(half_val, 4) if half_val is not None else None,
            "min_iou": round(min(ious), 4),
            "max_iou": round(max(ious), 4),
            "iou_range": round(max(ious) - min(ious), 4),
        }

    scores = {"baseline_iou": base_iou}

    # Scale: drop to half at what factor?
    scale_factors = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
    scores["scale"] = _half_life(summary, "scale", scale_factors)

    # Noise
    noise_sigmas = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    scores["noise"] = _half_life(summary, "noise", noise_sigmas)

    # Channel dropout
    drop_pcts = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    scores["ch_drop"] = _half_life(summary, "ch_drop", drop_pcts)

    # Top-K
    topk_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    scores["topk"] = _half_life(summary, "topk", topk_vals)

    # Discrete perturbations (single-point comparisons)
    for pname, label in [("channel_shuffle", "Channel shuffle"),
                          ("spatial_shuffle", "Spatial shuffle"),
                          ("pos_channels_only", "Pos-ch only")]:
        if pname in summary:
            m = summary[pname]["metrics"].get("iou", {})
            scores[label] = {
                "iou_mean": m.get("mean", 0) if m else 0,
                "iou_drop": round(base_iou - (m.get("mean", 0) if m else 0), 4),
                "retention": round((m.get("mean", 0) if m else 0) / max(base_iou, 1e-8), 4),
            }

    return scores


# ═══════════════════════════════════════════════════════════════════
# Printing
# ═══════════════════════════════════════════════════════════════════

def _print_sensitivity_report(summary: dict, scores: dict):
    SEP = "=" * 90
    print(f"\n{SEP}")
    print("  Decoder Prompt Sensitivity — how much does decoder depend on prompt?")
    print(f"{SEP}")

    base_iou = scores.get("baseline_iou", 0)
    print(f"\n  Baseline Pred IoU: {base_iou:.4f}")

    # ── Continuous perturbations ──
    print(f"\n  ┌─ Continuous Perturbation Curves{'─'*56}")
    for ptype, label in [("scale", "Scale (×factor)"), ("noise", "Noise (σ)"),
                          ("ch_drop", "Channel Dropout (%)"), ("topk", "Top-K channels")]:
        curve = scores.get(ptype)
        if curve is None: continue
        print(f"\n  [{label}]")
        print(f"    IoU range: {curve['min_iou']:.4f} → {curve['max_iou']:.4f}")
        if curve["half_life"] is not None:
            print(f"    50% retention at: {curve['half_life']:.4f}")
        else:
            print(f"    50% retention: never reached (decoder highly robust)")

        # Print the curve
        vals = curve["values"]
        ious = curve["ious"]
        print(f"    ", end="")
        for v, iou in zip(vals, ious):
            bar = "█" * max(1, int(iou / max(base_iou, 0.01) * 20))
            print(f"\n    {v:>6}: {iou:.4f} {bar}", end="")
        print()

    # ── Discrete perturbations ──
    print(f"\n  ┌─ Discrete Perturbations{'─'*65}")
    for pname, label in [("Channel shuffle", "Channel shuffle"),
                          ("Spatial shuffle", "Spatial shuffle"),
                          ("Pos-ch only", "Pos-ch only")]:
        if label in scores:
            s = scores[label]
            print(f"  {label:<25s}  IoU={s['iou_mean']:.4f}  "
                  f"Δ={s['iou_drop']:+.4f}  retention={s['retention']:.1%}")

    # ── Interpretation ──
    print(f"\n  ┌─ Interpretation{'─'*74}")
    ch_shuffle = scores.get("Channel shuffle", {}).get("retention", 0)
    sp_shuffle = scores.get("Spatial shuffle", {}).get("retention", 0)
    scale_half = scores.get("scale", {}).get("half_life", 0) or 0
    noise_half = scores.get("noise", {}).get("half_life", 0) or 0
    ch_drop_half = scores.get("ch_drop", {}).get("half_life", 0) or 0

    if ch_shuffle > 0.8:
        print(f"  ✓ Channel order NOT important (retention={ch_shuffle:.0%})")
        print(f"    → Decoder uses channel content, not channel identity")
    else:
        print(f"  ✗ Channel order IS important (retention={ch_shuffle:.0%})")
        print(f"    → Decoder assigns specific roles to specific channels")

    if sp_shuffle > 0.5:
        print(f"  ✓ Spatial structure NOT critical (retention={sp_shuffle:.0%})")
        print(f"    → Prompt works as global conditioning, not spatial mask")
    else:
        print(f"  ✗ Spatial structure IS critical (retention={sp_shuffle:.0%})")
        print(f"    → Prompt functions as spatial guidance map")

    if scale_half < 0.3:
        print(f"  ✗ Scale-sensitive (half-life={scale_half:.2f})")
        print(f"    → Decoder depends on prompt magnitude")
    else:
        print(f"  ✓ Scale-robust (half-life={scale_half:.2f})")

    if noise_half > 0.5:
        print(f"  ✓ Noise-robust (half-life={noise_half:.2f}σ)")
    else:
        print(f"  ✗ Noise-sensitive (half-life={noise_half:.2f}σ)")

    print(f"\n{SEP}")


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def _save_sensitivity_charts(summary, scores, all_results, vis_dir,
                             scale_factors, noise_sigmas, channel_drop_pcts, topk_values):
    """Create sensitivity summary figure."""
    base_iou = scores.get("baseline_iou", 0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Chart 1: Scale perturbation
    ax = axes[0, 0]
    _plot_curve(ax, summary, "scale", scale_factors, base_iou,
                "Scale factor", "Scale Perturbation",
                x_is_factor=True)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="baseline")
    ax.legend(fontsize=7)

    # Chart 2: Noise perturbation
    ax = axes[0, 1]
    _plot_curve(ax, summary, "noise", noise_sigmas, base_iou,
                "Noise σ (×channel std)", "Noise Perturbation")

    # Chart 3: Channel dropout
    ax = axes[0, 2]
    _plot_curve(ax, summary, "ch_drop", channel_drop_pcts, base_iou,
                "Fraction of channels dropped", "Channel Dropout")

    # Chart 4: Top-K channels
    ax = axes[1, 0]
    _plot_curve(ax, summary, "topk", topk_values, base_iou,
                "Number of channels kept (by |r| with GT)", "Top-K Channels (oracle)",
                log_x=True)
    ax.axvline(x=256, color="gray", linestyle="--", alpha=0.5, label="all 256")

    # Chart 5: Discrete perturbation bar chart
    ax = axes[1, 1]
    discrete_items = []
    for label in ["Channel shuffle", "Spatial shuffle", "Pos-ch only"]:
        if label in scores:
            s = scores[label]
            discrete_items.append((label, s["iou_mean"], s["retention"]))
    if discrete_items:
        labels = [d[0] for d in discrete_items]
        ious = [d[1] for d in discrete_items]
        colors = ["#3498db", "#e74c3c", "#2ecc71"][:len(labels)]
        bars = ax.bar(range(len(labels)), ious, color=colors, edgecolor="white")
        ax.axhline(y=base_iou, color="gray", linestyle="--", alpha=0.7, label=f"baseline ({base_iou:.3f})")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title("Discrete Perturbations", fontsize=10, fontweight="bold")
        ax.set_ylabel("IoU")
        ax.legend(fontsize=7)
        # Add retention labels
        for bar, ret in zip(bars, [d[2] for d in discrete_items]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{ret:.0%}", ha="center", fontsize=9, fontweight="bold")

    # Chart 6: Per-tile sensitivity scatter (scale=0 → scale=1)
    ax = axes[1, 2]
    if "scale_0.0" in all_results and "baseline" in all_results:
        zero_ious = [d["iou"] for d in all_results["scale_0.0"]]
        base_ious = [d["iou"] for d in all_results["baseline"]]
        ax.scatter(base_ious, zero_ious, alpha=0.6, s=15, c="steelblue")
        max_val = max(max(base_ious), max(zero_ious)) * 1.1
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3)
        ax.set_xlabel("Baseline IoU"); ax.set_ylabel("Prompt=0 IoU")
        ax.set_title(f"Per-Tile: Prompt=0 vs Baseline (n={len(base_ious)})",
                    fontsize=10, fontweight="bold")
        below = sum(1 for b, z in zip(base_ious, zero_ious) if z < b)
        ax.text(0.05, 0.95, f"Zero < Baseline: {below}/{len(base_ious)}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Decoder Prompt Sensitivity Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(vis_dir / "sensitivity_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_curve(ax, summary, prefix, values, base_iou, xlabel, title,
                x_is_factor=False, log_x=False):
    """Plot a perturbation curve with error bars."""
    means, stds, x_vals = [], [], []
    for v in values:
        key = f"{prefix}_{v}"
        if key in summary and "iou" in summary[key]["metrics"]:
            m = summary[key]["metrics"]["iou"]
            means.append(m["mean"])
            stds.append(m["std"])
            x_vals.append(v)

    if not means:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10, fontweight="bold")
        return

    ax.plot(x_vals, means, "o-", color="steelblue", markersize=6, linewidth=2)
    ax.fill_between(x_vals,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15, color="steelblue")
    ax.axhline(y=base_iou, color="green", linestyle="--", alpha=0.6, label=f"baseline ({base_iou:.3f})")

    if log_x:
        ax.set_xscale("log", base=2)

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Pred IoU", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")

    # Annotate 50% retention
    half_iou = base_iou / 2
    for i in range(len(means) - 1):
        if (means[i] >= half_iou and means[i+1] <= half_iou) or \
           (means[i] <= half_iou and means[i+1] >= half_iou):
            frac = (half_iou - means[i]) / (means[i+1] - means[i] + 1e-8)
            half_x = x_vals[i] + frac * (x_vals[i+1] - x_vals[i])
            ax.axvline(x=half_x, color="red", linestyle=":", alpha=0.5)
            ax.axhline(y=half_iou, color="red", linestyle=":", alpha=0.5)
            ax.annotate(f"50% at {half_x:.2f}",
                       xy=(half_x, half_iou), fontsize=7, color="red",
                       xytext=(5, -15), textcoords="offset points")
            break


if __name__ == "__main__":
    main()
