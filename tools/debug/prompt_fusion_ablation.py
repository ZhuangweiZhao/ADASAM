"""
PromptFusion Causal Ablation — 因果验证 GeoPrior 的贡献
========================================================

四种 dense_prompt 来源, 同一 bypass_head 解码, 通过空间指标对比
判断 GeoPrior 在融合中到底是帮助还是污染。

Ablation variants:
  1. GeoPrior only:  geometric_prior → bypass_head
  2. SPG only:       spg_out.semantic_prior → bypass_head
  3. Current (fused): PromptFusion(GeoPrior, SPG) → bypass_head
  4. Gated sweep:    α·GeoPrior + (1-α)·SPG → bypass_head  (α ∈ [0,1])

用法:
    python tools/debug/prompt_fusion_ablation.py \
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
# Helpers (shared with prompt_spatial_traceback.py)
# ═══════════════════════════════════════════════════════════════════

def _to_activation(x: torch.Tensor, method: str = "l2") -> np.ndarray:
    if x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().float()
    if method == "l2":
        return x.pow(2).mean(dim=0).sqrt().numpy()
    elif method == "mean_abs":
        return x.abs().mean(dim=0).numpy()
    elif method == "max_abs":
        return x.abs().max(dim=0)[0].numpy()
    return x.abs().mean(dim=0).numpy()


def _best_channel_corr(x: torch.Tensor, gt: np.ndarray, target_size: tuple) -> dict:
    if x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().float()
    C, H, W = x.shape

    gt_t = torch.from_numpy(gt.astype(np.float32))
    gt_rsz = F.interpolate(gt_t.unsqueeze(0).unsqueeze(0), (H, W), mode="area")[0, 0]
    gt_f = gt_rsz.numpy().ravel()

    cors = []
    for c in range(C):
        ch_f = x[c].numpy().ravel()
        std_c, std_g = ch_f.std(), gt_f.std()
        if std_c > 1e-8 and std_g > 1e-8:
            cors.append(np.corrcoef(ch_f, gt_f)[0, 1])
        else:
            cors.append(0.0)
    cors = np.array(cors)

    best_idx = int(np.argmax(cors))
    n_pos = int((cors > 0.1).sum())
    n_neg = int((cors < -0.1).sum())

    return {
        "best_ch": best_idx,
        "best_r": float(cors[best_idx]),
        "best_act": x[best_idx].numpy(),
        "n_pos_ch": n_pos,
        "n_neg_ch": n_neg,
        "top5_mean_r": float(np.mean(np.sort(cors)[-5:])),
        "mean_abs_r": float(np.mean(np.abs(cors))),
    }


def _spatial_metrics(act: np.ndarray, gt: np.ndarray, label: str = "") -> dict:
    H_a, W_a = act.shape
    gt_t = torch.from_numpy(gt.astype(np.float32))
    gt_rsz = F.interpolate(gt_t.unsqueeze(0).unsqueeze(0), (H_a, W_a),
                           mode="area")[0, 0].numpy()
    gt_bin = gt_rsz > 0.5

    a_f = act.ravel()
    g_f = gt_rsz.ravel()

    a_s, g_s = a_f.std(), g_f.std()
    pearson = float(np.corrcoef(a_f, g_f)[0, 1]) if a_s > 1e-8 and g_s > 1e-8 else float("nan")

    peak_y, peak_x = np.unravel_index(np.argmax(act), act.shape)
    peak_inside = bool(gt_bin[peak_y, peak_x])

    inside_m = float(act[gt_bin].mean()) if gt_bin.any() else 0.0
    outside_m = float(act[~gt_bin].mean()) if (~gt_bin).any() else 0.0
    in_out = inside_m / max(outside_m, 1e-8)

    topk_ious = {}
    for k in [10, 20, 30]:
        thresh = np.percentile(act, 100 - k)
        top = act >= thresh
        inter = (top & gt_bin).sum()
        union = (top | gt_bin).sum()
        topk_ious[f"top{k}_IoU"] = float(inter / union) if union > 0 else 0.0

    return {
        f"{label}pearson_r": round(pearson, 6) if pearson == pearson else None,
        f"{label}peak_inside": peak_inside,
        f"{label}inside_outside": round(in_out, 4),
        **{f"{label}{k}": v for k, v in topk_ious.items()},
    }


def _compute_dense_prompt_variants(
    geometric_prior: torch.Tensor | None,
    semantic_prior: torch.Tensor,
    prompt_fusion: torch.nn.Module | None,
    gate_alphas: list[float],
) -> dict[str, torch.Tensor]:
    """Build all dense_prompt ablation variants.

    Returns dict: variant_name → dense_prompt [1, C, H, W].
    """
    results = {}

    # 1. GeoPrior only
    if geometric_prior is not None:
        results["geo_only"] = geometric_prior

    # 2. SPG only
    results["spg_only"] = semantic_prior

    # 3. Current (PromptFusion)
    if prompt_fusion is not None and geometric_prior is not None:
        dp_fused, _ = prompt_fusion(geometric_prior, semantic_prior)
        results["current"] = dp_fused

    # 4. Gated sweep: a * Geo + (1-a) * SPG
    if geometric_prior is not None:
        for a in gate_alphas:
            dp_gated = a * geometric_prior + (1 - a) * semantic_prior
            results[f"gate_a{a:.2f}"] = dp_gated

    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="PromptFusion Causal Ablation")
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

    # Check which head is used
    use_bypass = model.bypass_head is not None
    has_pf = model.prompt_fusion is not None
    has_gp = model.geometric_prior is not None
    print(f"[model] bypass_head={use_bypass}, prompt_fusion={has_pf}, geometric_prior={has_gp}")

    # Gate alpha values to sweep
    gate_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    data_root = Path(args.data_root)
    if not data_root.is_absolute(): data_root = _REPO_ROOT / data_root
    val_ds = ISAID5iDataset(root=str(data_root), fold=fold, split="val", mode=mode)
    visible_classes = val_ds.visible_classes()

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "fusion_ablation"
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
        if len(present) >= 2: candidates.append((idx, present))
    random.Random(args.seed + 9999).shuffle(candidates)
    selected = candidates[:args.num_tiles]
    print(f"[select] {len(candidates)} candidates, using {len(selected)}")

    # ── Accumulators: per-variant list of per-tile metrics ──
    all_metrics: dict[str, list[dict]] = defaultdict(list)  # variant_name → per-tile metrics
    per_tile_data: list[dict] = []  # raw data for paired comparison

    for tile_idx, present_classes in tqdm(selected, desc="ablation"):
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

        # ── Manual forward to get intermediate tensors ──
        support_memory = model.support_encoder(sup_feat, sup_mask)

        geometric_prior = None
        if model.geometric_prior is not None:
            geometric_prior = model.geometric_prior(query_emb, support_memory)

        dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
        spg_out = model.spg(query_emb, support_memory, dense_pe)
        semantic_prior = spg_out.semantic_prior  # [1, C, 64, 64]

        # ── Build all dense_prompt variants ──
        variants = _compute_dense_prompt_variants(
            geometric_prior, semantic_prior, model.prompt_fusion, gate_alphas
        )

        tile_record = {"tile_idx": tile_idx, "class_id": main_cls,
                       "class_name": ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}"),
                       "tile_id": sample.get("tile_id", str(tile_idx))}

        for v_name, dense_prompt in variants.items():
            # Decode through bypass head (or SAM decoder)
            if model.bypass_head is not None:
                low_res = model.bypass_head(dense_prompt)
            else:
                support_proto = model._compute_support_prototype(sup_feat, sup_mask)
                sparse_token = dense_prompt.mean(dim=(2, 3))
                low_res, _ = model.sam_decoder(query_emb, sparse_token, dense_prompt,
                                               support_prototype=support_proto)

            # Upsample to original size
            pred_logits = F.interpolate(low_res.float(), size=(H, W),
                                        mode="bilinear", align_corners=False)[0, 0]
            vals = pred_logits.cpu()
            if vals.min() >= 0 and vals.max() <= 1:
                pred_np = (vals > 0.5).numpy()
            else:
                pred_np = (vals.sigmoid() > 0.5).numpy()

            # ── Spatial metrics ──
            # 1. L2 norm activation metrics
            dp_act_l2 = _to_activation(dense_prompt, "l2")
            metrics_l2 = _spatial_metrics(dp_act_l2, gt_main, "")

            # 2. Best-channel metrics
            ch_info = _best_channel_corr(dense_prompt, gt_main, (64, 64))
            metrics_bestch = _spatial_metrics(ch_info["best_act"], gt_main, "bestch_")

            # 3. Prediction metrics
            metrics_pred = _spatial_metrics(pred_np.astype(np.float32), gt_main, "pred_")

            # IoU of prediction
            pred_bool = pred_np.astype(bool)
            inter = (pred_bool & gt_main).sum()
            union = (pred_bool | gt_main).sum()
            pred_iou = float(inter / union) if union > 0 else 0.0

            combined = {
                "variant": v_name,
                "l2_pearson_r": metrics_l2.get("pearson_r"),
                "l2_peak_inside": metrics_l2.get("peak_inside"),
                "l2_inside_outside": metrics_l2.get("inside_outside"),
                "l2_top20_IoU": metrics_l2.get("top20_IoU"),
                "bestch_pearson_r": metrics_bestch.get("bestch_pearson_r"),
                "bestch_peak_inside": metrics_bestch.get("bestch_peak_inside"),
                "bestch_inside_outside": metrics_bestch.get("bestch_inside_outside"),
                "bestch_top20_IoU": metrics_bestch.get("bestch_top20_IoU"),
                "pred_pearson_r": metrics_pred.get("pred_pearson_r"),
                "pred_top20_IoU": metrics_pred.get("pred_top20_IoU"),
                "pred_inside_outside": metrics_pred.get("pred_inside_outside"),
                "pred_IoU": pred_iou,
                "n_pos_ch": ch_info["n_pos_ch"],
                "n_neg_ch": ch_info["n_neg_ch"],
                "top5_mean_r": ch_info["top5_mean_r"],
                "mean_abs_r": ch_info["mean_abs_r"],
            }
            all_metrics[v_name].append(combined)

            # Record key metrics for paired comparison
            if v_name in ("spg_only", "current", "geo_only"):
                tile_record[f"{v_name}_pred_IoU"] = pred_iou
                tile_record[f"{v_name}_l2_r"] = metrics_l2.get("pearson_r")
                tile_record[f"{v_name}_bestch_r"] = metrics_bestch.get("bestch_pearson_r")

        per_tile_data.append(tile_record)

        # ── Visualization (first 4 tiles) ──
        if args.save_vis and len(per_tile_data) <= 4:
            _save_tile_figure(img_np, gt_main, H, W, variants, gate_alphas,
                             model.bypass_head is not None, model, query_emb, sup_feat, sup_mask,
                             sample.get("tile_id", str(tile_idx)), main_cls, vis_dir)

    # ── Aggregate ──
    summary = _aggregate_variants(all_metrics)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Paired comparison (per-tile delta) ──
    paired_path = out_dir / "per_tile.json"
    with open(paired_path, "w") as f:
        json.dump(per_tile_data, f, indent=2, ensure_ascii=False)

    # ── Print ──
    _print_ablation_table(summary, all_metrics, per_tile_data)
    print(f"\n[Summary] {summary_path}")
    print(f"[Per-tile] {paired_path}")

    if args.save_vis:
        _save_summary_chart(summary, all_metrics, per_tile_data, gate_alphas, vis_dir)
        print(f"[vis] {vis_dir / 'summary_chart.png'}")

    print("[Done]")


# ═══════════════════════════════════════════════════════════════════
# Aggregation & Printing
# ═══════════════════════════════════════════════════════════════════

def _agg_vals(vals: list) -> dict | None:
    clean = [v for v in vals if v is not None and v == v]
    if not clean:
        return None
    return {
        "mean": round(float(np.mean(clean)), 4),
        "median": round(float(np.median(clean)), 4),
        "std": round(float(np.std(clean)), 4),
        "n": len(clean),
    }


def _aggregate_variants(all_metrics: dict) -> dict:
    summary = {}
    for v_name, items in all_metrics.items():
        if not items: continue
        sm = {}
        for key in items[0]:
            if key == "variant": continue
            agg = _agg_vals([d[key] for d in items])
            if agg: sm[key] = agg
        summary[v_name] = {"n_episodes": len(items), "metrics": sm}
    return summary


def _print_ablation_table(summary: dict, all_metrics: dict, per_tile_data: list):
    """Print ablation comparison table with causal interpretation."""
    SEP = "=" * 110
    print(f"\n{SEP}")
    print("  PromptFusion Causal Ablation — what happens when we bypass the fusion?")
    print(f"{SEP}")

    # ═══ Table 1: Core metrics for the 3 key variants ═══
    key_variants = ["geo_only", "spg_only", "current"]
    key_variant_labels = {
        "geo_only": "GeoPrior only",
        "spg_only": "SPG only",
        "current": "Current (Geo+SPG fused)",
    }

    key_metrics = [
        ("l2_pearson_r", "DP L2 r"),
        ("bestch_pearson_r", "DP best-ch r"),
        ("l2_top20_IoU", "DP Top20 IoU"),
        ("l2_inside_outside", "DP In/Out"),
        ("l2_peak_inside", "DP Peak GT%"),
        ("pred_IoU", "Pred IoU ★"),
        ("pred_pearson_r", "Pred Pearson"),
        ("pred_top20_IoU", "Pred Top20 IoU"),
        ("pred_inside_outside", "Pred In/Out"),
        ("mean_abs_r", "DP mean|r|"),
        ("top5_mean_r", "DP Top5 r"),
        ("n_pos_ch", "DP n_pos_ch"),
        ("n_neg_ch", "DP n_neg_ch"),
    ]

    print(f"\n  ┌─ Core Causal Comparison{'─'*88}")
    for metric_key, metric_label in key_metrics:
        print(f"\n  [{metric_label}]")
        header = f"  {'Variant':<28s}"
        for v in key_variants:
            if v in summary:
                header += f" {key_variant_labels.get(v, v):>14s}"
        print(header)
        print(f"  {'-'*70}")

        vals_row = f"  {'':28s}"
        for v in key_variants:
            if v not in summary: continue
            sm = summary[v]["metrics"]
            if metric_key in sm and sm[metric_key] is not None:
                vals_row += f" {sm[metric_key]['mean']:>14.4f}"
            else:
                vals_row += f" {'N/A':>14s}"
        print(vals_row)

    # ═══ Table 2: Gate sweep — best α vs current ═══
    gate_variants = [f"gate_a{a:.2f}" for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    gate_labels = [f"α=0.0 (pure SPG)", "α=0.2", "α=0.4", "α=0.6", "α=0.8", "α=1.0 (pure Geo)"]

    print(f"\n  ┌─ Gate Sweep: α·Geo + (1-α)·SPG{'─'*72}")
    for metric_key in ["pred_IoU", "l2_pearson_r", "bestch_pearson_r"]:
        metric_label = {"pred_IoU": "Pred IoU ★", "l2_pearson_r": "DP L2 r",
                       "bestch_pearson_r": "DP best-ch r"}.get(metric_key, metric_key)
        print(f"\n  [{metric_label}]")
        line = "  "
        for i, gv in enumerate(gate_variants):
            if gv in summary and metric_key in summary[gv]["metrics"]:
                line += f" α={[0.0,0.2,0.4,0.6,0.8,1.0][i]:.1f}: {summary[gv]['metrics'][metric_key]['mean']:.4f}"
                if i < len(gate_variants) - 1: line += " |"
        print(line)

    # ═══ Table 3: Paired comparison — per-tile winners ═══
    if per_tile_data:
        print(f"\n  ┌─ Per-Tile Winner Analysis{'─'*86}")
        spg_wins = 0
        geo_wins = 0
        current_wins = 0
        ties = 0
        spg_deltas = []
        for rec in per_tile_data:
            spg_iou = rec.get("spg_only_pred_IoU", 0)
            cur_iou = rec.get("current_pred_IoU", 0)
            geo_iou = rec.get("geo_only_pred_IoU", 0)
            if spg_iou is None or cur_iou is None: continue
            spg_deltas.append(spg_iou - cur_iou)
            best = max(spg_iou, cur_iou, geo_iou or 0)
            if best == spg_iou and spg_iou > cur_iou + 0.005:
                spg_wins += 1
            elif best == geo_iou and geo_iou and geo_iou > cur_iou + 0.005:
                geo_wins += 1
            elif best == cur_iou and cur_iou > max(spg_iou, geo_iou or 0) + 0.005:
                current_wins += 1
            else:
                ties += 1

        print(f"  SPG only  > current: {spg_wins}/{len(spg_deltas)} tiles "
              f"({100*spg_wins/max(len(spg_deltas),1):.0f}%)")
        print(f"  Geo only  > current: {geo_wins}/{len(spg_deltas)} tiles "
              f"({100*geo_wins/max(len(spg_deltas),1):.0f}%)")
        print(f"  Current   > both:   {current_wins}/{len(spg_deltas)} tiles "
              f"({100*current_wins/max(len(spg_deltas),1):.0f}%)")
        print(f"  Ties (±0.005):        {ties}/{len(spg_deltas)} tiles")
        if spg_deltas:
            mean_delta = np.mean(spg_deltas)
            print(f"  Mean IoU delta (SPG - Current): {mean_delta:+.4f} "
                  f"{'← SPG BETTER' if mean_delta > 0 else '← Current BETTER'}")

    # ═══ Causal conclusion ═══
    print(f"\n  ┌─ Causal Interpretation{'─'*88}")
    if "spg_only" in summary and "current" in summary:
        spg_iou = summary["spg_only"]["metrics"].get("pred_IoU", {})
        cur_iou = summary["current"]["metrics"].get("pred_IoU", {})
        geo_iou = summary.get("geo_only", {}).get("metrics", {}).get("pred_IoU", {})

        if spg_iou and cur_iou:
            delta = spg_iou["mean"] - cur_iou["mean"]
            if delta > 0.02:
                print(f"  ✓ SPG-only outperforms fused by {delta:+.4f} IoU")
                print(f"    → GeoPrior is TOXIC — it hurts more than it helps")
                print(f"    → Recommendation: drop GeoPrior or gate it strongly toward SPG")
            elif delta > 0.0:
                print(f"  ~ SPG-only slightly better ({delta:+.4f}) — mild GeoPrior toxicity")
            elif delta > -0.01:
                print(f"  ~ SPG-only ≈ fused ({delta:+.4f}) — GeoPrior is neutral")
                print(f"    → PromptFusion is learning to ignore GeoPrior")
            else:
                print(f"  ✗ Fused better than SPG-only ({delta:+.4f})")
                print(f"    → GeoPrior contributes useful signal despite poor Pearson r")

        if geo_iou and spg_iou:
            geo_vs_spg = geo_iou["mean"] - spg_iou["mean"]
            print(f"  Geo only vs SPG only: {geo_vs_spg:+.4f} IoU")

    print(f"\n{SEP}")


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def _decode_pred(dense_prompt, model, query_emb, sup_feat, sup_mask, H, W):
    """Decode dense_prompt through bypass_head or SAM decoder."""
    if model.bypass_head is not None:
        low_res = model.bypass_head(dense_prompt)
    else:
        support_proto = model._compute_support_prototype(sup_feat, sup_mask)
        sparse_token = dense_prompt.mean(dim=(2, 3))
        low_res, _ = model.sam_decoder(query_emb, sparse_token, dense_prompt,
                                       support_prototype=support_proto)
    pred_logits = F.interpolate(low_res.float(), size=(H, W),
                                mode="bilinear", align_corners=False)[0, 0]
    vals = pred_logits.cpu()
    if vals.min() >= 0 and vals.max() <= 1:
        return (vals > 0.5).numpy()
    return (vals.sigmoid() > 0.5).numpy()


def _save_tile_figure(img_np, gt_main, H, W, variants, gate_alphas,
                     use_bypass, model, query_emb, sup_feat, sup_mask,
                     tile_id, main_cls, vis_dir):
    """Save one tile's ablation comparison figure."""
    cls_name = ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}")
    n_cols = 7
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(21, 7))
    axes = axes.flatten()

    # Row 0: Query, GT, GeoOnly, SPGOnly, Current, Best gate, Gate sweep chart
    axes[0].imshow(img_np)
    axes[0].set_title("Query", fontsize=8); axes[0].axis("off")

    axes[1].imshow(gt_main, cmap="gray")
    axes[1].set_title(f"GT ({cls_name})", fontsize=8); axes[1].axis("off")

    # Determine key gate alphas for visualization
    key_gates = [0.0, 0.3, 0.5, 0.7, 1.0]
    viz_order = ["geo_only", "spg_only", "current"] + [f"gate_a{a:.2f}" for a in key_gates]
    viz_labels = ["GeoOnly", "SPGOnly", "Fused"] + [f"α={a:.1f}" for a in key_gates]

    for i, (vn, vl) in enumerate(zip(["geo_only", "spg_only", "current"], ["GeoOnly", "SPGOnly", "Fused"])):
        if vn not in variants: continue
        dp = variants[vn]
        pred = _decode_pred(dp, model, query_emb, sup_feat, sup_mask, H, W)
        inter = (pred & gt_main).sum()
        union = (pred | gt_main).sum()
        iou = inter / union if union > 0 else 0
        ax = axes[2 + i]
        ax.imshow(img_np)
        if pred.sum() > 0:
            overlay = np.zeros((H, W, 4), dtype=np.uint8)
            overlay[pred] = (255, 100, 100, 180)
            ax.imshow(overlay)
        ax.set_title(f"{vl}\nIoU={iou:.3f}", fontsize=8); ax.axis("off")

    # Best gate (α that maximizes IoU on this tile)
    best_alpha = None
    best_iou = -1
    for a in gate_alphas:
        vn = f"gate_a{a:.2f}"
        if vn not in variants: continue
        pred = _decode_pred(variants[vn], model, query_emb, sup_feat, sup_mask, H, W)
        inter = (pred & gt_main).sum()
        union = (pred | gt_main).sum()
        iou = inter / union if union > 0 else 0
        if iou > best_iou:
            best_iou = iou
            best_alpha = a

    if best_alpha is not None and f"gate_a{best_alpha:.2f}" in variants:
        dp = variants[f"gate_a{best_alpha:.2f}"]
        pred = _decode_pred(dp, model, query_emb, sup_feat, sup_mask, H, W)
        ax = axes[5]
        ax.imshow(img_np)
        if pred.sum() > 0:
            overlay = np.zeros((H, W, 4), dtype=np.uint8)
            overlay[pred] = (100, 255, 100, 180)
            ax.imshow(overlay)
        ax.set_title(f"Best α={best_alpha:.1f}\nIoU={best_iou:.3f}", fontsize=8); ax.axis("off")

    # Gate sweep curve
    ax = axes[6]
    alpha_vals = []
    iou_vals = []
    for a in gate_alphas:
        vn = f"gate_a{a:.2f}"
        if vn not in variants: continue
        pred = _decode_pred(variants[vn], model, query_emb, sup_feat, sup_mask, H, W)
        inter = (pred & gt_main).sum()
        union = (pred | gt_main).sum()
        alpha_vals.append(a)
        iou_vals.append(inter / union if union > 0 else 0)
    ax.plot(alpha_vals, iou_vals, "o-", color="steelblue", markersize=4)
    if "geo_only" in variants:
        geo_pred = _decode_pred(variants["geo_only"], model, query_emb, sup_feat, sup_mask, H, W)
        inter = (geo_pred & gt_main).sum()
        union = (geo_pred | gt_main).sum()
        ax.axhline(y=inter/union if union>0 else 0, color="red", linestyle="--", alpha=0.6, label="GeoOnly")
    if "spg_only" in variants:
        spg_pred = _decode_pred(variants["spg_only"], model, query_emb, sup_feat, sup_mask, H, W)
        inter = (spg_pred & gt_main).sum()
        union = (spg_pred | gt_main).sum()
        ax.axhline(y=inter/union if union>0 else 0, color="green", linestyle="--", alpha=0.6, label="SPGOnly")
    if "current" in variants:
        cur_pred = _decode_pred(variants["current"], model, query_emb, sup_feat, sup_mask, H, W)
        inter = (cur_pred & gt_main).sum()
        union = (cur_pred | gt_main).sum()
        ax.axhline(y=inter/union if union>0 else 0, color="orange", linestyle="--", alpha=0.6, label="Current")
    ax.set_xlabel("α (Geo weight)"); ax.set_ylabel("IoU")
    ax.set_title("Gate sweep", fontsize=8); ax.legend(fontsize=6)

    # Row 1: Dense prompt L2 norm visualizations
    axes[7].imshow(img_np)
    axes[7].set_title("Query", fontsize=8); axes[7].axis("off")

    axes[8].imshow(gt_main, cmap="gray")
    axes[8].set_title(f"GT ({cls_name})", fontsize=8); axes[8].axis("off")

    for i, (vn, vl) in enumerate(zip(["geo_only", "spg_only", "current"], ["Geo DP L2", "SPG DP L2", "Fused DP L2"])):
        if vn not in variants: continue
        dp = variants[vn]
        act = _to_activation(dp, "l2")
        im = axes[9 + i].imshow(act, cmap="RdBu_r")
        axes[9 + i].set_title(vl, fontsize=8); axes[9 + i].axis("off")
        plt.colorbar(im, ax=axes[9 + i], fraction=0.046)

    # Best gate DP L2
    if best_alpha is not None and f"gate_a{best_alpha:.2f}" in variants:
        act = _to_activation(variants[f"gate_a{best_alpha:.2f}"], "l2")
        im = axes[12].imshow(act, cmap="RdBu_r")
        axes[12].set_title(f"Best α={best_alpha:.1f} DP L2", fontsize=8); axes[12].axis("off")
        plt.colorbar(im, ax=axes[12], fraction=0.046)

    # Remaining axis: use for something useful or hide
    axes[13].axis("off")

    fig.suptitle(f"PromptFusion Ablation: {tile_id} | support={cls_name}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(vis_dir / f"{tile_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_summary_chart(summary, all_metrics, per_tile_data, gate_alphas, vis_dir):
    """Create summary charts."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Chart 1: Bar chart — pred IoU for key variants
    ax = axes[0, 0]
    key_variants = ["geo_only", "spg_only", "current"]
    labels = ["GeoOnly", "SPGOnly", "Fused"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    means, errs = [], []
    for v in key_variants:
        if v in summary and "pred_IoU" in summary[v]["metrics"]:
            m = summary[v]["metrics"]["pred_IoU"]
            means.append(m["mean"])
            errs.append(m["std"])
        else:
            means.append(0)
            errs.append(0)
    bars = ax.bar(range(len(means)), means, color=colors, edgecolor="white")
    ax.errorbar(range(len(means)), means, yerr=errs, fmt="none", ecolor="black", capsize=5)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Prediction IoU by Dense Prompt Source", fontsize=11, fontweight="bold")
    ax.set_ylabel("IoU")

    # Add value labels
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    # Chart 2: Gate sweep curve (aggregate)
    ax = axes[0, 1]
    gate_variants = [f"gate_a{a:.2f}" for a in gate_alphas]
    alpha_vals = gate_alphas
    iou_means, iou_stds = [], []
    for gv in gate_variants:
        if gv in summary and "pred_IoU" in summary[gv]["metrics"]:
            m = summary[gv]["metrics"]["pred_IoU"]
            iou_means.append(m["mean"])
            iou_stds.append(m["std"])
        else:
            iou_means.append(None)
            iou_stds.append(None)

    valid = [(a, m, s) for a, m, s in zip(alpha_vals, iou_means, iou_stds) if m is not None]
    if valid:
        av, mv, sv = zip(*valid)
        ax.plot(av, mv, "o-", color="steelblue", markersize=6, linewidth=2)
        ax.fill_between(av, [m - s for m, s in zip(mv, sv)],
                       [m + s for m, s in zip(mv, sv)], alpha=0.15, color="steelblue")

        # Horizontal lines for key variants
        for v, label, color in zip(key_variants, labels, colors):
            if v in summary and "pred_IoU" in summary[v]["metrics"]:
                ax.axhline(y=summary[v]["metrics"]["pred_IoU"]["mean"],
                          color=color, linestyle="--", alpha=0.7, label=label)
        ax.legend(fontsize=8)
    ax.set_xlabel("α (GeoPrior weight in gate)"); ax.set_ylabel("Pred IoU")
    ax.set_title("Gate Sweep: α·Geo + (1-α)·SPG → Pred IoU", fontsize=11, fontweight="bold")

    # Chart 3: Per-tile paired comparison (SPG vs Current)
    ax = axes[0, 2]
    spg_ious = [d.get("spg_only_pred_IoU", 0) for d in per_tile_data if d.get("spg_only_pred_IoU") is not None and d.get("current_pred_IoU") is not None]
    cur_ious = [d.get("current_pred_IoU", 0) for d in per_tile_data if d.get("spg_only_pred_IoU") is not None and d.get("current_pred_IoU") is not None]
    if spg_ious and cur_ious:
        max_val = max(max(spg_ious), max(cur_ious)) * 1.1
        ax.scatter(cur_ious, spg_ious, alpha=0.6, s=20, c="steelblue")
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3)
        ax.set_xlabel("Current (Fused) IoU"); ax.set_ylabel("SPG Only IoU")
        ax.set_title(f"Per-Tile: SPG vs Current (n={len(spg_ious)})", fontsize=11, fontweight="bold")

        # Count above/below diagonal
        above = sum(1 for s, c in zip(spg_ious, cur_ious) if s > c)
        below = sum(1 for s, c in zip(spg_ious, cur_ious) if s < c)
        ax.text(0.05, 0.95, f"SPG wins: {above}\nCurrent wins: {below}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Chart 4: DP L2 Pearson r comparison
    ax = axes[1, 0]
    for v, label, color in zip(key_variants, labels, colors):
        if v in summary and "l2_pearson_r" in summary[v]["metrics"]:
            m = summary[v]["metrics"]["l2_pearson_r"]
            ax.bar(label, m["mean"], color=color, edgecolor="white",
                   yerr=m["std"], capsize=5)
    ax.set_title("Dense Prompt L2 Pearson r vs GT", fontsize=11, fontweight="bold")
    ax.set_ylabel("Pearson r")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Chart 5: DP best-channel Pearson r
    ax = axes[1, 1]
    for v, label, color in zip(key_variants, labels, colors):
        if v in summary and "bestch_pearson_r" in summary[v]["metrics"]:
            m = summary[v]["metrics"]["bestch_pearson_r"]
            ax.bar(label, m["mean"], color=color, edgecolor="white",
                   yerr=m["std"], capsize=5)
    ax.set_title("Dense Prompt Best-Ch Pearson r vs GT", fontsize=11, fontweight="bold")
    ax.set_ylabel("Pearson r")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Chart 6: DP channel structure
    ax = axes[1, 2]
    x_pos = []
    heights = []
    colors_ch = []
    labels_ch = []
    for i, (v, label, color) in enumerate(zip(key_variants, labels, colors)):
        if v in summary and "n_pos_ch" in summary[v]["metrics"]:
            x_pos.extend([i * 3, i * 3 + 1])
            heights.extend([summary[v]["metrics"]["n_pos_ch"]["mean"],
                          summary[v]["metrics"]["n_neg_ch"]["mean"]])
            colors_ch.extend(["#2ecc71", "#e74c3c"])
            labels_ch.extend([f"{label}\npos", f"{label}\nneg"])
    ax.bar(range(len(heights)), heights, color=colors_ch, edgecolor="white")
    ax.set_xticks(range(len(heights)))
    ax.set_xticklabels(labels_ch, fontsize=7)
    ax.set_title("DP Channel Direction (n_pos / n_neg)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count (out of 256)")

    plt.suptitle("PromptFusion Causal Ablation Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(vis_dir / "summary_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
