"""
Prompt Spatial Traceback — 逐级定位空间漂移根源
================================================

在信息流的每一级 (GeoPrior → SPG probes → Dense Prompt → Prediction)
计算空间对齐指标, 定位空间信息丢失发生在哪一级。

指标 (每级):
  - Pearson r (vs GT distance transform)
  - Peak inside GT rate
  - Top-20% IoU
  - Inside/Outside activation ratio
  - Best-channel correlation (分布式编码检测)

用法:
    python tools/debug/prompt_spatial_traceback.py \
        --checkpoint <ckpt> --mode novel --k-shot 5 --num-tiles 20 --save-vis
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
from scipy import ndimage
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
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _to_activation(x: torch.Tensor, method: str = "l2") -> np.ndarray:
    """Reduce [C, H, W] or [1, C, H, W] to single-channel activation [H, W]."""
    if x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().float()  # [C, H, W]
    if method == "l2":
        return x.pow(2).mean(dim=0).sqrt().numpy()
    elif method == "mean_abs":
        return x.abs().mean(dim=0).numpy()
    elif method == "max_abs":
        return x.abs().max(dim=0)[0].numpy()
    return x.abs().mean(dim=0).numpy()


def _best_channel_corr(x: torch.Tensor, gt: np.ndarray, target_size: tuple) -> dict:
    """计算每个通道与 GT 的相关性, 返回最佳通道的统计。

    :param x: [C, H, W] tensor.
    :param gt: [H_gt, W_gt] bool GT mask.
    :param target_size: (h, w) of x's spatial dims.
    :return: {best_ch, best_r, best_act, n_pos, n_neg, top5_r}
    """
    if x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().float()
    C, H, W = x.shape

    # GT resized
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
    """单张激活图的空间对齐指标。"""
    H_a, W_a = act.shape
    gt_t = torch.from_numpy(gt.astype(np.float32))
    gt_rsz = F.interpolate(gt_t.unsqueeze(0).unsqueeze(0), (H_a, W_a),
                           mode="area")[0, 0].numpy()
    gt_bin = gt_rsz > 0.5

    # Flatten
    a_f = act.ravel()
    g_f = gt_rsz.ravel()

    # Pearson
    a_s, g_s = a_f.std(), g_f.std()
    pearson = float(np.corrcoef(a_f, g_f)[0, 1]) if a_s > 1e-8 and g_s > 1e-8 else float("nan")

    # Peak inside
    peak_y, peak_x = np.unravel_index(np.argmax(act), act.shape)
    peak_inside = bool(gt_bin[peak_y, peak_x])

    # Inside/Outside
    inside_m = float(act[gt_bin].mean()) if gt_bin.any() else 0.0
    outside_m = float(act[~gt_bin].mean()) if (~gt_bin).any() else 0.0
    in_out = inside_m / max(outside_m, 1e-8)

    # Top-K IoU
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


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Prompt Spatial Traceback")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/iSAID-5i")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--mode", default="novel")
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tiles", type=int, default=20)
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

    data_root = Path(args.data_root)
    if not data_root.is_absolute(): data_root = _REPO_ROOT / data_root
    val_ds = ISAID5iDataset(root=str(data_root), fold=fold, split="val", mode=mode)
    visible_classes = val_ds.visible_classes()

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "spatial_traceback"
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

    # ── Accumulators ──
    stages = ["geometric_prior", "spg_per_probe_best", "spg_semantic_prior",
              "dense_prompt", "pred_mask"]
    all_metrics: dict[str, list[dict]] = {s: [] for s in stages}

    for tile_idx, present_classes in tqdm(selected, desc="traceback"):
        sample = val_ds[tile_idx]
        H, W = sample["image"].shape[1], sample["image"].shape[2]
        img_np = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        xx, meta = preprocess_image(sample["image"])
        query_emb = backbone(xx.unsqueeze(0).to(device))["image_embedding"]
        if cat_adapter is not None: query_emb = cat_adapter(query_emb)

        # Pick class with largest GT area
        main_cls = max(present_classes, key=lambda c: val_ds.get_class_mask(tile_idx, c).sum())
        sup_data = support_cache.get(main_cls)
        if sup_data is None: continue
        sup_feat, sup_mask = sup_data
        gt_main = val_ds.get_class_mask(tile_idx, main_cls).numpy().astype(bool)

        # ── Manual forward with intermediate capture ──
        support_memory = model.support_encoder(sup_feat, sup_mask)

        # GeoPrior
        geometric_prior = None
        if model.geometric_prior is not None:
            geometric_prior = model.geometric_prior(query_emb, support_memory)

        # SPG (capture per-probe masks)
        dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
        spg_out = model.spg(query_emb, support_memory, dense_pe,
                            return_per_probe_masks=True)

        # PromptFusion
        if model.prompt_fusion is not None and geometric_prior is not None:
            dense_prompt, sparse_token = model.prompt_fusion(
                geometric_prior, spg_out.semantic_prior)
        else:
            dense_prompt = spg_out.semantic_prior

        # Prediction
        if model.bypass_head is not None:
            low_res = model.bypass_head(dense_prompt)
        else:
            support_proto = model._compute_support_prototype(sup_feat, sup_mask)
            low_res, _ = model.sam_decoder(query_emb, sparse_token, dense_prompt,
                                           support_prototype=support_proto)
        pred_logits = F.interpolate(low_res.float(), size=(H, W),
                                    mode="bilinear", align_corners=False)[0, 0]
        vals = pred_logits.cpu()
        if vals.min() >= 0 and vals.max() <= 1:
            pred_np = (vals > 0.5).numpy()
        else:
            pred_np = (vals.sigmoid() > 0.5).numpy()

        # ── E1: GeoPrior spatial metrics ──
        if geometric_prior is not None:
            gp_act_l2 = _to_activation(geometric_prior, "l2")  # [64, 64]
            gp_ch = _best_channel_corr(geometric_prior, gt_main, (64, 64))
            m = _spatial_metrics(gp_act_l2, gt_main, "gp_l2_")
            m.update(_spatial_metrics(gp_ch["best_act"], gt_main, "gp_bestch_"))
            m["gp_n_pos_ch"] = gp_ch["n_pos_ch"]
            m["gp_n_neg_ch"] = gp_ch["n_neg_ch"]
            m["gp_top5_mean_r"] = gp_ch["top5_mean_r"]
            m["gp_mean_abs_r"] = gp_ch["mean_abs_r"]
            m["tile_idx"] = tile_idx; m["class_id"] = main_cls
            all_metrics["geometric_prior"].append(m)
        else:
            gp_act_l2 = np.zeros((64, 64))
            gp_ch = None

        # ── E2: SPG per-probe (best probe) ──
        if spg_out.probe_masks is not None:
            probe_masks = spg_out.probe_masks.sigmoid().cpu()  # [N, 64, 64]
            N = probe_masks.shape[0]
            # Find best probe (highest correlation with GT)
            gt_t = torch.from_numpy(gt_main.astype(np.float32))
            gt64 = F.interpolate(gt_t.unsqueeze(0).unsqueeze(0), (64, 64), mode="area")[0, 0]
            gt64_f = gt64.numpy().ravel()
            probe_cors = []
            for p in range(N):
                pf = probe_masks[p].numpy().ravel()
                s_p, s_g = pf.std(), gt64_f.std()
                probe_cors.append(float(np.corrcoef(pf, gt64_f)[0, 1]) if s_p > 1e-8 and s_g > 1e-8 else 0.0)
            best_p = int(np.argmax(probe_cors))
            probe_best_act = probe_masks[best_p].numpy()
            m = _spatial_metrics(probe_best_act, gt_main, "probe_")
            m["probe_idx"] = best_p
            m["probe_r"] = round(probe_cors[best_p], 4)
            m["probe_mean_r"] = round(float(np.mean(probe_cors)), 4)
            m["probe_max_r"] = round(float(np.max(probe_cors)), 4)
            m["probe_min_r"] = round(float(np.min(probe_cors)), 4)
            m["tile_idx"] = tile_idx; m["class_id"] = main_cls
            all_metrics["spg_per_probe_best"].append(m)
        else:
            probe_best_act = np.zeros((64, 64))
            probe_cors = [0.0]

        # ── E3: SPG semantic_prior ──
        sp_sem = spg_out.semantic_prior
        sp_act_l2 = _to_activation(sp_sem, "l2")
        sp_ch = _best_channel_corr(sp_sem, gt_main, (64, 64))
        m = _spatial_metrics(sp_act_l2, gt_main, "sp_l2_")
        m.update(_spatial_metrics(sp_ch["best_act"], gt_main, "sp_bestch_"))
        m["sp_n_pos_ch"] = sp_ch["n_pos_ch"]
        m["sp_n_neg_ch"] = sp_ch["n_neg_ch"]
        m["sp_top5_mean_r"] = sp_ch["top5_mean_r"]
        m["sp_mean_abs_r"] = sp_ch["mean_abs_r"]
        m["tile_idx"] = tile_idx; m["class_id"] = main_cls
        all_metrics["spg_semantic_prior"].append(m)

        # ── E4: Dense Prompt ──
        dp_act_l2 = _to_activation(dense_prompt, "l2")
        dp_ch = _best_channel_corr(dense_prompt, gt_main, (64, 64))
        m = _spatial_metrics(dp_act_l2, gt_main, "dp_l2_")
        m.update(_spatial_metrics(dp_ch["best_act"], gt_main, "dp_bestch_"))
        m["dp_n_pos_ch"] = dp_ch["n_pos_ch"]
        m["dp_n_neg_ch"] = dp_ch["n_neg_ch"]
        m["dp_top5_mean_r"] = dp_ch["top5_mean_r"]
        m["dp_mean_abs_r"] = dp_ch["mean_abs_r"]
        m["tile_idx"] = tile_idx; m["class_id"] = main_cls
        all_metrics["dense_prompt"].append(m)

        # ── E5: Prediction ──
        m = _spatial_metrics(pred_np.astype(np.float32), gt_main, "pred_")
        m["tile_idx"] = tile_idx; m["class_id"] = main_cls
        all_metrics["pred_mask"].append(m)

        # ── Visualization (first 5 tiles) ──
        if args.save_vis and len(all_metrics["dense_prompt"]) <= 5:
            _save_tile_figure(img_np, gt_main, H, W,
                              gp_act_l2, gp_ch,
                              probe_best_act, probe_cors,
                              sp_act_l2, sp_ch,
                              dp_act_l2, dp_ch,
                              pred_np,
                              tile_idx, main_cls, vis_dir, sample.get("tile_id", str(tile_idx)))

    # ── Aggregate ──
    summary = _aggregate(all_metrics)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Print ──
    _print_trace(summary)
    print(f"\n[Summary] {summary_path}")

    if args.save_vis:
        _save_summary_chart(summary, all_metrics, vis_dir)
        print(f"[vis] {vis_dir / 'summary_chart.png'}")

    print("[Done]")


# ═══════════════════════════════════════════════════════════════════
# Aggregation & Printing
# ═══════════════════════════════════════════════════════════════════

def _agg_key(items: list[dict], key: str) -> dict:
    vals = [d[key] for d in items if d.get(key) is not None and d[key] == d[key]]
    if not vals:
        return {"mean": None, "n": 0}
    return {
        "mean": round(float(np.mean(vals)), 4),
        "median": round(float(np.median(vals)), 4),
        "std": round(float(np.std(vals)), 4),
        "n": len(vals),
    }


def _aggregate(all_metrics: dict) -> dict:
    summary = {}
    for stage, items in all_metrics.items():
        if not items: continue
        sm = {}
        for key in items[0]:
            if key in ("tile_idx", "class_id", "probe_idx"):
                continue
            sm[key] = _agg_key(items, key)
        summary[stage] = {"n_episodes": len(items), "metrics": sm}
    return summary


def _print_trace(summary: dict):
    """Print spatial traceback table."""
    SEP = "=" * 90
    print(f"\n{SEP}")
    print("  Prompt Spatial Traceback — where does spatial alignment break?")
    print(f"{SEP}")

    # Key metrics across stages
    stages_order = ["geometric_prior", "spg_per_probe_best", "spg_semantic_prior",
                    "dense_prompt", "pred_mask"]
    stage_labels = {
        "geometric_prior": "GeoPrior",
        "spg_per_probe_best": "SPG best probe",
        "spg_semantic_prior": "SPG semantic_prior",
        "dense_prompt": "Dense Prompt",
        "pred_mask": "Prediction",
    }

    key_metrics = [
        ("pearson_r", "Pearson r (L2)"),
        ("bestch_pearson_r", "Best-ch r"),
        ("peak_inside", "Peak inside GT"),
        ("inside_outside", "Inside/Outside"),
        ("top20_IoU", "Top-20% IoU"),
    ]

    for metric_key, metric_label in key_metrics:
        print(f"\n  [{metric_label}]")
        print(f"  {'Stage':<25s} {'Mean':>8s}  {'Median':>8s}  {'n':>5s}")
        print(f"  {'-'*48}")
        for stage in stages_order:
            if stage not in summary: continue
            sm = summary[stage]["metrics"]
            # Map metric names — try l2_ prefix, bestch_ prefix, probe_ prefix etc
            candidates = [k for k in sm if metric_key in k or k.endswith(metric_key)]
            for ck in candidates[:1]:
                v = sm[ck]
                label = stage_labels.get(stage, stage)
                print(f"  {label:<25s} {str(v.get('mean', 'N/A')):>8s}  "
                      f"{str(v.get('median', 'N/A')):>8s}  {v.get('n', 0):>5d}")

    # Channel statistics
    print(f"\n  [Channel Diversity]")
    print(f"  {'Stage':<25s} {'n_pos':>6s} {'n_neg':>6s} {'mean|r|':>8s} {'top5_r':>8s}")
    print(f"  {'-'*48}")
    for stage in ["geometric_prior", "spg_semantic_prior", "dense_prompt"]:
        if stage not in summary: continue
        sm = summary[stage]["metrics"]
        n_pos = sm.get(f"{'gp' if 'geo' in stage else 'sp' if 'spg' in stage else 'dp'}_n_pos_ch", {})
        n_neg = sm.get(f"{'gp' if 'geo' in stage else 'sp' if 'spg' in stage else 'dp'}_n_neg_ch", {})
        mean_r = sm.get(f"{'gp' if 'geo' in stage else 'sp' if 'spg' in stage else 'dp'}_mean_abs_r", {})
        top5 = sm.get(f"{'gp' if 'geo' in stage else 'sp' if 'spg' in stage else 'dp'}_top5_mean_r", {})
        label = stage_labels.get(stage, stage)
        print(f"  {label:<25s} {str(n_pos.get('mean', 'N/A')):>6s} "
              f"{str(n_neg.get('mean', 'N/A')):>6s} "
              f"{str(mean_r.get('mean', 'N/A')):>8s} "
              f"{str(top5.get('mean', 'N/A')):>8s}")

    print(f"\n{SEP}")


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def _save_tile_figure(img_np, gt_main, H, W,
                      gp_l2, gp_ch, probe_act, probe_cors,
                      sp_l2, sp_ch, dp_l2, dp_ch, pred_np,
                      tile_idx, main_cls, vis_dir, tile_id):
    """Save one tile's spatial traceback figure."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    cls_name = ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}")

    # Row 1: Query, GT, GP L2, GP best ch
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Query Image", fontsize=9); axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_main, cmap="gray")
    axes[0, 1].set_title(f"GT ({cls_name})", fontsize=9); axes[0, 1].axis("off")

    im = axes[0, 2].imshow(gp_l2, cmap="RdBu_r")
    axes[0, 2].set_title("GeoPrior (L2 norm)", fontsize=9); axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    if gp_ch:
        im = axes[0, 3].imshow(gp_ch["best_act"], cmap="RdBu_r")
        axes[0, 3].set_title(f"GeoPrior ch{gp_ch['best_ch']} (r={gp_ch['best_r']:.2f})",
                             fontsize=9); axes[0, 3].axis("off")
        plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    # Row 2: SPG best probe, SPG L2, SPG best ch, Dense Prompt L2
    im = axes[1, 0].imshow(probe_act, cmap="hot")
    best_r = max(probe_cors) if probe_cors else 0
    axes[1, 0].set_title(f"SPG best probe (r={best_r:.2f})", fontsize=9)
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(sp_l2, cmap="RdBu_r")
    axes[1, 1].set_title("SPG semantic_prior (L2)", fontsize=9)
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    im = axes[1, 2].imshow(sp_ch["best_act"], cmap="RdBu_r")
    axes[1, 2].set_title(f"SPG ch{sp_ch['best_ch']} (r={sp_ch['best_r']:.2f})",
                         fontsize=9); axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    im = axes[1, 3].imshow(dp_l2, cmap="RdBu_r")
    axes[1, 3].set_title("Dense Prompt (L2)", fontsize=9); axes[1, 3].axis("off")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046)

    # Row 3: DP best ch, Pred, GT overlay pred
    im = axes[2, 0].imshow(dp_ch["best_act"], cmap="RdBu_r")
    axes[2, 0].set_title(f"DP ch{dp_ch['best_ch']} (r={dp_ch['best_r']:.2f})",
                         fontsize=9); axes[2, 0].axis("off")
    plt.colorbar(im, ax=axes[2, 0], fraction=0.046)

    axes[2, 1].imshow(img_np)
    if pred_np.sum() > 0:
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        overlay[pred_np] = (255, 100, 100, 180)
        axes[2, 1].imshow(overlay)
    axes[2, 1].set_title("Prediction", fontsize=9); axes[2, 1].axis("off")

    axes[2, 2].imshow(img_np)
    gt_overlay = np.zeros((H, W, 4), dtype=np.uint8)
    gt_overlay[gt_main] = (100, 255, 100, 180)
    axes[2, 2].imshow(gt_overlay)
    axes[2, 2].set_title("GT overlay", fontsize=9); axes[2, 2].axis("off")

    # Per-channel correlation comparison
    if gp_ch and sp_ch and dp_ch:
        axes[2, 3].bar(["GeoPrior", "SPG", "DensePrompt"],
                       [gp_ch["top5_mean_r"], sp_ch["top5_mean_r"], dp_ch["top5_mean_r"]],
                       color=["steelblue", "coral", "darkgreen"])
        axes[2, 3].set_title("Top-5 mean channel r", fontsize=9)
        axes[2, 3].set_ylabel("Pearson r")

    fig.suptitle(f"Spatial Traceback: {tile_id}  support={cls_name}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(vis_dir / f"{tile_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_summary_chart(summary: dict, all_metrics: dict, vis_dir: Path):
    """Create summary bar chart comparing spatial metrics across stages."""
    stages_order = ["geometric_prior", "spg_per_probe_best", "spg_semantic_prior",
                    "dense_prompt", "pred_mask"]
    stage_labels = ["GeoPrior", "SPG\nbest probe", "SPG\nsem_prior", "Dense\nPrompt", "Prediction"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    # Chart 1: Pearson r (L2 norm)
    ax = axes[0]
    means, labels = [], []
    for stage in stages_order:
        if stage not in summary: continue
        sm = summary[stage]["metrics"]
        for k in sm:
            if "l2_pearson_r" in k or (k == "probe_pearson_r" and stage == "spg_per_probe_best"):
                means.append(sm[k].get("mean", 0) or 0)
                labels.append(stage_labels[stages_order.index(stage)])
                break
    colors = ["steelblue", "coral", "orange", "darkgreen", "darkred"][:len(means)]
    ax.bar(range(len(means)), means, color=colors, edgecolor="white")
    ax.set_xticks(range(len(means))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Pearson r (L2 norm vs GT)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Pearson r")

    # Chart 2: Best-channel Pearson r
    ax = axes[1]
    means_bc, labels_bc = [], []
    for stage in stages_order:
        if stage not in summary or stage == "pred_mask": continue
        sm = summary[stage]["metrics"]
        for k in sm:
            if "bestch_pearson_r" in k or (k == "probe_r" and stage == "spg_per_probe_best"):
                means_bc.append(sm[k].get("mean", 0) or 0)
                labels_bc.append(stage_labels[stages_order.index(stage)])
                break
    colors_bc = ["steelblue", "coral", "orange", "darkgreen"][:len(means_bc)]
    ax.bar(range(len(means_bc)), means_bc, color=colors_bc, edgecolor="white")
    ax.set_xticks(range(len(means_bc))); ax.set_xticklabels(labels_bc, fontsize=8)
    ax.set_title("Best-channel Pearson r vs GT")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Pearson r")

    # Chart 3: Peak inside GT
    ax = axes[2]
    means_pi, labels_pi = [], []
    for stage in stages_order:
        if stage not in summary: continue
        sm = summary[stage]["metrics"]
        for k in sm:
            if "peak_inside" in k and "l2" in k:
                means_pi.append(sm[k].get("mean", 0) or 0)
                labels_pi.append(stage_labels[stages_order.index(stage)])
                break
            elif k == "probe_peak_inside" and stage == "spg_per_probe_best":
                means_pi.append(sm[k].get("mean", 0) or 0)
                labels_pi.append(stage_labels[stages_order.index(stage)])
                break
    colors_pi = ["steelblue", "coral", "orange", "darkgreen", "darkred"][:len(means_pi)]
    ax.bar(range(len(means_pi)), means_pi, color=colors_pi, edgecolor="white")
    ax.set_xticks(range(len(means_pi))); ax.set_xticklabels(labels_pi, fontsize=8)
    ax.set_title("Peak inside GT rate")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)

    # Chart 4: Inside/Outside ratio
    ax = axes[3]
    means_io, labels_io = [], []
    for stage in stages_order:
        if stage not in summary: continue
        sm = summary[stage]["metrics"]
        for k in sm:
            if "inside_outside" in k and "l2" in k:
                means_io.append(sm[k].get("mean", 0) or 0)
                labels_io.append(stage_labels[stages_order.index(stage)])
                break
            elif k == "probe_inside_outside" and stage == "spg_per_probe_best":
                means_io.append(sm[k].get("mean", 0) or 0)
                labels_io.append(stage_labels[stages_order.index(stage)])
                break
    colors_io = ["steelblue", "coral", "orange", "darkgreen", "darkred"][:len(means_io)]
    ax.bar(range(len(means_io)), means_io, color=colors_io, edgecolor="white")
    ax.set_xticks(range(len(means_io))); ax.set_xticklabels(labels_io, fontsize=8)
    ax.set_title("Inside/Outside activation ratio")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="equal")
    ax.legend(fontsize=7)
    ax.set_ylabel("Ratio")

    # Chart 5: Top-20% IoU
    ax = axes[4]
    means_t20, labels_t20 = [], []
    for stage in stages_order:
        if stage not in summary: continue
        sm = summary[stage]["metrics"]
        for k in sm:
            if "top20_IoU" in k and "l2" in k:
                means_t20.append(sm[k].get("mean", 0) or 0)
                labels_t20.append(stage_labels[stages_order.index(stage)])
                break
            elif k == "probe_top20_IoU" and stage == "spg_per_probe_best":
                means_t20.append(sm[k].get("mean", 0) or 0)
                labels_t20.append(stage_labels[stages_order.index(stage)])
                break
    colors_t20 = ["steelblue", "coral", "orange", "darkgreen", "darkred"][:len(means_t20)]
    ax.bar(range(len(means_t20)), means_t20, color=colors_t20, edgecolor="white")
    ax.set_xticks(range(len(means_t20))); ax.set_xticklabels(labels_t20, fontsize=8)
    ax.set_title("Top-20% activation vs GT IoU")
    ax.set_ylabel("IoU")

    # Chart 6: Decoder gain (how much does each stage improve?)
    ax = axes[5]
    # Collect raw per-episode data for scatter: DP best-ch r vs prediction IoU
    if "dense_prompt" in all_metrics and "pred_mask" in all_metrics:
        dp_items = all_metrics["dense_prompt"]
        pred_items = all_metrics["pred_mask"]
        dp_rs = []
        pred_ious = []
        for dp, pr in zip(dp_items, pred_items):
            r = dp.get("dp_bestch_pearson_r")
            iou = pr.get("pred_top20_IoU")
            if r is not None and r == r and iou is not None and iou == iou:
                dp_rs.append(r)
                pred_ious.append(iou)
        if dp_rs:
            ax.scatter(dp_rs, pred_ious, alpha=0.5, s=15, c="steelblue")
            ax.set_xlabel("DP best-channel r"); ax.set_ylabel("Pred Top-20% IoU")
            ax.set_title(f"DP spatial quality → Pred quality\n(n={len(dp_rs)})")
            # Correlation
            if len(dp_rs) > 2:
                sr = np.corrcoef(dp_rs, pred_ious)[0, 1]
                ax.text(0.05, 0.95, f"r={sr:.3f}", transform=ax.transAxes,
                        fontsize=10, va="top")

    plt.suptitle("Spatial Traceback Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(vis_dir / "summary_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
