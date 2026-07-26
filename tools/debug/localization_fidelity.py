"""
Localization Fidelity Diagnostic — 空间定位精度诊断
====================================================

测量模型的**空间定位质量**, 不测语义正确性。

核心假说: 模型知道"该找哪一类" (Semantic Correct), 但"定位不准"
          (Localization Poor), 导致 mIoU 低。

四个实验:
  E1. Prompt Spatial Alignment — dense_prompt 激活图 vs GT 距离变换的相关性
  E2. Boundary F-score — 边界区域 vs 内部区域的 IoU 分解
  E3. Instance Recall — GT 实例中有多少被检测到
  E4. Prompt Peak Analysis — prompt 峰值距离 GT 中心的偏移分布

用法 | Usage::

    python tools/debug/localization_fidelity.py \\
        --checkpoint runs/stage2_fold1_k5_seed42/best_model.pt \\
        --mode novel --k-shot 5

    python tools/debug/localization_fidelity.py \\
        --checkpoint runs/stage2_fold1_k5_seed42/best_model.pt \\
        --mode novel --k-shot 5 --num-tiles 50 --save-vis

输出:
    localization_fidelity/summary.json  — 全部定量指标
    localization_fidelity/figures/      — 可视化 (with --save-vis)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
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
from adasam.datasets.isaid_5i import (
    ISAID5iDataset,
    ISAID5I_CATEGORIES,
)
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image, resize_mask


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_support_for_class(
    *,
    data_root: Path,
    fold: int,
    mode: str,
    class_id: int,
    k_shot: int,
    backbone: MobileSAMBackbone,
    cat_adapter: CATAdapter | None,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """为单个类别构建 support (features, masks)."""
    ds = ISAID5iDataset(root=str(data_root), fold=fold, split="train", mode=mode)
    tile_indices = ds.class_to_tiles(class_id)
    if not tile_indices:
        return None
    scenes: dict[str, list[int]] = defaultdict(list)
    for idx in tile_indices:
        src = ds._source_images.get(ds.tile_ids[idx], ds.tile_ids[idx])
        scenes[src].append(idx)
    rng = random.Random(seed)
    scene_keys = list(scenes)
    k = min(k_shot, len(scene_keys))
    chosen_scenes = rng.sample(scene_keys, k)
    images, masks = [], []
    for sid in chosen_scenes:
        idx = rng.choice(scenes[sid])
        sample = ds[idx]
        fg = ds.get_class_mask(idx, class_id)
        if fg is None or fg.sum() < 1:
            continue
        x, _ = preprocess_image(sample["image"])
        images.append(x.to(device))
        masks.append(fg)
    if not images:
        return None
    feats = backbone(torch.stack(images, dim=0))["image_embedding"]
    if cat_adapter is not None:
        feats = cat_adapter(feats)
    masks_grid = torch.stack(
        [resize_mask(m, (feats.shape[2], feats.shape[3])).to(device) for m in masks], dim=0,
    )
    return feats, masks_grid


def _prompt_activation(dense_prompt: torch.Tensor, method: str = "l2") -> np.ndarray:
    """将 dense_prompt [1, C, H, W] 压缩为单通道激活图.

    :param method: "l2" (L2 norm), "mean_abs" (mean absolute), "pca1" (first PC).
    :return: [H, W] numpy array.
    """
    x = dense_prompt[0].cpu().float()  # [C, H, W]
    C, H, W = x.shape
    if method == "l2":
        act = x.pow(2).mean(dim=0).sqrt()
    elif method == "mean_abs":
        act = x.abs().mean(dim=0)
    elif method == "pca1":
        x_flat = x.reshape(C, -1)  # [C, HW]
        x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
        try:
            _, _, V = torch.pca_lowrank(x_centered.T, q=1)
            act = (x_centered.T @ V).squeeze(1).reshape(H, W)
            act = act.abs()
        except Exception:
            act = x.abs().mean(dim=0)  # fallback
    else:
        act = x.abs().mean(dim=0)
    return act.numpy()


def _gt_distance_transform(gt: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """GT 的距离变换 (resize 到 target_size).

    正值 = 在 GT 内部, 值越大离边界越远 (中心区域)。
    负值 = 在 GT 外部。
    归一化到 [-1, 1]。

    :param gt: [H, W] bool or float binary mask.
    :param target_size: (h, w) output size.
    :return: [h, w] float distance transform.
    """
    gt_f = gt.astype(np.float32)
    gt_rsz = cv2.resize(gt_f, target_size[::-1], interpolation=cv2.INTER_AREA)
    gt_bin = gt_rsz > 0.5
    if not gt_bin.any():
        return np.zeros(target_size, dtype=np.float32)
    # Distance inside (positive), outside (negative)
    d_in = ndimage.distance_transform_edt(gt_bin)
    d_out = ndimage.distance_transform_edt(~gt_bin)
    dt = d_in - d_out
    # Normalize to [-1, 1] (clip at 3σ)
    max_abs = np.percentile(np.abs(dt), 95) or 1.0
    dt = np.clip(dt / max_abs, -1, 1)
    return dt.astype(np.float32)


def _boundary_mask(mask: np.ndarray, width: int = 5) -> np.ndarray:
    """提取 mask 的边界带 (bool array).

    :param mask: [H, W] bool.
    :param width: 边界带宽度 (px), 实际带宽度 ≈ 2*width.
    :return: [H, W] bool boundary band.
    """
    kernel = np.ones((width, width), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return (dilated > 0) & ~(eroded > 0)


def _connected_components(mask: np.ndarray, min_area: int = 16) -> list[np.ndarray]:
    """提取 mask 中的连通分量 (实例).

    :param mask: [H, W] bool.
    :param min_area: 最小面积 (px), 过滤噪声。
    :return: list of [H, W] bool per instance.
    """
    if not mask.any():
        return []
    labeled, n_labels = ndimage.label(mask)
    instances = []
    for i in range(1, n_labels + 1):
        inst = labeled == i
        if inst.sum() >= min_area:
            instances.append(inst)
    return instances


def _find_peaks(act_map: np.ndarray, min_distance: int = 3,
                threshold_abs: float | None = None) -> np.ndarray:
    """在激活图上找局部极大值。

    :param act_map: [H, W] activation.
    :param min_distance: peaks 之间的最小距离 (px).
    :param threshold_abs: 绝对阈值 (None = mean + 0.5*std).
    :return: [(y, x), ...] peak coordinates.
    """
    from scipy.ndimage import maximum_filter
    if threshold_abs is None:
        threshold_abs = float(act_map.mean() + 0.5 * act_map.std())
    # Find local maxima
    local_max = maximum_filter(act_map, size=min_distance * 2 + 1) == act_map
    peaks_mask = local_max & (act_map >= threshold_abs)
    coords = np.argwhere(peaks_mask)  # [(y, x), ...]
    return coords


# ═══════════════════════════════════════════════════════════════════
# E1: Prompt Spatial Alignment
# ═══════════════════════════════════════════════════════════════════

def compute_spatial_alignment(
    dense_prompt: torch.Tensor,
    gt: np.ndarray,
) -> dict:
    """E1: dense_prompt 激活图 vs GT 距离变换的空间对齐。

    :param dense_prompt: [1, C, 64, 64].
    :param gt: [256, 256] bool GT mask.
    :return: metrics dict.
    """
    # Prompt activation map
    act = _prompt_activation(dense_prompt, method="l2")  # [64, 64]

    # GT distance transform at 64×64
    dt = _gt_distance_transform(gt, (64, 64))  # [64, 64]

    # Flatten
    act_f = act.ravel()
    dt_f = dt.ravel()

    # Pearson correlation
    act_mean, dt_mean = act_f.mean(), dt_f.mean()
    act_std, dt_std = act_f.std(), dt_f.std()
    if act_std > 1e-8 and dt_std > 1e-8:
        pearson = float(np.corrcoef(act_f, dt_f)[0, 1])
    else:
        pearson = float("nan")

    # Spearman rank correlation
    from scipy.stats import spearmanr
    sr, _ = spearmanr(act_f, dt_f)
    spearman = float(sr)

    # Peak overlap: is the prompt peak inside GT?
    gt_rsz = cv2.resize(gt.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA) > 0.5
    peak_yx = np.unravel_index(np.argmax(act), act.shape)
    peak_inside = bool(gt_rsz[peak_yx[0], peak_yx[1]])

    # Mean activation inside vs outside GT
    inside_mean = float(act[gt_rsz].mean()) if gt_rsz.any() else 0.0
    outside_mean = float(act[~gt_rsz].mean()) if (~gt_rsz).any() else 0.0
    inside_ratio = inside_mean / max(outside_mean, 1e-8)

    # IoU between top-K% prompt activation and GT
    for pct in [10, 20, 30]:
        thresh = np.percentile(act, 100 - pct)
        top_mask = act >= thresh
        inter = float((top_mask & gt_rsz).sum())
        union = float((top_mask | gt_rsz).sum())
        iou_key = f"prompt_top{pct}_IoU"
        locals()[iou_key] = inter / union if union > 0 else 0.0

    return {
        "E1_pearson_r": round(pearson, 6) if pearson == pearson else None,
        "E1_spearman_rho": round(spearman, 6),
        "E1_peak_inside_GT": peak_inside,
        "E1_inside_outside_ratio": round(inside_ratio, 4),
        "E1_prompt_top10_IoU": round(locals()["prompt_top10_IoU"], 6),
        "E1_prompt_top20_IoU": round(locals()["prompt_top20_IoU"], 6),
        "E1_prompt_top30_IoU": round(locals()["prompt_top30_IoU"], 6),
    }


# ═══════════════════════════════════════════════════════════════════
# E2: Boundary F-score
# ═══════════════════════════════════════════════════════════════════

def compute_boundary_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    boundary_width: int = 5,
) -> dict:
    """E2: 边界 vs 内部的 IoU 分解。

    :param pred: [H, W] bool prediction.
    :param gt: [H, W] bool GT.
    :param boundary_width: 边界带宽度.
    :return: metrics dict.
    """
    H, W = gt.shape

    # Boundary bands
    gt_boundary = _boundary_mask(gt, boundary_width)
    pred_boundary = _boundary_mask(pred, boundary_width)

    # Interior (non-boundary)
    gt_interior = gt & ~gt_boundary
    pred_interior = pred & ~pred_boundary

    # Global IoU
    global_inter = float((pred & gt).sum())
    global_union = float((pred | gt).sum())
    global_iou = global_inter / global_union if global_union > 0 else 0.0

    # Boundary IoU (only within GT boundary band — where errors matter most)
    b_inter = float((pred & gt & gt_boundary).sum())
    b_union = float(((pred | gt) & gt_boundary).sum())
    b_iou = b_inter / b_union if b_union > 0 else float("nan")

    # Interior IoU
    i_inter = float((pred & gt & ~gt_boundary).sum())
    i_union = float(((pred | gt) & ~gt_boundary).sum())
    i_iou = i_inter / i_union if i_union > 0 else float("nan")

    # Boundary F-score (precision/recall on boundary pixels)
    tp = float((pred & gt_boundary).sum())
    fp = float((pred & ~gt_boundary).sum()) if pred.any() else 0.0
    fn = float((~pred & gt_boundary).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Error decomposition
    fp_boundary = float((pred & ~gt & gt_boundary).sum())    # FP on GT boundary
    fn_boundary = float((~pred & gt_boundary).sum())          # FN on GT boundary
    fp_interior = float((pred & ~gt & ~gt_boundary).sum())    # FP in interior
    fn_interior = float((~pred & gt & ~gt_boundary).sum())    # FN in interior

    total_error = fp_boundary + fn_boundary + fp_interior + fn_interior
    if total_error > 0:
        frac_boundary_err = (fp_boundary + fn_boundary) / total_error
        frac_interior_err = (fp_interior + fn_interior) / total_error
    else:
        frac_boundary_err = 0.0
        frac_interior_err = 0.0

    return {
        "E2_global_IoU": round(global_iou, 6),
        "E2_boundary_IoU": round(b_iou, 6) if b_iou == b_iou else None,
        "E2_interior_IoU": round(i_iou, 6) if i_iou == i_iou else None,
        "E2_boundary_F1": round(b_f1, 6),
        "E2_frac_error_boundary": round(frac_boundary_err, 4),
        "E2_frac_error_interior": round(frac_interior_err, 4),
        "E2_FP_GTboundary": int(fp_boundary),
        "E2_FN_GTboundary": int(fn_boundary),
    }


# ═══════════════════════════════════════════════════════════════════
# E3: Instance Recall
# ═══════════════════════════════════════════════════════════════════

def compute_instance_recall(
    pred: np.ndarray,
    gt: np.ndarray,
    iou_thr: float = 0.3,
    min_area: int = 16,
) -> dict:
    """E3: 实例级别召回率。

    GT 实例有多少被预测到了 (IoU > iou_thr)。

    :param pred: [H, W] bool prediction.
    :param gt: [H, W] bool GT.
    :param iou_thr: 匹配阈值.
    :param min_area: 最小实例面积.
    :return: metrics dict.
    """
    gt_instances = _connected_components(gt, min_area=min_area)
    pred_instances = _connected_components(pred, min_area=min_area)

    n_gt = len(gt_instances)
    n_pred = len(pred_instances)

    if n_gt == 0:
        return {
            "E3_n_GT_instances": 0,
            "E3_n_pred_instances": n_pred,
            "E3_instance_recall": None,
            "E3_instance_precision": None,
            "E3_instance_F1": None,
        }

    # Compute IoU matrix
    iou_mat = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i, gt_i in enumerate(gt_instances):
        for j, pred_j in enumerate(pred_instances):
            inter = (gt_i & pred_j).sum()
            union = (gt_i | pred_j).sum()
            iou_mat[i, j] = inter / union if union > 0 else 0.0

    # Greedy matching (descending IoU)
    matched_gt = set()
    matched_pred = set()
    ious = []
    for i, j in np.dstack(np.unravel_index(
        np.argsort(-iou_mat.ravel()), iou_mat.shape
    ))[0]:
        if iou_mat[i, j] >= iou_thr:
            if i not in matched_gt and j not in matched_pred:
                matched_gt.add(int(i))
                matched_pred.add(int(j))
                ious.append(float(iou_mat[i, j]))

    recall = len(matched_gt) / n_gt
    precision = len(matched_pred) / n_pred if n_pred > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    # Size-stratified recall
    gt_areas = np.array([inst.sum() for inst in gt_instances])
    small_thr = np.percentile(gt_areas, 33) if len(gt_areas) > 0 else 0
    large_thr = np.percentile(gt_areas, 67) if len(gt_areas) > 0 else 0

    small_matched = sum(1 for i in matched_gt if gt_areas[i] <= small_thr)
    large_matched = sum(1 for i in matched_gt if gt_areas[i] >= large_thr)
    n_small = sum(1 for a in gt_areas if a <= small_thr)
    n_large = sum(1 for a in gt_areas if a >= large_thr)

    return {
        "E3_n_GT_instances": n_gt,
        "E3_n_pred_instances": n_pred,
        "E3_instance_recall": round(recall, 6),
        "E3_instance_precision": round(precision, 6),
        "E3_instance_F1": round(f1, 6),
        "E3_mean_IoU_matched": round(float(np.mean(ious)), 6) if ious else None,
        "E3_small_recall": round(small_matched / n_small, 6) if n_small > 0 else None,
        "E3_large_recall": round(large_matched / n_large, 6) if n_large > 0 else None,
        "E3_small_threshold_px": int(small_thr),
        "E3_large_threshold_px": int(large_thr),
    }


# ═══════════════════════════════════════════════════════════════════
# E4: Prompt Peak Analysis
# ═══════════════════════════════════════════════════════════════════

def compute_prompt_peak_analysis(
    dense_prompt: torch.Tensor,
    gt: np.ndarray,
    pred: np.ndarray,
) -> dict:
    """E4: prompt 峰值与 GT 中心的距离。

    :param dense_prompt: [1, C, 64, 64].
    :param gt: [256, 256] bool GT.
    :param pred: [256, 256] bool prediction.
    :return: metrics dict.
    """
    act = _prompt_activation(dense_prompt, method="l2")  # [64, 64]
    H, W = gt.shape

    # Find prompt peaks at 64×64
    peaks_64 = _find_peaks(act, min_distance=3)
    # Scale to 256×256
    peaks = [(int(y * H / 64), int(x * W / 64)) for y, x in peaks_64]

    # GT instance centers
    gt_instances = _connected_components(gt, min_area=16)
    gt_centers = []
    for inst in gt_instances:
        ys, xs = np.where(inst)
        gt_centers.append((int(ys.mean()), int(xs.mean())))

    # For each GT center, find nearest prompt peak
    distances = []
    for gc in gt_centers:
        if peaks:
            min_d = min(np.sqrt((gc[0] - p[0])**2 + (gc[1] - p[1])**2) for p in peaks)
        else:
            min_d = float("nan")
        distances.append(min_d)

    valid_d = [d for d in distances if d == d]
    if valid_d:
        mean_dist = float(np.mean(valid_d))
        median_dist = float(np.median(valid_d))
        q25 = float(np.percentile(valid_d, 25))
        q75 = float(np.percentile(valid_d, 75))
    else:
        mean_dist = median_dist = q25 = q75 = float("nan")

    # Prompt peak count vs GT instance count
    n_peaks = len(peaks)
    n_gt_inst = len(gt_centers)

    # Coverage: fraction of GT instances within N pixels of a prompt peak
    cover_5px = sum(1 for d in valid_d if d <= 5) / max(len(valid_d), 1)
    cover_10px = sum(1 for d in valid_d if d <= 10) / max(len(valid_d), 1)
    cover_20px = sum(1 for d in valid_d if d <= 20) / max(len(valid_d), 1)

    # Also: distance from pred center to GT center for matched instances
    pred_instances = _connected_components(pred, min_area=16)
    pred_centers = []
    for inst in pred_instances:
        ys, xs = np.where(inst)
        pred_centers.append((int(ys.mean()), int(xs.mean())))

    # For each prediction center, distance to nearest GT center
    pred_to_gt_dists = []
    for pc in pred_centers:
        if gt_centers:
            min_d = min(np.sqrt((pc[0] - gc[0])**2 + (pc[1] - gc[1])**2) for gc in gt_centers)
            pred_to_gt_dists.append(min_d)

    avg_pred_offset = float(np.mean(pred_to_gt_dists)) if pred_to_gt_dists else float("nan")

    return {
        "E4_n_prompt_peaks": n_peaks,
        "E4_n_GT_instances": n_gt_inst,
        "E4_peak_GT_ratio": round(n_peaks / max(n_gt_inst, 1), 4),
        "E4_mean_peak_to_GT_center_px": round(mean_dist, 2) if mean_dist == mean_dist else None,
        "E4_median_peak_to_GT_center_px": round(median_dist, 2) if median_dist == median_dist else None,
        "E4_q25_peak_distance_px": round(q25, 2) if q25 == q25 else None,
        "E4_q75_peak_distance_px": round(q75, 2) if q75 == q75 else None,
        "E4_GT_cover_5px": round(cover_5px, 4),
        "E4_GT_cover_10px": round(cover_10px, 4),
        "E4_GT_cover_20px": round(cover_20px, 4),
        "E4_avg_pred_center_offset_px": round(avg_pred_offset, 2) if avg_pred_offset == avg_pred_offset else None,
    }


# ═══════════════════════════════════════════════════════════════════
# Manual forward — capture all intermediates without model changes
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _manual_forward(
    model: AdaSAMModel,
    query_features: torch.Tensor,
    support_features: torch.Tensor,
    support_masks_grid: torch.Tensor,
) -> dict:
    """手动执行前向, 捕获所有中间输出。

    :return: dict with keys:
        support_memory, geometric_prior, spg_out, dense_prompt, sparse_token, low_res
    """
    result = {}

    # 1. Support Encoder
    support_memory = model.support_encoder(support_features, support_masks_grid)
    result["support_memory"] = support_memory

    # 2. Geometric Prior
    if model.geometric_prior is not None:
        geometric_prior = model.geometric_prior(query_features, support_memory)
        result["geometric_prior"] = geometric_prior
    else:
        geometric_prior = None

    # 3. SPG
    dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
    spg_out = model.spg(query_features, support_memory, dense_pe)
    result["spg_out"] = spg_out

    # 4. PromptFusion → dense_prompt
    if model.prompt_fusion is not None and geometric_prior is not None:
        dense_prompt, sparse_token = model.prompt_fusion(
            geometric_prior, spg_out.semantic_prior
        )
    else:
        dense_prompt = model._build_dense_prompt(
            support_memory, support_features, support_masks_grid
        )
        if dense_prompt is None:
            dense_prompt = spg_out.semantic_prior
        sparse_token = dense_prompt.mean(dim=(2, 3))

    result["dense_prompt"] = dense_prompt
    result["sparse_token"] = sparse_token

    # 5. Decode
    if model.bypass_head is not None:
        low_res = model.bypass_head(dense_prompt)
    else:
        support_proto = model._compute_support_prototype(
            support_features, support_masks_grid
        ) if model.cfg.category_injection else None
        low_res, _ = model.sam_decoder(
            query_features, sparse_token, dense_prompt,
            support_prototype=support_proto,
        )
    result["low_res"] = low_res

    return result


# ═══════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_localization_fidelity(
    *,
    checkpoint_path: Path,
    data_root: Path,
    fold: int,
    mode: str,
    k_shot: int,
    seed: int,
    score_thr: float,
    device: torch.device,
    num_tiles: int = 50,
    num_classes_per_tile: int = 3,
    out_dir: Path | None = None,
    save_vis: bool = False,
) -> dict:
    """运行全部四个定位诊断实验。

    :return: summary dict with aggregated metrics for all experiments.
    """
    set_seed(seed)

    # ── Load model ──
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    bb_cfg = cfg.get("backbone", {})
    bb_path = Path(bb_cfg.get("checkpoint", "weights/mobile_sam.pt"))
    bb_path = bb_path if bb_path.is_absolute() else _REPO_ROOT / bb_path

    sam = build_mobile_sam(str(bb_path), bb_cfg.get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    embed_dim = int(cfg.get("support_encoder", {}).get("embed_dim", 256))
    model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(cfg)).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[WARN] missing keys: {missing}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected}")
    model.eval()

    cat_adapter = None
    if ckpt.get("cat_adapter") is not None:
        tcfg = cfg.get("train", {})
        adapter_cfg = tcfg.get("cat_adapter", {})
        cat_adapter = CATAdapter(
            dim=embed_dim, bottleneck=int(adapter_cfg.get("bottleneck", 64)),
        ).to(device)
        cat_adapter.load_state_dict(ckpt["cat_adapter"])
        cat_adapter.eval()

    # ── Data ──
    val_ds = ISAID5iDataset(root=str(data_root), fold=fold, split="val", mode=mode)
    visible_classes = val_ds.visible_classes()

    # ── Output dir ──
    if out_dir is None:
        out_dir = checkpoint_path.parent / "localization_fidelity"
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_vis:
        vis_dir = out_dir / "figures"
        vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir}")

    # ── Build support cache for ALL classes ──
    print(f"[setup] building support cache for {len(visible_classes)} classes...")
    support_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for cls in visible_classes:
        sup_data = build_support_for_class(
            data_root=data_root, fold=fold, mode=mode, class_id=cls,
            k_shot=k_shot, backbone=backbone, cat_adapter=cat_adapter,
            seed=seed, device=device,
        )
        if sup_data is not None:
            support_cache[cls] = sup_data
    print(f"[setup] support cache: {len(support_cache)}/{len(visible_classes)} classes")

    # ── Find query tiles with multiple classes ──
    tile_candidates = []
    for idx in range(len(val_ds)):
        present = []
        for cls in visible_classes:
            gt_m = val_ds.get_class_mask(idx, cls)
            if gt_m is not None and gt_m.sum() > 10:
                present.append(cls)
        if len(present) >= 2:
            tile_candidates.append((idx, present))

    rng = random.Random(seed + 9999)
    rng.shuffle(tile_candidates)
    selected = tile_candidates[:num_tiles]
    print(f"[select] {len(tile_candidates)} tiles with ≥2 classes, selected {len(selected)}")

    # ── Per-tile experiments ──
    # Accumulators
    all_e1: list[dict] = []
    all_e2: list[dict] = []
    all_e3: list[dict] = []
    all_e4: list[dict] = []
    # Raw data for histograms
    all_peak_distances: list[float] = []
    all_pred_offsets: list[float] = []
    per_class_e2: dict[int, list[dict]] = defaultdict(list)  # per-class boundary metrics

    for tile_idx, present_classes in tqdm(selected, desc="localization"):
        sample = val_ds[tile_idx]
        H, W = sample["image"].shape[1], sample["image"].shape[2]

        # Embed query once
        x, meta = preprocess_image(sample["image"])
        query_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
        if cat_adapter is not None:
            query_emb = cat_adapter(query_emb)

        # Test up to num_classes_per_tile classes
        test_classes = present_classes[:num_classes_per_tile]

        for cls in test_classes:
            sup_data = support_cache.get(cls)
            if sup_data is None:
                continue
            sup_feat, sup_mask = sup_data

            # GT
            gt_mask = val_ds.get_class_mask(tile_idx, cls)
            if gt_mask is None or gt_mask.sum() < 10:
                continue
            gt_np = gt_mask.numpy().astype(bool)

            # ── Manual forward (capture all intermediates) ──
            fwd = _manual_forward(model, query_emb, sup_feat, sup_mask)

            # ── Prediction mask ──
            low_res = fwd["low_res"]
            if low_res.shape[1] == 1:
                pred_logits = F.interpolate(
                    low_res.float(), size=(H, W), mode="bilinear", align_corners=False,
                )[0, 0]
                # Handle both raw logits (bypass) and post-sigmoid (SAM decoder)
                vals = pred_logits.cpu()
                if vals.min() >= 0 and vals.max() <= 1:
                    pred_np = (vals > 0.5).numpy()
                else:
                    pred_np = (vals.sigmoid() > 0.5).numpy()
            else:
                pred_np = np.zeros((H, W), dtype=bool)

            if not pred_np.any():
                continue  # empty prediction, skip metrics

            # ── E1: Spatial Alignment ──
            e1 = compute_spatial_alignment(fwd["dense_prompt"], gt_np)
            e1["tile_idx"] = tile_idx
            e1["class_id"] = cls
            e1["class_name"] = ISAID5I_CATEGORIES.get(cls, f"cls{cls}")
            all_e1.append(e1)

            # ── E2: Boundary F-score ──
            e2 = compute_boundary_metrics(pred_np, gt_np)
            e2["tile_idx"] = tile_idx
            e2["class_id"] = cls
            all_e2.append(e2)
            per_class_e2[cls].append(e2)

            # ── E3: Instance Recall ──
            e3 = compute_instance_recall(pred_np, gt_np)
            e3["tile_idx"] = tile_idx
            e3["class_id"] = cls
            all_e3.append(e3)

            # ── E4: Peak Analysis ──
            e4 = compute_prompt_peak_analysis(fwd["dense_prompt"], gt_np, pred_np)
            e4["tile_idx"] = tile_idx
            e4["class_id"] = cls
            all_e4.append(e4)

            # Collect for histograms
            if e4.get("E4_mean_peak_to_GT_center_px") is not None:
                for _ in range(e4.get("E4_n_GT_instances", 1)):
                    # We don't have per-instance distances here, just use mean
                    pass
            if e4.get("E4_avg_pred_center_offset_px") is not None:
                val = e4["E4_avg_pred_center_offset_px"]
                if val == val:
                    all_pred_offsets.append(val)

    # ── Aggregate ──
    def _agg(key: str, items: list[dict]) -> dict:
        vals = [d[key] for d in items if d.get(key) is not None and d[key] == d[key]]
        if not vals:
            return {"mean": None, "median": None, "n": 0}
        return {
            "mean": round(float(np.mean(vals)), 6),
            "median": round(float(np.median(vals)), 6),
            "std": round(float(np.std(vals)), 6),
            "q25": round(float(np.percentile(vals, 25)), 6),
            "q75": round(float(np.percentile(vals, 75)), 6),
            "n": len(vals),
        }

    # E1 aggregates
    e1_summary = {}
    for key in ["E1_pearson_r", "E1_spearman_rho", "E1_inside_outside_ratio",
                "E1_prompt_top10_IoU", "E1_prompt_top20_IoU", "E1_prompt_top30_IoU"]:
        e1_summary[key] = _agg(key, all_e1)
    # E1 peak_inside rate
    peak_inside = [d["E1_peak_inside_GT"] for d in all_e1 if "E1_peak_inside_GT" in d]
    e1_summary["E1_peak_inside_rate"] = round(np.mean(peak_inside), 4) if peak_inside else None

    # E2 aggregates
    e2_summary = {}
    for key in ["E2_global_IoU", "E2_boundary_IoU", "E2_interior_IoU",
                "E2_boundary_F1", "E2_frac_error_boundary", "E2_frac_error_interior"]:
        e2_summary[key] = _agg(key, all_e2)

    # E3 aggregates
    e3_summary = {}
    for key in ["E3_instance_recall", "E3_instance_precision", "E3_instance_F1",
                "E3_mean_IoU_matched", "E3_small_recall", "E3_large_recall"]:
        e3_summary[key] = _agg(key, all_e3)
    total_gt_instances = sum(d.get("E3_n_GT_instances", 0) for d in all_e3)
    total_pred_instances = sum(d.get("E3_n_pred_instances", 0) for d in all_e3)
    e3_summary["total_GT_instances"] = total_gt_instances
    e3_summary["total_pred_instances"] = total_pred_instances

    # E4 aggregates
    e4_summary = {}
    for key in ["E4_n_prompt_peaks", "E4_n_GT_instances", "E4_peak_GT_ratio",
                "E4_mean_peak_to_GT_center_px", "E4_median_peak_to_GT_center_px",
                "E4_q25_peak_distance_px", "E4_q75_peak_distance_px",
                "E4_GT_cover_5px", "E4_GT_cover_10px", "E4_GT_cover_20px",
                "E4_avg_pred_center_offset_px"]:
        e4_summary[key] = _agg(key, all_e4)

    # Per-class E2 (boundary vs interior)
    per_class_summary = {}
    for cls, items in sorted(per_class_e2.items()):
        per_class_summary[ISAID5I_CATEGORIES.get(cls, f"cls{cls}")] = {
            "E2_boundary_IoU": _agg("E2_boundary_IoU", items),
            "E2_interior_IoU": _agg("E2_interior_IoU", items),
            "E2_boundary_F1": _agg("E2_boundary_F1", items),
            "E2_frac_error_boundary": _agg("E2_frac_error_boundary", items),
            "n_samples": len(items),
        }

    # ── Summary ──
    summary = {
        "checkpoint": str(checkpoint_path),
        "fold": fold, "mode": mode, "k_shot": k_shot,
        "n_tiles": len(selected),
        "n_episodes_valid": len(all_e2),
        "E1_spatial_alignment": e1_summary,
        "E2_boundary_fidelity": e2_summary,
        "E3_instance_recall": e3_summary,
        "E4_prompt_peak_analysis": e4_summary,
        "per_class": per_class_summary,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Summary saved] {summary_path}")

    # ── Print key findings ──
    _print_summary(summary)

    # ── Visualizations ──
    if save_vis and all_e2:
        _save_summary_figures(all_e1, all_e2, all_e3, all_e4, per_class_summary, vis_dir)

    return summary


def _print_summary(summary: dict) -> None:
    """Pretty-print key findings."""
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  Localization Fidelity Diagnostic — Results")
    print(f"{SEP}")

    e1 = summary["E1_spatial_alignment"]
    print("\n  [E1] Prompt Spatial Alignment")
    print(f"    Pearson r (prompt vs GT distance transform):  "
          f"{_fmt(e1.get('E1_pearson_r', {}))}")
    print(f"    Spearman ρ:                                    "
          f"{_fmt(e1.get('E1_spearman_rho', {}))}")
    print(f"    Prompt peak inside GT:                        "
          f"{e1.get('E1_peak_inside_rate', 'N/A')}")
    print(f"    Inside/Outside activation ratio:              "
          f"{_fmt(e1.get('E1_inside_outside_ratio', {}))}")
    print(f"    Prompt top-20% vs GT IoU:                     "
          f"{_fmt(e1.get('E1_prompt_top20_IoU', {}))}")

    e2 = summary["E2_boundary_fidelity"]
    print(f"\n  [E2] Boundary Fidelity")
    print(f"    Global IoU:                                    "
          f"{_fmt(e2.get('E2_global_IoU', {}))}")
    print(f"    Boundary IoU (narrow band):                    "
          f"{_fmt(e2.get('E2_boundary_IoU', {}))}")
    print(f"    Interior IoU (non-boundary):                   "
          f"{_fmt(e2.get('E2_interior_IoU', {}))}")
    print(f"    Boundary F1:                                   "
          f"{_fmt(e2.get('E2_boundary_F1', {}))}")
    print(f"    Error fraction — boundary:                    "
          f"{_fmt(e2.get('E2_frac_error_boundary', {}))}")
    print(f"    Error fraction — interior:                    "
          f"{_fmt(e2.get('E2_frac_error_interior', {}))}")

    e3 = summary["E3_instance_recall"]
    print(f"\n  [E3] Instance Recall")
    print(f"    Total GT instances:   {e3.get('total_GT_instances', 'N/A')}")
    print(f"    Total pred instances: {e3.get('total_pred_instances', 'N/A')}")
    print(f"    Instance Recall:                               "
          f"{_fmt(e3.get('E3_instance_recall', {}))}")
    print(f"    Instance Precision:                            "
          f"{_fmt(e3.get('E3_instance_precision', {}))}")
    print(f"    Instance F1:                                   "
          f"{_fmt(e3.get('E3_instance_F1', {}))}")
    print(f"    Small instance recall:                         "
          f"{_fmt(e3.get('E3_small_recall', {}))}")
    print(f"    Large instance recall:                         "
          f"{_fmt(e3.get('E3_large_recall', {}))}")

    e4 = summary["E4_prompt_peak_analysis"]
    print(f"\n  [E4] Prompt Peak Analysis")
    print(f"    Mean peak → GT center distance:               "
          f"{_fmt(e4.get('E4_mean_peak_to_GT_center_px', {}))} px")
    print(f"    Median peak → GT center distance:             "
          f"{_fmt(e4.get('E4_median_peak_to_GT_center_px', {}))} px")
    print(f"    GT instances within 10px of a prompt peak:     "
          f"{_fmt(e4.get('E4_GT_cover_10px', {}))}")
    print(f"    GT instances within 20px of a prompt peak:     "
          f"{_fmt(e4.get('E4_GT_cover_20px', {}))}")
    print(f"    Avg pred center → GT center offset:            "
          f"{_fmt(e4.get('E4_avg_pred_center_offset_px', {}))} px")
    print(f"    Peak / GT instance ratio:                      "
          f"{_fmt(e4.get('E4_peak_GT_ratio', {}))}")

    print(f"\n{SEP}")


def _fmt(d: dict) -> str:
    """Format an aggregate dict as 'mean (n=N)'.  Also handles raw numbers."""
    if isinstance(d, (int, float)):
        return f"{d:.4f}" if d == d else "NaN"
    if not d or d.get("n", 0) == 0:
        return "N/A"
    return f"{d['mean']:.4f} (n={d['n']})"


def _save_summary_figures(
    all_e1: list[dict],
    all_e2: list[dict],
    all_e3: list[dict],
    all_e4: list[dict],
    per_class: dict,
    vis_dir: Path,
) -> None:
    """Generate summary histograms and scatter plots."""
    n_plots = 4 if all_e3 else 3
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # 1. E1: Pearson r histogram
    pearsons = [d["E1_pearson_r"] for d in all_e1
                if d.get("E1_pearson_r") is not None and d["E1_pearson_r"] == d["E1_pearson_r"]]
    axes[0].hist(pearsons, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5)
    axes[0].axvline(x=np.mean(pearsons) if pearsons else 0, color="darkblue", linestyle="-")
    axes[0].set_title(f"E1: Prompt-GT Pearson r\n(mean={np.mean(pearsons):.4f}, n={len(pearsons)})")
    axes[0].set_xlabel("Pearson r"); axes[0].set_ylabel("Count")

    # 2. E2: Boundary vs Interior IoU scatter
    b_ious = [d["E2_boundary_IoU"] for d in all_e2
              if d.get("E2_boundary_IoU") is not None and d["E2_boundary_IoU"] == d["E2_boundary_IoU"]]
    i_ious = [d["E2_interior_IoU"] for d in all_e2
              if d.get("E2_interior_IoU") is not None and d["E2_interior_IoU"] == d["E2_interior_IoU"]]
    if b_ious and i_ious:
        min_len = min(len(b_ious), len(i_ious))
        axes[1].scatter(b_ious[:min_len], i_ious[:min_len], alpha=0.5, s=10, c="steelblue")
        max_val = max(max(b_ious[:min_len]), max(i_ious[:min_len])) + 0.1
        axes[1].plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="y=x")
        axes[1].set_xlim(0, max_val); axes[1].set_ylim(0, max_val)
        axes[1].set_xlabel("Boundary IoU"); axes[1].set_ylabel("Interior IoU")
        axes[1].set_title(f"E2: Boundary vs Interior IoU\n(mean B-IoU={np.mean(b_ious):.4f}, I-IoU={np.mean(i_ious):.4f})")
        axes[1].legend()

    # 3. E2: Error fraction (boundary vs interior) — bar chart
    frac_b = np.mean([d["E2_frac_error_boundary"] for d in all_e2
                      if d.get("E2_frac_error_boundary") is not None])
    frac_i = np.mean([d["E2_frac_error_interior"] for d in all_e2
                      if d.get("E2_frac_error_interior") is not None])
    axes[2].bar(["Boundary Error", "Interior Error"], [frac_b, frac_i],
                color=["coral", "steelblue"], edgecolor="white")
    axes[2].set_title(f"E2: Error Source Decomposition\n({frac_b:.1%} boundary, {frac_i:.1%} interior)")
    axes[2].set_ylabel("Fraction of total error")

    # 4. E3: Instance recall histogram
    recalls = [d["E3_instance_recall"] for d in all_e3
               if d.get("E3_instance_recall") is not None and d["E3_instance_recall"] == d["E3_instance_recall"]]
    precs = [d["E3_instance_precision"] for d in all_e3
             if d.get("E3_instance_precision") is not None and d["E3_instance_precision"] == d["E3_instance_precision"]]
    axes[3].hist(recalls, bins=20, alpha=0.6, label=f"Recall (μ={np.mean(recalls):.3f})" if recalls else "Recall")
    if precs:
        axes[3].hist(precs, bins=20, alpha=0.6, label=f"Precision (μ={np.mean(precs):.3f})")
    axes[3].set_title(f"E3: Instance Recall & Precision (n={len(recalls)})")
    axes[3].set_xlabel("Score"); axes[3].set_ylabel("Count")
    axes[3].legend()

    # 5. E4: Peak distance histogram
    peak_dists = []
    for d in all_e4:
        mean_d = d.get("E4_mean_peak_to_GT_center_px")
        if mean_d is not None and mean_d == mean_d:
            peak_dists.append(mean_d)
    axes[4].hist(peak_dists, bins=30, color="coral", edgecolor="white", alpha=0.8)
    axes[4].axvline(x=10, color="orange", linestyle="--", label="10px")
    axes[4].axvline(x=20, color="red", linestyle="--", label="20px")
    if peak_dists:
        axes[4].axvline(x=np.mean(peak_dists), color="darkred", linestyle="-",
                        label=f"mean={np.mean(peak_dists):.1f}px")
    axes[4].set_title(f"E4: Prompt Peak → GT Center Distance\n(mean={np.mean(peak_dists):.1f}px, n={len(peak_dists)})" if peak_dists else "E4: Peak Distance")
    axes[4].set_xlabel("Distance (px)"); axes[4].set_ylabel("Count")
    axes[4].legend()

    # 6. Per-class boundary vs interior IoU
    classes = list(per_class.keys())
    if classes:
        b_means = [per_class[c]["E2_boundary_IoU"].get("mean", 0) or 0 for c in classes]
        i_means = [per_class[c]["E2_interior_IoU"].get("mean", 0) or 0 for c in classes]
        x = np.arange(len(classes))
        w = 0.35
        axes[5].bar(x - w/2, b_means, w, label="Boundary IoU", color="coral", edgecolor="white")
        axes[5].bar(x + w/2, i_means, w, label="Interior IoU", color="steelblue", edgecolor="white")
        axes[5].set_xticks(x)
        axes[5].set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        axes[5].set_title("E2: Per-Class Boundary vs Interior IoU")
        axes[5].set_ylabel("IoU"); axes[5].legend()

    plt.tight_layout()
    fig.savefig(vis_dir / "summary_figures.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[vis] saved: {vis_dir / 'summary_figures.png'}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Localization Fidelity Diagnostic — 空间定位精度诊断 (E1-E4)"
    )
    p.add_argument("--checkpoint", required=True, help="path to checkpoint .pt file")
    p.add_argument("--data-root", default="data/iSAID-5i")
    p.add_argument("--fold", type=int, default=None, help="fold override")
    p.add_argument("--mode", default=None, choices=["base", "novel", "all"],
                   help="mode override (default: from checkpoint)")
    p.add_argument("--k-shot", type=int, default=None, help="override k-shot")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-thr", type=float, default=0.1)
    p.add_argument("--num-tiles", type=int, default=50,
                   help="number of query tiles to evaluate (default: 50)")
    p.add_argument("--num-classes-per-tile", type=int, default=3,
                   help="max classes per tile (default: 3)")
    p.add_argument("--output-dir", default=None, help="custom output directory")
    p.add_argument("--save-vis", action="store_true",
                   help="save summary figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    fold = args.fold if args.fold is not None else ckpt.get("fold", 0)
    mode = args.mode if args.mode is not None else ckpt.get("mode", "novel")
    k_shot = args.k_shot if args.k_shot is not None else ckpt.get("k_shot", 5)
    del ckpt

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = _REPO_ROOT / data_root

    out_dir = Path(args.output_dir) if args.output_dir else None

    print(f"[setup] checkpoint: {ckpt_path}")
    print(f"[setup] fold={fold}  mode={mode}  k_shot={k_shot}  device={device}")

    run_localization_fidelity(
        checkpoint_path=ckpt_path,
        data_root=data_root,
        fold=fold, mode=mode, k_shot=k_shot,
        seed=args.seed,
        score_thr=args.score_thr,
        device=device,
        num_tiles=args.num_tiles,
        num_classes_per_tile=args.num_classes_per_tile,
        out_dir=out_dir,
        save_vis=args.save_vis,
    )

    print("[Done]")


if __name__ == "__main__":
    main()
