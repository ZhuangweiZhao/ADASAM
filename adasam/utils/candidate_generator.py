"""
Candidate Generator | 候选生成器.
==================================

从 Similarity Tensor [K, H, W] 生成候选区域 (candidates), 替代旧的 Matcher.select()。
Generates candidate regions from the Similarity Tensor [K, H, W], replacing Matcher.select().

关键改进 vs 旧 Matcher | Key improvements vs old Matcher:
    - 消费完整的 Similarity Tensor [K,64,64], 而非融合后的单张图。
      Consumes the full Similarity Tensor [K,64,64], not a fused single map.
    - 相对阈值 (μ+ασ) 替代固定阈值, 适应不同类别的相似度分布差异。
      Relative threshold (μ+ασ) replaces fixed threshold, adapting to per-class similarity scales.
    - 连通分量 → 每个 blob 一个候选, 而非贪心 top-K 峰值。
      Connected components → one candidate per blob, not greedy top-K peaks.
    - 每个候选携带 K 个 support 的相似度信号, 供 Prompt Generator 融合。
      Each candidate carries per-support similarity signals for the Prompt Generator.

坐标系 | Coordinate frame:
    返回的 coords / boxes 位于模型输入帧 (1024²), 与 PromptEncoder 期望一致。
    Returned coords / boxes are in the model input frame (1024²).
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
import torch


class CandidateSet(NamedTuple):
    """候选集 | Candidate set.

    :param coords: [N, 2] float, 输入帧 (x, y) — 候选中心点 | input-frame centroids.
    :param boxes: [N, 4] float, 输入帧 (x1, y1, x2, y2) — 候选边界框 | input-frame bboxes.
    :param scores: [N] float, region_score_raw ∈ [0, 1] | raw region confidence.
    :param per_support_sim: [N, K] float, 每 support 在该候选内的平均相似度 | per-support mean sim.
    :param query_features: [N, 256] float, 候选区域内池化后的 query 特征 | pooled query features.
    :param n_candidates: int, 候选数量 (便捷字段) | convenience field.
    """

    coords: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor
    per_support_sim: torch.Tensor
    query_features: torch.Tensor
    n_candidates: int


def _find_peaks_greedy(
    sim_map: np.ndarray,
    mask: np.ndarray,
    min_distance: int = 2,
    max_peaks: int = 8,
) -> list[tuple[int, int, float]]:
    """在 mask 区域内用贪心 NMS 找相似度局部峰值 | Greedy NMS peak finding within a mask.

    按相似度降序排列所有被 mask 覆盖的网格 cells, 依次取最高值作为峰值,
    然后抑制其周围 min_distance (Chebyshev) 内的所有 cells。
    Sort masked cells by similarity descending; take the highest as a peak,
    then suppress all cells within min_distance (Chebyshev).

    :param sim_map: [H, W] aggregated similarity (e.g., max over K supports).
    :param mask: [H, W] bool, region to search within.
    :param min_distance: suppression radius in grid cells.
    :param max_peaks: cap on number of peaks returned.
    :return: list of (gy, gx, sim_value) sorted by sim descending.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []

    sim_vals = sim_map[ys, xs]
    order = np.argsort(sim_vals)[::-1]  # descending

    peaks: list[tuple[int, int, float]] = []
    suppressed = np.zeros(len(ys), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        gy, gx = int(ys[idx]), int(xs[idx])
        sv = float(sim_map[gy, gx])
        if sv <= 0:
            continue

        peaks.append((gy, gx, sv))

        if len(peaks) >= max_peaks:
            break

        # Suppress neighbors within Chebyshev distance
        for j in range(len(ys)):
            if not suppressed[j]:
                dist = max(abs(int(ys[j]) - gy), abs(int(xs[j]) - gx))
                if dist <= min_distance:
                    suppressed[j] = True

    return peaks


def _remove_margin(labels: np.ndarray, margin: int = 1) -> np.ndarray:
    """将接触图像边界的连通分量标记为背景 (0) | Label CCs touching image border as background (0).

    对应 SAM 的 mask_decoder 会在图像边界产生伪影, 边界候选通常是噪声。
    SAM's mask_decoder produces artifacts near image boundaries; border candidates are usually noise.
    """
    if margin <= 0:
        return labels
    h, w = labels.shape
    # Top / bottom / left / right border labels
    border_labels = set()
    border_labels.update(labels[0:margin, :].flatten().tolist())
    border_labels.update(labels[-margin:, :].flatten().tolist())
    border_labels.update(labels[:, 0:margin].flatten().tolist())
    border_labels.update(labels[:, -margin:].flatten().tolist())
    border_labels.discard(0)  # background is fine
    for lbl in border_labels:
        labels[labels == lbl] = 0
    return labels


def generate_candidates(
    sim_tensor: torch.Tensor,
    query_feature: torch.Tensor,
    stride: float = 16.0,
    alpha: float = 1.0,
    min_area: int = 1,
    max_candidates: int = 64,
    border_margin: int = 1,
    peak_min_distance: int = 2,
    max_peaks_per_cc: int = 8,
) -> CandidateSet:
    """从 Similarity Tensor 生成候选 | Generate candidates from the Similarity Tensor.

    对每个 CC 内部用贪心 NMS 找多个相似度峰值, 每个峰值生成一个独立候选。
    这解决了密集小目标 (如车辆) 的 blob 粘连问题: 即使 binary union 把相邻实例
    合并为一个 CC, 每个实例的峰值仍会生成独立的候选。
    Within each CC, greedy NMS finds multiple similarity peaks; each peak becomes an
    independent candidate. This solves blob merging for dense small objects (e.g. vehicles):
    even when the binary union merges adjacent instances into one CC, each instance's
    peak still produces its own candidate.

    :param sim_tensor: [K, H, W] similarity maps (one per support, NOT fused).
    :param query_feature: [1, C, H, W] (或 [C, H, W]) query image embedding.
    :param stride: 输入帧 / 网格 步长 (1024/64 = 16) | input-frame-per-grid-cell stride.
    :param alpha: 相对阈值系数 τ = μ + α·σ | relative threshold coefficient.
    :param min_area: 最小候选面积 (网格 cells) | minimum candidate area in grid cells.
    :param max_candidates: 最大候选数 (按 score 截断) | max candidates (truncated by score).
    :param border_margin: 边界去除边距 (网格 cells) | border removal margin in grid cells.
    :param peak_min_distance: NMS 抑制半径 (网格 cells) | NMS suppression radius.
    :param max_peaks_per_cc: 每个 CC 最大峰值数 | max peaks per connected component.
    :return: CandidateSet. 零候选时回退为全局最大峰值 | falls back to global-max peak if zero.
    """
    if sim_tensor.ndim != 3:
        raise ValueError(
            f"expected sim_tensor [K, H, W], got {tuple(sim_tensor.shape)}"
        )

    K, H, W = sim_tensor.shape
    C = query_feature.shape[-3] if query_feature.ndim == 3 else query_feature.shape[-3]
    if query_feature.ndim == 4:
        query_3d = query_feature[0]  # [C, H, W]
    else:
        query_3d = query_feature    # [C, H, W]

    device = sim_tensor.device

    # ── Step 1: Relative threshold per support, then union ──
    sim_np = sim_tensor.detach().cpu().numpy()  # [K, H, W]

    binary_union = np.zeros((H, W), dtype=np.uint8)
    for k in range(K):
        s = sim_np[k]
        mu = float(s.mean())
        sigma = float(s.std())
        tau = mu + alpha * sigma
        binary_union |= (s > tau).astype(np.uint8)

    # ── Step 1.5: Aggregated similarity for peak finding ──
    sim_agg = sim_np.max(axis=0)  # [H, W] — max over K supports

    # ── Step 2: Connected components ──
    num_labels, labels = cv2.connectedComponents(binary_union, connectivity=8)

    # Remove border-touching components
    labels = _remove_margin(labels, margin=border_margin)

    remaining = set(np.unique(labels)) - {0}
    if not remaining:
        return _fallback_candidate(sim_tensor, query_3d, device, stride, C)

    # ── Step 3: Per-CC → multi-peak candidates ──
    candidates: list[dict] = []

    for label_id in sorted(remaining):
        grid_mask = (labels == label_id)
        area = int(grid_mask.sum())

        if area < min_area:
            continue

        # Find similarity peaks within this CC
        peaks = _find_peaks_greedy(
            sim_agg, grid_mask,
            min_distance=peak_min_distance,
            max_peaks=max_peaks_per_cc,
        )

        if not peaks:
            continue

        # CC-level bbox (used when only 1 peak — keeps original behavior)
        rows, cols = np.where(grid_mask)
        gx1_cc, gx2_cc = int(cols.min()), int(cols.max())
        gy1_cc, gy2_cc = int(rows.min()), int(rows.max())

        for peak_gy, peak_gx, peak_val in peaks:
            # --- Point from peak (more precise than center-of-mass) ---
            centroid_x = (float(peak_gx) + 0.5) * stride
            centroid_y = (float(peak_gy) + 0.5) * stride

            # --- Bbox: local neighborhood around peak ---
            # When multiple peaks exist in one CC, use a local box per peak
            # so the box prompt doesn't cover unrelated instances.
            if len(peaks) == 1:
                # Single peak → use the full CC bbox (original behavior)
                bbox_x1 = float(gx1_cc) * stride
                bbox_y1 = float(gy1_cc) * stride
                bbox_x2 = float(gx2_cc + 1) * stride
                bbox_y2 = float(gy2_cc + 1) * stride
            else:
                # Multiple peaks → local box around each peak
                half = max(1, peak_min_distance)
                px1 = max(0, peak_gx - half)
                py1 = max(0, peak_gy - half)
                px2 = min(W - 1, peak_gx + half + 1)
                py2 = min(H - 1, peak_gy + half + 1)
                bbox_x1 = float(px1) * stride
                bbox_y1 = float(py1) * stride
                bbox_x2 = float(px2 + 1) * stride
                bbox_y2 = float(py2 + 1) * stride

            # --- Per-support similarity in local neighborhood ---
            py1_l = max(0, peak_gy - 1)
            py2_l = min(H, peak_gy + 2)
            px1_l = max(0, peak_gx - 1)
            px2_l = min(W, peak_gx + 2)
            local_mask = np.zeros((H, W), dtype=bool)
            local_mask[py1_l:py2_l, px1_l:px2_l] = True
            local_mask &= grid_mask  # intersect with CC foreground

            local_mask_t = torch.from_numpy(local_mask).to(device)
            n_local = int(local_mask.sum())
            per_support_sim = []
            if n_local > 0:
                for k in range(K):
                    mean_sim_k = sim_tensor[k][local_mask_t].mean().item()
                    per_support_sim.append(mean_sim_k)
            else:
                per_support_sim = [peak_val] * K

            max_sim = max(per_support_sim) if per_support_sim else peak_val
            mean_sim = sum(per_support_sim) / K if per_support_sim else peak_val
            region_score_raw = 0.5 * float(mean_sim) + 0.5 * float(max_sim)

            # --- Pooled query feature from local neighborhood ---
            if n_local > 0:
                qf_pooled = query_3d[:, local_mask_t].mean(dim=1)  # [C]
            else:
                qf_pooled = query_3d[:, peak_gy, peak_gx]  # [C] single-point fallback

            candidates.append({
                "centroid_xy": (centroid_x, centroid_y),
                "bbox_xyxy": (bbox_x1, bbox_y1, bbox_x2, bbox_y2),
                "per_support_sim": per_support_sim,
                "score": region_score_raw,
                "query_feature": qf_pooled,
            })

    if not candidates:
        return _fallback_candidate(sim_tensor, query_3d, device, stride, C)

    # Sort by score descending, truncate
    candidates.sort(key=lambda c: c["score"], reverse=True)
    candidates = candidates[:max_candidates]

    # ── Pack into tensors ──
    N = len(candidates)
    coords_t = torch.tensor(
        [c["centroid_xy"] for c in candidates], device=device, dtype=torch.float32
    )
    boxes_t = torch.tensor(
        [c["bbox_xyxy"] for c in candidates], device=device, dtype=torch.float32
    )
    scores_t = torch.tensor(
        [c["score"] for c in candidates], device=device, dtype=torch.float32
    )
    per_support_t = torch.tensor(
        [c["per_support_sim"] for c in candidates], device=device, dtype=torch.float32
    )
    qf_t = torch.stack([c["query_feature"] for c in candidates], dim=0)  # [N, C]

    return CandidateSet(
        coords=coords_t,
        boxes=boxes_t,
        scores=scores_t,
        per_support_sim=per_support_t,
        query_features=qf_t,
        n_candidates=N,
    )


def _fallback_candidate(
    sim_tensor: torch.Tensor,
    query_3d: torch.Tensor,
    device: torch.device,
    stride: float,
    C: int,
) -> CandidateSet:
    """零候选回退: 全局最大峰值点 → 单点候选 | Zero-candidate fallback: global-max peak → single point.

    选择 sim_tensor 中最大相似度所在的 support map 的 argmax 位置。
    Picks the argmax location of the support map with the highest similarity.
    """
    K, H, W = sim_tensor.shape

    # Best support: the one with the highest max similarity
    max_per_k = sim_tensor.reshape(K, -1).max(dim=1).values  # [K]
    best_k = int(torch.argmax(max_per_k).item())
    best_map = sim_tensor[best_k]  # [H, W]

    flat_idx = int(torch.argmax(best_map).item())
    gy, gx = divmod(flat_idx, W)

    cx = (float(gx) + 0.5) * stride
    cy = (float(gy) + 0.5) * stride

    coords = torch.tensor([[cx, cy]], device=device, dtype=torch.float32)
    boxes = torch.tensor([[cx - 16, cy - 16, cx + 16, cy + 16]],
                         device=device, dtype=torch.float32)

    # Per-support sim at the fallback point
    grid_mask_t = torch.zeros(H, W, dtype=torch.bool, device=device)
    grid_mask_t[gy, gx] = True
    per_support_sim = sim_tensor[:, grid_mask_t].mean(dim=1)  # [K]

    # Score from similarity at the peak point (not 0.0)
    mean_sim = float(per_support_sim.mean().item())
    max_sim = float(per_support_sim.max().item())
    fallback_score = 0.5 * mean_sim + 0.5 * max_sim
    scores = torch.tensor([fallback_score], device=device, dtype=torch.float32)

    qf = query_3d[:, grid_mask_t].mean(dim=1)  # [C]

    return CandidateSet(
        coords=coords,
        boxes=boxes,
        scores=scores,
        per_support_sim=per_support_sim.unsqueeze(0),  # [1, K]
        query_features=qf.unsqueeze(0),                  # [1, C]
        n_candidates=1,
    )


class CandidateGenerator:
    """候选生成器 (类形式, 与 Matcher 风格一致) | Candidate generator (class form, consistent with Matcher style).

    :param alpha: 相对阈值系数 | relative threshold coefficient.
    :param min_area: 最小候选面积 | minimum candidate area in grid cells.
    :param max_candidates: 最大候选数 | max candidates.
    :param border_margin: 边界去除边距 | border removal margin.
    :param stride: 输入帧 / 网格步长 | stride.
    :param peak_min_distance: NMS 抑制半径 (网格 cells) | NMS suppression radius.
    :param max_peaks_per_cc: 每个 CC 最大峰值数 | max peaks per CC.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        min_area: int = 1,
        max_candidates: int = 64,
        border_margin: int = 1,
        stride: float = 16.0,
        peak_min_distance: int = 2,
        max_peaks_per_cc: int = 8,
    ) -> None:
        self.alpha = alpha
        self.min_area = min_area
        self.max_candidates = max_candidates
        self.border_margin = border_margin
        self.stride = stride
        self.peak_min_distance = peak_min_distance
        self.max_peaks_per_cc = max_peaks_per_cc

    def generate(
        self,
        sim_tensor: torch.Tensor,
        query_feature: torch.Tensor,
    ) -> CandidateSet:
        """从 Similarity Tensor 生成候选 | Generate candidates from Similarity Tensor.

        :param sim_tensor: [K, H, W].
        :param query_feature: [1, C, H, W] or [C, H, W].
        :return: CandidateSet.
        """
        return generate_candidates(
            sim_tensor=sim_tensor,
            query_feature=query_feature,
            stride=self.stride,
            alpha=self.alpha,
            min_area=self.min_area,
            max_candidates=self.max_candidates,
            border_margin=self.border_margin,
            peak_min_distance=self.peak_min_distance,
            max_peaks_per_cc=self.max_peaks_per_cc,
        )
