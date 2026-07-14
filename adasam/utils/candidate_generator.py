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
) -> CandidateSet:
    """从 Similarity Tensor 生成候选 | Generate candidates from the Similarity Tensor.

    :param sim_tensor: [K, H, W] similarity maps (one per support, NOT fused).
    :param query_feature: [1, C, H, W] (或 [C, H, W]) query image embedding.
    :param stride: 输入帧 / 网格 步长 (1024/64 = 16) | input-frame-per-grid-cell stride.
    :param alpha: 相对阈值系数 τ = μ + α·σ | relative threshold coefficient.
    :param min_area: 最小候选面积 (网格 cells) | minimum candidate area in grid cells.
    :param max_candidates: 最大候选数 (按 score 截断) | max candidates (truncated by score).
    :param border_margin: 边界去除边距 (网格 cells) | border removal margin in grid cells.
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

    # ── Step 2: Connected components ──
    num_labels, labels = cv2.connectedComponents(binary_union, connectivity=8)
    # labels: [H, W] int, 0 = background

    # Remove border-touching components
    labels = _remove_margin(labels, margin=border_margin)

    # Relabel after removal (gap in label IDs, but fine for iteration)
    remaining = set(np.unique(labels)) - {0}
    if not remaining:
        return _fallback_candidate(sim_tensor, query_3d, device, stride, C)

    # ── Step 3: Per-candidate descriptor ──
    candidates: list[dict] = []

    for label_id in sorted(remaining):
        grid_mask = (labels == label_id)
        area = int(grid_mask.sum())

        if area < min_area:
            continue

        # --- Geometry (grid → input frame) ---
        rows, cols = np.where(grid_mask)
        gx1, gx2 = int(cols.min()), int(cols.max())
        gy1, gy2 = int(rows.min()), int(rows.max())

        # centroid (center of mass in grid, then → input frame)
        gy_c = float(rows.mean())
        gx_c = float(cols.mean())
        centroid_x = (gx_c + 0.5) * stride
        centroid_y = (gy_c + 0.5) * stride

        # bbox xyxy in input frame
        bbox_x1 = float(gx1) * stride
        bbox_y1 = float(gy1) * stride
        bbox_x2 = float(gx2 + 1) * stride
        bbox_y2 = float(gy2 + 1) * stride

        w_px = bbox_x2 - bbox_x1
        h_px = bbox_y2 - bbox_y1
        aspect = max(w_px, h_px) / max(min(w_px, h_px), 1.0)

        # --- Per-support similarity statistics ---
        grid_mask_t = torch.from_numpy(grid_mask).to(device)
        per_support_sim = []
        for k in range(K):
            mean_sim_k = sim_tensor[k][grid_mask_t].mean().item()
            per_support_sim.append(mean_sim_k)

        max_sim = max(per_support_sim)
        mean_sim = sum(per_support_sim) / K
        region_score_raw = 0.5 * float(mean_sim) + 0.5 * float(max_sim)

        # --- Pooled query feature ---
        qf_pooled = query_3d[:, grid_mask_t].mean(dim=1)  # [C]

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
    """

    def __init__(
        self,
        alpha: float = 1.0,
        min_area: int = 1,
        max_candidates: int = 64,
        border_margin: int = 1,
        stride: float = 16.0,
    ) -> None:
        self.alpha = alpha
        self.min_area = min_area
        self.max_candidates = max_candidates
        self.border_margin = border_margin
        self.stride = stride

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
        )
