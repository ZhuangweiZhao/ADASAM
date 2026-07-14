"""
原型-查询匹配 | Prototype-Query Matcher.
========================================

把 support 原型与 query 图像嵌入的相似度峰值转化为 SAM 点提示。
Turn similarity peaks between the support prototype and the query image embedding into
SAM point prompts. 这是 PerSAM/Matcher 范式的定位环节 —— 每个峰值 → 一个正点提示 →
SAM 解码出一个实例。Each peak becomes one positive point prompt → SAM decodes one instance.

坐标系 | Coordinate frame:
    返回坐标位于**模型输入帧** (1024²), 与 PromptEncoder.forward_with_coords 期望一致
    (它按 input_image_size 归一化)。网格 (gy,gx) → 像素中心 ((gx+.5)·stride, (gy+.5)·stride)。
    Returned coords are in the model INPUT frame (1024²), matching what PromptEncoder expects.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class PromptPoints(NamedTuple):
    """点提示 | Point prompts.

    :param coords: [N, 2] float, 输入帧 (x, y) | input-frame (x, y).
    :param labels: [N] float, 1=正点 | 1 = positive point.
    :param sims: [N] float, 各点相似度 (评分用) | per-point similarity (for scoring).
    """

    coords: torch.Tensor
    labels: torch.Tensor
    sims: torch.Tensor


def similarity_map(embedding: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
    """原型与嵌入逐位置余弦相似度 | Per-location cosine similarity between prototype and embedding.

    :param embedding: [C, gh, gw] 图像嵌入 | image embedding.
    :param prototype: [C] 类原型 | class prototype.
    :return: [gh, gw] 余弦相似度 ∈ [-1, 1] | cosine similarity map.
    """
    if embedding.ndim != 3:
        raise ValueError(f"expected embedding [C, gh, gw], got {tuple(embedding.shape)}")
    emb = F.normalize(embedding, dim=0)          # 逐位置归一化 | normalize over channels
    proto = F.normalize(prototype, dim=0)
    return torch.einsum("c,chw->hw", proto, emb)


class Matcher:
    """相似度图 → Top-K 点提示 (贪心 NMS) | Similarity map → Top-K point prompts (greedy NMS).

    :param top_k: 最多选取的点数 | max number of points.
    :param sim_threshold: 低于此相似度不再新增点 (至少保留 1 个) | stop adding below this sim (keep ≥1).
    :param min_distance: NMS 抑制半径 (网格单元) | NMS suppression radius in grid cells.
    """

    def __init__(self, top_k: int = 10, sim_threshold: float = 0.5, min_distance: int = 1) -> None:
        self.top_k = top_k
        self.sim_threshold = sim_threshold
        self.min_distance = min_distance

    def select(self, sim_map: torch.Tensor, stride: float) -> PromptPoints:
        """选取点提示 | Select point prompts.

        :param sim_map: [gh, gw] 相似度图 | similarity map.
        :param stride: 输入帧/网格 步长 (1024/64=16) | input-frame-per-grid-cell stride.
        :return: PromptPoints, 至少含 1 个点 (全局最大兜底) | at least one point (global-max fallback).
        """
        gh, gw = sim_map.shape
        work = sim_map.clone()
        r = self.min_distance
        neg_inf = float("-inf")

        gys: list[int] = []
        gxs: list[int] = []
        vals: list[float] = []
        for _ in range(self.top_k):
            flat = int(torch.argmax(work))
            gy, gx = divmod(flat, gw)
            v = float(work[gy, gx])
            if v == neg_inf:
                break                                       # 已全部抑制 | fully suppressed
            if gys and v < self.sim_threshold:
                break                                       # 达到阈值下限 | below threshold
            gys.append(gy); gxs.append(gx); vals.append(v)
            work[max(0, gy - r):gy + r + 1, max(0, gx - r):gx + r + 1] = neg_inf

        if not gys:                                         # 兜底: 全局最大 | fallback: global max
            flat = int(torch.argmax(sim_map))
            gy, gx = divmod(flat, gw)
            gys, gxs, vals = [gy], [gx], [float(sim_map[gy, gx])]

        device = sim_map.device
        gx_t = torch.tensor(gxs, device=device, dtype=torch.float32)
        gy_t = torch.tensor(gys, device=device, dtype=torch.float32)
        coords = torch.stack([(gx_t + 0.5) * stride, (gy_t + 0.5) * stride], dim=1)  # [N,2] (x,y)
        labels = torch.ones(len(gys), device=device, dtype=torch.float32)            # 全正点 | positive
        sims = torch.tensor(vals, device=device, dtype=torch.float32)
        return PromptPoints(coords=coords, labels=labels, sims=sims)
