"""
匈牙利匹配器 | Hungarian matcher.
==================================

Mask2Former 式一对一二分匹配: objectness + 成对 BCE + 成对 Dice 代价,
scipy linear_sum_assignment 求最优指派。
Mask2Former-style one-to-one bipartite matching: objectness + pairwise BCE +
pairwise dice costs, solved with scipy's linear_sum_assignment.

单图 (episode 为类条件, 无图批维), 纯类无参数, 全程 no_grad。
Single-image (episodes are class-conditional; no image-batch dim); a plain
parameter-free class, fully under no_grad.

参考 | Reference: thirdparty/Mask2Former/mask2former/modeling/matcher.py (MIT).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from scipy.optimize import linear_sum_assignment

from adasam.losses.seg_losses import pairwise_dice_cost, pairwise_sigmoid_bce_cost


@dataclass(frozen=True)
class MatcherConfig:
    """匹配代价权重 | matching cost weights.

    :param cost_objectness: objectness 代价权重 | objectness cost weight.
    :param cost_mask: 成对 BCE 代价权重 | pairwise BCE cost weight.
    :param cost_dice: 成对 Dice 代价权重 | pairwise dice cost weight.
    """

    cost_objectness: float = 2.0
    cost_mask: float = 5.0
    cost_dice: float = 5.0

    @classmethod
    def from_dict(cls, d: dict) -> "MatcherConfig":
        """从 yaml loss 段构建 (键带 match_cost_ 前缀) | build from the yaml loss block."""
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for k, v in d.items():
            name = k.removeprefix("match_") if k.startswith("match_") else k
            if name in known:
                kwargs[name] = v
        return cls(**kwargs)


class HungarianMatcher:
    """一对一实例匹配 | One-to-one instance matching.

    :param cfg: :class:`MatcherConfig`.
    """

    def __init__(self, cfg: MatcherConfig) -> None:
        self.cfg = cfg

    @torch.no_grad()
    def match(
        self,
        mask_logits: torch.Tensor,
        objectness_logits: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """求最优一对一指派 | Solve the optimal one-to-one assignment.

        :param mask_logits: [N, h, w] 预测掩码 logits | predicted mask logits.
        :param objectness_logits: [N] objectness logits.
        :param gt_masks: [M, h, w] 软目标 ∈ [0,1], 1 ≤ M ≤ N | soft targets.
        :return: (pred_idx [M], gt_idx [M]) long — 匹配对索引 | matched pair indices.
        """
        n, m = mask_logits.shape[0], gt_masks.shape[0]
        if m > n:
            # Cap GT to top-n by area (largest first), keeping the most salient instances
            areas = gt_masks.flatten(1).sum(dim=1)
            _, top_idx = areas.topk(n)
            gt_masks = gt_masks[top_idx]
            m = n
        if m == 0:
            raise ValueError("gt_masks is empty; caller must skip empty episodes")

        pred_flat = mask_logits.flatten(1)                       # [N, hw]
        gt_flat = gt_masks.flatten(1).to(pred_flat.dtype)        # [M, hw]

        # objectness 代价: 概率越高代价越低 (DETR 惯例) | higher prob → lower cost
        cost_obj = -objectness_logits.sigmoid().unsqueeze(1).expand(n, m)
        cost = (
            self.cfg.cost_objectness * cost_obj
            + self.cfg.cost_mask * pairwise_sigmoid_bce_cost(pred_flat, gt_flat)
            + self.cfg.cost_dice * pairwise_dice_cost(pred_flat, gt_flat)
        )                                                        # [N, M]
        # 训练早期极端 logits 可能产生 NaN/Inf, 钳制保证指派可解
        # clamp NaN/Inf from extreme early-training logits so assignment stays solvable
        cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4)

        pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())
        device = mask_logits.device
        return (
            torch.as_tensor(pred_idx, dtype=torch.long, device=device),
            torch.as_tensor(gt_idx, dtype=torch.long, device=device),
        )
