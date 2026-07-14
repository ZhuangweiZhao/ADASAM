"""
分割损失 | Segmentation losses.
================================

用于逐实例掩码监督的二值损失, 集中定义 (AdaTile-FastSAM 中曾散落在各训练脚本内)。
Binary losses for per-instance mask supervision, centralized here (previously copy-pasted
across AdaTile-FastSAM training scripts).

遥感 focal 关键常量 | Remote-sensing focal key constants:
    - eps=1e-4 (非 1e-8): 把 1/(1-p) 的梯度从 ~1e8 压到 ~1e4, 抑制稀有前景梯度爆炸。
      caps the 1/(1-p) gradient from ~1e8 to ~1e4, preventing rare-FG gradient explosion.
    - gamma=5.0: 极端前景/背景不平衡下更强的背景抑制 | stronger BG suppression.

约定 | Convention: 所有函数接收 **logits** (未过 sigmoid), 内部做 sigmoid。
All functions take logits (pre-sigmoid) and apply sigmoid internally.
形状 [N, H, W] (N 个实例) 或 [H, W] | shapes [N, H, W] (N instances) or [H, W].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """归约 | Reduce."""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"unknown reduction '{reduction}'")


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 5.0,
    alpha: float | None = None,
    eps: float = 1e-4,
    reduction: str = "mean",
) -> torch.Tensor:
    """二值 focal loss | Binary focal loss.

    :param logits: 预测 logits | prediction logits.
    :param targets: {0,1} 目标, 同形状 | binary targets, same shape.
    :param gamma: 聚焦指数 | focusing exponent.
    :param alpha: 正类权重 (None=不加权) | positive-class weight (None = unweighted).
    :param eps: 概率截断, 防梯度爆炸 | probability clamp to prevent gradient explosion.
    """
    prob = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    ce = -(targets * torch.log(prob) + (1.0 - targets) * torch.log(1.0 - prob))
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = (1.0 - p_t).pow(gamma) * ce
    if alpha is not None:
        a_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = a_t * loss
    return _reduce(loss, reduction)


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """Soft Dice loss (逐实例) | Soft Dice loss (per instance).

    对 [N, H, W]: 每个实例独立计算 Dice, 再按 reduction 归约。
    For [N, H, W]: Dice per instance, then reduced.
    """
    prob = torch.sigmoid(logits)
    if prob.ndim == 2:                          # [H, W] → [1, H, W]
        prob, targets = prob.unsqueeze(0), targets.unsqueeze(0)
    prob_f = prob.flatten(1)
    tgt_f = targets.flatten(1).to(prob_f.dtype)
    inter = (prob_f * tgt_f).sum(dim=1)
    denom = prob_f.sum(dim=1) + tgt_f.sum(dim=1)
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)   # [N]
    return _reduce(dice, reduction)


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    focal_weight: float = 1.0,
    dice_weight: float = 1.0,
    gamma: float = 5.0,
    alpha: float | None = None,
    eps: float = 1e-4,
) -> torch.Tensor:
    """focal + dice 组合 | Combined focal + dice loss."""
    fl = focal_loss(logits, targets, gamma=gamma, alpha=alpha, eps=eps)
    dl = dice_loss(logits, targets)
    return focal_weight * fl + dice_weight * dl


@torch.no_grad()
def mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """二值掩码 IoU (逐实例) | Per-instance binary mask IoU.

    :param pred_mask: [N, H, W] bool/float 预测 | predicted masks.
    :param gt_mask: [N, H, W] bool/float 目标 | target masks.
    :return: [N] IoU. 用作 IoU-head 的回归目标 | regression target for the IoU head.
    """
    p = (pred_mask > 0.5).flatten(1).float()
    g = (gt_mask > 0.5).flatten(1).float()
    inter = (p * g).sum(dim=1)
    union = p.sum(dim=1) + g.sum(dim=1) - inter
    return inter / (union + eps)
