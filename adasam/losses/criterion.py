"""
集合预测损失准则 | Set-prediction criterion.
==============================================

匈牙利匹配后的多任务损失 (Mask2Former SetCriterion 思路, 适配二值 objectness 与
SAM IoU head):
Multi-task loss after Hungarian matching (Mask2Former SetCriterion adapted to
binary objectness and the SAM IoU head):

    - focal + dice: SAM 解码器 256² 掩码 (匹配对) | on matched SAM 256² masks.
    - objectness BCE: 匹配=1 / 未匹配=0, eos_coef 降权背景 | eos-weighted BCE.
    - IoU-head MSE: 预测 IoU 对真实 mask IoU 回归 | regress predicted IoU.
    - aux 深监督: DPG 每层 64² 掩码重新匹配 + 最终层掩码用主匹配索引耦合监督。
      Deep supervision: per-layer re-matched DPG 64² masks + final-layer DPG
      masks coupled via the main match indices.

GT 用软面积平均降采样 (不阈值化), 保住 64²/256² 网格上的微小实例。
GT is soft area-average downsampled (no thresholding) so tiny instances stay
visible on the 64²/256² grids.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.losses.hungarian_matcher import HungarianMatcher
from adasam.losses.seg_losses import dice_loss, focal_loss, mask_iou
from adasam.prompt import DPGOutput
from adasam.utils.debug_trace import tracer


@dataclass(frozen=True)
class CriterionConfig:
    """损失权重与常数 | loss weights and constants.

    focal_gamma/focal_eps 沿用仓库遥感常数 (见 seg_losses.py 模块注释)。
    focal_gamma/focal_eps keep the repo's remote-sensing constants.
    """

    focal_weight: float = 1.0
    dice_weight: float = 1.0
    objectness_weight: float = 2.0
    iou_weight: float = 1.0
    aux_weight: float = 1.0
    prompt_weight: float = 0.5  # V3: BCE+Dice on dense_prompt projection mask
    var_weight: float = 0.0     # V3.1 spatial variance loss (实验性): 负空间方差惩罚,
                                # 鼓励 dense_prompt 学习空间变化而非退化为全局广播。
                                # 0=关闭; 0.1~1.0 实验推荐值。
                                # Negative spatial-variance penalty: encourages the dense
                                # prompt to learn spatial variation instead of collapsing
                                # to a global broadcast. 0=off; 0.1-1.0 for experiments.
    eos_coef: float = 0.1
    focal_gamma: float = 5.0
    focal_eps: float = 1.0e-4

    @classmethod
    def from_dict(cls, d: dict) -> "CriterionConfig":
        """从 yaml loss 段构建, 忽略未知键 | build from the yaml loss block."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class SetCriterion(nn.Module):
    """匈牙利匹配 + 多任务损失 | Hungarian matching + multi-task loss.

    :param matcher: :class:`HungarianMatcher`.
    :param cfg: :class:`CriterionConfig`.
    """

    def __init__(self, matcher: HungarianMatcher, cfg: CriterionConfig) -> None:
        super().__init__()
        self.matcher = matcher
        self.cfg = cfg

    # ── 辅助 | Helpers ──

    @staticmethod
    def _soft_resize(masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """[M, H, W] → [M, h, w] 软面积平均降采样 | soft area-average downsampling.

        area 模式 = 自适应平均池化: 每个输出格取输入窗口均值, 任何前景像素都留下
        与面积占比成正比的信号 — 双线性只采 4 邻点, 会跳过采样点之间的微小实例。
        Area mode = adaptive average pooling: every FG pixel leaves a signal
        proportional to its area fraction; bilinear samples only 4 neighbors and
        can skip tiny instances between sample points.
        """
        return F.interpolate(masks.unsqueeze(0).float(), size, mode="area")[0]

    def _focal_dice(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """加权 focal + dice (匹配对) | weighted focal + dice on matched pairs."""
        fl = focal_loss(logits, targets, gamma=self.cfg.focal_gamma, eps=self.cfg.focal_eps)
        dl = dice_loss(logits, targets)
        return self.cfg.focal_weight * fl + self.cfg.dice_weight * dl

    def _objectness_bce(
        self, objectness_logits: torch.Tensor, matched_pred_idx: torch.Tensor
    ) -> torch.Tensor:
        """eos 加权 objectness BCE | eos-weighted objectness BCE.

        匹配 query 目标 1 权重 1.0; 未匹配 query 目标 0 权重 eos_coef。
        Matched queries: target 1, weight 1.0; unmatched: target 0, weight eos_coef.
        """
        n = objectness_logits.shape[0]
        target = torch.zeros(n, dtype=objectness_logits.dtype, device=objectness_logits.device)
        target[matched_pred_idx] = 1.0
        weight = torch.full_like(target, self.cfg.eos_coef)
        weight[matched_pred_idx] = 1.0
        bce = F.binary_cross_entropy_with_logits(objectness_logits, target, reduction="none")
        return (weight * bce).sum() / weight.sum()

    # ── 梯度冲突诊断 | Gradient conflict diagnostics ──

    @staticmethod
    def _compute_grad_cosine(
        focal: torch.Tensor,
        dice: torch.Tensor,
        obj: torch.Tensor,
        iou_head: torch.Tensor,
        aux_total: torch.Tensor,
        prompt_loss: torch.Tensor,
        cfg: CriterionConfig,
        dpg_params: list[torch.Tensor],
    ) -> dict[str, torch.Tensor] | None:
        """计算 Main/Aux/Prompt 在 DPG 参数上的梯度余弦相似度.
        Compute gradient cosine similarity between Main/Aux/Prompt on DPG params.

        分别对主损失 (focal+dice+obj+iou) / aux / prompt 求 DPG 参数的梯度,
        然后计算两两之间的余弦相似度 — 检测梯度冲突 (负值 = 互相伤害)。
        Differentiate main/aux/prompt losses w.r.t. DPG params, then compute
        pairwise cosine similarity — detects gradient conflict (negative = harm).
        """
        params = [p for p in dpg_params if p.requires_grad]
        if not params:
            return None

        main_loss = (
            cfg.focal_weight * focal + cfg.dice_weight * dice
            + cfg.objectness_weight * obj + cfg.iou_weight * iou_head
        )
        aux_l = cfg.aux_weight * aux_total
        prompt_l = cfg.prompt_weight * prompt_loss

        components = {"main": main_loss, "aux": aux_l, "prompt": prompt_l}

        # 对每个分量分别求 DPG 参数梯度, 用 torch.autograd.grad (不改 .grad 属性)
        # Differentiate each component w.r.t. DPG params individually
        grads: dict[str, torch.Tensor | None] = {}
        for name, loss_tensor in components.items():
            if not loss_tensor.requires_grad:
                grads[name] = None
                continue
            g = torch.autograd.grad(
                loss_tensor, params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            # g is tuple(grad_wrt_p1, grad_wrt_p2, ...) — flatten into one vector
            flat_parts = [gi.flatten() for gi in g if gi is not None]
            grads[name] = torch.cat(flat_parts) if flat_parts else None

        # 两两余弦相似度 | pairwise cosine similarities
        result: dict[str, torch.Tensor] = {}
        pairs = [
            ("main", "aux", "main_vs_aux"),
            ("main", "prompt", "main_vs_prompt"),
            ("aux", "prompt", "aux_vs_prompt"),
        ]
        for a, b, key in pairs:
            ga, gb = grads.get(a), grads.get(b)
            if ga is not None and gb is not None and ga.numel() > 0 and gb.numel() > 0:
                # 数值稳定: 防止除零 | numerically stable cosine
                cos = F.cosine_similarity(ga.unsqueeze(0), gb.unsqueeze(0), dim=1)
                result[key] = cos.clamp(-1.0, 1.0)
            else:
                result[key] = torch.zeros(1)

        return result

    # ── 前向 | Forward ──

    def forward(
        self,
        sam_mask_logits: torch.Tensor,
        iou_pred: torch.Tensor,
        dpg_out: DPGOutput,
        gt_masks: torch.Tensor,
        dpg_params: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """计算总损失与分项 | Compute the total loss and its components.

        :param sam_mask_logits: [N, 256, 256] SAM 解码掩码 logits (low_res[:, 0])。
        :param iou_pred: [N] SAM IoU head 预测 (iou_pred[:, 0])。
        :param dpg_out: :class:`DPGOutput` (objectness + 64² 掩码 + aux)。
        :param gt_masks: [M, H, W] 该类 GT 实例掩码, 1 ≤ M ≤ N | GT instance masks.
        :return: {"loss", "focal", "dice", "obj", "iou_head", "aux", "n_matched",
                  "mean_obj_matched", "mean_obj_unmatched"} — 除 "loss" 外均已 detach。
                  All entries except "loss" are detached.
        """
        cfg = self.cfg
        gt_256 = self._soft_resize(gt_masks, tuple(sam_mask_logits.shape[-2:]))
        gt_grid = self._soft_resize(gt_masks, tuple(dpg_out.mask_logits.shape[-2:]))

        # ── 主匹配 (SAM 256² 掩码) | main match on SAM 256² masks ──
        pred_idx, gt_idx = self.matcher.match(
            sam_mask_logits, dpg_out.objectness_logits, gt_256
        )

        focal = focal_loss(
            sam_mask_logits[pred_idx], gt_256[gt_idx],
            gamma=cfg.focal_gamma, eps=cfg.focal_eps,
        )
        dice = dice_loss(sam_mask_logits[pred_idx], gt_256[gt_idx])
        obj = self._objectness_bce(dpg_out.objectness_logits, pred_idx)

        # IoU head: 对匹配对的真实 mask IoU 回归 | regress the matched pairs' true IoU
        iou_target = mask_iou(torch.sigmoid(sam_mask_logits[pred_idx]), gt_256[gt_idx] > 0.5)
        iou_head = F.mse_loss(iou_pred[pred_idx], iou_target)

        # ── Prompt auxiliary mask loss (V3): BCE + Dice on dense prompt projection ──
        # 将 dense prompt 投影掩码与 GT 并集对齐, 迫使 prompt 学习类别-空间判别.
        # Align dense prompt projection with GT union → forces class-spatial discrimination.
        if dpg_out.prompt_mask is not None and cfg.prompt_weight > 0:
            prompt_grid = self._soft_resize(
                gt_masks, tuple(dpg_out.prompt_mask.shape[-2:])
            )                                                          # [M, h, w]
            # Merge all GT instances → single foreground mask
            gt_union = prompt_grid.amax(dim=0, keepdim=True)           # [1, h, w]
            prompt_logits = dpg_out.prompt_mask[0]                      # [1, h, w]
            prompt_focal = focal_loss(
                prompt_logits, gt_union, gamma=cfg.focal_gamma, eps=cfg.focal_eps,
            )
            prompt_dice = dice_loss(prompt_logits, gt_union)
            prompt_loss = cfg.focal_weight * prompt_focal + cfg.dice_weight * prompt_dice
        else:
            prompt_loss = torch.tensor(0.0, device=sam_mask_logits.device)
            prompt_focal = torch.tensor(0.0, device=sam_mask_logits.device)
            prompt_dice = torch.tensor(0.0, device=sam_mask_logits.device)

        # ── Spatial variance loss (V3.1 实验): 惩罚空间平坦化 ──
        # 若不加显式约束, 网络倾向于将 dense prompt 退化为全局广播 (所有空间位置
        # 相同), 因为这是优化 BCE+Dice 时最容易的路径。
        # Without explicit constraint, the network collapses the dense prompt to a
        # global broadcast — the easiest path for BCE+Dice optimization.
        #
        # 使用尺度无关的变异系数 (CV = std/mean) 而非原始 std:
        # - 原始 std 梯度 ∝ prompt 幅值 → prompt 小时梯度消失
        # - CV 归一化后梯度不受幅值影响 → 即使 prompt 值~0.003 也有有效梯度
        # Coefficient-of-variation (CV = std/mean) is scale-invariant:
        # raw std gradient ∝ prompt magnitude → vanishes when small;
        # CV is normalized → effective gradient even at 0.003-scale values.
        #
        # 同时惩罚全局平坦 (per-channel CV) 和局部平坦 (Total Variation):
        # Penalize both global flatness (per-channel CV) and local smoothness (TV).
        if dpg_out.dense_prompt is not None and cfg.var_weight > 0:
            dp = dpg_out.dense_prompt  # [1, C, H, W]

            # 1) Per-channel coefficient of variation: std/mean → scale-invariant
            #    Low CV = channel broadcasts same value everywhere → penalize
            ch_std = dp.std(dim=(-2, -1))                     # [1, C]
            ch_mean = dp.abs().mean(dim=(-2, -1)) + 1e-8      # [1, C]
            cv = ch_std / ch_mean                              # [1, C], ~0(flat) ~1+(varied)
            cv_loss = -cv.mean()                               # maximize CV

            # 2) Total Variation (local): penalize if neighboring positions are identical
            #    Gradient magnitude ~O(1) regardless of value scale — much stronger
            #    signal than variance-based losses for tiny values.
            tv_h = (dp[:, :, 1:, :] - dp[:, :, :-1, :]).abs().mean()
            tv_w = (dp[:, :, :, 1:] - dp[:, :, :, :-1]).abs().mean()
            tv_loss = -(tv_h + tv_w)                           # maximize local variation

            var_loss = cv_loss + tv_loss
        else:
            var_loss = torch.tensor(0.0, device=sam_mask_logits.device)

        # ── aux: DPG 最终层掩码用主匹配索引耦合监督 | final DPG masks, main indices ──
        aux_total = self._focal_dice(dpg_out.mask_logits[pred_idx], gt_grid[gt_idx])

        # ── aux: 每个中间层重新匹配 (深监督) | per-layer re-match (deep supervision) ──
        for layer in dpg_out.aux:
            l_pred, l_gt = self.matcher.match(
                layer["mask_logits"], layer["objectness_logits"], gt_grid
            )
            aux_total = aux_total + self._focal_dice(
                layer["mask_logits"][l_pred], gt_grid[l_gt]
            )
            aux_total = aux_total + cfg.objectness_weight * self._objectness_bce(
                layer["objectness_logits"], l_pred
            )

        loss = (
            cfg.focal_weight * focal
            + cfg.dice_weight * dice
            + cfg.objectness_weight * obj
            + cfg.iou_weight * iou_head
            + cfg.aux_weight * aux_total
            + cfg.prompt_weight * prompt_loss
            + cfg.var_weight * var_loss
        )

        # ── 梯度冲突诊断 | Gradient conflict diagnostics ──
        # 分别对主损失 / aux / prompt 求 DPG 梯度, 计算余弦相似度
        # Compute per-component DPG gradients and their cosine similarities
        if dpg_params is not None and dpg_out.dense_prompt is not None:
            grad_cos = self._compute_grad_cosine(
                focal, dice, obj, iou_head,
                aux_total, prompt_loss, cfg, dpg_params,
            )
            if grad_cos:
                tracer.section("SetCriterion — Gradient Cosine (DPG params)")
                tracer.tensor_dict("grad_cos", {
                    "main_vs_aux":    grad_cos["main_vs_aux"].detach(),
                    "main_vs_prompt": grad_cos["main_vs_prompt"].detach(),
                    "aux_vs_prompt":  grad_cos["aux_vs_prompt"].detach(),
                })

        # ── 监控指标 (objectness 塌缩监视) | monitoring (objectness-collapse watch) ──
        with torch.no_grad():
            probs = dpg_out.objectness_logits.sigmoid()
            matched_mask = torch.zeros_like(probs, dtype=torch.bool)
            matched_mask[pred_idx] = True
            mean_matched = probs[matched_mask].mean()
            mean_unmatched = (
                probs[~matched_mask].mean() if (~matched_mask).any()
                else torch.zeros_like(mean_matched)
            )

        # ── Debug trace: loss breakdown ──
        tracer.section("SetCriterion — Loss Breakdown")
        tracer.tensor_dict("loss", {
            "total":    loss.detach(),
            "focal":    focal.detach(),
            "dice":     dice.detach(),
            "obj":      obj.detach(),
            "iou_head": iou_head.detach(),
            "aux":      aux_total.detach(),
            "prompt":   prompt_loss.detach(),
            "var":      var_loss.detach(),
        })
        tracer.tensor("n_matched", torch.as_tensor(float(pred_idx.numel())).unsqueeze(0))

        return {
            "loss": loss,
            "focal": focal.detach(),
            "dice": dice.detach(),
            "obj": obj.detach(),
            "iou_head": iou_head.detach(),
            "aux": aux_total.detach(),
            "prompt_focal": prompt_focal.detach(),
            "prompt_dice": prompt_dice.detach(),
            "prompt": prompt_loss.detach(),
            "var": var_loss.detach(),
            "n_matched": torch.as_tensor(float(pred_idx.numel())),
            "mean_obj_matched": mean_matched,
            "mean_obj_unmatched": mean_unmatched,
        }
