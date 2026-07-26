"""
语义分割损失 | Semantic Segmentation Loss.
============================================

统一损失: L = L_main + λ₁·L_prior + λ₂·L_reg

- L_main: CE + Focal + Dice (最终 SAM 输出 mask 与 GT 之间)
- L_prior: BCE + Dice (SPG prior_mask 与 GT 之间, deep supervision)
- L_reg: 正则化项 (预留: TV, variance, consistency)

Unified loss: L = L_main + λ₁·L_prior + λ₂·L_reg

- L_main: CE + Focal + Dice (between final SAM mask and GT)
- L_prior: BCE + Dice (between SPG prior_mask and GT, deep supervision)
- L_reg: regularization (reserved: TV, variance, consistency)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.losses.seg_losses import dice_loss, focal_loss


class SemanticSegLoss(nn.Module):
    """语义分割损失 | Semantic segmentation loss.

    输入 | Input:
        pred: [B, 2, H, W] FG/BG logits (channel 0=BG, channel 1=FG)
        gt:   [B, H, W] {0, 1} binary ground-truth

    注意 | Note:
        - gt 是合并后的单张 binary mask, 不区分实例边界。
        - 不需要 Hungarian matching, 不需要 objectness, 不需要 IoU head。
    """

    def __init__(
        self,
        prior_weight: float = 0.3,
        reg_weight: float = 0.0,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_gamma: float = 5.0,
        focal_eps: float = 1.0e-4,
        ignore_index: int = 255,
        div_weight: float = 0.0,
        cov_weight: float = 0.0,
        ent_weight: float = 0.0,
    ):
        super().__init__()
        self.prior_weight = prior_weight
        self.reg_weight = reg_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma
        self.focal_eps = focal_eps
        self.ignore_index = ignore_index
        self.div_weight = div_weight
        self.cov_weight = cov_weight
        self.ent_weight = ent_weight

    # ── L_main: CE + Focal + Dice ──

    def _compute_main_loss(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """主损失: CE + focal + dice on final SAM output."""
        ce = F.cross_entropy(pred, gt.long(), ignore_index=self.ignore_index)

        fg_logits = pred[:, 1, :, :]
        focal = focal_loss(
            fg_logits, gt.float(),
            gamma=self.focal_gamma, eps=self.focal_eps,
        )
        dice = dice_loss(fg_logits, gt.float())

        loss = ce + self.focal_weight * focal + self.dice_weight * dice

        return {
            "loss": loss,
            "ce": ce.detach(),
            "focal": focal.detach(),
            "dice": dice.detach(),
        }

    # ── L_prior: BCE + Dice on SPG unified prior masks (deep supervision) ──

    def _compute_prior_loss(
        self, prior_masks: list[torch.Tensor], gt: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, float]]]:
        """先验深监督损失 | Prior deep supervision loss.

        对 SPG prior_aux 中每层的 unified prior_mask 计算 BCE + Dice。
        prior_mask 已经是聚合后的 unified mask [1, gh, gw], 无需 max-pool。
        Each prior_mask is already a unified mask [1, gh, gw];
        no max-pool aggregation needed.

        :param prior_masks: list of [1, H, W] SPG unified prior masks (per layer).
        :param gt: [B, H, W] binary GT.
        :return: (total_prior_loss, per_layer_breakdown).
        """
        total = torch.tensor(0.0, device=gt.device)
        breakdown = []
        for i, mask_values in enumerate(prior_masks):
            # mask_values: [1, Ha, Wa] — already unified, no max-pool needed
            Ha, Wa = mask_values.shape[1], mask_values.shape[2]
            gt_resized = F.interpolate(
                gt.unsqueeze(1).float(), (Ha, Wa), mode="area"
            ).squeeze(1)  # [B, Ha, Wa]

            # Direct supervision on unified prior (was: max-pool over N queries)
            fg_values = mask_values[0]  # [Ha, Wa]

            bce = F.binary_cross_entropy_with_logits(fg_values, gt_resized[0])
            dice = dice_loss(fg_values, gt_resized[0].float())

            layer_loss = bce + self.dice_weight * dice
            total = total + layer_loss
            breakdown.append({
                "layer": i,
                "bce": float(bce.detach()),
                "dice": float(dice.detach()),
            })

        return total, breakdown

    # ── L_probe: probe spatial diversity + coverage + entropy ──

    def _compute_probe_diversity_loss(
        self,
        probe_masks: torch.Tensor,
        probe_logits: torch.Tensor | None,
        gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Probe diversity loss: encourages spatial specialization among probes.

        Three components:
          L_div  = mean pairwise cosine of flattened mask vectors (lower → diverse)
          L_cov  = BCE(union_of_probes, GT) — prevents collapsing to background
          L_ent  = -H(per-probe activation distribution) — prevents probe death

        :param probe_masks: [N, gh, gw] per-probe mask logits (final SPG layer).
        :param probe_logits: [N] per-probe confidence logits (informational).
        :param gt: [B, H, W] binary GT, resized to match probe spatial dims.
        :return: dict with "L_div", "L_cov", "L_ent".
        """
        N, gh, gw = probe_masks.shape
        masks_prob = probe_masks.sigmoid()  # [N, gh, gw] in [0,1]

        # ── L_div: mean pairwise cosine similarity ──
        masks_flat = masks_prob.reshape(N, gh * gw)          # [N, gh*gw]
        masks_norm = F.normalize(masks_flat, dim=1, eps=1e-8)
        cos_sim = masks_norm @ masks_norm.T                   # [N, N]
        triu_idx = torch.triu_indices(N, N, offset=1)
        cos_pairs = cos_sim[triu_idx[0], triu_idx[1]]         # [N*(N-1)/2]
        L_div = cos_pairs.mean()

        # ── L_cov: BCE(union, GT) ──
        union = masks_prob.max(dim=0)[0]                       # [gh, gw] max over probes
        gt_resized = F.interpolate(
            gt.unsqueeze(0).unsqueeze(0).float(),
            size=(gh, gw), mode="area",
        ).squeeze(0).squeeze(0)                                 # [gh, gw]
        L_cov = F.binary_cross_entropy(
            union.clamp(1e-7, 1 - 1e-7), gt_resized.clamp(0, 1),
        )

        # ── L_ent: negative entropy of per-probe activation distribution ──
        if probe_logits is not None:
            softmax_w = torch.softmax(probe_logits, dim=0)     # [N]
            H = -(softmax_w * torch.log(softmax_w + 1e-8)).sum()
            L_ent = -H
        else:
            mean_act = masks_prob.mean(dim=(1, 2))             # [N]
            weights = mean_act / mean_act.sum().clamp(min=1e-8)
            H = -(weights * torch.log(weights + 1e-8)).sum()
            L_ent = -H

        return {
            "L_div": L_div,
            "L_cov": L_cov,
            "L_ent": L_ent,
        }

    # ── Forward ──

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        prior_masks: list[torch.Tensor] | None = None,
        prior_mask: torch.Tensor | None = None,
        prior_weight: float | None = None,
        probe_masks: torch.Tensor | None = None,
        probe_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """前向计算 | Forward.

        :param pred: [B, 2, H, W] SAM decoder FG/BG logits.
        :param gt: [B, H, W] binary {0, 1} ground-truth.
        :param prior_masks: SPG prior_aux 中每层的 prior_mask 列表 (N,Ha,Wa).
        :param prior_mask: [B, 1, Hp, Wp] SPG prior_mask_head 输出 (顶层 prior).
        :param prior_weight: L_prior 权重覆盖 (默认 self.prior_weight).
        :param probe_masks: [N, gh, gw] or None. Per-probe mask logits for diversity loss.
        :param probe_logits: [N] or None. Per-probe confidence logits.
        :return: {
            "loss": L_main + λ₁·L_prior + λ₂·L_reg + λ_div·L_div + λ_cov·L_cov + λ_ent·L_ent,
            "L_main": ..., "L_prior": ..., "L_reg": ..., "L_div": ..., "L_cov": ..., "L_ent": ...,
        }
        """
        # L_main
        main = self._compute_main_loss(pred, gt)
        loss = main["loss"]

        # L_prior
        prior_total = torch.tensor(0.0, device=pred.device)
        prior_breakdown: list[dict] = []
        _pw = prior_weight if prior_weight is not None else self.prior_weight

        all_prior_masks: list[torch.Tensor] = []
        if prior_masks is not None:
            all_prior_masks.extend(prior_masks)
        if prior_mask is not None and _pw > 0:
            # prior_mask [B, 1, Hp, Wp] → squeeze to [1, Hp, Wp] (unified, no max-pool)
            pm = prior_mask[0]  # [1, Hp, Wp]
            all_prior_masks.append(pm)

        if all_prior_masks and _pw > 0:
            prior_total, prior_breakdown = self._compute_prior_loss(all_prior_masks, gt)
            loss = loss + _pw * prior_total

        # L_probe: probe spatial diversity + coverage + entropy
        L_div = torch.tensor(0.0, device=pred.device)
        L_cov = torch.tensor(0.0, device=pred.device)
        L_ent = torch.tensor(0.0, device=pred.device)
        if probe_masks is not None and (
            self.div_weight > 0 or self.cov_weight > 0 or self.ent_weight > 0
        ):
            probe_losses = self._compute_probe_diversity_loss(
                probe_masks, probe_logits, gt
            )
            L_div = probe_losses["L_div"]
            L_cov = probe_losses["L_cov"]
            L_ent = probe_losses["L_ent"]
            loss = loss + self.div_weight * L_div + self.cov_weight * L_cov + self.ent_weight * L_ent

        # L_reg (预留)
        reg_total = torch.tensor(0.0, device=pred.device)

        return {
            "loss": loss,
            "L_main": main["loss"].detach(),
            "L_prior": prior_total.detach() if isinstance(prior_total, torch.Tensor) else prior_total,
            "L_reg": reg_total,
            "L_div": L_div.detach(),
            "L_cov": L_cov.detach(),
            "L_ent": L_ent.detach(),
            # Detailed breakdown (for logging)
            "main_ce": main["ce"],
            "main_focal": main["focal"],
            "main_dice": main["dice"],
            "prior_layers": prior_breakdown,
        }
