"""
Activation Maximization — What Prompt Does the Decoder Want?
=============================================================

核心问题: Decoder "喜欢"什么样的 prompt? 不匹配 GT, 直接最大化 Decoder 输出。

与 decoder_inversion 的区别:
  - Inversion: 优化 prompt 匹配 GT → 找到"最优" prompt → 比较 gap
  - AM: 优化 prompt 最大化特定目标 → 发现 decoder 的内在偏好

目标:
  (a) FG Activation: max sum(sigmoid(logits)) — decoder 想把 FG 放在哪?
  (b) Spatial Concentration: max(max/mean activation) — 偏好集中还是分散?
  (c) Channel Sparsity: 观察 AM prompt 的通道使用模式

方法:
  1. 从噪声/零初始化 prompt [1, 256, 64, 64]
  2. 冻结 decoder, 优化 prompt 最大化目标 + L2 正则化
  3. 200 步 Adam, 多种初始化 (random/zero)
  4. 分析优化后 prompt 的空间模式、通道结构、与 GT prompt 的关系

用法:
    python tools/analysis/diag_activation_maximization.py \
        --stage2-ckpt runs/stage2_fold1_k5_seed42/best_model.pt \
        --data-root data/iSAID-5i --fold 1 --mode novel --k-shot 5 \
        --num-tiles 20 --optim-steps 200
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

from adasam.datasets.isaid_5i import ISAID5I_CATEGORIES
from adasam.utils.transforms import preprocess_image

from tools.analysis._decoder_diag_base import (
    DiagContext,
    add_common_args,
    build_diag_context,
    build_support_cache,
    select_tiles,
    extract_dense_prompt,
    eval_iou,
    print_header,
    print_metric_table,
    aggregate_metrics,
    save_results,
)


# ═══════════════════════════════════════════════════════════════════
# AM Optimizer
# ═══════════════════════════════════════════════════════════════════


class ActivationMaximizer:
    """Optimize prompt to maximize decoder output — no GT matching."""

    def __init__(self, ctx: DiagContext, query_emb, sup_feat, sup_mask, device):
        self.ctx = ctx
        self.model = ctx.model
        # Detach to prevent backward-graph-reuse errors across optim steps
        self.query_emb = query_emb.detach()
        self.sup_feat = sup_feat.detach()
        self.sup_mask = sup_mask.detach()
        self.device = device

        # Pre-compute support proto
        self.support_proto = self.model._compute_support_prototype(
            sup_feat, sup_mask
        ) if self.model.bypass_head is None else None

        # Disable category injection for clean optimization
        self._saved_cat = self.model.sam_decoder._category_enabled
        self.model.sam_decoder._category_enabled = False

    def decode(self, prompt):
        if self.model.bypass_head is not None:
            return self.model.bypass_head(prompt)
        else:
            sparse_token = prompt.mean(dim=(2, 3))
            low_res, _ = self.model.sam_decoder(
                self.query_emb, sparse_token, prompt,
                support_prototype=self.support_proto,
            )
            return low_res  # [1, 1, 256, 256]

    def optimize(self, objective: str = "fg_activation",
                 l2_weight: float = 0.001, steps: int = 200, lr: float = 0.01) -> dict:
        """Optimize prompt from random noise to maximize decoder output.

        :param objective: "fg_activation" | "spatial_concentration" | "channel_sparsity"
        :param l2_weight: L2 regularization on prompt to prevent explosion.
        """
        C, H, W = 256, 64, 64
        prompt = torch.randn(1, C, H, W, device=self.device) * 0.1
        prompt = prompt.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([prompt], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

        trace = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = self.decode(prompt)  # [1, 1, 256, 256]

            if objective == "fg_activation":
                # Maximize total FG activation
                prob = logits[0, 0].sigmoid()
                obj = -prob.mean()  # negative because we minimize
                reg = l2_weight * prompt.pow(2).mean()
                loss = obj + reg

            elif objective == "spatial_concentration":
                # Maximize (max activation / mean activation) → encourage peaks
                prob = logits[0, 0].sigmoid()
                spatial_max = prob.max()
                spatial_mean = prob.mean()
                obj = -(spatial_max / (spatial_mean + 1e-6))
                reg = l2_weight * prompt.pow(2).mean()
                loss = obj + reg

            elif objective == "channel_sparsity":
                # Encourage sparse channel usage: minimize entropy of channel energies
                ch_energy = prompt[0].pow(2).mean(dim=(1, 2))  # [C]
                ch_prob = ch_energy / (ch_energy.sum() + 1e-10)
                ent = -(ch_prob * torch.log(ch_prob + 1e-10)).sum()
                # Minimize entropy → sparser channels; but also maximize output
                prob = logits[0, 0].sigmoid()
                obj = -prob.mean() + 0.1 * ent
                reg = l2_weight * prompt.pow(2).mean()
                loss = obj + reg

            else:
                raise ValueError(f"Unknown objective: {objective}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_([prompt], 1.0)
            optimizer.step()
            scheduler.step()
            del logits

            if step % 20 == 0 or step == steps - 1:
                with torch.no_grad():
                    prob_out = self.decode(prompt)[0, 0].sigmoid()
                    trace.append({
                        "step": step,
                        "loss": float(loss),
                        "fg_mean": float(prob_out.mean()),
                        "fg_max": float(prob_out.max()),
                        "prompt_l2": float(prompt.pow(2).mean().sqrt()),
                    })

        with torch.no_grad():
            final_logits = self.decode(prompt)
            final_prob = final_logits[0, 0].sigmoid()

        return {
            "am_prompt": prompt.detach().clone(),
            "final_prob_mean": float(final_prob.mean()),
            "final_prob_max": float(final_prob.max()),
            "final_prob_spatial": final_prob.cpu().numpy().tolist(),  # [256, 256]
            "trace": trace,
            "objective": objective,
            "n_steps": steps,
        }

    def restore(self):
        self.model.sam_decoder._category_enabled = self._saved_cat


# ═══════════════════════════════════════════════════════════════════
# AM Prompt Analysis
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def analyze_am_prompt(
    am_prompt: torch.Tensor,
    actual_prompt: torch.Tensor,
) -> dict:
    """Compare AM-optimized prompt with actual prompt.

    :param am_prompt: [1, C, H, W] optimized by AM.
    :param actual_prompt: [1, C, H, W] from PromptFusion.
    """
    C = am_prompt.shape[1]
    a = am_prompt[0].cpu().float()
    p = actual_prompt[0].cpu().float()

    # Cosine similarity
    cos_global = float(F.cosine_similarity(
        a.reshape(1, -1), p.reshape(1, -1)
    ))

    # Channel energy Gini
    a_energy = a.pow(2).mean(dim=(1, 2))
    p_energy = p.pow(2).mean(dim=(1, 2))

    # Effective rank
    a_svd = torch.svd(a.reshape(C, -1))[1]
    p_svd = torch.svd(p.reshape(C, -1))[1]
    a_eff = _eff_rank(a_svd)
    p_eff = _eff_rank(p_svd)

    # Spatial frequency
    a_sf = _spatial_freq(am_prompt)
    p_sf = _spatial_freq(actual_prompt)

    # Prominent channels: how many channels have > 10% of max energy?
    a_active = int((a_energy > 0.1 * a_energy.max()).sum())
    p_active = int((p_energy > 0.1 * p_energy.max()).sum())

    return {
        "cos_to_actual": cos_global,
        "am_eff_rank": float(a_eff),
        "actual_eff_rank": float(p_eff),
        "am_spatial_freq": float(a_sf),
        "actual_spatial_freq": float(p_sf),
        "am_n_active_channels": a_active,
        "actual_n_active_channels": p_active,
        "magnitude_ratio": float(a.pow(2).mean().sqrt() / max(p.pow(2).mean().sqrt(), 1e-8)),
    }


def _eff_rank(s: torch.Tensor) -> float:
    s2 = s ** 2
    cumsum = torch.cumsum(s2, dim=0) / s2.sum()
    return float((cumsum < 0.9).sum() + 1)


def _spatial_freq(x: torch.Tensor) -> float:
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                             dtype=torch.float32, device=x.device)
    lap = laplacian.unsqueeze(0).unsqueeze(0)
    total_var = 0.0
    for c in range(min(x.shape[1], 32)):
        ch = x[0, c].unsqueeze(0).unsqueeze(0)
        lap_resp = F.conv2d(ch, lap, padding=1)
        total_var += float(lap_resp.var())
    return total_var / min(x.shape[1], 32)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Activation Maximization — What prompt does the decoder want?"
    )
    add_common_args(parser)
    parser.add_argument("--optim-steps", type=int, default=200)
    parser.add_argument("--objectives", nargs="+",
                       default=["fg_activation", "spatial_concentration"],
                       help="AM objectives to test")
    parser.add_argument("--l2-weight", type=float, default=0.001,
                       help="L2 regularization weight")
    args = parser.parse_args()

    ctx = build_diag_context(args, require_adapter=True, split="val")
    build_support_cache(ctx)
    select_tiles(ctx, num_tiles=args.num_tiles)

    per_tile = []

    for tile_idx, present_classes in tqdm(ctx.selected_tiles, desc="activation max"):
        sample = ctx.dataset[tile_idx]
        H, W = sample["image"].shape[1], sample["image"].shape[2]

        xx, _ = preprocess_image(sample["image"])
        query_emb = ctx.backbone(xx.unsqueeze(0).to(ctx.device))["image_embedding"]
        if ctx.adapter is not None:
            query_emb = ctx.adapter(query_emb)

        main_cls = max(present_classes, key=lambda c:
                       ctx.dataset.get_class_mask(tile_idx, c).sum())
        gt = ctx.dataset.get_class_mask(tile_idx, main_cls).numpy().astype(bool)
        sup_data = ctx.support_cache.get(main_cls)
        if sup_data is None:
            continue
        sup_feat, sup_mask = sup_data

        # Get actual prompt for comparison
        actual_prompt, _, _ = extract_dense_prompt(ctx, query_emb, sup_feat, sup_mask)

        am = ActivationMaximizer(ctx, query_emb, sup_feat, sup_mask, ctx.device)

        tile_entry = {"tile_idx": tile_idx, "class_id": main_cls,
                      "class_name": ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}")}

        for objective in args.objectives:
            result = am.optimize(
                objective=objective, l2_weight=args.l2_weight,
                steps=args.optim_steps,
            )
            analysis = analyze_am_prompt(result["am_prompt"], actual_prompt)

            tile_entry[f"{objective}_fg_mean"] = result["final_prob_mean"]
            tile_entry[f"{objective}_fg_max"] = result["final_prob_max"]
            tile_entry[f"{objective}_cos_to_actual"] = analysis["cos_to_actual"]
            tile_entry[f"{objective}_eff_rank"] = analysis["am_eff_rank"]
            tile_entry[f"{objective}_spatial_freq"] = analysis["am_spatial_freq"]
            tile_entry[f"{objective}_n_active_ch"] = analysis["am_n_active_channels"]

        am.restore()
        per_tile.append(tile_entry)

    # Aggregate
    summary = {"n_tiles": len(per_tile)}
    summary["metrics"] = aggregate_metrics(per_tile)

    s_path, p_path = save_results(ctx.out_dir, summary, per_tile)

    # Print report
    print_header("Activation Maximization — Decoder Prompt Preferences")
    for objective in args.objectives:
        print(f"\n  ┌─ Objective: {objective}")
        keys = [k for k in summary.get("metrics", {})
                if k.startswith(f"{objective}_")]
        print_metric_table({k: summary["metrics"][k] for k in keys})

    print(f"\n[Summary] {s_path}")
    print(f"[Per-tile] {p_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
