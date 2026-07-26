"""
PromptFusion-Only Training Probe — 因果验证 Rank Collapse 是原因还是结果
==========================================================================

核心实验: 冻结 Decoder + SPG + Geo, 只训练 PromptFusion, 观察 effective rank 是否升高.

假设:
  - 如果 rank 从 2 升到 8/16/32 → PromptFusion 本身没问题, 只是以前梯度到不了
  - 如果 rank 始终 ≈ 2      → PromptFusion 结构有问题 (需要重新设计)

方法:
  1. 加载 Stage 2 checkpoint
  2. 冻结所有参数, 除 PromptFusion
  3. 用 Stage 2 的 episodic 训练方式训练 PromptFusion
  4. 每 N 步在 validation tiles 上测量:
     - Effective rank (SVD, 90% variance)
     - Per-channel energy Gini
     - mIoU
  5. 对比初始 vs 最终 rank

用法:
    python tools/debug/train_promptfusion_only.py \
        --stage2-ckpt runs/stage2_fold1_k5_seed42/best_model.pt \
        --data-root data/iSAID-5i --fold 1 --mode novel --k-shot 5 \
        --train-steps 200 --eval-interval 20 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5I_CATEGORIES
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image, resize_mask

from tools.analysis._decoder_diag_base import (
    DiagContext,
    build_diag_context,
    build_support_cache,
    select_tiles,
    extract_dense_prompt,
    print_header,
    save_results,
    aggregate_metrics,
)


# ═══════════════════════════════════════════════════════════════════
# Effective Rank (per-sample SVD)
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_effective_rank(prompt: torch.Tensor) -> float:
    """Per-sample effective rank: # of singular values for 90% variance.

    :param prompt: [1, C, H, W]
    :return: effective rank (float)
    """
    C = prompt.shape[1]
    flat = prompt[0].reshape(C, -1).float()  # [C, HW]
    _, S, _ = torch.svd(flat)
    s2 = S ** 2
    cumsum = torch.cumsum(s2, dim=0) / s2.sum()
    return float((cumsum < 0.9).sum() + 1)


@torch.no_grad()
def compute_channel_gini(prompt: torch.Tensor) -> float:
    """Gini coefficient of per-channel energy distribution.

    :param prompt: [1, C, H, W]
    """
    C = prompt.shape[1]
    energy = prompt[0].pow(2).reshape(C, -1).mean(dim=1)  # [C]
    sorted_e = torch.sort(energy)[0]
    n = len(sorted_e)
    if sorted_e.sum() < 1e-10:
        return 0.0
    idx = torch.arange(1, n + 1, device=sorted_e.device, dtype=torch.float32)
    return float(1 - 2 * torch.sum(sorted_e * idx) / (n * sorted_e.sum()) + (n + 1) / n)


# ═══════════════════════════════════════════════════════════════════
# Validation probe
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def probe_rank_and_iou(
    ctx: DiagContext,
    eval_tiles: list[tuple[int, list[int]]],
) -> dict:
    """Measure effective rank, channel Gini, and IoU on eval tiles.

    :return: aggregated metrics dict.
    """
    ranks = []
    ginis = []
    ious = []

    for tile_idx, present_classes in eval_tiles:
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

        dense_prompt, _, _ = extract_dense_prompt(ctx, query_emb, sup_feat, sup_mask)

        # Effective rank
        ranks.append(compute_effective_rank(dense_prompt))

        # Channel Gini
        ginis.append(compute_channel_gini(dense_prompt))

        # IoU
        from tools.analysis._decoder_diag_base import eval_iou
        ious.append(eval_iou(ctx.model, dense_prompt, query_emb, sup_feat, sup_mask, gt, H, W))

    return {
        "eff_rank": {"mean": float(np.mean(ranks)), "std": float(np.std(ranks))},
        "ch_gini": {"mean": float(np.mean(ginis)), "std": float(np.std(ginis))},
        "mIoU": {"mean": float(np.mean(ious)), "std": float(np.std(ious))},
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Train PromptFusion only — causal test of rank collapse"
    )
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--data-root", default="data/iSAID-5i")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--mode", default="novel")
    parser.add_argument("--k-shot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-eval-tiles", type=int, default=20,
                       help="Number of validation tiles for probing")
    parser.add_argument("--train-steps", type=int, default=200,
                       help="Total training steps")
    parser.add_argument("--eval-interval", type=int, default=20,
                       help="Probe rank every N steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate for PromptFusion")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[setup] device={device}")

    # ── Build context (loads model with frozen params) ──
    ctx = build_diag_context(args, require_adapter=True, split="val")
    build_support_cache(ctx)

    # ── Prepare eval tiles ──
    select_tiles(ctx, num_tiles=args.num_eval_tiles + 30)  # extra for training
    eval_tiles = ctx.selected_tiles[:args.num_eval_tiles]
    train_tiles = ctx.selected_tiles[args.num_eval_tiles:]

    model = ctx.model

    # ── Verify PromptFusion exists ──
    if model.prompt_fusion is None:
        print("ERROR: Model has no PromptFusion module. Nothing to train.")
        sys.exit(1)

    # ── Count PromptFusion params ──
    pf_params = sum(p.numel() for p in model.prompt_fusion.parameters())
    print(f"[model] PromptFusion params: {pf_params:,}")

    # ── Unfreeze only PromptFusion ──
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.prompt_fusion.parameters():
        p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[freeze] trainable params: {trainable:,} (should equal {pf_params:,})")
    assert trainable == pf_params, f"Expected {pf_params} trainable, got {trainable}"

    # ── Initial probe ──
    print_header("Initial Probe (before training)")
    init_probe = probe_rank_and_iou(ctx, eval_tiles)
    print(f"  Eff. Rank:  {init_probe['eff_rank']['mean']:.2f} ± {init_probe['eff_rank']['std']:.2f}")
    print(f"  Ch. Gini:   {init_probe['ch_gini']['mean']:.4f}")
    print(f"  mIoU:       {init_probe['mIoU']['mean']:.4f}")

    # ── Training ──
    print_header(f"Training PromptFusion ({args.train_steps} steps, lr={args.lr})")

    optimizer = torch.optim.Adam(model.prompt_fusion.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_steps)

    probe_log = []
    train_losses = []

    # Build training dataset (train split)
    train_ds = ISAID5iDataset(
        root=str(ctx.data_root), fold=ctx.fold, split="train", mode=ctx.mode,
    )

    # Pre-extract support features for efficiency
    train_support = {}  # {class_id: [(query_emb, gt, H, W), ...]}
    for cls in ctx.visible_classes:
        tiles = train_ds.class_to_tiles(cls)
        if not tiles:
            continue
        train_support[cls] = []
        for idx in tiles[:min(20, len(tiles))]:  # cap per class
            sample = train_ds[idx]
            gt = train_ds.get_class_mask(idx, cls)
            if gt is None or gt.sum() < 100:
                continue
            xx, _ = preprocess_image(sample["image"])
            qe = ctx.backbone(xx.unsqueeze(0).to(device))["image_embedding"]
            if ctx.adapter is not None:
                qe = ctx.adapter(qe)
            train_support[cls].append((qe, gt, sample["image"].shape[1], sample["image"].shape[2]))

    print(f"[data] training tiles per class: {[(c, len(v)) for c, v in train_support.items()]}")

    pbar = tqdm(range(args.train_steps), desc="training PF")
    for step in pbar:
        # Sample a random class and query tile
        cls = random.choice(list(train_support.keys()))
        if not train_support[cls]:
            continue
        qe, gt_np, H, W = random.choice(train_support[cls])
        gt = gt_np.numpy().astype(bool)

        sup_data = ctx.support_cache.get(cls)
        if sup_data is None:
            continue
        sup_feat, sup_mask = sup_data

        # Forward: build prompt through PromptFusion
        support_memory = model.support_encoder(sup_feat, sup_mask)
        geometric_prior = model.geometric_prior(qe, support_memory) if model.geometric_prior is not None else None
        dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
        spg_out = model.spg(qe, support_memory, dense_pe)
        semantic_prior = spg_out.semantic_prior

        # PromptFusion (only trainable part)
        if geometric_prior is not None:
            dense_prompt, sparse_token = model.prompt_fusion(geometric_prior, semantic_prior)
        else:
            dense_prompt = semantic_prior
            sparse_token = dense_prompt.mean(dim=(2, 3))

        # Decoder (frozen — no grad)
        with torch.no_grad():
            support_proto = model._compute_support_prototype(sup_feat, sup_mask)
            saved_cat = model.sam_decoder._category_enabled
            model.sam_decoder._category_enabled = False
            low_res, _ = model.sam_decoder(qe, sparse_token, dense_prompt,
                                           support_prototype=support_proto)
            model.sam_decoder._category_enabled = saved_cat

        # Loss
        gt_t = torch.from_numpy(gt.astype(np.float32)).to(device)
        gt_256 = F.interpolate(gt_t.unsqueeze(0).unsqueeze(0), (256, 256), mode="area")[0, 0]
        fg = low_res[0, 0]
        bce = F.binary_cross_entropy_with_logits(fg, gt_256)
        prob = fg.sigmoid()
        inter = (prob * gt_256).sum()
        dice = 1.0 - (2 * inter + 1) / (prob.sum() + gt_256.sum() + 1)
        loss = bce + dice

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.prompt_fusion.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(float(loss))
        pbar.set_postfix(loss=f"{float(loss):.4f}")

        # ── Periodic probe ──
        if (step + 1) % args.eval_interval == 0 or step == 0 or step == args.train_steps - 1:
            probe = probe_rank_and_iou(ctx, eval_tiles)
            probe["step"] = step + 1
            probe["train_loss"] = float(np.mean(train_losses[-10:])) if train_losses else 0
            probe_log.append(probe)

            pbar.write(
                f"  step {step+1:4d} | "
                f"rank={probe['eff_rank']['mean']:.1f}±{probe['eff_rank']['std']:.1f} | "
                f"gini={probe['ch_gini']['mean']:.3f} | "
                f"IoU={probe['mIoU']['mean']:.4f}±{probe['mIoU']['std']:.4f} | "
                f"loss={probe['train_loss']:.4f}"
            )

    # ── Final probe ──
    print_header("Final Probe (after training)")
    final_probe = probe_rank_and_iou(ctx, eval_tiles)
    print(f"  Eff. Rank:  {final_probe['eff_rank']['mean']:.2f} ± {final_probe['eff_rank']['std']:.2f}")
    print(f"  Ch. Gini:   {final_probe['ch_gini']['mean']:.4f}")
    print(f"  mIoU:       {final_probe['mIoU']['mean']:.4f}")

    # ── Summary ──
    delta_rank = final_probe['eff_rank']['mean'] - init_probe['eff_rank']['mean']
    delta_iou = final_probe['mIoU']['mean'] - init_probe['mIoU']['mean']

    print_header("Causal Diagnosis: Rank Collapse — Cause or Symptom?")
    print(f"  Δ Eff. Rank:  {delta_rank:+.2f}  (initial={init_probe['eff_rank']['mean']:.1f} → final={final_probe['eff_rank']['mean']:.1f})")
    print(f"  Δ mIoU:       {delta_iou:+.4f}  (initial={init_probe['mIoU']['mean']:.4f} → final={final_probe['mIoU']['mean']:.4f})")

    if delta_rank > 2:
        print(f"\n  ★ Rank GREW from {init_probe['eff_rank']['mean']:.1f} to {final_probe['eff_rank']['mean']:.1f}")
        print(f"    → PromptFusion structure is FINE. Previous low rank was a GRADIENT problem.")
        print(f"    → Next step: fix gradient flow to PromptFusion (not redesign it).")
    elif delta_rank > 0.5:
        print(f"\n  ★ Rank grew slightly ({delta_rank:+.1f})")
        print(f"    → PromptFusion has some capacity to use more channels.")
        print(f"    → Gradient improvement would help, but PF structure may also limit expressivity.")
    else:
        print(f"\n  ★ Rank did NOT increase significantly ({delta_rank:+.1f})")
        print(f"    → PromptFusion STRUCTURE is the bottleneck (gradient or not).")
        print(f"    → Next step: redesign PromptFusion (residual projection, wider bottleneck, orthogonal constraint).")

    # ── Save ──
    probe_log.append({"step": "final", **final_probe})

    # Re-freeze everything for safe saving
    for p in model.parameters():
        p.requires_grad_(False)

    save_results(
        ctx.out_dir,
        {
            "init": init_probe,
            "final": final_probe,
            "delta_rank": delta_rank,
            "delta_iou": delta_iou,
            "n_train_steps": args.train_steps,
            "lr": args.lr,
            "train_loss_final": float(np.mean(train_losses[-20:])) if train_losses else 0,
        },
        probe_log,
    )

    print(f"\n[Output] {ctx.out_dir}")
    print("[Done]")


if __name__ == "__main__":
    main()
