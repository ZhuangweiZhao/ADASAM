"""
Dense Prompt 通道归因 v2 | Dense Prompt Channel Attribution v2.
================================================================

三个控制实验：
  1. Bottom-K only — 后半部分通道是否真的无用？
  2. Random-K baseline — Top-K 优势来自排序还是通道数减少？
  3. Trainable Gate — 模型自己会选择关闭哪些通道？

前提：先运行 diag_dense_prompt_classifiability.py 生成 channel_importance.pt

用法 | Usage:
    python tools/analysis/diag_channel_attribution.py \
        --stage2-ckpt runs/stage2_fold1_k5_seed42/best_model.pt \
        --data-root data/iSAID-5i --fold 1 --k-shot 5
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5I_CATEGORIES
from adasam.model.adasam_model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image, resize_mask


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def build_support_for_class(
    dataset, class_id: int, k_shot: int, device: torch.device,
    backbone: MobileSAMBackbone, adapter, rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    tiles = dataset.class_to_tiles(class_id)
    if len(tiles) < 1:
        return None
    scenes: dict[str, list[int]] = defaultdict(list)
    for idx in tiles:
        tile_id = dataset.tile_ids[idx]
        src = dataset._source_images.get(tile_id, str(idx))
        scenes[src].append(idx)
    chosen = []
    scene_list = list(scenes.keys())
    rng.shuffle(scene_list)
    for src in scene_list:
        if len(chosen) >= k_shot:
            break
        chosen.append(rng.choice(scenes[src]))
    if len(chosen) < 1:
        return None
    images, masks = [], []
    for idx in chosen:
        sample = dataset[idx]
        fg = dataset.get_class_mask(idx, class_id)
        if fg is None:
            continue
        x, _ = preprocess_image(sample["image"])
        images.append(x.to(device))
        masks.append(fg)
    if not images:
        return None
    feats = backbone(torch.stack(images, dim=0))["image_embedding"]
    if adapter is not None:
        feats = adapter(feats)
    masks_grid = torch.stack(
        [resize_mask(m, (feats.shape[2], feats.shape[3])).to(device) for m in masks], dim=0)
    if masks_grid.sum() < 1.0:
        return None
    return feats, masks_grid


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    return inter / max(union, 1.0)


def decode(model, q_emb, dp):
    """Run decoder → binary mask [256, 256]."""
    if model.bypass_head is not None:
        low_res = model.bypass_head(dp)
    else:
        low_res, _ = model.sam_decoder(q_emb, dp.mean(dim=(2, 3)), dp)
    return (low_res[0, 0] > 0).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Dense Prompt Channel Attribution v2")
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--importance-file", type=str, default=None)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--n-random", type=int, default=5,
                        help="Number of random subsets per K for Random-K baseline")
    parser.add_argument("--gate-iters", type=int, default=30,
                        help="Trainable gate optimization iterations")
    parser.add_argument("--skip-gate", action="store_true",
                        help="Skip trainable gate experiment (saves ~5 min)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)
    print(f"Device: {device}")

    # ── Load channel importance ──
    if args.importance_file:
        imp_path = Path(args.importance_file)
    else:
        imp_path = Path(args.stage2_ckpt).parent / "channel_importance.pt"
    if not imp_path.exists():
        print(f"ERROR: {imp_path} not found. Run diag_dense_prompt_classifiability.py first.")
        sys.exit(1)

    imp_data = torch.load(imp_path, map_location="cpu", weights_only=False)
    ranked_channels = imp_data["ranked_channels"].numpy()
    channel_importance = imp_data["channel_importance"].numpy()
    probe_acc = imp_data.get("probe_accuracy", 0.0)
    print(f"Loaded: {imp_path}  (probe acc={probe_acc:.1f}%)")

    # ── Load model ──
    ckpt_path = Path(args.stage2_ckpt)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    weights_path = str(_REPO_ROOT / cfg.get("backbone", {}).get(
        "checkpoint", "weights/mobile_sam.pt"))
    sam = build_mobile_sam(weights_path, cfg.get("backbone", {}).get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    model_cfg = AdaSAMModelConfig.from_dict(cfg)
    model = AdaSAMModel(sam, model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    adapter = None
    adapter_state = ckpt.get("cat_adapter")
    if adapter_state is not None:
        adapter_cfg = ckpt.get("config", {}).get("adapter", {})
        adapter = CATAdapter(dim=256, bottleneck=int(adapter_cfg.get("bottleneck", 64))).to(device)
        adapter.load_state_dict(adapter_state)
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad_(False)

    data_root = str(_REPO_ROOT / args.data_root) if not Path(args.data_root).is_absolute() else args.data_root
    dataset = ISAID5iDataset(root=data_root, fold=args.fold, split="val", mode="base")
    val_classes = sorted(dataset.visible_classes())
    cls_to_idx = {cls: i for i, cls in enumerate(val_classes)}

    tile_scores = []
    for idx in range(len(dataset)):
        classes_present = []
        for cls in val_classes:
            gt = dataset.get_class_mask(idx, cls)
            if gt is not None and gt.sum() > 100:
                classes_present.append(cls)
        if len(classes_present) >= 2:
            tile_scores.append((idx, len(classes_present), classes_present))
    tile_scores.sort(key=lambda x: -x[1])
    eval_offset = 30
    eval_tiles = tile_scores[eval_offset:eval_offset + args.n_eval]
    if len(eval_tiles) < args.n_eval:
        eval_tiles = tile_scores[-args.n_eval:] if len(tile_scores) >= args.n_eval else tile_scores
    print(f"Eval tiles: {len(eval_tiles)}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # Pre-compute all (q_emb, dp_full, gt_fg, sup_cls) for eval tiles
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("Pre-computing Dense Prompts for all eval samples ...")
    print("=" * 60)

    samples = []  # list of {q_emb, dp_full, gt_fg, sup_cls_idx}
    for tile_idx, n_cls_on_tile, cls_list in tqdm(eval_tiles, desc="pre-compute"):
        sample_data = dataset[tile_idx]
        x, _ = preprocess_image(sample_data["image"])
        gt_fg = np.zeros((256, 256), dtype=bool)
        for cls in val_classes:
            gt = dataset.get_class_mask(tile_idx, cls)
            if gt is not None:
                gt_fg = gt_fg | gt.numpy().astype(bool)
        if gt_fg.sum() < 100:
            continue
        with torch.no_grad():
            q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_emb = adapter(q_emb)
            dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
            for sup_cls in cls_list[:3]:
                sup_data = build_support_for_class(
                    dataset, sup_cls, args.k_shot, device, backbone, adapter, rng)
                if sup_data is None:
                    continue
                sup_feat, sup_mask = sup_data
                sm = model.support_encoder(sup_feat, sup_mask)
                gp = model.geometric_prior(q_emb, sm) if model.geometric_prior else None
                spg_out = model.spg(q_emb, sm, dense_pe)
                if model.prompt_fusion is not None and gp is not None:
                    dp_full, _ = model.prompt_fusion(gp, spg_out.semantic_prior)
                else:
                    dp_full = model._build_dense_prompt(sm, sup_feat, sup_mask)
                    if dp_full is None:
                        dp_full = spg_out.semantic_prior
                samples.append({
                    "q_emb": q_emb.cpu(),
                    "dp_full": dp_full.cpu(),
                    "gt_fg": gt_fg,
                    "sup_cls_idx": cls_to_idx[sup_cls],
                })
                del sup_feat, sup_mask, sm, spg_out, dp_full
                if gp is not None:
                    del gp
        del q_emb, dense_pe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    N_samples = len(samples)
    print(f"Pre-computed {N_samples} samples")

    # Build helper: apply channel mask → decode → FB-IoU
    def eval_mask(channel_indices):
        """Decode all samples with only `channel_indices` active, return mean FB-IoU."""
        mask = torch.zeros(256, device=device)
        mask[channel_indices] = 1.0
        mask_view = mask.view(1, 256, 1, 1)
        ious = []
        for s in samples:
            dp_m = s["dp_full"].to(device) * mask_view
            q = s["q_emb"].to(device)
            pred = decode(model, q, dp_m)
            ious.append(mask_iou(pred, s["gt_fg"]))
        return np.mean(ious) if ious else 0.0

    # ═══════════════════════════════════════════════════════════════
    # Experiment 0: Full baseline + Top-K / Drop-TopK (re-run from cache)
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("Experiment 0: Top-K only, Drop-TopK, Bottom-K only")
    print("=" * 60)

    K_VALUES = [0, 8, 16, 32, 64, 128, 256]

    topk_results = {}
    drop_results = {}
    bottom_results = {}

    # Compute full baseline first (K=256)
    ious_full = []
    for s in samples:
        pred = decode(model, s["q_emb"].to(device), s["dp_full"].to(device))
        ious_full.append(mask_iou(pred, s["gt_fg"]))
    full_fb_iou = np.mean(ious_full)

    for K in K_VALUES:
        if K == 256:
            topk_results[K] = full_fb_iou
            drop_results[K] = full_fb_iou
            bottom_results[K] = full_fb_iou
        elif K == 0:
            dp_zero = torch.zeros(1, 256, 64, 64, device=device)
            ious = []
            for s in samples:
                pred = decode(model, s["q_emb"].to(device), dp_zero)
                ious.append(mask_iou(pred, s["gt_fg"]))
            topk_results[K] = np.mean(ious)
            bottom_results[K] = np.mean(ious)
            drop_results[K] = full_fb_iou
        else:
            topk_idx = ranked_channels[:K]
            bottom_idx = ranked_channels[-K:]
            drop_idx = ranked_channels[K:]

            # Top-K
            ious_topk = []
            mask_t = torch.zeros(256, device=device)
            mask_t[topk_idx] = 1.0
            mt = mask_t.view(1, 256, 1, 1)
            for s in samples:
                pred = decode(model, s["q_emb"].to(device), s["dp_full"].to(device) * mt)
                ious_topk.append(mask_iou(pred, s["gt_fg"]))
            topk_results[K] = np.mean(ious_topk)

            # Bottom-K
            ious_bot = []
            mask_b = torch.zeros(256, device=device)
            mask_b[bottom_idx] = 1.0
            mb = mask_b.view(1, 256, 1, 1)
            for s in samples:
                pred = decode(model, s["q_emb"].to(device), s["dp_full"].to(device) * mb)
                ious_bot.append(mask_iou(pred, s["gt_fg"]))
            bottom_results[K] = np.mean(ious_bot)

            # Drop-TopK (keep rest)
            ious_drop = []
            mask_d = torch.ones(256, device=device)
            mask_d[topk_idx] = 0.0
            md = mask_d.view(1, 256, 1, 1)
            for s in samples:
                pred = decode(model, s["q_emb"].to(device), s["dp_full"].to(device) * md)
                ious_drop.append(mask_iou(pred, s["gt_fg"]))
            drop_results[K] = np.mean(ious_drop)

    print(f"\n  {'K':>5s}  {'Top-K':>10s}  {'Drop-TopK':>12s}  {'Bottom-K':>12s}  {'Retention':>10s}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*10}")
    for K in sorted(K_VALUES):
        if K == 256:
            continue
        tk = topk_results[K]
        dk = drop_results[K]
        bk = bottom_results[K]
        ret = tk / max(full_fb_iou, 1e-8) * 100
        print(f"  {K:>5d}  {tk:>10.4f}  {dk:>12.4f}  {bk:>12.4f}  {ret:>9.1f}%")

    print(f"\n  Full (256): {full_fb_iou:.4f}    Zero baseline: {topk_results[0]:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Experiment 1: Bottom-K vs Zero — are bottom channels noise or weak signal?
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'─' * 60}")
    print("Experiment 1: Bottom-K vs All-Zero")
    print("  (Bottom-K > Zero → bottom channels carry usable signal)")
    print("  (Bottom-K ≈ Zero → bottom channels are near-useless for decoder)")
    print("─" * 60)

    zero_fb = topk_results[0]
    for K in [8, 16, 32, 64, 128]:
        bk = bottom_results[K]
        ratio_to_zero = bk / max(zero_fb, 1e-8)
        excess = (bk - zero_fb) / max(full_fb_iou - zero_fb, 1e-8) * 100  # % of usable range
        bar = "█" * max(1, int(ratio_to_zero * 10))
        print(f"  Bottom-{K:>3d}: {bk:.4f}  ({ratio_to_zero:.2f}× zero, {excess:.0f}% of full−zero gap)  {bar}")

    # ═══════════════════════════════════════════════════════════════
    # Experiment 2: Random-K baseline
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("Experiment 2: Random-K baseline (n={args.n_random} per K)")
    print("  (Top-K outside Random CI → linear probe ranking is causal)")
    print("  (Top-K within Random CI → gain is from channel count reduction)")
    print("=" * 60)

    RANDOM_K = [16, 32, 64, 128]
    print(f"\n  {'K':>5s}  {'Top-K':>10s}  {'Random mean±std':>18s}  {'Z-score':>9s}  {'Verdict':>30s}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*18}  {'─'*9}  {'─'*30}")

    for K in RANDOM_K:
        random_vals = []
        for _ in range(args.n_random):
            rand_idx = np_rng.choice(256, size=K, replace=False)
            rv = eval_mask(rand_idx)
            random_vals.append(rv)
        r_mean = np.mean(random_vals)
        r_std = np.std(random_vals)
        t_val = topk_results[K]
        z_score = (t_val - r_mean) / max(r_std, 1e-8)

        if z_score > 3:
            verdict = "✅ Top-K >> Random (ranking WORKS)"
        elif z_score > 1.5:
            verdict = "⚠️  Top-K > Random (weak advantage)"
        elif z_score > -1.5:
            verdict = "≈ No difference (channel count effect)"
        else:
            verdict = "❌ Top-K WORSE than random"

        print(f"  {K:>5d}  {t_val:>10.4f}  {r_mean:>8.4f} ± {r_std:<8.4f}  {z_score:>+8.2f}  {verdict}")

    # ═══════════════════════════════════════════════════════════════
    # Experiment 3: Trainable Channel Gate
    # ═══════════════════════════════════════════════════════════════

    if not args.skip_gate:
        print(f"\n{'=' * 60}")
        print("Experiment 3: Trainable Channel Gate")
        print("  (Which channels does the model choose to shut off?)")
        print("=" * 60)

        # Initialize gate: all channels start at 0.5 (sigmoid(0))
        gate_raw = nn.Parameter(torch.zeros(256, device=device))
        optimizer = torch.optim.Adam([gate_raw], lr=0.1)
        # Pre-collect tensors on GPU for fast iteration
        all_dp = torch.cat([s["dp_full"] for s in samples], dim=0).to(device)  # [N, 256, 64, 64]
        all_q = torch.cat([s["q_emb"] for s in samples], dim=0).to(device)      # [N, 256, 64, 64]
        all_gt = [s["gt_fg"] for s in samples]

        N = all_dp.shape[0]
        print(f"  Training gate on {N} samples, {args.gate_iters} iters ...")

        gate_history = []  # track gate values over iterations
        for it in range(args.gate_iters):
            gate = torch.sigmoid(gate_raw)
            gated_dp = all_dp * gate.view(1, 256, 1, 1)

            total_loss = 0.0
            total_iou = 0.0
            for i in range(N):
                dp_i = gated_dp[i:i+1]
                q_i = all_q[i:i+1]
                if model.bypass_head is not None:
                    low_res = model.bypass_head(dp_i)
                else:
                    low_res, _ = model.sam_decoder(q_i, dp_i.mean(dim=(2, 3)), dp_i)
                pred = low_res[0, 0]
                gt_t = torch.from_numpy(all_gt[i].astype(np.float32)).to(device)
                # Resize pred to GT size if needed
                if pred.shape != gt_t.shape:
                    pred = F.interpolate(pred[None, None], size=gt_t.shape,
                                         mode="bilinear", align_corners=False)[0, 0]
                loss = F.binary_cross_entropy_with_logits(pred, gt_t)
                total_loss += loss
                with torch.no_grad():
                    total_iou += mask_iou((pred > 0).cpu().numpy(), all_gt[i])

            avg_loss = total_loss / N
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()

            with torch.no_grad():
                gate_val = torch.sigmoid(gate_raw).detach()
                n_active = int((gate_val > 0.3).sum().item())
                gate_history.append(gate_val.cpu().numpy().copy())

            if (it + 1) % 10 == 0:
                print(f"    iter {it+1:>3d}: loss={avg_loss.item():.4f}  "
                      f"FB-IoU={total_iou/N:.4f}  active_ch={n_active}")

        # Final gate analysis
        final_gate = torch.sigmoid(gate_raw).detach().cpu().numpy()
        gate_sorted_idx = np.argsort(-final_gate)  # high gate → low gate

        # Correlation: gate value vs linear probe importance (simple Spearman)
        def spearmanr(x, y):
            from scipy.stats import spearmanr as _sp
            return _sp(x, y)
        try:
            corr, pval = spearmanr(final_gate, channel_importance)
        except ImportError:
            # Fallback: Pearson on ranks
            rx = np.argsort(np.argsort(final_gate))
            ry = np.argsort(np.argsort(channel_importance))
            corr = np.corrcoef(rx, ry)[0, 1]
            pval = float('nan')
        print(f"\n  Spearman correlation(gate, linear_probe_importance): ρ={corr:.3f} (p={pval})")

        # Overlap: bottom-32 by gate vs bottom-32 by probe
        gate_bottom32 = set(gate_sorted_idx[-32:].tolist())
        probe_bottom32 = set(ranked_channels[-32:].tolist())
        overlap32 = len(gate_bottom32 & probe_bottom32)
        print(f"  Gate bottom-32 ∩ Probe bottom-32: {overlap32}/32")
        print(f"  Gate bottom-32 ∩ Probe top-32:   {len(gate_bottom32 & set(ranked_channels[:32].tolist()))}/32")

        # How many channels did gate shut off?
        n_off = int((final_gate < 0.1).sum())
        n_on = int((final_gate > 0.9).sum())
        print(f"  Channels fully OFF (<0.1): {n_off}")
        print(f"  Channels fully ON  (>0.9): {n_on}")

        # Are shut-off channels the same as linear probe's bottom?
        if n_off > 0:
            off_idx = set(np.where(final_gate < 0.1)[0].tolist())
            off_in_probe_bottom = len(off_idx & set(ranked_channels[-n_off:].tolist()))
            print(f"  Of {n_off} shut-off channels, {off_in_probe_bottom} are in probe bottom-{n_off} "
                  f"({off_in_probe_bottom/max(n_off,1)*100:.0f}%)")

        # Decode with learned gate
        gate_final_t = torch.from_numpy(final_gate).float().to(device).view(1, 256, 1, 1)
        ious_gated = []
        for s in samples:
            dp_g = s["dp_full"].to(device) * gate_final_t
            q = s["q_emb"].to(device)
            pred = decode(model, q, dp_g)
            ious_gated.append(mask_iou(pred, s["gt_fg"]))
        mean_gated = np.mean(ious_gated) if ious_gated else 0.0
        print(f"\n  FB-IoU with learned gate: {mean_gated:.4f}  (full={full_fb_iou:.4f}, "
              f"Δ={((mean_gated-full_fb_iou)/max(full_fb_iou,1e-8)*100):+.1f}%)")

        del all_dp, all_q, gate_raw
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("SUMMARY: Three Hypotheses Tested")
    print("=" * 60)

    # H1: Bottom channels are toxic
    bk128 = bottom_results.get(128, 0)
    z128 = topk_results[0]
    if bk128 < z128 * 1.15:
        print(f"\n  H1 (Bottom channels ≈ zero): ✅ CONFIRMED — Bottom-128 FB-IoU ({bk128:.4f}) ≈ zero ({z128:.4f})")
    elif bk128 < full_fb_iou * 0.8:
        print(f"\n  H1 (Bottom channels ≈ weak): ⚠️  PARTIAL — Bottom-128 has some signal but much weaker than full")
    else:
        print(f"\n  H1 (Bottom channels ≈ toxic): ❌ REJECTED — Bottom-128 ({bk128:.4f}) ≈ full ({full_fb_iou:.4f})")

    # H2: Top-K advantage is from ranking, not channel count → see Exp 2 Z-scores
    print(f"\n  H2 (Top-K advantage from ranking): See Experiment 2 Z-scores above")
    print(f"  H3 (Gate learns to shut off probe-bottom channels): See Experiment 3 overlap above")
    print(f"\n  Key metric: If Bottom-128 ≈ Zero AND Top-K >> Random-K")
    print(f"    → linear probe ranking is CAUSAL for decoder performance")
    print(f"    → Train Stage 2 with channel sparsity (gating loss) on Dense Prompt")

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
