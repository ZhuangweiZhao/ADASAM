"""
PromptFusion Conv1×1 瓶颈验证 | PromptFusion Conv1×1 Bottleneck Verification.
===============================================================================

验证 Conv1×1 (512→256) 到底是在压缩信息还是丢失信息。

三个实验:
  Exp A: CKA — Concat vs Conv1×1 表示的相似度
  Exp B: 重建 — 训练 MLP 从 Conv1×1 输出恢复原始 Concat
  Exp C: 权重分析 — Conv1×1 权重矩阵的奇异值谱

如果 CKA≈1 且重建误差低 → Conv1×1 只是压缩 (lossless)
如果 CKA<<1 且重建误差高 → Conv1×1 在丢弃信息 (lossy)

用法 | Usage:
    python tools/analysis/diag_promptfusion_bottleneck.py \
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
# CKA Implementation
# ═══════════════════════════════════════════════════════════════════

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two representations.

    X: [N, d1] or [d1, N] — will be treated as features × samples
    Y: [N, d2] or [d2, N]
    Returns scalar in [0, 1].
    """
    # Ensure shape is [features, samples]
    if X.shape[0] < X.shape[1]:
        X = X.T  # [d1, N]
    if Y.shape[0] < Y.shape[1]:
        Y = Y.T  # [d2, N]

    # Center
    X = X - X.mean(dim=1, keepdim=True)
    Y = Y - Y.mean(dim=1, keepdim=True)

    # Gram matrices
    K = X.T @ X  # [N, N]
    L = Y.T @ Y  # [N, N]

    # HSIC
    hsic = torch.sum(K * L)

    # Normalization
    norm_x = torch.sqrt(torch.sum(K * K))
    norm_y = torch.sqrt(torch.sum(L * L))
    if norm_x < 1e-10 or norm_y < 1e-10:
        return 0.0

    return float(hsic / (norm_x * norm_y))


def svcca(X: torch.Tensor, Y: torch.Tensor, n_components: int = 64) -> float:
    """SVCCA: SVD → top-k components → CCA → mean correlation.

    X: [d1, N], Y: [d2, N]
    """
    d1, N = X.shape
    d2 = Y.shape[1] if Y.ndim == 2 else Y.shape[0]

    # Ensure [features, samples]
    if X.shape[1] != N:
        X = X.T
    if Y.shape[1] != N:
        Y = Y.T

    # Center
    X = X - X.mean(dim=1, keepdim=True)
    Y = Y - Y.mean(dim=1, keepdim=True)

    # SVD to reduce dimension
    k_x = min(n_components, X.shape[0])
    k_y = min(n_components, Y.shape[0])

    Ux, Sx, Vx = torch.linalg.svd(X, full_matrices=False)
    Uy, Sy, Vy = torch.linalg.svd(Y, full_matrices=False)

    X_reduced = Ux[:, :k_x].T @ X  # [k_x, N]
    Y_reduced = Uy[:, :k_y].T @ Y  # [k_y, N]

    # CCA via SVD of cross-covariance
    Sigma_xx = X_reduced @ X_reduced.T / (N - 1) + 1e-6 * torch.eye(k_x, device=X.device)
    Sigma_yy = Y_reduced @ Y_reduced.T / (N - 1) + 1e-6 * torch.eye(k_y, device=Y.device)
    Sigma_xy = X_reduced @ Y_reduced.T / (N - 1)

    # Whitening
    Lx = torch.linalg.cholesky(Sigma_xx)
    Ly = torch.linalg.cholesky(Sigma_yy)

    # Canonical correlations = singular values of inv(Lx) @ Sigma_xy @ inv(Ly).T
    whitened = torch.linalg.solve(Lx, Sigma_xy)
    whitened = torch.linalg.solve(Ly, whitened.T).T
    _, S_cca, _ = torch.linalg.svd(whitened, full_matrices=False)

    return float(S_cca[:min(k_x, k_y)].mean().item())


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


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PromptFusion Bottleneck Verification")
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--recon-epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    print(f"Device: {device}")

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

    pf = model.prompt_fusion
    if pf is None or pf.mode != "concat":
        print("ERROR: PromptFusion not enabled or not concat mode")
        sys.exit(1)

    data_root = str(_REPO_ROOT / args.data_root) if not Path(args.data_root).is_absolute() else args.data_root
    dataset = ISAID5iDataset(root=data_root, fold=args.fold, split="val", mode="base")
    val_classes = sorted(dataset.visible_classes())

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
    n_queries = min(args.n_queries, len(tile_scores))
    selected = tile_scores[:n_queries]
    print(f"Multi-class tiles: {len(tile_scores)}, using {n_queries}")

    # ═══════════════════════════════════════════════════════════════
    # Collect (concat_512, conv1_out_256) pairs
    # ═══════════════════════════════════════════════════════════════

    concat_list = []      # [512, H*W] per sample
    conv1_list = []       # [256, H*W] per sample
    labels = []

    print(f"\nCollecting Concat & Conv1×1 intermediates ...")
    for tile_idx, n_cls, cls_list in tqdm(selected, desc="collecting"):
        sample = dataset[tile_idx]
        x, _ = preprocess_image(sample["image"])
        with torch.no_grad():
            q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_emb = adapter(q_emb)
            dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
            for sup_cls in cls_list[:4]:
                sup_data = build_support_for_class(
                    dataset, sup_cls, args.k_shot, device, backbone, adapter, rng)
                if sup_data is None:
                    continue
                sup_feat, sup_mask = sup_data
                sm = model.support_encoder(sup_feat, sup_mask)
                gp = model.geometric_prior(q_emb, sm) if model.geometric_prior else None
                sp = model.spg(q_emb, sm, dense_pe).semantic_prior
                if gp is None:
                    continue

                concat = torch.cat([gp, sp], dim=1)  # [1, 512, 64, 64]
                conv1_out = F.conv2d(concat, pf.fusion_conv[0].weight, bias=None)  # [1, 256, 64, 64]

                Cc, H, W = concat.shape[1], concat.shape[2], concat.shape[3]
                Co = conv1_out.shape[1]

                concat_flat = concat[0].reshape(Cc, -1).cpu()  # [512, 4096]
                conv1_flat = conv1_out[0].reshape(Co, -1).cpu()  # [256, 4096]

                # Subsample spatial positions to keep CKA feasible
                n_spatial = H * W
                n_sample = min(n_spatial, 2048)
                idx_sample = torch.randperm(n_spatial)[:n_sample]

                concat_list.append(concat_flat[:, idx_sample])  # [512, ~2048]
                conv1_list.append(conv1_flat[:, idx_sample])    # [256, ~2048]
                labels.append(sup_cls)

                del gp, sp, concat, conv1_out, sup_feat, sup_mask, sm

        del q_emb, dense_pe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    N = len(concat_list)
    print(f"Collected {N} sample pairs")

    # ═══════════════════════════════════════════════════════════════
    # Exp A: CKA — Concat vs Conv1×1
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 72}")
    print("EXP A: CKA / SVCCA — Concat vs Conv1×1")
    print("=" * 72)

    cka_vals = []
    svcca_vals = []
    for i in range(N):
        X = concat_list[i].to(device)   # [512, S]
        Y = conv1_list[i].to(device)    # [256, S]
        cka_vals.append(linear_cka(X, Y))
        try:
            svcca_vals.append(svcca(X, Y, n_components=min(64, X.shape[1])))
        except Exception:
            pass

    mean_cka = np.mean(cka_vals)
    std_cka = np.std(cka_vals)
    mean_svcca = np.mean(svcca_vals) if svcca_vals else 0.0

    print(f"\n  Linear CKA: {mean_cka:.4f} ± {std_cka:.4f}")
    print(f"  SVCCA (top-64): {mean_svcca:.4f}")

    if mean_cka > 0.95:
        print(f"  → ✅ Conv1×1 preserves representation structure (near-lossless compression)")
    elif mean_cka > 0.7:
        print(f"  → ⚠️  Partial structure preservation — some information transformed")
    elif mean_cka > 0.4:
        print(f"  → ❌ Significant structure change — substantial information loss")
    else:
        print(f"  → ‼️  Near-orthogonal representations — catastrophic information loss")

    # CKA between different samples as baseline
    inter_sample_cka = []
    for i in range(min(N, 10)):
        for j in range(i + 1, min(N, 10)):
            inter_sample_cka.append(linear_cka(concat_list[i].to(device), concat_list[j].to(device)))
    baseline_cka = np.mean(inter_sample_cka) if inter_sample_cka else 0.0
    print(f"\n  Baseline CKA (different samples, concat): {baseline_cka:.4f}")
    print(f"  CKA(concat, conv1) / CKA(baseline): {mean_cka/max(baseline_cka,1e-8):.2f}×")

    # ═══════════════════════════════════════════════════════════════
    # Exp B: Reconstruction — MLP(256 → 512) recover Concat
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 72}")
    print("EXP B: Reconstruction — MLP(256→512→512) recover Concat from Conv1×1")
    print("=" * 72)

    # Train/test split by sample index (80/20)
    n_train = int(N * 0.8)
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, N))

    # Prepare data: each sample = [S, dim] matrix
    # We'll do per-pixel reconstruction: Conv1×1[256] → predict Concat[512]
    # Stack all training samples' pixels
    X_train_pixels = torch.cat([conv1_list[i].T for i in train_idx], dim=0)  # [total_pixels, 256]
    Y_train_pixels = torch.cat([concat_list[i].T for i in train_idx], dim=0)  # [total_pixels, 512]

    # Train reconstruction MLP
    recon_mlp = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 512),
    ).to(device)
    opt = torch.optim.Adam(recon_mlp.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.recon_epochs)

    # Use batches for speed
    batch_size = 4096
    n_pixels = X_train_pixels.shape[0]

    for ep in range(args.recon_epochs):
        perm = torch.randperm(n_pixels)
        total_loss = 0.0
        n_batches = 0
        for b in range(0, n_pixels, batch_size):
            idx = perm[b:b + batch_size]
            xb = X_train_pixels[idx].to(device)
            yb = Y_train_pixels[idx].to(device)
            pred = recon_mlp(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if (ep + 1) % 100 == 0:
            print(f"    epoch {ep+1}: MSE={total_loss/n_batches:.6f}")

    # Evaluate reconstruction
    recon_mlp.eval()
    # Baseline: predict mean of concat
    with torch.no_grad():
        # Test MSE
        test_mse = 0.0
        test_cos = 0.0
        n_test_pixels = 0
        for i in test_idx:
            x_i = conv1_list[i].T.to(device)  # [S, 256]
            y_i = concat_list[i].T.to(device)  # [S, 512]
            pred_i = recon_mlp(x_i)
            test_mse += F.mse_loss(pred_i, y_i, reduction='sum').item()
            test_cos += float(F.cosine_similarity(pred_i, y_i, dim=1).mean().item()) * x_i.shape[0]
            n_test_pixels += x_i.shape[0]

        test_mse /= n_test_pixels
        test_cos /= n_test_pixels

        # Baseline: always predict the training mean
        train_mean = Y_train_pixels.mean(dim=0, keepdim=True).to(device)  # [1, 512]
        baseline_mse = 0.0
        for i in test_idx:
            y_i = concat_list[i].T.to(device)  # [S, 512]
            pred_mean = train_mean.expand(y_i.shape[0], -1)
            baseline_mse += F.mse_loss(pred_mean, y_i, reduction='sum').item()
        baseline_mse /= n_test_pixels

    print(f"\n  Reconstruction MSE (MLP): {test_mse:.6f}")
    print(f"  Baseline MSE (predict mean): {baseline_mse:.6f}")
    print(f"  Cos similarity (pred, true): {test_cos:.4f}")
    r2 = 1.0 - test_mse / max(baseline_mse, 1e-8)
    print(f"  R² (vs mean baseline): {r2:.4f}")

    if r2 > 0.9:
        print(f"  → ✅ Conv1×1 is near-lossless — MLP recovers most of concat")
    elif r2 > 0.5:
        print(f"  → ⚠️  Partial information loss — MLP recovers {r2*100:.0f}% beyond mean")
    elif r2 > 0.2:
        print(f"  → ❌ Significant information loss — Conv1×1 discards substantial info")
    else:
        print(f"  → ‼️  Catastrophic loss — Conv1×1 destroys almost all concat structure")

    # ═══════════════════════════════════════════════════════════════
    # Exp C: Weight Matrix SVD Analysis
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 72}")
    print("EXP C: Conv1×1 Weight Matrix SVD")
    print("=" * 72)

    W = pf.fusion_conv[0].weight.data  # [256, 512]
    W_flat = W.reshape(256, 512).float()

    # SVD
    U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
    S_np = S.cpu().numpy()

    # Cumulative energy
    total_energy = (S_np ** 2).sum()
    cumsum = np.cumsum(S_np ** 2) / total_energy

    # Find effective rank at 90% and 99%
    r90 = int((cumsum < 0.90).sum()) + 1
    r99 = int((cumsum < 0.99).sum()) + 1
    r50 = int((cumsum < 0.50).sum()) + 1

    print(f"\n  Total singular values: {len(S_np)}")
    print(f"  σ_max: {S_np[0]:.4f}  σ_min: {S_np[-1]:.4f}  "
          f"condition number: {S_np[0]/max(S_np[-1],1e-8):.1f}")
    print(f"  Effective rank (50% energy): {r50}")
    print(f"  Effective rank (90% energy): {r90}")
    print(f"  Effective rank (99% energy): {r99}")

    # Print top-10 singular values
    print(f"\n  Top-10 singular values:")
    for i in range(min(10, len(S_np))):
        pct = S_np[i]**2 / total_energy * 100
        print(f"    σ_{i+1}: {S_np[i]:.4f}  ({pct:.1f}% energy)")

    # How many sv's needed for 99%
    print(f"\n  → Conv1×1 weight matrix effective rank = {r90} (90%), {r99} (99%)")
    if r99 < 50:
        print(f"     Weight matrix IS low-rank — projection inherently discards dimensions")
        print(f"     {256 - r99} output dims are near-zero — widening projection won't help")
    elif r90 < 128:
        print(f"     Weight matrix moderately low-rank — some capacity unused")
        print(f"     Widening projection MIGHT help if combined with retraining")
    else:
        print(f"     Weight matrix is full-rank — bottleneck is NOT in the linear projection")
        print(f"     Low effective rank of output comes from INPUT structure, not weight rank")

    # ── Also: effective rank of the OUTPUT (not weight) ──
    # This confirms whether low rank comes from weight or from input
    print(f"\n  ── Output-side analysis ──")
    # For each sample, compute SVD of conv1 output
    out_ranks = []
    for i in range(min(N, 20)):
        X = conv1_list[i].float()  # [256, ~2048]
        try:
            _, So, _ = torch.linalg.svd(X, full_matrices=False)
            total_o = (So ** 2).sum()
            cum_o = torch.cumsum(So ** 2, dim=0) / total_o
            r90_o = int((cum_o < 0.90).sum().item()) + 1
            out_ranks.append(r90_o)
        except Exception:
            pass
    mean_out_rank = np.mean(out_ranks) if out_ranks else 0
    print(f"  Mean effective rank of Conv1×1 OUTPUT (per sample): {mean_out_rank:.1f}")

    # ── Summary ──
    print(f"\n{'─' * 72}")
    print("SYNTHESIS:")
    print(f"  CKA: {mean_cka:.3f}  |  SVCCA: {mean_svcca:.3f}  |  "
          f"Recon R²: {r2:.3f}  |  W effective rank: {r90}")
    print(f"\n  If CKA>0.9 AND R²>0.9 AND W-rank>128:")
    print(f"    → Conv1×1 is a near-lossless compressor. Rank collapse comes from INPUT.")
    print(f"    → Fix: make SPG output more diverse, or increase projection width.")
    print(f"\n  If CKA<0.5 OR R²<0.5:")
    print(f"    → Conv1×1 discards significant information. This IS the bottleneck.")
    print(f"    → Fix: replace with wider projection + orthogonal init + regularization.")
    print(f"\n  If W-rank<50:")
    print(f"    → Weight matrix itself is low-rank. Pretraining failed to utilize capacity.")
    print(f"    → Fix: reinitialize Conv1×1 with orthogonal init, retrain.")

    del recon_mlp
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
