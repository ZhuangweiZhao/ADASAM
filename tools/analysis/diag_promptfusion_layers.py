"""
PromptFusion 逐层信息追踪 | PromptFusion Layer-wise Information Tracking.
==========================================================================

拆解 PromptFusion concat 模式的每一步:
    concat  → Conv1×1(512→256) → ReLU → Conv3×3(256→256) = dense_prompt

逐层统计:
    - Linear Probe: GAP → Linear(N_classes) 交叉验证准确率
    - Cross-Class Cos: 不同 support 类产出表示的余弦相似度 (越低越好)
    - Decoder FB-IoU: 将中间特征作为 dense_prompt 喂给 decoder
    - Effective Rank: SVD 能量 90% 所需的奇异值数

用法 | Usage:
    python tools/analysis/diag_promptfusion_layers.py \
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


def effective_rank(feature_map: torch.Tensor, energy_frac: float = 0.90) -> int:
    """Effective rank of a [C, H, W] feature map via SVD of [H*W, C] reshaped matrix."""
    C, H, W = feature_map.shape
    X = feature_map.reshape(C, -1).T.float()  # [H*W, C]
    # Center
    X = X - X.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(X, full_matrices=False)
    except RuntimeError:
        return C
    total = (S ** 2).sum()
    cumulative = torch.cumsum(S ** 2, dim=0)
    rank = int((cumulative / total < energy_frac).sum().item()) + 1
    return min(rank, C)


def decode_with_prompt(model, q_emb: torch.Tensor, dp: torch.Tensor, device: torch.device):
    """Run decoder → binary mask [256, 256]."""
    if model.bypass_head is not None:
        low_res = model.bypass_head(dp)
    else:
        sparse = dp.mean(dim=(2, 3))
        low_res, _ = model.sam_decoder(q_emb, sparse, dp)
    return (low_res[0, 0] > 0).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PromptFusion Layer-wise Tracking")
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=30)
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

    # Get PromptFusion weights for manual sub-step execution
    pf = model.prompt_fusion
    if pf is None:
        print("ERROR: PromptFusion not enabled in this checkpoint")
        sys.exit(1)
    print(f"PromptFusion mode: {pf.mode}")

    data_root = str(_REPO_ROOT / args.data_root) if not Path(args.data_root).is_absolute() else args.data_root
    dataset = ISAID5iDataset(root=data_root, fold=args.fold, split="val", mode="base")
    val_classes = sorted(dataset.visible_classes())
    cls_to_idx = {cls: i for i, cls in enumerate(val_classes)}
    idx_to_name = {i: ISAID5I_CATEGORIES.get(cls, str(cls)) for cls, i in cls_to_idx.items()}

    # ── Find multi-class query tiles ──
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
    # Collect intermediates at each PromptFusion sub-step
    # ═══════════════════════════════════════════════════════════════

    stage_names = ["geo_prior", "spg_prior", "concat", "conv1", "relu", "conv3"]
    stage_dims  = [256,         256,         512,      256,     256,    256]

    # Per-stage: list of {gap_vec: [C] cpu, label: int, q_emb: [1,256,64,64] cpu, gt_fg: [256,256] bool}
    stage_data: dict[str, list[dict]] = {s: [] for s in stage_names}

    # For cross-class cosine: store per-(tile, class) intermediates
    # tile_cc_data[tile_idx] = {class_id: {stage: tensor_cpu}}
    tile_cc_data: dict[int, dict[int, dict[str, torch.Tensor]]] = defaultdict(dict)

    print(f"\nCollecting PromptFusion intermediates ...")
    for tile_idx, n_cls, cls_list in tqdm(selected, desc="collecting"):
        sample = dataset[tile_idx]
        x, _ = preprocess_image(sample["image"])

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

            for sup_cls in cls_list[:4]:
                sup_data = build_support_for_class(
                    dataset, sup_cls, args.k_shot, device, backbone, adapter, rng)
                if sup_data is None:
                    continue
                sup_feat, sup_mask = sup_data
                support_memory = model.support_encoder(sup_feat, sup_mask)

                gp = model.geometric_prior(q_emb, support_memory) if model.geometric_prior else None
                spg_out = model.spg(q_emb, support_memory, dense_pe)
                sp = spg_out.semantic_prior

                if gp is None:
                    del sup_feat, sup_mask, support_memory, spg_out
                    continue

                # ── Manual sub-step execution of PromptFusion ──
                # Step 0: inputs
                gp_cpu = gp.cpu()
                sp_cpu = sp.cpu()

                # Step 1: concat
                concat = torch.cat([gp, sp], dim=1)  # [1, 512, 64, 64]

                # Step 2: Conv1×1
                conv1_w = pf.fusion_conv[0].weight  # [256, 512, 1, 1]
                conv1_out = F.conv2d(concat, conv1_w, bias=None)  # [1, 256, 64, 64]

                # Step 3: ReLU
                relu_out = F.relu(conv1_out, inplace=False)

                # Step 4: Conv3×3
                conv3_w = pf.fusion_conv[2].weight  # [256, 256, 3, 3]
                conv3_out = F.conv2d(relu_out, conv3_w, bias=None, padding=1)  # [1, 256, 64, 64]

                intermediates = {
                    "geo_prior": gp_cpu,
                    "spg_prior": sp_cpu,
                    "concat": concat.cpu(),
                    "conv1": conv1_out.cpu(),
                    "relu": relu_out.cpu(),
                    "conv3": conv3_out.cpu(),
                }

                # Store for linear probe + decoder eval
                q_cpu = q_emb.cpu()
                label = cls_to_idx[sup_cls]
                for sname in stage_names:
                    feat = intermediates[sname]  # [1, C, 64, 64]
                    gap = feat.mean(dim=(2, 3)).squeeze(0)  # [C]
                    stage_data[sname].append({
                        "gap": gap,
                        "label": label,
                        "q_emb": q_cpu,
                        "gt_fg": gt_fg.copy(),
                    })

                # Store for cross-class cosine
                tile_cc_data[tile_idx][sup_cls] = {
                    s: intermediates[s] for s in stage_names
                }

                del gp, sp, concat, conv1_out, relu_out, conv3_out
                del sup_feat, sup_mask, support_memory, spg_out

        del q_emb, dense_pe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    N = len(stage_data["conv3"])
    print(f"Collected {N} samples per stage")

    # ═══════════════════════════════════════════════════════════════
    # Metric 1: Linear Probe Accuracy (5-fold CV)
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 80}")
    print("METRIC 1: Linear Probe Accuracy (GAP → Linear(N_classes), 5-fold CV)")
    print("=" * 80)

    lp_results = {}
    for sname, sdim in zip(stage_names, stage_dims):
        gaps = torch.stack([d["gap"] for d in stage_data[sname]], dim=0)  # [N, C]
        labels = torch.tensor([d["label"] for d in stage_data[sname]], dtype=torch.long)
        N_s = gaps.shape[0]
        if N_s < 10:
            lp_results[sname] = (0.0, 0.0)
            continue

        n_folds = 5
        accs = []
        perm = torch.randperm(N_s)
        fold_size = N_s // n_folds
        for f in range(n_folds):
            val_idx = perm[f*fold_size:(f+1)*fold_size]
            train_idx = torch.cat([perm[:f*fold_size], perm[(f+1)*fold_size:]])

            X_train, y_train = gaps[train_idx].to(device), labels[train_idx].to(device)
            X_val, y_val = gaps[val_idx].to(device), labels[val_idx].to(device)

            probe = nn.Linear(sdim, len(val_classes)).to(device)
            nn.init.xavier_uniform_(probe.weight, gain=0.1)
            nn.init.zeros_(probe.bias)
            opt = torch.optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-4)

            for _ in range(200):
                loss = F.cross_entropy(probe(X_train), y_train)
                opt.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                preds = probe(X_val).argmax(dim=1)
                accs.append(float((preds == y_val).float().mean().item()))
            del probe, opt, X_train, y_train, X_val, y_val

        lp_results[sname] = (np.mean(accs) * 100, np.std(accs) * 100)

    # ═══════════════════════════════════════════════════════════════
    # Metric 2: Cross-Class Cosine (lower = better separation)
    # ═══════════════════════════════════════════════════════════════

    print("METRIC 2: Cross-Class Cosine ...")

    cc_results = {}
    for sname in stage_names:
        coses = []
        for tile_idx, cls_data in tile_cc_data.items():
            cls_ids = sorted(cls_data.keys())
            for ci in range(len(cls_ids)):
                for cj in range(ci + 1, len(cls_ids)):
                    a = cls_data[cls_ids[ci]].get(sname)
                    b = cls_data[cls_ids[cj]].get(sname)
                    if a is None or b is None:
                        continue
                    a0 = a[0] if a.ndim == 4 else a
                    b0 = b[0] if b.ndim == 4 else b
                    C = a0.shape[0]
                    af = a0.reshape(C, -1).float()
                    bf = b0.reshape(C, -1).float()
                    cos = float((F.normalize(af, dim=1) * F.normalize(bf, dim=1)).sum(dim=1).mean().item())
                    coses.append(cos)
        cc_results[sname] = (np.mean(coses), np.std(coses)) if coses else (0.0, 0.0)

    # ═══════════════════════════════════════════════════════════════
    # Metric 3: Decoder FB-IoU (use intermediate as dense_prompt)
    # ═══════════════════════════════════════════════════════════════

    print("METRIC 3: Decoder FB-IoU ...")

    decoder_results = {}
    for sname, sdim in zip(stage_names, stage_dims):
        if sdim != 256:
            decoder_results[sname] = (0.0, 0.0)
            continue
        ious = []
        for d in stage_data[sname]:
            feat = d["gap"]  # [C] — need to reconstruct spatial
            # Reconstruct [1, C, 64, 64] from GAP is impossible.
            # Instead we need the full spatial feature. Let's re-collect.
            pass
        decoder_results[sname] = (0.0, 0.0)

    # ── Re-run with spatial features for decoder eval ──
    # Store spatial features during collection
    # We'll do a quick second pass for decoder IoU
    print("  (re-collecting spatial features for decoder eval) ...")
    decoder_samples = []  # list of {stage: spatial_tensor_cpu, q_emb_cpu, gt_fg}
    subset = selected[:10]  # 10 tiles for speed
    for tile_idx, n_cls, cls_list in tqdm(subset, desc="decoder-collect"):
        sample = dataset[tile_idx]
        x, _ = preprocess_image(sample["image"])
        gt_fg = np.zeros((256, 256), dtype=bool)
        for cls in val_classes:
            gt = dataset.get_class_mask(tile_idx, cls)
            if gt is not None:
                gt_fg = gt_fg | gt.numpy().astype(bool)
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
                sp = model.spg(q_emb, sm, dense_pe).semantic_prior
                if gp is None:
                    continue
                concat = torch.cat([gp, sp], dim=1)
                conv1_out = F.conv2d(concat, pf.fusion_conv[0].weight, bias=None)
                relu_out = F.relu(conv1_out, inplace=False)
                conv3_out = F.conv2d(relu_out, pf.fusion_conv[2].weight, bias=None, padding=1)
                decoder_samples.append({
                    "geo_prior": gp.cpu(),
                    "spg_prior": sp.cpu(),
                    "conv1": conv1_out.cpu(),
                    "relu": relu_out.cpu(),
                    "conv3": conv3_out.cpu(),
                    "q_emb": q_emb.cpu(),
                    "gt_fg": gt_fg.copy(),
                })
                del gp, sp, concat, conv1_out, relu_out, conv3_out
                del sup_feat, sup_mask, sm
        del q_emb, dense_pe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    decoder_results = {}
    for sname in stage_names:
        if stage_dims[stage_names.index(sname)] != 256:
            decoder_results[sname] = (0.0, 0.0)
            continue
        ious = []
        for ds in decoder_samples:
            feat = ds[sname].to(device)  # [1, C, 64, 64]
            q = ds["q_emb"].to(device)
            pred = decode_with_prompt(model, q, feat, device)
            ious.append(mask_iou(pred, ds["gt_fg"]))
        decoder_results[sname] = (np.mean(ious), np.std(ious)) if ious else (0.0, 0.0)

    # ═══════════════════════════════════════════════════════════════
    # Metric 4: Effective Rank
    # ═══════════════════════════════════════════════════════════════

    print("METRIC 4: Effective Rank (90% energy) ...")

    rank_results = {}
    for sname, sdim in zip(stage_names, stage_dims):
        ranks = []
        # Use spatial features from decoder_samples
        for ds in decoder_samples:
            feat = ds[sname]  # [1, C, 64, 64] on CPU
            r = effective_rank(feat[0], energy_frac=0.90)
            ranks.append(r)
        rank_results[sname] = (np.mean(ranks), np.std(ranks))

    # ═══════════════════════════════════════════════════════════════
    # Report
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 96}")
    print("PROMPT FUSION LAYER-WISE INFORMATION TRACKING")
    print("=" * 96)

    stage_labels = {
        "geo_prior": "Geo Prior (input A)",
        "spg_prior": "SPG Prior (input B)",
        "concat": "Concat [A;B] (512D)",
        "conv1": "Conv1×1 → 256D",
        "relu": "ReLU",
        "conv3": "Conv3×3 → DensePrompt",
    }

    print(f"\n  {'Stage':<25s} {'LP Acc':>9s}  {'CC Cos ↓':>9s}  "
          f"{'FB-IoU':>9s}  {'EffRank':>9s}")
    print(f"  {'─'*25} {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")

    for sname in stage_names:
        lp_mean, lp_std = lp_results[sname]
        cc_mean, cc_std = cc_results[sname]
        di_mean, di_std = decoder_results[sname]
        rk_mean, rk_std = rank_results[sname]

        label = stage_labels.get(sname, sname)
        lp_str = f"{lp_mean:>5.1f}±{lp_std:.1f}%" if lp_mean > 0 else "    N/A"
        cc_str = f"{cc_mean:>.4f}" if cc_mean > 0 else "    N/A"
        di_str = f"{di_mean:>.4f}" if di_mean > 0 else "    N/A"
        rk_str = f"{rk_mean:>5.0f}" if rk_mean > 0 else "    N/A"

        print(f"  {label:<25s} {lp_str:>9s}  {cc_str:>9s}  {di_str:>9s}  {rk_str:>9s}")

    # ── Delta Analysis ──
    print(f"\n{'─' * 96}")
    print("STEP-WISE DELTA (positive = improvement, negative = degradation)")
    print("─" * 96)

    metrics_to_track = [
        ("lp_results", "Linear Probe Acc", "↑"),
        ("cc_results", "Cross-Class Cos", "↓"),
        ("decoder_results", "Decoder FB-IoU", "↑"),
        ("rank_results", "Effective Rank", "↑"),
    ]

    for metric_name, metric_label, direction in metrics_to_track:
        print(f"\n  {metric_label} ({direction}):")
        prev_val = None
        prev_name = None
        for sname in stage_names:
            data = {"lp_results": lp_results, "cc_results": cc_results,
                    "decoder_results": decoder_results, "rank_results": rank_results}[metric_name]
            val = data[sname][0] if data[sname][0] > 0 else None
            if val is not None and prev_val is not None and prev_val > 0:
                delta_pct = (val - prev_val) / max(abs(prev_val), 1e-8) * 100
                bar = "↑" if delta_pct > 0 else "↓"
                print(f"    {prev_name:<25s} → {stage_labels.get(sname, sname):<25s}: "
                      f"{delta_pct:+.1f}% {bar}")
            if val is not None and val > 0:
                prev_val = val
                prev_name = stage_labels.get(sname, sname)

    print(f"\n{'─' * 96}")
    print("KEY QUESTION: Which sub-step creates the Linear Probe jump from SPG (~3%) → DensePrompt (~27%)?")
    print("Look for the step with the largest positive delta in Linear Probe Acc.")

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
