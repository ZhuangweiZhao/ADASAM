"""
Stage 2 Few-Shot Pipeline 诊断工具 | Stage 2 Pipeline Diagnostic.
===================================================================

逐环节诊断 Stage 2 的每一处可能出现问题的地方, 定位瓶颈。
Step-by-step diagnosis of Stage 2 pipeline to locate the bottleneck.

检查项 | Checks:
    1. Support Memory Discriminability — 同类/异类 prototype 余弦相似度
    2. GeometricPrior Quality — support-query correlation map 是否对准 GT
    3. SPG Probe Behavior — 语义探针是否在目标区域激活
    4. Prompt Quality — dense/sparse prompt 是否包含有用信号
    5. Decoder Sensitivity — SAM Decoder 是否真正利用 prompt

用法 | Usage:
    # 只检查 Support Memory (不需要 Stage 2 checkpoint)
    python tools/analysis/diag_stage2.py --fold 1 --stage1-ckpt runs/stage1_fold1_seed42/best_adapter.pt

    # 完整检查 (需要 Stage 2 checkpoint)
    python tools/analysis/diag_stage2.py --fold 1 --stage2-ckpt runs/stage2_fold1_k5_seed42/best.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5I_CATEGORIES
from adasam.model.adasam_model import AdaSAMModel, AdaSAMModelConfig
from adasam.support_encoder import SupportEncoder, SupportEncoderConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image


# ═══════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════

def get_support_query(
    dataset: ISAID5iDataset,
    class_id: int,
    k_shot: int,
    device: torch.device,
    backbone: MobileSAMBackbone,
    adapter: CATAdapter | None,
    rng: random.Random,
) -> dict | None:
    """Sample K support tiles + 1 query tile for one class, extract features."""
    tiles = dataset.class_to_tiles(class_id)
    if len(tiles) < k_shot + 1:
        return None

    sampled = rng.sample(tiles, k_shot + 1)
    support_indices = sampled[:k_shot]
    query_idx = sampled[k_shot]

    # ── Support features ──
    s_feats, s_masks = [], []
    for idx in support_indices:
        sample = dataset[idx]
        x, _ = preprocess_image(sample["image"])
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = backbone(x)["image_embedding"]  # [1, 256, 64, 64]
            if adapter is not None:
                emb = adapter(emb)
        # Resize class mask to feature grid
        mask_256 = dataset.get_class_mask(idx, class_id)  # [256, 256]
        mask_64 = F.interpolate(
            mask_256.unsqueeze(0).unsqueeze(0).float(),
            size=(64, 64), mode="nearest",
        ).squeeze() > 0.5
        s_feats.append(emb[0])           # [256, 64, 64]
        s_masks.append(mask_64.float())  # [64, 64]

    # ── Query features ──
    q_sample = dataset[query_idx]
    q_x, _ = preprocess_image(q_sample["image"])
    q_x = q_x.unsqueeze(0).to(device)
    with torch.no_grad():
        q_emb = backbone(q_x)["image_embedding"]
        if adapter is not None:
            q_emb = adapter(q_emb)
    q_mask_256 = dataset.get_class_mask(query_idx, class_id)
    q_mask_64 = F.interpolate(
        q_mask_256.unsqueeze(0).unsqueeze(0).float(),
        size=(64, 64), mode="nearest",
    ).squeeze() > 0.5

    return {
        "class_id": class_id,
        "support_features": torch.stack(s_feats, dim=0),     # [K, 256, 64, 64]
        "support_masks": torch.stack(s_masks, dim=0),         # [K, 64, 64]
        "query_features": q_emb,                               # [1, 256, 64, 64]
        "query_mask": q_mask_64,                               # [64, 64]
        "support_indices": support_indices,
        "query_idx": query_idx,
    }


# ═══════════════════════════════════════════════════════════════════
# Check 1: Support Memory Discriminability
# ═══════════════════════════════════════════════════════════════════

def check_support_memory(
    backbone: MobileSAMBackbone,
    adapter: CATAdapter | None,
    dataset: ISAID5iDataset,
    support_encoder: SupportEncoder,
    classes: list[int],
    k_shot: int,
    device: torch.device,
    rng: random.Random,
    n_episodes: int = 5,
) -> dict:
    """检查 Support Memory 的类间可分离性."""
    print("\n" + "=" * 70)
    print("  CHECK 1: Support Memory Discriminability")
    print("=" * 70)

    all_memories: dict[int, list[torch.Tensor]] = {}  # {class_id: [memory_vectors]}

    for cls_id in tqdm(classes, desc="support memory"):
        tiles = dataset.class_to_tiles(cls_id)
        if len(tiles) < k_shot + 1:
            continue

        for ep in range(n_episodes):
            episode = get_support_query(dataset, cls_id, k_shot, device, backbone, adapter, rng)
            if episode is None:
                continue

            with torch.no_grad():
                memory = support_encoder(
                    episode["support_features"], episode["support_masks"]
                )  # [M, C]
                # Mean-pool memory tokens → single prototype [C]
                proto = memory.mean(dim=0)
                all_memories.setdefault(cls_id, []).append(F.normalize(proto, dim=0).cpu())

    # ── Compute intra/inter class similarity ──
    intra_sims, inter_sims = [], []
    cls_ids_list = sorted(all_memories.keys())
    for i, c in enumerate(cls_ids_list):
        mems = torch.stack(all_memories[c], dim=0)  # [N_ep, C]
        if mems.shape[0] >= 2:
            sim_mat = mems @ mems.T
            triu = sim_mat[np.triu_indices(mems.shape[0], k=1)]
            intra_sims.extend(triu.tolist())
        for j, c2 in enumerate(cls_ids_list):
            if j <= i:
                continue
            mems2 = torch.stack(all_memories[c2], dim=0)
            cross = (mems @ mems2.T).flatten()
            inter_sims.extend(cross.tolist())

    intra_arr, inter_arr = np.array(intra_sims), np.array(inter_sims)
    ratio = intra_arr.mean() / max(inter_arr.mean(), 1e-8) if len(intra_arr) > 0 else 0

    print(f"  intra-class cos-sim:  {intra_arr.mean():.4f} ± {intra_arr.std():.4f}  (n={len(intra_arr)})")
    print(f"  inter-class cos-sim:  {inter_arr.mean():.4f} ± {inter_arr.std():.4f}  (n={len(inter_arr)})")
    print(f"  separability ratio:   {ratio:.2f}x")

    if ratio < 1.2:
        verdict = "⚠️  FAILED — SupportEncoder has collapsed class distinctions"
    elif ratio < 2.0:
        verdict = "⚡ WEAK — some class separation but noisy"
    elif ratio < 5.0:
        verdict = "✓  OK — usable class separation in memory space"
    else:
        verdict = "✓✓ GOOD — strong class separation"
    print(f"  → {verdict}\n")

    return {"intra_mean": float(intra_arr.mean()), "inter_mean": float(inter_arr.mean()),
            "ratio": float(ratio), "verdict": verdict}


# ═══════════════════════════════════════════════════════════════════
# Check 2: GeometricPrior Quality
# ═══════════════════════════════════════════════════════════════════

def check_geometric_prior(
    backbone: MobileSAMBackbone,
    adapter: CATAdapter | None,
    dataset: ISAID5iDataset,
    support_encoder: SupportEncoder,
    class_id: int,
    k_shot: int,
    device: torch.device,
    rng: random.Random,
) -> dict:
    """检查 support-query cosine similarity 是否对准 GT 区域."""

    print(f"\n  --- GeometricPrior: class {class_id} ({ISAID5I_CATEGORIES.get(class_id, '?')}) ---")

    episode = get_support_query(dataset, class_id, k_shot, device, backbone, adapter, rng)
    if episode is None:
        print(f"  SKIP: not enough tiles for class {class_id}")
        return {}

    with torch.no_grad():
        memory = support_encoder(episode["support_features"], episode["support_masks"])

        # ── 1. Raw cosine similarity (no projection, no merge) ──
        q_raw = episode["query_features"]               # [1, C, gh, gw]
        proto = memory.mean(dim=0)                       # [C]

        q_flat = q_raw.reshape(1, 256, -1)               # [1, C, N]
        q_norm = F.normalize(q_flat, dim=1)              # [1, C, N]
        p_norm = F.normalize(proto, dim=0)               # [C]

        raw_sim = torch.einsum("bcn,c->bn", q_norm, p_norm)  # [1, N]
        raw_sim = raw_sim.reshape(1, 1, 64, 64)          # [1, 1, 64, 64]
        raw_map = (raw_sim - raw_sim.min()) / (raw_sim.max() - raw_sim.min() + 1e-8)
        raw_map = raw_map[0, 0].cpu()                    # [64, 64]

        # ── 2. GeometricPriorModule (with projections + merge) ──
        from adasam.prompt import GeometricPriorModule
        gp = GeometricPriorModule(embed_dim=256).to(device)
        geo_prior = gp(episode["query_features"], memory)
        if geo_prior.shape[1] == 1:
            geo_map = geo_prior[0, 0].cpu()
        else:
            geo_map = geo_prior[0].mean(dim=0).cpu()

    gt = episode["query_mask"].cpu().float()

    def _stats(sim_map, name):
        fg = sim_map[gt > 0.5]
        bg = sim_map[gt < 0.5]
        fgm = fg.mean().item() if len(fg) > 0 else 0
        bgm = bg.mean().item() if len(bg) > 0 else 0
        contrast = fgm - bgm
        best_iou, best_th = 0.0, 0.5
        for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pred = sim_map > (sim_map.max() * th)
            inter = (pred & (gt > 0.5)).sum().item()
            union = (pred | (gt > 0.5)).sum().item()
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou, best_th = iou, th
        print(f"  {name}: FG={fgm:.4f}  BG={bgm:.4f}  contrast={contrast:.4f}  best_IoU={best_iou:.4f}@{best_th}")
        return contrast, best_iou

    c1, i1 = _stats(raw_map, "raw cos-sim ")
    c2, i2 = _stats(geo_map, "GeoPrior   ")

    if c1 > 0.05:
        print(f"  → Raw cosine similarity IS informative — GeometricPrior projections need training")
    elif c1 > 0.01:
        print(f"  → Raw cosine similarity weakly informative — pixel-level matching is inherently noisy")
    else:
        print(f"  → Raw cosine similarity has NO signal — feature space issue, not projection issue")

    return {"raw_contrast": c1, "raw_best_iou": i1,
            "gp_contrast": c2, "gp_best_iou": i2,
            "verdict": "raw" if c1 > 0.05 else "none"}


# ═══════════════════════════════════════════════════════════════════
# Check 3: SPG Probe Behavior
# ═══════════════════════════════════════════════════════════════════

def check_spg_probes(
    model: AdaSAMModel,
    backbone: MobileSAMBackbone,
    adapter: CATAdapter | None,
    dataset: ISAID5iDataset,
    class_id: int,
    k_shot: int,
    device: torch.device,
    rng: random.Random,
) -> dict:
    """检查 SPG 语义探针的激活模式."""
    print(f"\n  --- SPG Probes: class {class_id} ({ISAID5I_CATEGORIES.get(class_id, '?')}) ---")

    episode = get_support_query(dataset, class_id, k_shot, device, backbone, adapter, rng)
    if episode is None:
        print(f"  SKIP: not enough tiles")
        return {}

    with torch.no_grad():
        spg_out = model.forward_train(
            episode["query_features"],
            episode["support_features"],
            episode["support_masks"],
        )[0]  # SPGOutput

    prior_mask = spg_out.prior_mask  # [1, 1, 64, 64]
    if prior_mask is not None:
        pm = prior_mask[0, 0].cpu()
        gt = episode["query_mask"].cpu().float()

        # Prior mask statistics
        pm_min, pm_max = pm.min().item(), pm.max().item()
        pm_mean, pm_std = pm.mean().item(), pm.std().item()

        # Prior mask on FG vs BG
        fg_pm = pm[gt > 0.5]
        bg_pm = pm[gt < 0.5]
        fg_pm_mean = fg_pm.mean().item() if len(fg_pm) > 0 else 0
        bg_pm_mean = bg_pm.mean().item() if len(bg_pm) > 0 else 0

        print(f"  prior_mask (raw logits): min={pm_min:.3f} max={pm_max:.3f} mean={pm_mean:.3f}±{pm_std:.3f}")
        print(f"  prior_mask FG={fg_pm_mean:.4f}  BG={bg_pm_mean:.4f}  contrast={fg_pm_mean-bg_pm_mean:.4f}")

        # Prior aux snapshots
        print(f"  prior_aux layers: {len(spg_out.prior_aux)}")
        for i, aux in enumerate(spg_out.prior_aux):
            aux_mask = aux["prior_mask"]
            if aux_mask is not None:
                aux_m = aux_mask[0, 0].cpu() if aux_mask.ndim == 4 else aux_mask[0].cpu()
                print(f"    L{i}: min={aux_m.min():.3f} max={aux_m.max():.3f} mean={aux_m.mean():.3f}")

        contrast = fg_pm_mean - bg_pm_mean
        if contrast > 0.05:
            verdict = "✓  SPG prior_mask shows correct FG/BG discrimination"
        elif contrast > 0:
            verdict = "⚡ SPG prior_mask has weak FG signal"
        else:
            verdict = "⚠️  SPG prior_mask has NO or INVERTED FG signal"
        print(f"  → {verdict}")

    else:
        verdict = "N/A — prior_mask is None"

    return {"verdict": verdict}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2 Pipeline Diagnostic")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--stage1-ckpt", default=None, help="Stage 1 adapter checkpoint (optional)")
    p.add_argument("--stage2-ckpt", default=None, help="Stage 2 model checkpoint (for full diagnosis)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--weights", default=None, help="MobileSAM weights path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-id", type=int, default=None, help="single class to diagnose (default: all)")
    p.add_argument("--check", default="all", choices=["all", "memory", "geometric", "spg"],
                   help="which check to run (default: all)")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    rng = random.Random(args.seed)

    data_root = args.data_root or str(_REPO_ROOT / "data" / "iSAID-5i")
    weights = args.weights or str(_REPO_ROOT / "weights" / "mobile_sam.pt")

    # ── Backbone ──
    sam = build_mobile_sam(weights, "vit_t", device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    backbone.eval()

    # ── Adapter ──
    adapter = None
    adapter_name = "raw_SAM"
    if args.stage1_ckpt:
        ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
        adapter_cfg = ckpt.get("config", {}).get("adapter", {})
        adapter = CATAdapter(dim=256, bottleneck=int(adapter_cfg.get("bottleneck", 64))).to(device)
        adapter.load_state_dict(ckpt["adapter"])
        adapter.eval()
        adapter_name = f"stage1_ep{ckpt.get('epoch','?')}"

    # ── Dataset ──
    ds = ISAID5iDataset(root=data_root, fold=args.fold, split="train", mode="base")
    classes = sorted(ds.visible_classes())
    print(f"\n{'='*70}")
    print(f"  Stage 2 Pipeline Diagnostic")
    print(f"  fold={args.fold}  adapter={adapter_name}  k-shot={args.k_shot}")
    print(f"  classes={classes}  tiles={len(ds)}")
    print(f"{'='*70}")

    # ── SupportEncoder (fresh or from checkpoint) ──
    if args.stage2_ckpt:
        ckpt2 = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
        cfg2 = AdaSAMModelConfig.from_dict(ckpt2.get("config", {}))
        model = AdaSAMModel(sam, cfg2).to(device)
        model.load_state_dict(ckpt2["model"])
        model.eval()
        support_encoder = model.support_encoder
        print(f"  [model] loaded from {args.stage2_ckpt}")
    else:
        # Fresh SupportEncoder (untrained — baseline behavior)
        se_cfg = SupportEncoderConfig(embed_dim=256, n_support_tokens=16, n_encoder_layers=0)
        support_encoder = SupportEncoder(se_cfg).to(device)
        support_encoder.eval()
        model = None
        print(f"  [support_encoder] fresh (untrained) — n_layers=0, n_tokens=16")

    # ── Run checks ──
    target_classes = [args.class_id] if args.class_id else classes

    if args.check in ("all", "memory"):
        check_support_memory(backbone, adapter, ds, support_encoder,
                            target_classes, args.k_shot, device, rng)

    if args.check in ("all", "geometric"):
        for c in target_classes:
            check_geometric_prior(backbone, adapter, ds, support_encoder,
                                 c, args.k_shot, device, rng)

    if args.check in ("all", "spg"):
        if model is not None:
            for c in target_classes:
                check_spg_probes(model, backbone, adapter, ds, c, args.k_shot, device, rng)
        else:
            print("\n  [SPG check] SKIP — requires --stage2-ckpt for full model")


if __name__ == "__main__":
    main()
