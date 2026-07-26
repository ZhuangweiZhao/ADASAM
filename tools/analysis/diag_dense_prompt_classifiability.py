"""
Dense Prompt 线性可分性诊断 | Dense Prompt Linear Classifiability.
===================================================================

核心问题：Dense Prompt 到底有没有类别信息？

方法：冻结全部网络 → 对 DensePrompt 做 GAP → Linear(256, N_classes) → 交叉熵。
如果 Linear 分类准确率 >> 随机基线 (1/N)，说明 DensePrompt 里类别信息存在，
只是 decoder 利用效率低。如果 ≈ 随机，说明类别信息确实丢失了。

用法 | Usage:
    python tools/analysis/diag_dense_prompt_classifiability.py \
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
    backbone: MobileSAMBackbone, adapter,
    rng: random.Random,
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
        [resize_mask(m, (feats.shape[2], feats.shape[3])).to(device) for m in masks],
        dim=0,
    )
    if masks_grid.sum() < 1.0:
        return None
    return feats, masks_grid


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Dense Prompt Linear Classifiability"
    )
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=30,
                        help="Number of multi-class query tiles for data collection")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Linear probe training epochs")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    print(f"Device: {device}")

    # ── Load checkpoint ──
    ckpt_path = Path(args.stage2_ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    weights_path = str(_REPO_ROOT / cfg.get("backbone", {}).get(
        "checkpoint", "weights/mobile_sam.pt"))
    sam = build_mobile_sam(
        weights_path, cfg.get("backbone", {}).get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(
        sam.image_encoder, sam.image_encoder.img_size).to(device)

    model_cfg = AdaSAMModelConfig.from_dict(cfg)
    model = AdaSAMModel(sam, model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    adapter_state = ckpt.get("cat_adapter")
    if adapter_state is not None:
        adapter_cfg = ckpt.get("config", {}).get("adapter", {})
        adapter = CATAdapter(
            dim=256, bottleneck=int(adapter_cfg.get("bottleneck", 64)),
        ).to(device)
        adapter.load_state_dict(adapter_state)
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = None

    data_root = str(_REPO_ROOT / args.data_root) if not Path(args.data_root).is_absolute() \
        else args.data_root
    dataset = ISAID5iDataset(
        root=data_root, fold=args.fold, split="val", mode="base")
    val_classes = sorted(dataset.visible_classes())
    N_CLASSES = len(val_classes)
    # Build class_id → 0..N-1 index mapping
    cls_to_idx = {cls: i for i, cls in enumerate(val_classes)}
    idx_to_name = {i: ISAID5I_CATEGORIES.get(cls, str(cls))
                   for cls, i in cls_to_idx.items()}

    print(f"Val classes ({N_CLASSES}): {[idx_to_name[i] for i in range(N_CLASSES)]}")
    print(f"Random baseline: {100.0/N_CLASSES:.1f}%")

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
    # Collect (dense_prompt_gap, support_class_label) pairs
    # ═══════════════════════════════════════════════════════════════

    X_list, y_list = [], []
    # source: "geometric_prior" | "spg" | "dense_prompt"
    for source in ["geometric_prior", "spg_semantic_prior", "dense_prompt"]:
        X_list.append([])
        y_list.append([])

    print(f"\n{'=' * 72}")
    print("Collecting intermediate features ...")
    print("=" * 72)

    for tile_idx, n_cls_on_tile, cls_list in tqdm(selected, desc="collecting"):
        sample = dataset[tile_idx]
        x, _ = preprocess_image(sample["image"])

        with torch.no_grad():
            q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_emb = adapter(q_emb)

            dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()

            for sup_cls in cls_list[:4]:  # max 4 supports per tile
                sup_data = build_support_for_class(
                    dataset, sup_cls, args.k_shot, device, backbone, adapter, rng)
                if sup_data is None:
                    continue
                sup_feat, sup_mask = sup_data

                support_memory = model.support_encoder(sup_feat, sup_mask)

                # GeoPrior
                gp = model.geometric_prior(q_emb, support_memory) if model.geometric_prior else None

                # SPG
                spg_out = model.spg(q_emb, support_memory, dense_pe)

                # PromptFusion
                if model.prompt_fusion is not None and gp is not None:
                    dp, _ = model.prompt_fusion(gp, spg_out.semantic_prior)
                else:
                    dp = model._build_dense_prompt(support_memory, sup_feat, sup_mask)
                    if dp is None:
                        dp = spg_out.semantic_prior

                # GAP each intermediate → [256] vector
                label = cls_to_idx[sup_cls]

                # 0: geometric_prior
                if gp is not None:
                    vec_gp = gp.mean(dim=(2, 3)).squeeze(0).cpu()  # [256]
                    X_list[0].append(vec_gp)
                    y_list[0].append(label)

                # 1: spg_semantic_prior
                vec_spg = spg_out.semantic_prior.mean(dim=(2, 3)).squeeze(0).cpu()  # [256]
                X_list[1].append(vec_spg)
                y_list[1].append(label)

                # 2: dense_prompt
                vec_dp = dp.mean(dim=(2, 3)).squeeze(0).cpu()  # [256]
                X_list[2].append(vec_dp)
                y_list[2].append(label)

                del sup_feat, sup_mask, support_memory, spg_out, dp
                if gp is not None:
                    del gp

        del q_emb, dense_pe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # Train linear probe on each source
    # ═══════════════════════════════════════════════════════════════

    source_names = ["geometric_prior", "spg_semantic_prior", "dense_prompt"]
    source_labels = ["GeoPrior", "SPG semantic_prior", "Dense Prompt"]

    print(f"\n{'=' * 72}")
    print("LINEAR PROBE RESULTS")
    print(f"  Model: Linear(256 → {N_CLASSES}), trained {args.epochs} epochs, lr={args.lr}")
    print("=" * 72)

    for src_idx, (src_name, src_label) in enumerate(zip(source_names, source_labels)):
        X_all = torch.stack(X_list[src_idx], dim=0)  # [N, 256]
        y_all = torch.tensor(y_list[src_idx], dtype=torch.long)  # [N]
        N = X_all.shape[0]

        if N < 10:
            print(f"\n  {src_label}: only {N} samples, skipping")
            continue

        # Train/test split (stratified by tile would be ideal, but simple random is ok here)
        # For a fair probe: 70/30 random split, 5-fold to get mean±std
        n_runs = 5
        accs = []

        for run in range(n_runs):
            # Shuffle
            perm = torch.randperm(N)
            X_shuf = X_all[perm]
            y_shuf = y_all[perm]

            split = int(N * 0.7)
            X_train, y_train = X_shuf[:split].to(device), y_shuf[:split].to(device)
            X_test, y_test = X_shuf[split:].to(device), y_shuf[split:].to(device)

            # Linear probe
            probe = nn.Linear(256, N_CLASSES).to(device)
            nn.init.xavier_uniform_(probe.weight, gain=0.1)
            nn.init.zeros_(probe.bias)
            opt = torch.optim.Adam(probe.parameters(), lr=args.lr, weight_decay=1e-4)

            for ep in range(args.epochs):
                probe.train()
                logits = probe(X_train)
                loss = F.cross_entropy(logits, y_train)

                opt.zero_grad()
                loss.backward()
                opt.step()

            # Evaluate
            probe.eval()
            with torch.no_grad():
                test_logits = probe(X_test)
                preds = test_logits.argmax(dim=1)
                acc = float((preds == y_test).float().mean().item())
                accs.append(acc)

            del probe, opt, X_train, y_train, X_test, y_test, test_logits

        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        baseline = 100.0 / N_CLASSES

        # Diagnosis
        ratio_to_baseline = mean_acc / baseline
        if ratio_to_baseline > 5:
            diag = f"✅ STRONG class info ({ratio_to_baseline:.1f}× baseline)"
        elif ratio_to_baseline > 2:
            diag = f"⚠️  MODERATE class info ({ratio_to_baseline:.1f}× baseline)"
        elif ratio_to_baseline > 1.3:
            diag = f"❌ WEAK class info ({ratio_to_baseline:.1f}× baseline)"
        else:
            diag = f"‼️  NO class info — near random"

        bar = "█" * max(1, int(mean_acc / 2))
        print(f"\n  {src_label:<20s}: {mean_acc:.1f}% ± {std_acc:.1f}%  "
              f"(baseline={baseline:.1f}%)  {diag}")
        print(f"    {bar}")

    # ── Summary + Save channel importance for downstream attribution ──
    print(f"\n{'─' * 72}")
    print("DECISION TREE:")
    print(f"  If Dense Prompt accuracy >> random → class info EXISTS, decoder is bottleneck")
    print(f"  If Dense Prompt accuracy ≈ random → class info truly LOST in prompt chain")
    print(f"  If GeoPrior accuracy >> Dense Prompt → SPG/PromptFusion DESTROYS info")
    print(f"  If GeoPrior accuracy << Dense Prompt → SPG/PromptFusion CREATES info")

    # ── Save final probe weights + channel rankings for attribution ──
    # Train one final probe on ALL dense_prompt data
    dp_idx = 2  # dense_prompt
    X_dp = torch.stack(X_list[dp_idx], dim=0).to(device)
    y_dp = torch.tensor(y_list[dp_idx], dtype=torch.long).to(device)

    final_probe = nn.Linear(256, N_CLASSES).to(device)
    nn.init.xavier_uniform_(final_probe.weight, gain=0.1)
    nn.init.zeros_(final_probe.bias)
    opt_final = torch.optim.Adam(final_probe.parameters(), lr=args.lr, weight_decay=1e-4)

    final_probe.train()
    for ep in range(args.epochs):
        logits = final_probe(X_dp)
        loss = F.cross_entropy(logits, y_dp)
        opt_final.zero_grad()
        loss.backward()
        opt_final.step()

    final_probe.eval()
    with torch.no_grad():
        preds = final_probe(X_dp).argmax(dim=1)
        final_acc = float((preds == y_dp).float().mean().item()) * 100
    print(f"\nFinal probe (all data): {final_acc:.1f}%")

    W = final_probe.weight.data.cpu().numpy()
    importance = np.linalg.norm(W, axis=0)  # [256]
    ranked = np.argsort(-importance)

    save_path = Path(args.stage2_ckpt).parent / "channel_importance.pt"
    torch.save({
        "probe_weight": final_probe.weight.data.cpu(),
        "probe_bias": final_probe.bias.data.cpu(),
        "channel_importance": torch.from_numpy(importance),
        "ranked_channels": torch.from_numpy(ranked.copy()),
        "probe_accuracy": final_acc,
        "val_classes": val_classes,
        "class_names": [ISAID5I_CATEGORIES.get(c, str(c)) for c in val_classes],
    }, save_path)
    print(f"Channel importance saved to: {save_path}")

    # Free GPU memory
    del X_all, y_all, X_dp, y_dp, final_probe, opt_final
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
