"""
Prompt Channel Grid — 可视化 dense_prompt 全部 256 通道
========================================================

验证假说: 空间信息是否是跨通道分布式编码的 (L2 norm 检测不到)?

输出: 16×16 通道网格 + GT + L2 norm + prediction 对比

用法:
    python tools/debug/prompt_channel_grid.py \
        --checkpoint <ckpt> --mode novel --k-shot 5 --num-tiles 2
"""

from __future__ import annotations

import argparse
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5I_CATEGORIES
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image, resize_mask


@torch.no_grad()
def build_support_for_class(*, data_root, fold, mode, class_id, k_shot,
                              backbone, cat_adapter, seed, device):
    ds = ISAID5iDataset(root=str(data_root), fold=fold, split="train", mode=mode)
    tiles = ds.class_to_tiles(class_id)
    if not tiles: return None
    scenes = defaultdict(list)
    for idx in tiles:
        src = ds._source_images.get(ds.tile_ids[idx], ds.tile_ids[idx])
        scenes[src].append(idx)
    rng = random.Random(seed)
    keys = list(scenes)
    k = min(k_shot, len(keys))
    chosen = rng.sample(keys, k)
    images, masks = [], []
    for sid in chosen:
        idx = rng.choice(scenes[sid])
        s = ds[idx]
        fg = ds.get_class_mask(idx, class_id)
        if fg is None or fg.sum() < 1: continue
        x, _ = preprocess_image(s["image"])
        images.append(x.to(device)); masks.append(fg)
    if not images: return None
    feats = backbone(torch.stack(images, dim=0))["image_embedding"]
    if cat_adapter is not None: feats = cat_adapter(feats)
    mg = torch.stack([resize_mask(m, (feats.shape[2], feats.shape[3])).to(device) for m in masks], dim=0)
    return feats, mg


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/iSAID-5i")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--mode", default="novel")
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tiles", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    fold = args.fold if args.fold is not None else ckpt.get("fold", 0)
    mode = args.mode if args.mode is not None else ckpt.get("mode", "novel")
    k_shot = args.k_shot if args.k_shot is not None else ckpt.get("k_shot", 5)

    bb_cfg = cfg.get("backbone", {})
    bb_path = Path(bb_cfg.get("checkpoint", "weights/mobile_sam.pt"))
    bb_path = bb_path if bb_path.is_absolute() else _REPO_ROOT / bb_path
    sam = build_mobile_sam(str(bb_path), bb_cfg.get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    embed_dim = int(cfg.get("support_encoder", {}).get("embed_dim", 256))
    model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(cfg)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    cat_adapter = None
    if ckpt.get("cat_adapter") is not None:
        tcfg = cfg.get("train", {})
        cat_adapter = CATAdapter(dim=embed_dim, bottleneck=int(tcfg.get("cat_adapter", {}).get("bottleneck", 64))).to(device)
        cat_adapter.load_state_dict(ckpt["cat_adapter"]); cat_adapter.eval()

    data_root = Path(args.data_root)
    if not data_root.is_absolute(): data_root = _REPO_ROOT / data_root
    val_ds = ISAID5iDataset(root=str(data_root), fold=fold, split="val", mode=mode)
    visible_classes = val_ds.visible_classes()

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "prompt_channels"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build support cache
    support_cache = {}
    for cls in visible_classes:
        sd = build_support_for_class(data_root=data_root, fold=fold, mode=mode,
                                       class_id=cls, k_shot=k_shot, backbone=backbone,
                                       cat_adapter=cat_adapter, seed=args.seed, device=device)
        if sd: support_cache[cls] = sd

    # Find tiles with multiple classes
    candidates = []
    for idx in range(len(val_ds)):
        present = [c for c in visible_classes if val_ds.get_class_mask(idx, c) is not None and val_ds.get_class_mask(idx, c).sum() > 50]
        if len(present) >= 2: candidates.append((idx, present))
    random.Random(args.seed + 9999).shuffle(candidates)

    for tile_idx, present_classes in candidates[:args.num_tiles]:
        sample = val_ds[tile_idx]
        H, W = sample["image"].shape[1], sample["image"].shape[2]
        img_np = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        x, meta = preprocess_image(sample["image"])
        query_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
        if cat_adapter is not None: query_emb = cat_adapter(query_emb)

        # Build GT combined
        gt_combined = np.zeros((H, W), dtype=np.uint8)
        for cls in present_classes:
            gm = val_ds.get_class_mask(tile_idx, cls)
            if gm is not None and gm.sum() > 0: gt_combined[gm.numpy().astype(bool)] = cls

        # Pick main class (largest GT area)
        main_cls = max(present_classes, key=lambda c: val_ds.get_class_mask(tile_idx, c).sum())
        sup_data = support_cache.get(main_cls)
        if sup_data is None: continue
        sup_feat, sup_mask = sup_data

        # Manual forward to get dense_prompt
        support_memory = model.support_encoder(sup_feat, sup_mask)
        geometric_prior = model.geometric_prior(query_emb, support_memory) if model.geometric_prior else None
        dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
        spg_out = model.spg(query_emb, support_memory, dense_pe)
        if model.prompt_fusion is not None and geometric_prior is not None:
            dense_prompt, sparse_token = model.prompt_fusion(geometric_prior, spg_out.semantic_prior)
        else:
            dense_prompt = spg_out.semantic_prior

        dp = dense_prompt[0].cpu().float()  # [256, 64, 64]

        # Prediction
        if model.bypass_head is not None:
            low_res = model.bypass_head(dense_prompt)
        else:
            support_proto = model._compute_support_prototype(sup_feat, sup_mask)
            low_res, _ = model.sam_decoder(query_emb, sparse_token, dense_prompt, support_prototype=support_proto)
        pred_logits = F.interpolate(low_res.float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        pred_np = (pred_logits.sigmoid() > 0.5).cpu().numpy()

        # GT for main class
        gt_main = val_ds.get_class_mask(tile_idx, main_cls).numpy().astype(bool)

        # ── Create figure ──
        fig = plt.figure(figsize=(24, 28))

        # Row 1: Overview
        ax_img = fig.add_subplot(5, 5, (1, 2))
        ax_img.imshow(img_np); ax_img.set_title("Query Image", fontsize=10); ax_img.axis("off")

        ax_gt = fig.add_subplot(5, 5, (3, 4))
        ax_gt.imshow(gt_combined, cmap="tab20", vmin=0, vmax=15)
        ax_gt.set_title(f"GT (all classes)", fontsize=10); ax_gt.axis("off")

        # Prediction
        ax_pred = fig.add_subplot(5, 5, 5)
        ax_pred.imshow(img_np)
        if pred_np.sum() > 0:
            overlay = np.zeros((H, W, 4), dtype=np.uint8)
            overlay[pred_np] = (255, 100, 100, 180)
            ax_pred.imshow(overlay)
        ax_pred.set_title(f"Pred (sup={ISAID5I_CATEGORIES.get(main_cls, '?')})", fontsize=10); ax_pred.axis("off")

        # L2 norm (what we used before)
        ax_l2 = fig.add_subplot(5, 5, 6)
        l2_norm = dp.pow(2).mean(dim=0).sqrt().numpy()
        ax_l2.imshow(l2_norm, cmap="hot")
        ax_l2.set_title("L2 norm (channel-mean)", fontsize=10); ax_l2.axis("off")

        # GT resized to 64x64
        ax_gt64 = fig.add_subplot(5, 5, 7)
        gt64 = torch.from_numpy(gt_main.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        gt64 = F.interpolate(gt64, (64, 64), mode="area")[0, 0].numpy()
        ax_gt64.imshow(gt64, cmap="gray")
        ax_gt64.set_title(f"GT ({ISAID5I_CATEGORIES.get(main_cls, '?')}) 64x64", fontsize=10); ax_gt64.axis("off")

        # ── Compute per-channel correlation with GT64 ──
        gt64_f = gt64.ravel()
        channel_corrs = []
        for c in range(256):
            ch_f = dp[c].numpy().ravel()
            std_c, std_g = ch_f.std(), gt64_f.std()
            if std_c > 1e-8 and std_g > 1e-8:
                corr = np.corrcoef(ch_f, gt64_f)[0, 1]
            else:
                corr = 0.0
            channel_corrs.append(corr)
        channel_corrs = np.array(channel_corrs)

        # Sort channels by correlation
        top16_idx = np.argsort(-channel_corrs)[:16]   # most GT-aligned
        bot16_idx = np.argsort(channel_corrs)[:16]    # most anti-aligned

        # Top-16 channels
        for i, ci in enumerate(top16_idx):
            ax = fig.add_subplot(5, 16, 16*3 + i + 1)
            ax.imshow(dp[ci].numpy(), cmap="RdBu_r", vmin=-dp.abs().max()*0.5, vmax=dp.abs().max()*0.5)
            ax.set_title(f"ch{ci}\nr={channel_corrs[ci]:.2f}", fontsize=6)
            ax.axis("off")

        fig.text(0.5, 0.69, f"Top-16 channels (highest GT correlation)", ha="center", fontsize=11, fontweight="bold")

        # Bottom-16 channels
        for i, ci in enumerate(bot16_idx):
            ax = fig.add_subplot(5, 16, 16*4 + i + 1)
            ax.imshow(dp[ci].numpy(), cmap="RdBu_r", vmin=-dp.abs().max()*0.5, vmax=dp.abs().max()*0.5)
            ax.set_title(f"ch{ci}\nr={channel_corrs[ci]:.2f}", fontsize=6)
            ax.axis("off")

        fig.text(0.5, 0.88, f"Bottom-16 channels (lowest GT correlation)", ha="center", fontsize=11, fontweight="bold")

        # ── Correlation histogram ──
        ax_hist = fig.add_subplot(5, 5, (8, 10))
        ax_hist.hist(channel_corrs, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax_hist.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        pos_pct = (channel_corrs > 0.05).mean() * 100
        neg_pct = (channel_corrs < -0.05).mean() * 100
        ax_hist.axvline(x=0.05, color="green", linestyle=":", alpha=0.5)
        ax_hist.axvline(x=-0.05, color="red", linestyle=":", alpha=0.5)
        ax_hist.set_title(f"Per-channel Pearson r with GT\n"
                          f"r>0.05: {pos_pct:.1f}%  r<-0.05: {neg_pct:.1f}%  neutral: {100-pos_pct-neg_pct:.1f}%\n"
                          f"max(r)={channel_corrs.max():.3f}  min(r)={channel_corrs.min():.3f}", fontsize=10)
        ax_hist.set_xlabel("Pearson r"); ax_hist.set_ylabel("N channels")

        tile_id = sample.get("tile_id", str(tile_idx))
        cls_name = ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}")
        fig.suptitle(f"Prompt Channel Grid: {tile_id}  |  support={cls_name}  |  fold={fold} mode={mode}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = out_dir / f"{tile_id}_sup{main_cls}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")
        print(f"  Channel-GT correlation: max={channel_corrs.max():.3f}, min={channel_corrs.min():.3f}, "
              f"|r|>0.05: {((np.abs(channel_corrs) > 0.05).mean()*100):.1f}%")

    print("[Done]")


if __name__ == "__main__":
    main()
