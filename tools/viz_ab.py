"""
[DEPRECATED] SAM-RSP 改进可视化 | Visualize SAM-RSP Improvements.
=================================================================
⚠️ 此工具引用旧 DPG API (dpg_out.fg_queries, dpg_out.fg_logits, model.coarse_prior,
   model.dpg.feedback_conv), 与新 SPG 架构不兼容, 运行会崩溃。需更新后才能使用。

可视化 DPG 管线中的关键中间产物:
  - RSP Map: support-query 相似度生成的粗空间先验 (CoarsePriorModule 输出)
  - Predicted Mask: 模型最终预测
  - GT Mask: 真实标注
  - Query Image: 原始查询图

用法 | Usage::

    # 单张图可视化
    python tools/viz_ab.py \
        --checkpoint runs/ab_new/.../best_model.pt \
        --k-shot 1 --num-tiles 4

    # 对比两个模型 (baseline vs new)
    python tools/viz_ab.py \
        --checkpoint runs/ab_new/.../best_model.pt \
        --checkpoint-baseline runs/ab_baseline/.../best_model.pt \
        --k-shot 1 --num-tiles 4

输出 | Output:
    runs/viz_ab/ 下的 PNG 对比图
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import (
    ISAID5iDataset,
    ISAID5I_CATEGORIES,
    ISAID5I_FOLDS,
)
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image, resize_mask


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def inference_with_rsp(
    model: AdaSAMModel,
    backbone: MobileSAMBackbone,
    cat_adapter: torch.nn.Module | None,
    query_image: torch.Tensor,
    support_features: torch.Tensor,
    support_masks: torch.Tensor,
    device: torch.device,
) -> dict:
    """推理并返回所有中间产物 | Inference with intermediate outputs.

    :return: {
        "mask": [H, W] bool predicted mask,
        "rsp_map": [64, 64] RSP spatial prior,
        "score": float confidence score,
    }
    """
    # Embed query
    x, _ = preprocess_image(query_image)
    emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
    if cat_adapter is not None:
        emb = cat_adapter(emb)

    # Support memory
    sup_feat = support_features.to(device)
    sup_mask = support_masks.to(device)
    support_memory = model.support_encoder(sup_feat, sup_mask)

    # ---- Coarse Prior (RSP) ----
    if model.coarse_prior is not None:
        enriched, rsp_map = model.coarse_prior(emb, support_memory)
        rsp_np = rsp_map[0, 0].cpu().numpy()  # [64, 64]
    else:
        enriched = emb
        rsp_np = np.zeros((64, 64), dtype=np.float32)

    # DPG + SAM decoder
    dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
    dpg_out = model.dpg(enriched, support_memory, dense_pe)

    if dpg_out.dense_prompt is not None:
        no_mask = model.sam_decoder.prompt_encoder.no_mask_embed.weight.view(1, -1, 1, 1)
        dense_override = no_mask + dpg_out.dense_prompt
    else:
        dense_override = None

    low_res, iou_pred = model.sam_decoder(enriched, dpg_out.fg_queries, dense_override)

    # Score filtering — 使用较低阈值，因为 IoU 预测头在早期训练中未校准
    # Use lower threshold since IoU head is uncalibrated in early training
    scores = dpg_out.fg_logits.sigmoid() * iou_pred[:, 0].clamp(0.0, 1.0)
    # Sort by score, take top-K instances
    sorted_idx = scores.argsort(descending=True)
    keep = sorted_idx[:8]  # top-8 queries regardless of absolute score

    H, W = 256, 256  # iSAID-5i native
    if len(keep) > 0:
        logits = model.sam_decoder.upscale_logits(low_res[keep], (1024, 1024), (H, W))
        masks = (logits > model.sam_decoder.mask_threshold).cpu().numpy()
        kept_scores = scores[keep].cpu().numpy()
        pred_mask = np.any(masks, axis=0)
        best_score = float(kept_scores.max()) if len(kept_scores) > 0 else 0.0
    else:
        pred_mask = np.zeros((H, W), dtype=bool)
        best_score = 0.0

    return {
        "mask": pred_mask,
        "rsp_map": rsp_np,
        "score": best_score,
    }


def draw_tile(ax, img: np.ndarray, title: str, cmap: str = "gray",
              vmin: float = 0.0, vmax: float = 1.0) -> None:
    """绘制单个子图 | Draw a single subplot tile."""
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def make_figure(
    query_img: np.ndarray,       # [H, W, 3] uint8
    gt_mask: np.ndarray,         # [H, W] bool
    rsp_map: np.ndarray,         # [64, 64] float32
    pred_mask: np.ndarray,       # [H, W] bool
    title: str,
    score: float,
    class_name: str = "",
) -> np.ndarray:
    """生成一张对比图 | Generate one comparison figure.

    布局:
      [Query Image]  [RSP Map (upsampled)]
      [GT Mask]      [Predicted Mask]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = query_img.shape[:2]

    # Upsample RSP to image size
    rsp_up = cv2.resize(rsp_map, (W, H), interpolation=cv2.INTER_LINEAR)

    # Overlay pred on image
    overlay = query_img.copy()
    overlay[pred_mask] = (overlay[pred_mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

    gt_overlay = query_img.copy()
    gt_overlay[gt_mask] = (gt_overlay[gt_mask] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{title}  |  score={score:.3f}  {class_name}", fontsize=12, fontweight="bold")

    draw_tile(axes[0, 0], query_img, "Query Image")
    draw_tile(axes[0, 1], rsp_up, "RSP Map (Coarse Prior)", cmap="hot", vmin=0.0, vmax=1.0)
    draw_tile(axes[0, 2], gt_overlay, "GT Mask (yellow overlay)")

    # RSP + GT overlay
    rsp_gt = np.stack([rsp_up, gt_mask.astype(np.float32), np.zeros_like(rsp_up)], axis=-1)
    draw_tile(axes[1, 0], rsp_gt, "RSP vs GT (R=prior, G=GT)")

    draw_tile(axes[1, 1], pred_mask.astype(np.float32), "Predicted Mask")
    draw_tile(axes[1, 2], overlay, "Prediction Overlay (green)")

    plt.tight_layout()
    fig.canvas.draw()
    vis = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]
    plt.close(fig)
    return vis


def main() -> None:
    print("=" * 60)
    print("[DEPRECATED] viz_ab.py 引用旧 DPG API, 与新 SPG 架构不兼容。")
    print("请使用 tools/viz_neuseg.py 或 tools/eval_isaid_5i.py --save-vis 替代。")
    print("=" * 60)
    import sys; sys.exit(1)

    p = argparse.ArgumentParser(description="Visualize SAM-RSP improvements")
    p.add_argument("--checkpoint", required=True, help="新模型 checkpoint 路径")
    p.add_argument("--checkpoint-baseline", default=None,
                   help="(可选) baseline 模型 checkpoint，并排对比")
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--num-tiles", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="data/iSAID-5i")
    p.add_argument("--class-id", type=int, default=None,
                   help="指定类别 (1-15), 不指定则随机")
    p.add_argument("--output-dir", default="runs/viz_ab")
    args = p.parse_args()

    set_seed(args.seed)
    device = _get_device()
    data_root = _resolve(args.data_root)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    def load_model(ckpt_path: str) -> tuple:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt["config"]
        bb_cfg = cfg.get("backbone", {})
        bb_path = _resolve(bb_cfg.get("checkpoint", "weights/mobile_sam.pt"))

        sam = build_mobile_sam(str(bb_path), bb_cfg.get("model_type", "vit_t"), device)
        backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
        embed_dim = int(cfg.get("support_encoder", {}).get("embed_dim", 256))
        model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(cfg)).to(device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()

        cat_adapter = None
        if ckpt.get("cat_adapter") is not None:
            tcfg = cfg.get("train", {})
            adapter_cfg = tcfg.get("cat_adapter", {})
            cat_adapter = CATAdapter(
                dim=embed_dim,
                bottleneck=int(adapter_cfg.get("bottleneck", 64)),
            ).to(device)
            cat_adapter.load_state_dict(ckpt["cat_adapter"])
            cat_adapter.eval()

        has_cp = model.coarse_prior is not None
        has_fb = model.dpg.feedback_conv is not None if hasattr(model.dpg, 'feedback_conv') else False
        print(f"[load] {Path(ckpt_path).parent.name}: coarse_prior={has_cp}, feedback={has_fb}")
        return model, backbone, cat_adapter, cfg, embed_dim

    model_new, backbone, cat_adapter, cfg, embed_dim = load_model(args.checkpoint)

    if args.checkpoint_baseline:
        model_bl, backbone_bl, cat_bl, _, _ = load_model(args.checkpoint_baseline)
    else:
        model_bl = None

    # ── Data ──
    fold_def = ISAID5I_FOLDS[args.fold]
    train_ds = ISAID5iDataset(root=str(data_root), fold=args.fold, split="train", mode="novel")
    val_ds = ISAID5iDataset(root=str(data_root), fold=args.fold, split="val", mode="novel")
    novel_classes = val_ds.visible_classes()
    print(f"Novel classes: {[ISAID5I_CATEGORIES[c] for c in novel_classes]}")

    # ── Pick query tiles ──
    rng = np.random.RandomState(args.seed)
    if args.class_id is not None and args.class_id in novel_classes:
        target_cls = args.class_id
    else:
        target_cls = int(rng.choice(novel_classes))

    cls_tiles = val_ds.class_to_tiles(target_cls)
    chosen = rng.choice(cls_tiles, size=min(args.num_tiles, len(cls_tiles)), replace=False)
    print(f"Class: {ISAID5I_CATEGORIES[target_cls]} ({target_cls}), tiles: {len(cls_tiles)}, chosen: {list(chosen)}")

    # ── Build support (from train split) ──
    support_tiles = train_ds.class_to_tiles(target_cls)
    support_idx = rng.choice(support_tiles, size=min(args.k_shot, len(support_tiles)), replace=False)

    support_images, support_masks_list = [], []
    for si in support_idx:
        sup = train_ds[int(si)]
        fg = None
        for inst in sup["regions"]:
            if inst["category_id"] == target_cls:
                fg = inst["mask"].clone() if fg is None else (fg | inst["mask"])
        if fg is None:
            continue
        x_s, _ = preprocess_image(sup["image"])
        support_images.append(x_s.to(device))
        support_masks_list.append(fg.float())

    if not support_images:
        print(f"ERROR: No valid support tiles for class {target_cls}")
        return

    sup_feat = backbone(torch.stack(support_images, dim=0))["image_embedding"]
    sup_mask_grid = torch.stack(
        [resize_mask(m, (sup_feat.shape[2], sup_feat.shape[3])).to(device)
         for m in support_masks_list], dim=0,
    )

    # ── Render each query ──
    vis_images_new, vis_images_bl = [], []

    for qi in chosen:
        sample = val_ds[int(qi)]
        query_img_np = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_mask = None
        for inst in sample["regions"]:
            if inst["category_id"] == target_cls:
                gt_mask = inst["mask"].numpy() if gt_mask is None else (gt_mask | inst["mask"].numpy())
        if gt_mask is None:
            gt_mask = np.zeros((256, 256), dtype=bool)

        # New model inference
        result_new = inference_with_rsp(
            model_new, backbone, cat_adapter,
            sample["image"], sup_feat, sup_mask_grid, device,
        )

        tile_id = sample.get("tile_id", f"tile_{qi}")
        cname = ISAID5I_CATEGORIES.get(target_cls, f"cls{target_cls}")

        vis_new = make_figure(
            query_img_np, gt_mask, result_new["rsp_map"], result_new["mask"],
            f"[NEW] {tile_id}", result_new["score"], cname,
        )
        vis_images_new.append(vis_new)

        # Baseline model inference (if provided)
        if model_bl is not None:
            result_bl = inference_with_rsp(
                model_bl, backbone_bl if backbone_bl is not None else backbone,
                cat_bl, sample["image"], sup_feat, sup_mask_grid, device,
            )
            vis_bl = make_figure(
                query_img_np, gt_mask, result_bl["rsp_map"], result_bl["mask"],
                f"[BASELINE] {tile_id}", result_bl["score"], cname,
            )
            vis_images_bl.append(vis_bl)

    # ── Save ──
    for i, vis in enumerate(vis_images_new):
        path = output_dir / f"new_{target_cls}_{cname}_tile{i}.png"
        cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"[saved] {path}")

    for i, vis in enumerate(vis_images_bl):
        path = output_dir / f"baseline_{target_cls}_{cname}_tile{i}.png"
        cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"[saved] {path}")

    # ── Summary ──
    n_total = len(vis_images_new) + len(vis_images_bl)
    print(f"\n[OK] Done! {n_total} figures saved to {output_dir}")
    print(f"   Class: {cname} ({target_cls})")
    print(f"   Support tiles: {len(support_images)}")


if __name__ == "__main__":
    main()
