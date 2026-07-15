"""
推理可视化 | Inference Visualization.
======================================

可视化 Dense Prompt Generation 管线: DPG 64² 内部掩码热图、objectness 分数分布、
GT 实例、最终 SAM 解码掩码 (按 score 着色)。
Visualize the Dense Prompt Generation pipeline: DPG 64² internal-mask heatmap,
objectness score distribution, GT instances, and final SAM-decoded masks
(colored by score).

用法 | Usage::

    python tools/visualize.py \
        --checkpoint runs/train_fold0_k5_all_seed42/best_model.pt \
        --k-shot 5 --seed 42 --cls 1 --max-tiles 5
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter  # noqa: E402
from adasam.backbone import build_mobile_sam, MobileSAMBackbone  # noqa: E402
from adasam.datasets import ISAID_CATEGORIES  # noqa: E402
from adasam.model import AdaSAMModel, AdaSAMModelConfig  # noqa: E402
from adasam.prototype import PrototypeBuilder  # noqa: E402
from adasam.utils import set_seed  # noqa: E402
from adasam.utils.transforms import preprocess_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AdaSAM Dense-Prompt Inference Visualizer")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cls", type=int, default=None,
                   help="visualize a specific class (None = pick largest)")
    p.add_argument("--max-tiles", type=int, default=5,
                   help="max query tiles to visualize")
    p.add_argument("--score-thr", type=float, default=0.3,
                   help="filter threshold on sigmoid(objectness) × iou_pred")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def draw_mask_overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.4,
                      border: bool = True) -> np.ndarray:
    """在 RGB 图上叠加彩色掩码 | Overlay colored mask on RGB image."""
    overlay = rgb.copy()
    overlay[mask] = color
    result = cv2.addWeighted(rgb, 1 - alpha, overlay, alpha, 0)
    if border:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
    return result


def score_bar_panel(scores: np.ndarray, score_thr: float, size: tuple[int, int]) -> np.ndarray:
    """objectness×iou 分数条形图面板 | score bar-chart panel (sorted descending)."""
    h, w = size
    panel = np.full((h, w, 3), 24, dtype=np.uint8)
    n = len(scores)
    if n == 0:
        return panel
    order = np.argsort(scores)[::-1]
    bar_w = max(1, (w - 20) // n)
    base_y = h - 30
    for rank, idx in enumerate(order):
        s = float(scores[idx])
        bh = int(s * (h - 60))
        x0 = 10 + rank * bar_w
        color = (0, 220, 120) if s >= score_thr else (110, 110, 110)
        cv2.rectangle(panel, (x0, base_y - bh), (x0 + bar_w - 1, base_y), color, -1)
    thr_y = base_y - int(score_thr * (h - 60))
    cv2.line(panel, (10, thr_y), (w - 10, thr_y), (255, 200, 0), 1)
    kept = int((scores >= score_thr).sum())
    cv2.putText(panel, f"query scores: {kept}/{n} kept (thr={score_thr})", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return panel


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    if "prompt_generator" not in cfg:
        raise ValueError(f"old-format checkpoint (no 'prompt_generator' section): {args.checkpoint}")
    embed_dim = int(cfg.get("prototype", {}).get("embed_dim", 256))
    mtype = cfg.get("backbone", {}).get("model_type", "vit_t")
    weights_path = _REPO_ROOT / cfg.get("backbone", {}).get("checkpoint", "weights/mobile_sam.pt")
    data_root = Path(args.data_root) if args.data_root else Path(
        cfg.get("data", {}).get("data_root", "data/iSAID_instance_fewshot"))

    # ── Output dir ──
    run_name = Path(args.checkpoint).parent.name
    out_dir = Path(args.output_dir) if args.output_dir else (
        _REPO_ROOT / "runs" / f"vis_{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vis] output → {out_dir}")

    # ── Build model ──
    sam = build_mobile_sam(str(weights_path), mtype, device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(cfg)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    proto_builder = PrototypeBuilder(embed_dim)

    cat_adapter = None
    if "cat_adapter" in ckpt:
        adapter_cfg = cfg.get("train", {}).get("cat_adapter", {})
        cat_adapter = CATAdapter(
            dim=embed_dim, bottleneck=int(adapter_cfg.get("bottleneck", 64)),
        ).to(device)
        cat_adapter.load_state_dict(ckpt["cat_adapter"])
        cat_adapter.eval()
        print(f"[vis] CAT-Adapter loaded: "
              f"params={sum(p.numel() for p in cat_adapter.parameters()):,}")

    def embed(rgb: np.ndarray):
        x, meta = preprocess_image(rgb)
        with torch.no_grad():
            emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if cat_adapter is not None:
                emb = cat_adapter(emb)
        return emb, meta

    # ── Load GT COCO ──
    from pycocotools.coco import COCO
    coco_gt = COCO(str(data_root / "annotations" / "instances_val.json"))
    tile_id_to_stem = {k: Path(v["file_name"]).stem for k, v in coco_gt.imgs.items()}

    class_to_tiles: dict[int, list[int]] = defaultdict(list)
    for ann in coco_gt.dataset.get("annotations", []):
        cat = ann.get("category_id", 0)
        if 1 <= cat <= 15:
            class_to_tiles[cat].append(ann["image_id"])

    # ── Pick class ──
    if args.cls is not None and args.cls in class_to_tiles:
        cls = args.cls
    else:
        cls = max(class_to_tiles, key=lambda c: len(class_to_tiles[c]))
    cls_name = ISAID_CATEGORIES.get(cls, f"cls{cls}")
    print(f"[vis] class = {cls} ({cls_name})")

    # ── Sample support + query tiles (scene-disjoint) ──
    cls_tiles = list(set(class_to_tiles[cls]))
    rng = random.Random(args.seed)
    rng.shuffle(cls_tiles)
    tile_to_scene = {tid: coco_gt.imgs[tid].get("orig_image_id", tid) for tid in cls_tiles}

    support_tiles, used_scenes = [], set()
    for tid in cls_tiles:
        if tile_to_scene[tid] not in used_scenes:
            support_tiles.append(tid)
            used_scenes.add(tile_to_scene[tid])
        if len(support_tiles) >= args.k_shot:
            break
    query_tiles = []
    for tid in cls_tiles:
        if tile_to_scene[tid] not in used_scenes and tid not in support_tiles:
            query_tiles.append(tid)
        if len(query_tiles) >= args.max_tiles:
            break
    print(f"[vis] {len(support_tiles)} support tiles, {len(query_tiles)} query tiles")

    def load_tile_rgb(image_id: int) -> np.ndarray:
        stem = tile_id_to_stem[image_id]
        bgr = cv2.imread(str(data_root / "images" / "val" / f"{stem}.png"), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def class_fg_mask(image_id: int) -> np.ndarray | None:
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id, catIds=[cls]))
        if not anns:
            return None
        img = coco_gt.imgs[image_id]
        fg = np.zeros((img["height"], img["width"]), dtype=bool)
        for ann in anns:
            fg |= coco_gt.annToMask(ann).astype(bool)
        return fg

    # ── Prototype ──
    embs, masks = [], []
    for tid in support_tiles:
        fg = class_fg_mask(tid)
        if fg is None:
            continue
        emb_s, _ = embed(load_tile_rgb(tid))
        embs.append(emb_s[0])
        masks.append(torch.from_numpy(fg).float())
    if len(embs) < args.k_shot:
        print(f"[vis] ERROR: only {len(embs)} valid supports, need {args.k_shot}")
        return
    prototype = proto_builder.build(embs, masks).to(device)

    # ── Visualize each query tile ──
    for qi, qid in enumerate(query_tiles):
        print(f"\n[vis] tile {qi + 1}/{len(query_tiles)}: image_id={qid}")
        rgb = load_tile_rgb(qid)
        H, W = rgb.shape[:2]
        emb, meta = embed(rgb)

        with torch.no_grad():
            dpg_out, low_res, iou_pred = model.forward_train(emb, prototype)
            scores_all = (dpg_out.objectness_logits.sigmoid()
                          * iou_pred[:, 0].clamp(0.0, 1.0)).cpu().numpy()
            keep = scores_all >= args.score_thr
            logits = model.sam_decoder.upscale_logits(
                low_res, meta.input_size, meta.original_size)
            pred_masks = (logits > 0.0).cpu().numpy()             # [N, H, W]
            dpg_heat = dpg_out.mask_logits.sigmoid().max(dim=0).values.cpu().numpy()  # [64,64]

        n_kept = int(keep.sum())
        print(f"  queries={len(scores_all)}, kept={n_kept}, "
              f"score range=[{scores_all.min():.3f}, {scores_all.max():.3f}]")

        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=qid, catIds=[cls], iscrowd=0))
        gt_cls_masks = [coco_gt.annToMask(a).astype(bool) for a in gt_anns]
        gt_cls_masks = [m for m in gt_cls_masks if m.sum() > 0]

        # ── Panel 1: DPG 64² internal-mask heatmap ──
        heat = cv2.resize(dpg_heat, (W, H), interpolation=cv2.INTER_LINEAR)
        heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        panel_heat = cv2.addWeighted(rgb, 0.55, heat_color, 0.45, 0)
        cv2.putText(panel_heat, "DPG 64x64 mask heatmap (max over queries)", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ── Panel 2: objectness × iou score bars ──
        panel_scores = score_bar_panel(scores_all, args.score_thr, (H, W))

        # ── Panel 3: GT ──
        panel_gt = rgb.copy()
        gt_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for gi, gm in enumerate(gt_cls_masks):
            panel_gt = draw_mask_overlay(panel_gt, gm, gt_colors[gi % len(gt_colors)], alpha=0.3)
        cv2.putText(panel_gt, f"GT ({cls_name}) x{len(gt_cls_masks)}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ── Panel 4: kept predictions (colored by rank) + GT contours ──
        panel_pred = rgb.copy()
        kept_idx = np.argsort(scores_all)[::-1]
        kept_idx = [i for i in kept_idx if keep[i]]
        for rank, i in enumerate(kept_idx):
            hue = (rank * 137) % 360
            hsv = np.uint8([[[hue // 2, 220, 240]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            color = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            panel_pred = draw_mask_overlay(panel_pred, pred_masks[i], color, alpha=0.25)
        for gm in gt_cls_masks:
            contours, _ = cv2.findContours(gm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel_pred, contours, -1, (0, 255, 0), 2)
        cv2.putText(panel_pred, f"Predictions x{n_kept} (GT=green)", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        canvas = np.hstack([panel_heat, panel_scores, panel_gt, panel_pred])
        out_path = out_dir / f"tile{qi:02d}_cls{cls}_{cls_name}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"  saved → {out_path}")

    print(f"\n[vis] done. {len(query_tiles)} tiles saved to {out_dir}")


if __name__ == "__main__":
    main()
