"""
V2 推理可视化 | V2 Inference Visualization.
============================================

可视化 CandidateGenerator 的 CC 区域、质心、边界框，以及最终解码的掩码。
Visualize CC regions, centroids, bboxes from CandidateGenerator, and final masks.

用法 | Usage::

    python tools/visualize_v2.py \
        --checkpoint runs/train_fold0_k5_all_seed42/best_model.pt \
        --k-shot 5 --seed 42 --cls 1 --max-tiles 5
"""

from __future__ import annotations

import argparse
import json
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

from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets import ISAID_CATEGORIES, ISAIDInstanceDataset
from adasam.decoder import PromptMaskDecoder
from adasam.prototype import PrototypeBuilder
from adasam.prototype.correlation import CorrelationBuilder
from adasam.utils.candidate_generator import CandidateGenerator
from adasam.utils.transforms import preprocess_image, resize_mask
from adasam.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="V2 Inference Visualizer")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cls", type=int, default=None,
                   help="visualize a specific class (None = pick largest)")
    p.add_argument("--max-tiles", type=int, default=5,
                   help="max query tiles to visualize")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def load_gt_instances(coco, image_id: int) -> list[dict]:
    """从 COCO GT 读取该图逐实例掩码 | Per-instance GT masks for an image."""
    anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id, iscrowd=0))
    out = []
    for ann in anns:
        cat = ann.get("category_id", 0)
        if cat < 1 or cat > 15:
            continue
        m = coco.annToMask(ann).astype(bool)
        if m.sum() == 0:
            continue
        out.append({"category_id": cat, "mask": m, "area": float(ann.get("area", m.sum()))})
    return out


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


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    embed_dim = int(cfg.get("prototype", {}).get("embed_dim", 256))
    mtype = cfg.get("backbone", {}).get("model_type", "vit_t")
    weights_path = _REPO_ROOT / cfg.get("backbone", {}).get("checkpoint", "weights/mobile_sam.pt")

    data_root = Path(args.data_root) if args.data_root else (
        _REPO_ROOT / cfg.get("data", {}).get("data_root", "data/iSAID_instance_fewshot"))

    # ── Output dir ──
    run_name = Path(args.checkpoint).parent.name
    out_dir = Path(args.output_dir) if args.output_dir else (
        _REPO_ROOT / "runs" / f"vis_{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vis] output → {out_dir}")

    # ── Build model ──
    sam_ft = build_mobile_sam(str(weights_path), mtype, device)
    backbone = MobileSAMBackbone(sam_ft.image_encoder, sam_ft.image_encoder.img_size).to(device)
    image_size = backbone.img_size

    decoder = PromptMaskDecoder(
        sam_ft.prompt_encoder, sam_ft.mask_decoder,
        embed_dim=embed_dim, image_size=image_size,
    ).to(device)
    decoder.load_state_dict(ckpt["model"])
    decoder.eval()

    proto_builder = PrototypeBuilder(embed_dim)
    correlation = CorrelationBuilder()
    candidate_gen = CandidateGenerator(
        alpha=1.0, min_area=1, max_candidates=64, stride=16.0,
        peak_min_distance=2, max_peaks_per_cc=8,
    )

    # ── Load dataset ──
    dataset = ISAIDInstanceDataset(
        root=str(data_root), split="val", fold=int(cfg["data"].get("fold", 0)), mode="all",
    )

    # ── Load GT COCO ──
    from pycocotools.coco import COCO
    gt_path = str(data_root / "annotations" / "instances_val.json")
    coco_gt = COCO(gt_path)
    stem_to_id = {Path(v["file_name"]).stem: k for k, v in coco_gt.imgs.items()}
    tile_id_to_stem = {k: Path(v).stem for k, v in {
        coco_gt.imgs[k]["id"]: coco_gt.imgs[k]["file_name"]
        for k in coco_gt.imgs}.items()}

    # ── Build class index for sampling ──
    class_to_tiles: dict[int, list[int]] = defaultdict(list)
    for ann in coco_gt.dataset.get("annotations", []):
        cat = ann.get("category_id", 0)
        if 1 <= cat <= 15:
            class_to_tiles[cat].append(ann["image_id"])

    # ── Pick class ──
    visible = sorted(dataset.visible_classes())
    if args.cls is not None and args.cls in visible:
        cls = args.cls
    else:
        # Pick the class with most tiles
        cls = max(visible, key=lambda c: len(class_to_tiles.get(c, [])))
    cls_name = ISAID_CATEGORIES.get(cls, f"cls{cls}")
    print(f"[vis] class = {cls} ({cls_name})")

    # ── Sample support + query tiles ──
    cls_tiles = list(set(class_to_tiles.get(cls, [])))
    rng = random.Random(args.seed)
    rng.shuffle(cls_tiles)

    # Build scene-to-tile index for scene-disjoint sampling
    tile_to_scene = {}
    for img_id in cls_tiles:
        img = coco_gt.imgs[img_id]
        tile_to_scene[img_id] = img.get("orig_image_id", img_id)

    # Pick K support tiles from different scenes
    support_tiles, used_scenes = [], set()
    for tid in cls_tiles:
        scene = tile_to_scene[tid]
        if scene not in used_scenes:
            support_tiles.append(tid)
            used_scenes.add(scene)
        if len(support_tiles) >= args.k_shot:
            break

    # Pick query tiles from different scenes
    query_tiles = []
    for tid in cls_tiles:
        scene = tile_to_scene[tid]
        if scene not in used_scenes and tid not in support_tiles:
            query_tiles.append(tid)
        if len(query_tiles) >= args.max_tiles:
            break

    print(f"[vis] {len(support_tiles)} support tiles, {len(query_tiles)} query tiles")

    # ── Build class prototype + support features ──
    def load_tile_rgb(image_id: int) -> np.ndarray:
        stem = tile_id_to_stem[image_id]
        bgr = cv2.imread(str(data_root / "images" / "val" / f"{stem}.png"), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def embed(rgb: np.ndarray):
        x, meta = preprocess_image(rgb)
        emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
        return emb, meta

    def class_fg_mask(image_id: int, c: int) -> np.ndarray | None:
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id, catIds=[c]))
        if not anns:
            return None
        img = coco_gt.imgs[image_id]
        fg = np.zeros((img["height"], img["width"]), dtype=bool)
        for ann in anns:
            fg |= coco_gt.annToMask(ann).astype(bool)
        return fg

    embs, masks = [], []
    for tid in support_tiles:
        fg = class_fg_mask(tid, cls)
        if fg is None:
            continue
        emb_s, _ = embed(load_tile_rgb(tid))
        embs.append(emb_s[0])
        masks.append(torch.from_numpy(fg).float())

    if len(embs) < args.k_shot:
        print(f"[vis] ERROR: only {len(embs)} valid supports, need {args.k_shot}")
        return

    prototype = proto_builder.build(embs, masks)
    support_feats = torch.stack(embs, dim=0).to(device)  # [K, C, 64, 64]
    prototype = prototype.to(device)

    # ── Visualize each query tile ──
    for qi, qid in enumerate(query_tiles):
        print(f"\n[vis] tile {qi + 1}/{len(query_tiles)}: image_id={qid}")

        rgb = load_tile_rgb(qid)
        emb, meta = embed(rgb)
        H, W = meta.original_size

        gt_insts = load_gt_instances(coco_gt, qid)
        gt_cls_masks = [g["mask"] for g in gt_insts if g["category_id"] == cls]

        # ── Correlation → sim_tensor ──
        sim_tensor = correlation.build(support_feats, prototype, emb, support_masks=masks)  # [K, 64, 64]

        # ── Candidate Generation ──
        candidates = candidate_gen.generate(sim_tensor, emb)

        print(f"  candidates = {candidates.n_candidates}")
        print(f"  sim_tensor range = [{sim_tensor.min().item():.4f}, {sim_tensor.max().item():.4f}]")
        print(f"  sim_tensor mean = {sim_tensor.mean().item():.4f}")

        # ── Decode masks ──
        N = candidates.n_candidates
        point_xy = candidates.coords
        box_xyxy = candidates.boxes
        prompt_token = prototype.unsqueeze(0).expand(N, -1).contiguous()
        labels = torch.ones(N, device=device, dtype=torch.float32)

        with torch.no_grad():
            low_res, iou_pred = decoder.decode_v2(
                emb, point_xy, labels, box_xyxy, prompt_token=prompt_token,
            )
            logits = decoder.upscale_logits(low_res, meta.input_size, meta.original_size)
            pred_masks = (logits > 0.0).cpu().numpy()  # [N, H, W] bool
            iou_scores = iou_pred[:, 0].clamp(0.0, 1.0).cpu().numpy()
            region_scores = candidates.scores.clamp(0.0, 1.0).cpu().numpy()
            final_scores = iou_scores * region_scores

        print(f"  mask areas = [{[int(m.sum()) for m in pred_masks]}]")
        print(f"  scores = {[f'{s:.3f}' for s in final_scores]}")

        # ═══════════════════════════════════════════════════════════
        # Figure Layout (5 panels wide)
        #   Row A: sim (mean), sim (max), binary union, CC labels, GT masks
        #   Row B: candidates + bboxes + centroids on query image
        #   Row C: top-5 predicted masks
        # ═══════════════════════════════════════════════════════════

        sim_np = sim_tensor.cpu().numpy()  # [K, 64, 64]
        sim_mean = sim_np.mean(axis=0)     # [64, 64]
        sim_max = sim_np.max(axis=0)       # [64, 64]

        # Binary union (same logic as CandidateGenerator)
        binary_union = np.zeros((64, 64), dtype=np.uint8)
        alpha_val = 1.0
        for k in range(sim_np.shape[0]):
            s = sim_np[k]
            tau = s.mean() + alpha_val * s.std()
            binary_union |= (s > tau).astype(np.uint8)

        num_labels, labels_cc = cv2.connectedComponents(binary_union, connectivity=8)

        # Upsample grid maps to tile resolution for overlay
        def grid_to_tile(grid_map):
            return cv2.resize(grid_map.astype(np.float32), (W, H),
                              interpolation=cv2.INTER_NEAREST if grid_map.dtype == np.int32
                              else cv2.INTER_LINEAR)

        sim_mean_tile = grid_to_tile(sim_mean)
        sim_max_tile = grid_to_tile(sim_max)
        binary_tile = grid_to_tile(binary_union)

        # Candidate bbox/centroid coords: input frame (1024²) → tile frame (896²)
        scale_x = W / image_size
        scale_y = H / image_size
        cand_centroids_tile = [(int(c[0].item() * scale_x), int(c[1].item() * scale_y))
                                for c in candidates.coords]
        cand_boxes_tile = [(int(b[0].item() * scale_x), int(b[1].item() * scale_y),
                             int(b[2].item() * scale_x), int(b[3].item() * scale_y))
                            for b in candidates.boxes]

        # ── Row A: similarity + union ──
        row_a = []
        # Panel 1: mean sim heatmap
        sim_mean_viz = (sim_mean_tile - sim_mean_tile.min()) / (sim_mean_tile.max() - sim_mean_tile.min() + 1e-8)
        heat = cv2.applyColorMap((sim_mean_viz * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended_mean = cv2.addWeighted(rgb, 0.5, cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), 0.5, 0)
        cv2.putText(blended_mean, "sim_mean", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        row_a.append(blended_mean)

        # Panel 2: max sim heatmap
        sim_max_viz = (sim_max_tile - sim_max_tile.min()) / (sim_max_tile.max() - sim_max_tile.min() + 1e-8)
        heat = cv2.applyColorMap((sim_max_viz * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended_max = cv2.addWeighted(rgb, 0.5, cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), 0.5, 0)
        cv2.putText(blended_max, "sim_max", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        row_a.append(blended_max)

        # Panel 3: binary union
        binary_color = np.zeros((H, W, 3), dtype=np.uint8)
        binary_color[binary_tile > 0] = [0, 255, 128]
        blended_bin = cv2.addWeighted(rgb, 0.7, binary_color, 0.3, 0)
        cv2.putText(blended_bin, "binary union (CC input)", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 128), 1)
        row_a.append(blended_bin)

        # Panel 4: GT masks
        gt_viz = rgb.copy()
        gt_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for gi, gm in enumerate(gt_cls_masks):
            gt_viz = draw_mask_overlay(gt_viz, gm, gt_colors[gi % len(gt_colors)], alpha=0.3)
        cv2.putText(gt_viz, f"GT ({cls_name}) x{len(gt_cls_masks)}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        row_a.append(gt_viz)

        # ── Row B: candidates on RGB ──
        cand_viz = rgb.copy()
        # Draw GT masks faintly
        for gm in gt_cls_masks:
            contours_gt, _ = cv2.findContours(gm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(cand_viz, contours_gt, -1, (0, 255, 0), 1)
        # Draw candidate boxes + centroids
        cand_colors = [
            (255, 0, 0), (0, 180, 255), (255, 180, 0), (180, 0, 255), (0, 255, 255),
            (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255),
        ]
        for ci in range(N):
            col = cand_colors[ci % len(cand_colors)]
            bx1, by1, bx2, by2 = cand_boxes_tile[ci]
            cx, cy = cand_centroids_tile[ci]
            cv2.rectangle(cand_viz, (bx1, by1), (bx2, by2), col, 2)
            cv2.circle(cand_viz, (cx, cy), 6, col, -1)
            cv2.circle(cand_viz, (cx, cy), 8, (255, 255, 255), 1)
        cv2.putText(cand_viz, f"Candidates x{N} (box+centroid, GT=green)", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        row_a.append(cand_viz)

        # ── Row B: top-K predicted masks ──
        top_k = min(5, N)
        sort_idx = np.argsort(final_scores)[::-1]
        cols = min(top_k, 5)
        pred_row = []
        for ri in range(top_k):
            si = sort_idx[ri]
            pred_viz = rgb.copy()
            pm = pred_masks[si]
            pred_viz = draw_mask_overlay(pred_viz, pm, (255, 100, 0), alpha=0.35, border=True)
            # Draw GT contours
            for gm in gt_cls_masks:
                contours_gt, _ = cv2.findContours(gm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(pred_viz, contours_gt, -1, (0, 255, 0), 1)
            cv2.putText(pred_viz, f"pred#{ri} s={final_scores[si]:.3f} area={int(pred_masks[si].sum())}",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            pred_row.append(pred_viz)
        # Pad if less than 5
        while len(pred_row) < 5:
            blank = np.zeros_like(rgb)
            cv2.putText(blank, "(no prediction)", (20, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            pred_row.append(blank)

        # ── Stack rows ──
        # Row A: 5 images (sim_mean, sim_max, binary, GT, candidates)
        # Row B: 5 images (top-5 predictions)
        row_a_img = np.hstack(row_a)
        row_b_img = np.hstack(pred_row)

        # Make panels same width
        target_w = row_a_img.shape[1]
        if row_b_img.shape[1] != target_w:
            row_b_img = cv2.resize(row_b_img, (target_w, row_b_img.shape[0]))

        canvas = np.vstack([row_a_img, row_b_img])

        out_path = out_dir / f"tile{qi:02d}_cls{cls}_{cls_name}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"  saved → {out_path}")

    print(f"\n[vis] done. {len(query_tiles)} tiles saved to {out_dir}")


if __name__ == "__main__":
    main()
