"""
实例分割评估器 V3 | Instance Segmentation Evaluator V3.
========================================================

AdaSAM 唯一评估入口, 严格复用冻结的评估协议 V3 (与 AdaTile-FastSAM 逐字一致的度量语义):
The single evaluation entry of AdaSAM, strictly reusing the frozen Protocol V3
(metric semantics byte-identical to AdaTile-FastSAM):

    1. 实例级, 绝不 union | Instance-level, never union masks.
    2. 一对一贪心匹配 (TP/FP/FN) | greedy one-to-one matching.
    3. COCO AP 家族由官方 pycocotools 计算 (adasam.metrics.coco_eval) | official COCO AP.
    4. Instance mIoU: 每 GT 取最大 IoU 预测再平均 | per-GT max-IoU, then averaged.
    5. Zero-shot 非 oracle: MobileSAM everything-mode 类无关输出 | non-oracle everything-mode.
    6. 固定评估清单 (frozen manifest) → 跨运行/跨模型完全一致的 query 集合。

模型输出如何成为实例 | How model output becomes instances:
    PromptMaskDecoder 对每个"原型相似度峰值"点提示直接解码出一个实例掩码 —— 无需连通域分解、
    无 oracle、无 GT 提示。Each prototype-sim peak point directly decodes one instance mask.

输出 | Output: <output_dir>/instance_metrics.json (schema 与 V3 完全一致 | identical schema).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets import ISAID_CATEGORIES
from adasam.decoder import PromptMaskDecoder
from adasam.logging import get_logger
from adasam.logging.backends import ConsoleBackend, FileBackend
from adasam.metrics import COCOInstanceEvaluator, greedy_match, instance_miou
from adasam.prototype import PrototypeBuilder
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ═══════════════════════════════════════════════════════════════════
# 冻结协议辅助 (语义逐字复用) | Frozen protocol helpers (verbatim semantics)
# ═══════════════════════════════════════════════════════════════════

def _det_hash(s: str) -> int:
    """确定性哈希 (md5, 跨 OS/Python 稳定) | Deterministic md5 hash (stable across OS/Python).

    禁用内置 hash(): 其对 str 带每进程随机盐, 会破坏可复现的采样。
    Avoids built-in hash(), which is per-process salted for str and breaks reproducible sampling.
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def load_manifest(path: Path) -> list[str]:
    """读取评估清单 → tile stem 列表 | Read evaluation manifest → tile stems."""
    with open(path, encoding="utf-8") as f:
        names = json.load(f)
    return [Path(n).stem for n in names]


def save_manifest(path: Path, stems: list[str]) -> None:
    """保存评估清单 (排序后的 png 名) | Save manifest (sorted png names)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f"{s}.png" for s in sorted(stems)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2, ensure_ascii=False)


def load_gt_instances(coco, image_id: int) -> list[dict]:
    """从 COCO GT 读取该图逐实例掩码 | Per-instance GT masks for an image (pycocotools).

    :return: list of {category_id, mask(bool[H,W]), area}; 类别限 1-15 | categories 1-15 only.
    """
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


# ═══════════════════════════════════════════════════════════════════
# 评估器 | Evaluator
# ═══════════════════════════════════════════════════════════════════

class Evaluator:
    """AdaSAM 协议 V3 评估器 | AdaSAM Protocol-V3 evaluator.

    :param args: 解析后的 CLI 参数 | parsed CLI args (see build_arg_parser).
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        set_seed(args.seed)
        self.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                                   else "cpu")
        self.split = args.split
        self.k_shot = args.k_shot

        # ── checkpoint 加载一次, 其 config 作为默认来源 | load checkpoint once; its config is the default source ──
        ckpt = torch.load(args.checkpoint, map_location=self.device)
        cfg = ckpt.get("config", {})
        pcfg = cfg.get("prototype", {})
        self.embed_dim = int(pcfg.get("embed_dim", 256))
        mtype = cfg.get("backbone", {}).get("model_type", "vit_t")
        weights_path = self._resolve(cfg.get("backbone", {}).get("checkpoint", "weights/mobile_sam.pt"))
        self.data_root = Path(args.data_root) if args.data_root else self._resolve(
            cfg.get("data", {}).get("data_root", "data/iSAID_instance_fewshot"))

        # ── 输出 / 日志 | Output / logging ──
        self.out_dir = Path(args.output_dir) if args.output_dir else (
            _REPO_ROOT / "runs" / f"eval_{Path(args.checkpoint).parent.name}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("evaluator")
        if not self.logger.backends:
            self.logger.add_backend(ConsoleBackend())
            self.logger.add_backend(FileBackend(str(self.out_dir / "eval.jsonl")))

        # ── 微调模型: backbone + decoder (从 checkpoint 恢复) | fine-tuned model ──
        sam_ft = build_mobile_sam(weights_path, mtype, self.device)
        self.backbone = MobileSAMBackbone(sam_ft.image_encoder, sam_ft.image_encoder.img_size).to(self.device)
        self.image_size = self.backbone.img_size
        self.decoder = PromptMaskDecoder(
            sam_ft.prompt_encoder, sam_ft.mask_decoder,
            embed_dim=self.embed_dim, image_size=self.image_size,
            top_k_points=args.top_k if args.top_k is not None else int(pcfg.get("top_k_points", 10)),
            sim_threshold=args.sim_threshold if args.sim_threshold is not None
            else float(pcfg.get("sim_threshold", 0.5)),
            min_distance=int(pcfg.get("min_distance", 1)),
            n_proto_tokens=int(pcfg.get("n_proto_tokens", 1)),
        ).to(self.device)
        self.decoder.load_state_dict(ckpt["model"])
        self.decoder.eval()
        self.proto_builder = PrototypeBuilder(self.embed_dim)

        # ── COCO GT + 索引 | COCO GT + indices ──
        gt_path = str(self.data_root / "annotations" / f"instances_{self.split}.json")
        self.ft_eval = COCOInstanceEvaluator(gt_path, iouType="segm")
        self.coco = self.ft_eval.coco_gt
        self.stem_to_id = {Path(v["file_name"]).stem: k for k, v in self.coco.imgs.items()}
        self.class_index = self._build_class_index()

        # ── 零样本 (独立 clean sam, 避免用到微调后的 mask_decoder) | zero-shot clean sam ──
        self.zs_eval = None
        self._zs_generator = None
        if not args.no_zero_shot:
            self.zs_eval = COCOInstanceEvaluator(gt_path, iouType="segm")
            sam_zs = build_mobile_sam(weights_path, mtype, self.device)
            from mobile_sam import SamAutomaticMaskGenerator  # vendored (path already injected)
            self._zs_generator = SamAutomaticMaskGenerator(
                sam_zs, points_per_side=args.zs_points_per_side)

        self.logger.log_info(
            "config",
            f"AdaSAM Eval V3 | K={self.k_shot} seed={args.seed} split={self.split} "
            f"device={self.device} → {self.out_dir}")

    # ── 路径 | Paths ──

    @staticmethod
    def _resolve(path: str | Path) -> Path:
        """相对路径解析到仓库根 | resolve a relative path against the repo root."""
        p = Path(path)
        return p if p.is_absolute() else (_REPO_ROOT / p)

    # ── 类别索引: cls → {源全图 → {tile stem}} | class → {source scene → {stems}} ──

    def _build_class_index(self) -> dict[int, dict[int, set[str]]]:
        idx: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
        for ann in self.coco.dataset.get("annotations", []):
            cat = ann.get("category_id", 0)
            if cat < 1 or cat > 15:
                continue
            img = self.coco.imgs[ann["image_id"]]
            source = img.get("orig_image_id", img["id"])
            idx[cat][source].add(Path(img["file_name"]).stem)
        return idx

    # ── 嵌入 | Embedding ──

    def _embed(self, rgb_uint8: np.ndarray):
        """RGB HWC uint8 → (embedding[1,256,64,64], meta) | frozen embedding + preprocess meta."""
        x, meta = preprocess_image(rgb_uint8)
        emb = self.backbone(x.unsqueeze(0).to(self.device))["image_embedding"]
        return emb, meta

    def _load_tile_rgb(self, stem: str) -> np.ndarray:
        bgr = cv2.imread(str(self.data_root / "images" / self.split / f"{stem}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"tile image not found: {stem}.png")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _class_fg_mask(self, image_id: int, cls: int) -> Optional[torch.Tensor]:
        """该 tile 内某类实例并集 FG 掩码 | union FG mask of a class in a tile."""
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id, catIds=[cls]))
        if not anns:
            return None
        img = self.coco.imgs[image_id]
        fg = np.zeros((img["height"], img["width"]), dtype=bool)
        for ann in anns:
            fg |= self.coco.annToMask(ann).astype(bool)
        return torch.from_numpy(fg).float()

    # ── 每类原型 + query 集合 (场景不相交) | per-class prototypes + query set ──

    def build_prototypes(self, manifest_stems: Optional[set[str]]):
        rng = random.Random(self.args.seed)
        class_protos: dict[int, torch.Tensor] = {}
        query_stems: set[str] = set(manifest_stems) if manifest_stems is not None else set()

        for cls, src_to_stems in sorted(self.class_index.items()):
            sources = list(src_to_stems)
            if len(sources) < self.k_shot + 1:
                continue

            cls_stems = {s for stems in src_to_stems.values() for s in stems}
            if manifest_stems is None:
                # 首次: 确定性采样 query 源图 | first run: deterministic query sampling
                n_q = min(self.args.per_class, len(sources) - self.k_shot)
                q_sources = random.Random(self.args.seed + 99999).sample(sources, n_q)
                for s in q_sources:
                    q_stem = random.Random(self.args.seed + _det_hash(str(s))).choice(
                        sorted(src_to_stems[s]))
                    query_stems.add(q_stem)
                query_src_set = set(q_sources)
            else:
                query_src_set = {img_src for img_src in sources
                                 if src_to_stems[img_src] & manifest_stems}

            support_pool = [s for s in sources if s not in query_src_set]
            if len(support_pool) < self.k_shot:
                continue
            support_sources = rng.sample(support_pool, self.k_shot)
            support_stems = [s for src in support_sources for s in sorted(src_to_stems[src])]

            embs, masks = [], []
            for stem in support_stems:
                iid = self.stem_to_id.get(stem)
                if iid is None:
                    continue
                fg = self._class_fg_mask(iid, cls)
                if fg is None:
                    continue
                emb, _ = self._embed(self._load_tile_rgb(stem))
                embs.append(emb[0])
                masks.append(fg)
            if embs:
                class_protos[cls] = self.proto_builder.build(embs, masks)
                self.logger.log_info("proto", f"class {cls:>2d} ({ISAID_CATEGORIES.get(cls,'?')}): "
                                              f"K={self.k_shot} from {len(support_sources)} scenes")
        return class_protos, query_stems

    # ── 逐 tile 推理 | Per-tile inference ──

    @torch.no_grad()
    def _predict_tile(self, rgb: np.ndarray, class_protos: dict[int, torch.Tensor]):
        """→ {cls: [(mask bool[H,W], score float), ...]} | per-class predicted instances."""
        emb, meta = self._embed(rgb)
        preds: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
        for cls, proto in class_protos.items():
            out = self.decoder(emb, proto, meta.input_size, meta.original_size)
            for i in range(out.masks.shape[0]):
                m = out.masks[i].cpu().numpy()
                if m.sum() < self.args.min_area:
                    continue
                preds[cls].append((m, float(out.scores[i])))
        return preds

    # ── 主流程 | Run ──

    def run(self) -> dict:
        args = self.args
        manifest_path = Path(args.manifest) if args.manifest else (
            self.data_root / f"evaluation_manifest_{self.split}.json")
        manifest_existed = manifest_path.exists()
        manifest_stems = set(load_manifest(manifest_path)) if manifest_existed else None

        class_protos, query_stems = self.build_prototypes(manifest_stems)
        if not manifest_existed:
            save_manifest(manifest_path, sorted(query_stems))
            self.logger.log_info("manifest", f"generated fixed eval set: {len(query_stems)} tiles")

        query_list = sorted(query_stems)
        if args.limit and args.limit < len(query_list):
            self.logger.log_warn("limit",
                                 f"PARTIAL eval: {args.limit}/{len(query_list)} tiles "
                                 f"(debug only, NOT a paper number)")
            query_list = query_list[: args.limit]

        per_class_gt_ious = defaultdict(list)
        per_class_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0, "n_pred": 0})
        evaluated_image_ids = []

        self.logger.log_info("eval", f"evaluating {len(query_list)} query tiles ...")
        for qi, stem in enumerate(query_list):
            image_id = self.stem_to_id.get(stem)
            if image_id is None:
                continue
            evaluated_image_ids.append(image_id)
            rgb = self._load_tile_rgb(stem)

            preds = self._predict_tile(rgb, class_protos)
            for cls, insts in preds.items():
                for mask, score in insts:
                    self.ft_eval.add_prediction(image_id, cls, mask, score)

            gt = load_gt_instances(self.coco, image_id)
            gt_by_class = defaultdict(list)
            for g in gt:
                gt_by_class[g["category_id"]].append(g["mask"])

            for cls in set(gt_by_class) | set(preds):
                gm = gt_by_class.get(cls, [])
                pm = [m for m, _ in preds.get(cls, [])]
                ps = [s for _, s in preds.get(cls, [])]
                if gm:
                    per_gt, _ = instance_miou(pm, gm)
                    per_class_gt_ious[cls].extend(per_gt)
                mres = greedy_match(pm, ps, gm, iou_thr=args.iou_thr)
                c = per_class_counts[cls]
                for k in ("tp", "fp", "fn", "n_gt", "n_pred"):
                    c[k] += mres[k]

            if self.zs_eval is not None:
                for it in self._zero_shot(rgb):
                    self.zs_eval.add_prediction(image_id, 1, it["mask"], it["score"])

            if (qi + 1) % 20 == 0:
                self.logger.log_info("progress", f"{qi + 1}/{len(query_list)} tiles")

        is_full = not (args.limit and args.limit < len(query_stems))
        return self._assemble(evaluated_image_ids, per_class_gt_ious, per_class_counts,
                              manifest_path, is_full_eval=is_full)

    # ── 零样本 everything-mode | Zero-shot everything-mode ──

    def _zero_shot(self, rgb: np.ndarray) -> list[dict]:
        if self._zs_generator is None:
            return []
        out = []
        for m in self._zs_generator.generate(rgb):
            seg = np.asarray(m["segmentation"], dtype=bool)
            if seg.sum() == 0:
                continue
            out.append({"mask": seg, "score": float(m.get("predicted_iou", 1.0))})
        return out

    # ── 指标组装 + 写盘 (schema 与 V3 一致) | Assemble metrics + write (V3 schema) ──

    def _assemble(self, image_ids, per_class_gt_ious, per_class_counts, manifest_path,
                  is_full_eval: bool = True) -> dict:
        self.logger.log_info("coco", "running COCO AP (fine-tuned) ...")
        ft_ap = self.ft_eval.evaluate(verbose=True, image_ids=image_ids)
        ft_ap_ag = self.ft_eval.evaluate_class_agnostic(verbose=False, image_ids=image_ids)
        ft_per_cat = self.ft_eval.get_per_category_ap(image_ids=image_ids)

        all_ious = [x for v in per_class_gt_ious.values() for x in v]
        overall_miou = float(np.mean(all_ious)) if all_ious else 0.0
        per_class_miou = {c: (float(np.mean(v)) if v else 0.0) for c, v in per_class_gt_ious.items()}
        class_mean_miou = float(np.mean(list(per_class_miou.values()))) if per_class_miou else 0.0

        per_class_out = {}
        for cls in sorted(set(per_class_counts) | set(per_class_miou)):
            c = per_class_counts[cls]
            _tp, _fp, _fn = c["tp"], c["fp"], c["fn"]
            _p = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
            _r = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
            _f1 = 2.0 * _p * _r / (_p + _r) if (_p + _r) > 0 else 0.0
            per_class_out[str(cls)] = {
                "name": ISAID_CATEGORIES.get(cls, f"cls{cls}"),
                "n_gt": c["n_gt"], "n_pred": c["n_pred"],
                "tp": _tp, "fp": _fp, "fn": _fn,
                "precision": round(_p, 4), "recall": round(_r, 4), "f1": round(_f1, 4),
                "AP50": round(float(ft_per_cat.get(cls, 0.0)), 4),
                "instance_miou": round(per_class_miou.get(cls, 0.0), 4),
            }

        # overall precision / recall / f1 (aggregated across all classes)
        _all_tp = sum(c["tp"] for c in per_class_counts.values())
        _all_fp = sum(c["fp"] for c in per_class_counts.values())
        _all_fn = sum(c["fn"] for c in per_class_counts.values())
        _op = _all_tp / (_all_tp + _all_fp) if (_all_tp + _all_fp) > 0 else 0.0
        _or = _all_tp / (_all_tp + _all_fn) if (_all_tp + _all_fn) > 0 else 0.0
        _of1 = 2.0 * _op * _or / (_op + _or) if (_op + _or) > 0 else 0.0

        result = {
            "protocol": "instance_v3",
            "checkpoint": self.args.checkpoint,
            "backbone": "mobile_sam_vit_t",
            "k_shot": self.k_shot, "seed": self.args.seed,
            "eval_split": self.split,
            "is_full_eval": is_full_eval,
            "manifest": str(manifest_path),
            "n_query_tiles": len(image_ids),
            "iou_thr": self.args.iou_thr, "min_area": self.args.min_area,
            "finetuned": {
                **{k: round(v, 4) for k, v in ft_ap.items() if k != "n_predictions"},
                "n_predictions": ft_ap["n_predictions"],
                "AP_class_agnostic": round(ft_ap_ag["AP"], 4),
                "AP50_class_agnostic": round(ft_ap_ag["AP50"], 4),
                "instance_miou_overall": round(overall_miou, 4),
                "instance_miou_class_mean": round(class_mean_miou, 4),
                "precision": round(_op, 4),
                "recall": round(_or, 4),
                "f1": round(_of1, 4),
                "per_class": per_class_out,
            },
        }
        if self.zs_eval is not None:
            zs = self.zs_eval.evaluate_class_agnostic(verbose=False, image_ids=image_ids)
            result["zero_shot"] = {
                "AP_class_agnostic": round(zs["AP"], 4),
                "AP50_class_agnostic": round(zs["AP50"], 4),
                "AP75_class_agnostic": round(zs["AP75"], 4),
                "AR_max100": round(zs["AR_max100"], 4),
                "n_predictions": zs["n_predictions"],
                "note": "MobileSAM everything-mode, non-oracle, class-agnostic only",
            }

        (self.out_dir / "instance_metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        for k in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"):
            self.logger.log_metric(f"ft_{k}", float(ft_ap[k]), tags=["instance", "finetuned"])
        self.logger.log_metric("ft_instance_miou", overall_miou, tags=["instance"])
        self.logger.log_info(
            "done",
            f"FT AP={ft_ap['AP']:.4f} AP50={ft_ap['AP50']:.4f} "
            f"AP_agnostic={ft_ap_ag['AP']:.4f} InstMIoU={overall_miou:.4f} "
            f"→ {self.out_dir / 'instance_metrics.json'}")
        self.logger.flush()
        return result


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AdaSAM Instance Segmentation Evaluator V3")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None, help="default: from checkpoint config")
    p.add_argument("--split", default="val")
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--per-class", type=int, default=20, help="max query tiles/class (first-gen only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--min-area", type=int, default=16)
    p.add_argument("--top-k", type=int, default=None, help="override matcher top-k points")
    p.add_argument("--sim-threshold", type=float, default=None, help="override matcher sim threshold")
    p.add_argument("--no-zero-shot", action="store_true")
    p.add_argument("--zs-points-per-side", type=int, default=16, help="everything-mode grid density")
    p.add_argument("--limit", type=int, default=0, help="cap query tiles (debug only, 0 = full)")
    return p
