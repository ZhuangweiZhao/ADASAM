"""
COCO 实例分割评估器 | COCO Instance Segmentation Evaluator.
=============================================================

封装 pycocotools COCOeval，提供标准实例分割指标：
Wraps pycocotools COCOeval to provide standard instance segmentation metrics:

    - AP @ [IoU=0.50:0.95] (主要 COCO 指标 | main COCO metric)
    - AP50, AP75
    - AP_small, AP_medium, AP_large (按目标大小 | by object size)
    - AR_max1, AR_max10, AR_max100

用法 | Usage::

    from adasam.metrics.coco_eval import COCOInstanceEvaluator

    evaluator = COCOInstanceEvaluator("path/to/instances_val.json", iouType="segm")
    for image_id, preds in predictions.items():
        for pred in preds:
            evaluator.add_prediction(
                image_id=image_id,
                category_id=pred["category_id"],
                mask=pred["mask"],         # [H, W] bool numpy array
                score=pred["score"],       # float [0, 1]
            )
    results = evaluator.evaluate()
    print(f"AP: {results['AP']:.4f}, AP50: {results['AP50']:.4f}")

与 FastSAM 原生 COCOeval 的关系 | Relationship with FastSAM's COCOeval:
    本模块独立封装，不依赖 FastSAM thirdLibrary。
    与 thirdLibrary/FastSAM/ultralytics/yolo/v8/segment/val.py 功能等价，
    但提供更简洁的 API 和单张图像添加接口。
    This module is self-contained, independent of FastSAM thirdLibrary.
    Functionally equivalent to FastSAM's COCOeval but with a cleaner API.
"""

from __future__ import annotations

import copy
import logging
import numpy as np
from typing import Optional


def _json_default_bytes(o):
    """json.dump 兜底: 将 bytes 解码为 str | json.dump fallback: decode bytes to str.

    pycocotools 的 COCOeval(iouType='segm') 会就地把 GT 的 polygon 分割替换为 RLE dict,
    其 counts 为 bytes; 因此 evaluate() 之后深拷贝的 GT 含 bytes, 需在序列化时解码.
    COCOeval(segm) mutates GT segmentations in place to RLE dicts whose counts are bytes;
    a GT deep-copied after evaluate() therefore contains bytes that must be decoded on dump.
    """
    if isinstance(o, bytes):
        return o.decode("utf-8")
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _summarize(coco_eval, verbose: bool) -> None:
    """调用 COCOeval.summarize() 以填充 .stats; 非 verbose 时抑制其 stdout.
    Call COCOeval.summarize() to populate .stats; suppress its stdout when not verbose.

    注意: .stats 仅由 summarize() 填充, 因此即使静默也必须调用, 否则 stats 为空列表.
    Note: .stats is only populated by summarize(), so it must run even when quiet,
    otherwise .stats is an empty list (→ IndexError downstream).
    """
    if verbose:
        coco_eval.summarize()
    else:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.summarize()


def mask_to_bbox(mask: np.ndarray) -> list[float]:
    """
    二值掩码 → COCO 格式 bbox [x, y, width, height]。
    Binary mask → COCO-format bbox [x, y, width, height].

    使用轮廓查找获取紧凑边界框。| Uses contour finding for tight bounding box.

    :param mask: [H, W] bool 或 uint8 二值掩码 | bool or uint8 binary mask.
    :return: [x, y, w, h] bbox in COCO format (top-left corner + size).
    """
    import cv2

    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [0.0, 0.0, 0.0, 0.0]

    # 取最大轮廓的边界框 | Use the largest contour's bounding box
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return [float(x), float(y), float(w), float(h)]


class COCOInstanceEvaluator:
    """
    COCO 实例分割评估器 | COCO Instance Segmentation Evaluator.

    封装 pycocotools 的 COCO / COCOeval，用于标准实例分割评估。
    Wraps pycocotools COCO / COCOeval for standard instance segmentation evaluation.

    使用流程 | Usage Flow:
        1. __init__(gt_anno_path) — 加载 GT 标注 | Load GT annotations
        2. add_prediction(...) — 逐实例添加预测 | Add predictions instance by instance
        3. evaluate() — 运行 COCOeval，返回标准指标 | Run COCOeval, return standard metrics
        4. reset() — 清空预测缓存 | Clear prediction cache

    Parameters
    ----------
    gt_anno_path : str
        COCO 格式 GT 标注文件路径 (e.g. "instances_val.json").
        Path to COCO-format GT annotation file.
    iouType : str
        评估类型: "segm" (掩码) 或 "bbox" (边界框).
        Evaluation type: "segm" (mask) or "bbox" (bounding box).
    """

    def __init__(self, gt_anno_path: str, iouType: str = "segm"):
        from pycocotools.coco import COCO

        self.coco_gt = COCO(gt_anno_path)
        self.iouType = iouType
        self.predictions: list[dict] = []

        # ── 缓存 GT 类别集合 | Cache GT category set ──
        self._gt_cat_ids = set(self.coco_gt.getCatIds())

        # ── 日志 | Log ──
        n_imgs = len(self.coco_gt.getImgIds())
        n_cats = len(self._gt_cat_ids)
        n_anns = len(self.coco_gt.getAnnIds())
        print(f"[COCOInstanceEvaluator] Loaded GT: {n_imgs} images, "
              f"{n_cats} categories, {n_anns} annotations (iouType={iouType})")

    def add_prediction(
        self,
        image_id: int,
        category_id: int,
        mask: np.ndarray,
        score: float,
        bbox: Optional[list[float]] = None,
    ) -> None:
        """
        添加单个实例预测 | Add a single instance prediction.

        将二值掩码 RLE 编码后存储。每个预测对应一个目标实例。
        RLE-encodes the binary mask and stores it. One prediction per object instance.

        :param image_id: COCO 图像 ID（与 GT 中一致）| COCO image ID (must match GT).
        :param category_id: 类别 ID（1-15 for iSAID）| Category ID.
        :param mask: [H, W] bool 或 uint8 二值掩码 | bool or uint8 binary mask.
        :param score: 置信度 [0, 1] | Confidence score.
        :param bbox: [x, y, w, h] COCO 格式边界框。None → 自动从 mask 计算。
            [x, y, w, h] COCO-format bbox. None → auto-compute from mask.
        """
        from pycocotools.mask import encode

        # ── 输入验证 | Input validation ──
        if mask.sum() == 0:
            return  # 跳过空掩码 | Skip empty masks

        # ── RLE 编码 | RLE Encoding ──
        # pycocotools 要求 Fortran-order uint8 数组
        # pycocotools requires Fortran-order uint8 array
        rle = encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")

        # ── Bbox (如果需要) | Bbox (if needed) ──
        if bbox is None:
            bbox = mask_to_bbox(mask)

        self.predictions.append({
            "image_id": int(image_id),
            "category_id": int(category_id),
            "segmentation": rle,
            "score": float(np.clip(score, 0.0, 1.0)),
            "bbox": [round(x, 3) for x in bbox],
        })

    def add_predictions_batch(
        self,
        image_id: int,
        masks: list[np.ndarray],
        category_ids: list[int],
        scores: list[float],
        bboxes: Optional[list[list[float]]] = None,
    ) -> None:
        """
        批量添加同一张图像的多个实例预测 | Batch-add multiple instance predictions for one image.

        :param image_id: COCO 图像 ID | COCO image ID.
        :param masks: 二值掩码列表 | List of binary masks.
        :param category_ids: 类别 ID 列表 | List of category IDs.
        :param scores: 置信度列表 | List of confidence scores.
        :param bboxes: 边界框列表 (可选) | List of bboxes (optional).
        """
        if bboxes is None:
            bboxes = [None] * len(masks)

        for mask, cat_id, score, bbox in zip(masks, category_ids, scores, bboxes):
            self.add_prediction(image_id, cat_id, mask, score, bbox)

    def evaluate(self, verbose: bool = True,
                 image_ids: Optional[list[int]] = None) -> dict:
        """
        运行 COCO 评估 | Run COCO evaluation.

        :param verbose: 是否打印 COCOeval 的标准输出 | Whether to print COCOeval's standard output.
        :param image_ids: 若给定, 仅在这些图像上评估 (AP 分母 = 这些图像).
            None → 在 GT 全部图像上评估 (原行为, 向后兼容).
            If given, evaluate only on these images (AP denominator = these images).
            None → evaluate on all GT images (original behavior, backward compatible).
        :return: dict with keys:
            - "AP": AP @ IoU=0.50:0.95 (area=all, maxDets=100)
            - "AP50": AP @ IoU=0.50
            - "AP75": AP @ IoU=0.75
            - "AP_small": AP for small objects (area < 32²)
            - "AP_medium": AP for medium objects (32² <= area < 96²)
            - "AP_large": AP for large objects (area >= 96²)
            - "AR_max1": AR given 1 detection per image
            - "AR_max10": AR given 10 detections per image
            - "AR_max100": AR given 100 detections per image
            - "n_predictions": 评估的预测总数 | Total predictions evaluated
        """
        from pycocotools.cocoeval import COCOeval

        if len(self.predictions) == 0:
            print("[COCOInstanceEvaluator] WARNING: 0 predictions — returning zeros")
            return {
                "AP": 0.0, "AP50": 0.0, "AP75": 0.0,
                "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0,
                "AR_max1": 0.0, "AR_max10": 0.0, "AR_max100": 0.0,
                "n_predictions": 0,
            }

        # ── 加载预测为 COCO 对象 | Load predictions as COCO object ──
        coco_pred = self.coco_gt.loadRes(self.predictions)

        # ── 运行 COCOeval | Run COCOeval ──
        coco_eval = COCOeval(self.coco_gt, coco_pred, self.iouType)
        # 限定评估图像集 (AP 分母 = 评估的 query tiles) | Restrict evaluated image set
        if image_ids is not None:
            coco_eval.params.imgIds = sorted(int(i) for i in image_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()

        # summarize() 填充 .stats (即使静默也必须调用) | populate .stats (must run even when quiet)
        _summarize(coco_eval, verbose)

        # ── 提取指标 | Extract metrics ──
        # COCOeval.stats = [AP, AP50, AP75, AP_small, AP_medium, AP_large,
        #                    AR_max1, AR_max10, AR_max100, ...]
        stats = coco_eval.stats

        return {
            "AP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "AP_small": float(stats[3]),
            "AP_medium": float(stats[4]),
            "AP_large": float(stats[5]),
            "AR_max1": float(stats[6]),
            "AR_max10": float(stats[7]),
            "AR_max100": float(stats[8]),
            "n_predictions": len(self.predictions),
        }

    def reset(self) -> None:
        """清空预测缓存 | Clear prediction cache."""
        self.predictions = []

    def evaluate_class_agnostic(self, verbose: bool = True,
                                image_ids: Optional[list[int]] = None) -> dict:
        """
        Class-agnostic COCO 评估 | Class-Agnostic COCO Evaluation.

        将所有 GT 和预测的 category_id 统一映射为 1，
        评估 segment-anything 能力（不考虑类别分类正确性）。
        Remaps all GT and prediction category_ids to 1,
        evaluating segment-anything capability regardless of class accuracy.

        这通过创建临时 GT 拷贝实现，不修改原始 COCO GT 对象。
        This creates a temporary GT copy, leaving the original COCO GT unchanged.

        :param image_ids: 若给定, 仅在这些图像上评估 | If given, evaluate only on these images.
        :return: same dict format as evaluate().
        """
        import copy, tempfile, json, os

        if len(self.predictions) == 0:
            return self.evaluate(verbose=False)

        # ── 复制预测并全部映射为 cls=1 | Copy preds and remap all to cls=1 ──
        preds_remapped = copy.deepcopy(self.predictions)
        for p in preds_remapped:
            p["category_id"] = 1

        # ── 创建 class-agnostic GT (全部映射为 cls=1)
        # Create class-agnostic GT (all remapped to cls=1)
        gt_agnostic = copy.deepcopy(self.coco_gt.dataset)
        for ann in gt_agnostic.get("annotations", []):
            ann["category_id"] = 1
        for cat in gt_agnostic.get("categories", []):
            cat["id"] = 1
            cat["name"] = "object"
        # 去重 categories | Deduplicate categories
        seen = set()
        unique_cats = []
        for cat in gt_agnostic.get("categories", []):
            if cat["id"] not in seen:
                unique_cats.append(cat)
                seen.add(cat["id"])
        gt_agnostic["categories"] = unique_cats

        # ── 写入临时文件 | Write to temp file ──
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(gt_agnostic, f, default=_json_default_bytes)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            coco_gt_ag = COCO(f.name)
            coco_pred = coco_gt_ag.loadRes(preds_remapped)
            coco_eval = COCOeval(coco_gt_ag, coco_pred, self.iouType)
            coco_eval.params.catIds = [1]
            if image_ids is not None:
                coco_eval.params.imgIds = sorted(int(i) for i in image_ids)
            coco_eval.evaluate()
            coco_eval.accumulate()

            if verbose:
                print("\n--- Class-Agnostic COCO Evaluation ---")
            _summarize(coco_eval, verbose)

            stats = coco_eval.stats
            return {
                "AP": float(stats[0]),
                "AP50": float(stats[1]),
                "AP75": float(stats[2]),
                "AP_small": float(stats[3]),
                "AP_medium": float(stats[4]),
                "AP_large": float(stats[5]),
                "AR_max1": float(stats[6]),
                "AR_max10": float(stats[7]),
                "AR_max100": float(stats[8]),
                "n_predictions": len(preds_remapped),
            }
        finally:
            os.unlink(f.name)  # 清理临时文件 | Clean up temp file

    def get_per_category_ap(self, image_ids: Optional[list[int]] = None) -> dict[int, float]:
        """
        获取每类 AP@50 | Get per-category AP@50.

        对每个类别单独运行 COCOeval。注意：此操作较慢（每次 evaluate 都重建索引）。
        Get per-category AP@50 by running COCOeval per category.
        Note: this is relatively slow (rebuilds index for each evaluate).

        :param image_ids: 若给定, 仅在这些图像上评估 | If given, evaluate only on these images.
        :return: {category_id: AP50} 映射.
        """
        if len(self.predictions) == 0:
            return {}

        from pycocotools.cocoeval import COCOeval

        per_cat_ap = {}
        all_preds = self.predictions
        _img_ids = None if image_ids is None else sorted(int(i) for i in image_ids)

        for cat_id in sorted(self._gt_cat_ids):
            # 过滤出该类别的预测 | Filter predictions for this category
            cat_preds = [p for p in all_preds if p["category_id"] == cat_id]
            if not cat_preds:
                per_cat_ap[cat_id] = 0.0
                continue

            try:
                # 使用独立的 GT 副本, 避免被之前的 evaluate() 修改影响
                # Use a clean GT copy — avoids mutation from prior evaluate() calls.
                from pycocotools.coco import COCO
                coco_gt_clean = COCO()
                coco_gt_clean.dataset = copy.deepcopy(self.coco_gt.dataset)
                coco_gt_clean.createIndex()
                coco_pred = coco_gt_clean.loadRes(cat_preds)
                coco_eval = COCOeval(coco_gt_clean, coco_pred, self.iouType)
                coco_eval.params.catIds = [cat_id]
                if _img_ids is not None:
                    coco_eval.params.imgIds = _img_ids
                coco_eval.evaluate()
                coco_eval.accumulate()
                _summarize(coco_eval, verbose=False)      # populate .stats (required before indexing)
                per_cat_ap[cat_id] = float(coco_eval.stats[1])  # AP50
            except (IndexError, KeyError, RuntimeError) as exc:
                logging.warning("per_category_ap: category %d failed — %s: %s",
                                cat_id, type(exc).__name__, exc)
                per_cat_ap[cat_id] = 0.0

        return per_cat_ap

    def __repr__(self) -> str:
        return (f"COCOInstanceEvaluator(gt={len(self.coco_gt.getImgIds())}imgs, "
                f"preds={len(self.predictions)}, iouType={self.iouType})")
