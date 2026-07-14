"""
adasam.metrics — 评测指标 (V3 冻结核心) | Evaluation Metrics (V3 frozen core).
==============================================================================

只包含论文评估协议 V3 所需的**冻结**度量, 语义与 AdaTile-FastSAM 逐字一致:
Only the **frozen** metrics required by Evaluation Protocol V3, byte-identical
semantics to AdaTile-FastSAM:

    - COCOInstanceEvaluator — 官方 pycocotools AP 封装 | official pycocotools AP wrapper
    - pairwise_iou / greedy_match / instance_miou — 纯 numpy 实例级度量 | pure-numpy metrics

设计约束 | Design constraint:
    coco_eval.py 是**唯一**允许直接调用 pycocotools.COCOeval 的文件。
    coco_eval.py is the ONLY file allowed to call pycocotools.COCOeval directly.
"""

from adasam.metrics.coco_eval import (
    COCOInstanceEvaluator,
    mask_to_bbox,
)
from adasam.metrics.instance_match import (
    pairwise_iou,
    greedy_match,
    instance_miou,
)

__all__ = [
    "COCOInstanceEvaluator",
    "mask_to_bbox",
    "pairwise_iou",
    "greedy_match",
    "instance_miou",
]
