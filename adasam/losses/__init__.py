"""adasam.losses — 语义分割损失 | Semantic segmentation losses (L_main + L_prior + L_reg)."""

from adasam.losses.seg_losses import (
    focal_loss,
    dice_loss,
    combined_loss,
    mask_iou,
)
from adasam.losses.semantic_loss import SemanticSegLoss
from adasam.losses.prototype_query_loss import PrototypeQuerySemanticLoss
from adasam.losses.label_efficient_loss import (
    DefectPromptAlignmentLoss,
    LabelEfficientSegmentationLoss,
    PrototypeCompactnessLoss,
)
from adasam.losses.boundary_loss import BoundaryLoss, boundary_f1_counts, semantic_boundary_target
from adasam.losses.budget_distillation_loss import MagnitudeTeacherDistillationLoss
from adasam.losses.region_loss import LovaszSoftmaxLoss

__all__ = [
    "focal_loss",
    "dice_loss",
    "combined_loss",
    "mask_iou",
    "SemanticSegLoss",
    "PrototypeQuerySemanticLoss",
    "LabelEfficientSegmentationLoss",
    "DefectPromptAlignmentLoss",
    "PrototypeCompactnessLoss",
    "BoundaryLoss",
    "boundary_f1_counts",
    "semantic_boundary_target",
    "MagnitudeTeacherDistillationLoss",
    "LovaszSoftmaxLoss",
]
