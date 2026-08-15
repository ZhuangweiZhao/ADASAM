"""Tests for the independent non-episodic label-efficient segmentation pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from adasam.datasets.industrial import LabelRatioSubset, fixed_validation_split_indices
from adasam.losses import LabelEfficientSegmentationLoss
from adasam.models import LabelEfficientSAM
from adasam.losses import MagnitudeTeacherDistillationLoss
from tools.train_segmentation import task_utility_routing_loss
from adasam.backbone.mobile_sam import LabelEfficientMobileSAMBackbone


class FakeFrozenBackbone(nn.Module):
    img_size = 224

    def __init__(self) -> None:
        super().__init__()
        self.encoder_trainable = False
        self.stem = nn.Conv2d(3, 256, 1)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = F.adaptive_avg_pool2d(self.stem(image), (14, 14))
        return {
            "P3": F.adaptive_avg_pool2d(embedding[:, :128], (28, 28)),
            "P4": F.adaptive_avg_pool2d(embedding[:, :160], (14, 14)),
            "embedding": embedding,
        }

    def train(self, mode: bool = True):
        return super().train(mode if self.encoder_trainable else False)

    def set_encoder_trainable(self, trainable: bool = True) -> None:
        self.encoder_trainable = bool(trainable)
        for parameter in self.parameters():
            parameter.requires_grad_(self.encoder_trainable)


def test_forward_512_and_batch_four_backward() -> None:
    model = LabelEfficientSAM(FakeFrozenBackbone(), num_classes=4, decoder_dim=32)
    image = torch.rand(4, 3, 512, 512)
    logits = model(image)
    assert logits.shape == (4, 4, 512, 512)
    target = torch.randint(0, 4, (4, 512, 512))
    LabelEfficientSegmentationLoss()(logits, target).backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.adapter.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())


def test_parameter_counts_partition_total() -> None:
    model = LabelEfficientSAM(FakeFrozenBackbone(), num_classes=4, decoder_dim=32)
    counts = model.parameter_counts()
    assert counts["total"] == counts["trainable"] + counts["frozen"]
    assert counts["trainable"] > 0
    assert counts["frozen"] > 0


def test_full_encoder_finetuning_receives_gradients_and_can_refreeze() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version="sum",
    )
    frozen_count = model.parameter_counts()["trainable"]
    model.set_encoder_trainable(True)
    model.train()
    assert model.backbone.training
    assert model.parameter_counts()["trainable"] > frozen_count
    model(torch.rand(2, 3, 64, 64)).mean().backward()
    assert all(parameter.grad is not None for parameter in model.backbone.parameters())

    model.set_encoder_trainable(False)
    model.train()
    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


def test_input_adapter_starts_as_identity_and_receives_gradients() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_input_adapter=True,
    )
    image = torch.rand(1, 3, 64, 64)
    expected = F.interpolate(image, (224, 224), mode="bilinear", align_corners=False)
    adapted = model.input_adapter(expected)
    assert torch.allclose(adapted, expected)
    model(image).mean().backward()
    assert model.input_adapter.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_real_label_efficient_backbone_forward_allows_input_gradients() -> None:
    class Encoder(nn.Module):
        def forward_multi_scale(self, image):
            return {
                "stage1": image,
                "stage2": F.avg_pool2d(image, 2),
                "stage3": F.avg_pool2d(image, 4),
            }

    backbone = LabelEfficientMobileSAMBackbone(Encoder(), img_size=32)
    image = torch.randn(1, 3, 32, 32, requires_grad=True)
    sum(value.mean() for value in backbone(image).values()).backward()
    assert image.grad is not None


def test_feature_scale_ablation_has_clean_parameter_and_gradient_scope() -> None:
    embedding = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32, feature_scales="embedding"
    )
    full = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        feature_scales="p3_p4_embedding",
    )
    assert embedding.parameter_counts()["trainable"] < full.parameter_counts()["trainable"]
    logits = embedding(torch.rand(2, 3, 128, 128))
    logits.mean().backward()
    assert set(embedding.adapter.adapters) == {"embedding"}
    assert set(embedding.decoder.lateral) == {"embedding"}
    assert all(parameter.grad is not None for parameter in embedding.adapter.parameters())


def test_no_adapter_embedding_decoder_forward() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, feature_scales="embedding",
    )
    assert model(torch.rand(2, 3, 128, 128)).shape == (2, 4, 128, 128)


@pytest.mark.parametrize(
    "feature_scales",
    ["p3", "p4", "embedding", "p3_p4", "p4_embedding", "p3_p4_embedding"],
)
def test_hierarchical_feature_probes_forward_backward(feature_scales: str) -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, feature_scales=feature_scales,
    )
    logits = model(torch.rand(2, 3, 128, 128))
    assert logits.shape == (2, 4, 128, 128)
    logits.mean().backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())


@pytest.mark.parametrize("fusion_version", ["concat", "sum", "global", "image_conditioned", "scsr", "scsr_v2", "scsr_task"])
def test_controlled_fusion_variants(fusion_version: str) -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version=fusion_version,
    )
    logits = model(torch.rand(2, 3, 128, 128))
    assert logits.shape == (2, 4, 128, 128)
    logits.mean().backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_semantic_budget_forward_backward_and_routing(budget: int) -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version="semantic_budget",
        representation_budget=budget,
    )
    logits = model(torch.rand(2, 3, 128, 128))
    assert logits.shape == (2, 4, 128, 128)
    logits.mean().backward()
    routing = model.decoder.last_routing
    assert routing is not None
    assert routing["budget"] == budget
    assert routing["weights"].shape[1] == 3
    assert torch.isfinite(routing["entropy"]).all()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_semantic_budget_rejects_incomplete_feature_set() -> None:
    with pytest.raises(ValueError, match="semantic_budget requires"):
        LabelEfficientSAM(
            FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
            use_cat_adapter=False, feature_scales="p4_embedding",
            fusion_version="semantic_budget",
        )


@pytest.mark.parametrize("policy", ["adaptive", "static", "magnitude", "random"])
@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75, 1.0])
def test_semantic_budget_spatial_policies_enforce_retention(policy: str, ratio: float) -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version="semantic_budget",
        representation_budget=3, spatial_policy=policy,
        feature_retention_ratio=ratio,
    )
    model(torch.rand(2, 3, 128, 128))
    routing = model.decoder.last_routing
    assert routing["importance_maps"].shape == routing["retained_masks"].shape
    assert routing["retained_masks"].shape[1] == 2
    spatial_positions = routing["retained_masks"].shape[-2] * routing["retained_masks"].shape[-1]
    expected = max(1, round(spatial_positions * ratio)) / spatial_positions
    assert torch.allclose(
        routing["retention_ratio"],
        torch.full_like(routing["retention_ratio"], expected),
    )
    assert routing["spatial_policy"] == policy
    assert routing["lateral_projected_positions"].shape == (2, 2)
    if ratio < 1.0 and policy != "magnitude":
        assert routing["sparse_lateral_projection"] is True


@pytest.mark.parametrize("ratio", [0.0, 1.1])
def test_semantic_budget_rejects_invalid_retention_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="feature_retention_ratio"):
        LabelEfficientSAM(
            FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
            use_cat_adapter=False, fusion_version="semantic_budget",
            feature_retention_ratio=ratio,
        )


def test_semantic_budget_uses_external_static_template() -> None:
    template = torch.stack((torch.eye(8), torch.flip(torch.eye(8), dims=(0,))))
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version="semantic_budget",
        spatial_policy="static", feature_retention_ratio=0.25,
        static_importance_map=template,
    )
    model(torch.rand(1, 3, 128, 128))
    assert model.decoder.semantic_static_logits is None
    assert torch.equal(model.decoder.semantic_static_template[:, 0], template)
    assert model.decoder.last_routing["retained_masks"].sum() > 0


def test_distilled_magnitude_uses_teacher_only_during_training() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version="semantic_budget",
        spatial_policy="distilled_magnitude", feature_retention_ratio=0.25,
    )
    model.train()
    logits = model(torch.rand(2, 3, 128, 128))
    routing = model.decoder.last_routing
    assert routing["student_logits"].shape == routing["teacher_masks"].shape
    assert routing["teacher_masks"].shape[1] == 2
    assert routing["sparse_lateral_projection"] is False
    assert 0.0 <= float(routing["teacher_student_mask_iou"]) <= 1.0
    loss = logits.mean() + MagnitudeTeacherDistillationLoss()(
        routing["student_logits"], routing["teacher_masks"]
    )
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.decoder.semantic_budget_student.parameters()
    )

    model.eval()
    with torch.no_grad():
        model(torch.rand(2, 3, 128, 128))
    routing = model.decoder.last_routing
    assert "teacher_masks" not in routing
    assert "teacher_importance_maps" not in routing
    assert routing["sparse_lateral_projection"] is True
    assert torch.all(routing["retention_ratio"] == 0.25)


def test_scsr_starts_uniform_and_records_finite_routing() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version="scsr",
    )
    model(torch.rand(2, 3, 128, 128))
    routing = model.decoder.last_routing
    assert routing is not None
    weights = routing["weights"]
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / 3.0))
    assert torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]))
    assert torch.isfinite(routing["entropy"]).all()


def test_scsr_is_under_ten_thousand_additional_parameters() -> None:
    fixed = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version="global",
    )
    scsr = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version="scsr",
    )
    assert scsr.parameter_counts()["trainable"] - fixed.parameter_counts()["trainable"] < 10_000


def test_scsr_v2_starts_symmetric_and_records_temperature() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version="scsr_v2",
    )
    logits = model(torch.rand(2, 3, 128, 128))
    assert logits.shape == (2, 4, 128, 128)
    routing = model.decoder.last_routing
    assert routing is not None
    weights = routing["weights"]
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / 3.0))
    assert torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]))
    assert torch.isfinite(routing["entropy"]).all()
    assert torch.isfinite(routing["temperature"])
    logits.mean().backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_scsr_v2_is_under_ten_thousand_additional_parameters() -> None:
    fixed = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version="global",
    )
    scsr_v2 = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version="scsr_v2",
    )
    assert scsr_v2.parameter_counts()["trainable"] - fixed.parameter_counts()["trainable"] < 10_000


def test_scsr_task_exposes_scale_logits_and_utility_loss() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version="scsr_task",
    )
    prediction = model(torch.rand(2, 3, 128, 128))
    target = torch.randint(0, 4, (2, 128, 128))
    routing = model.decoder.last_routing
    assert routing is not None
    assert len(routing["scale_logits"]) == 3
    losses = task_utility_routing_loss(model, target)
    assert losses is not None
    assert torch.isfinite(losses["route_loss"])
    assert torch.isfinite(losses["aux_loss"])
    assert torch.isfinite(losses["target_entropy"])
    (prediction.mean() + losses["route_loss"] + losses["aux_loss"]).backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_scsr_task_hard_routing_uses_straight_through_one_hot_weights() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        use_cat_adapter=False, fusion_version="scsr_task",
    )
    model.decoder.task_routing_hard = True
    with torch.no_grad():
        model.decoder.scsr_v2_bias.copy_(torch.tensor([10.0, 0.0, 0.0]))
    model(torch.rand(1, 3, 128, 128))
    routing = model.decoder.last_routing
    assert routing is not None
    assert torch.allclose(routing["weights"][:, 0], torch.ones_like(routing["weights"][:, 0]))
    assert torch.allclose(routing["weights"][:, 1:], torch.zeros_like(routing["weights"][:, 1:]))
    assert not torch.allclose(routing["route_weights"], routing["weights"])


def test_scsr_rejects_incomplete_feature_set() -> None:
    for fusion_version in ("scsr", "scsr_v2", "scsr_task"):
        with pytest.raises(ValueError, match="SCSR requires"):
            LabelEfficientSAM(
                FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
                use_cat_adapter=False, feature_scales="p4_embedding", fusion_version=fusion_version,
            )


def test_post_fusion_adapter_forward_and_scope() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32,
        adapter_placement="post_fusion",
    )
    assert model.use_pre_fusion_adapter is False
    assert model.decoder.post_fusion_adapter is not None
    logits = model(torch.rand(2, 3, 128, 128))
    assert logits.shape == (2, 4, 128, 128)
    logits.mean().backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.post_fusion_adapter.parameters())


def test_dapg_v2_dpm_batch_four_forward_backward() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(),
        num_classes=4,
        decoder_dim=32,
        prompt_version="v2",
        num_prompt=8,
        prototype_version="dpm",
    )
    image = torch.rand(4, 3, 512, 512)
    target = torch.randint(0, 4, (4, 512, 512))
    logits, prompts, prototype_aux = model.forward_with_auxiliary(image, target)
    assert logits.shape == (4, 4, 512, 512)
    assert prompts is not None and prompts["dense_prompt"].ndim == 4
    assert prototype_aux is not None
    assert prototype_aux["similarity"].shape[:2] == (4, 4)
    LabelEfficientSegmentationLoss()(logits, target).backward()
    assert model.prototype_memory is not None
    assert model.prototype_memory.prior_projection.weight.grad is not None


def test_label_ratio_subsets_are_nested() -> None:
    dataset = list(range(1000))
    one = LabelRatioSubset(dataset, 1, seed=7)
    five = LabelRatioSubset(dataset, 5, seed=7)
    assert len(one) == 10
    assert len(five) == 50
    assert set(one.indices).issubset(five.indices)


def test_extended_label_ratios_are_nested() -> None:
    dataset = list(range(1000))
    ten = LabelRatioSubset(dataset, 10, seed=7)
    twenty = LabelRatioSubset(dataset, 20, seed=7)
    fifty = LabelRatioSubset(dataset, 50, seed=7)
    assert len(twenty) == 200
    assert len(fifty) == 500
    assert set(ten.indices).issubset(twenty.indices)
    assert set(twenty.indices).issubset(fifty.indices)


def test_fixed_validation_protocol_is_fixed_and_training_sets_are_nested() -> None:
    train_5, validation_5, pool_5 = fixed_validation_split_indices(1000, 5, seed=123)
    train_10, validation_10, pool_10 = fixed_validation_split_indices(1000, 10, seed=123)
    _, validation_other_seed, _ = fixed_validation_split_indices(1000, 10, seed=456)
    assert validation_5 == validation_10 == validation_other_seed
    assert len(validation_5) == 200
    assert len(pool_5) == len(pool_10) == 800
    assert len(train_5) == 40
    assert len(train_10) == 80
    assert set(train_5).issubset(train_10)
    assert set(train_10).isdisjoint(validation_10)
