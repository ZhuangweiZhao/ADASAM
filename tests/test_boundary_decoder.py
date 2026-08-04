from __future__ import annotations

import torch

from adasam.losses import BoundaryLoss, boundary_f1_counts, semantic_boundary_target
from adasam.models.decoder import BoundaryAwareSemanticDecoder, LightweightSemanticDecoder


def decoder_features(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "P3": torch.randn(batch_size, 128, 16, 16),
        "P4": torch.randn(batch_size, 160, 8, 8),
        "embedding": torch.randn(batch_size, 256, 4, 4),
    }


def test_boundary_target_excludes_ignore_transitions() -> None:
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, 2:6, 2:6] = 1
    target[:, :, 6:] = 255
    boundary, valid = semantic_boundary_target(target, ignore_index=255)
    assert not boundary[:, :, 6:].any()
    assert not valid[:, :, 6:].any()
    assert boundary[:, 2:6, 2:6].any()


def test_boundary_loss_is_finite_and_differentiable() -> None:
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.randint(0, 7, (2, 16, 16))
    target[:, :3, :3] = 255
    loss = BoundaryLoss(ignore_index=255)(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_boundary_f1_is_one_for_identical_masks() -> None:
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, 4:12, 4:12] = 1
    matched_pred, predicted, matched_target, target_count = boundary_f1_counts(
        target, target, tolerance=2
    )
    assert matched_pred == predicted
    assert matched_target == target_count
    assert predicted > 0


def test_boundary_decoder_zero_initialized_fusion_and_parameter_budget() -> None:
    auxiliary = BoundaryAwareSemanticDecoder(7, enable_boundary_fusion=False)
    boundary = BoundaryAwareSemanticDecoder(7, enable_boundary_fusion=True)
    boundary.load_state_dict(auxiliary.state_dict())
    features = decoder_features()
    auxiliary_logits = auxiliary(features, (64, 64))
    boundary_logits = boundary(features, (64, 64))
    assert auxiliary_logits.shape == (2, 7, 64, 64)
    assert boundary.last_boundary_logits is not None
    assert boundary.last_boundary_logits.shape == (2, 1, 64, 64)
    assert float(boundary.alpha.detach()) == 0.0
    assert torch.equal(auxiliary_logits, boundary_logits)

    baseline = LightweightSemanticDecoder(7)
    added = sum(p.numel() for p in boundary.parameters()) - sum(
        p.numel() for p in baseline.parameters()
    )
    assert 0 < added < 100_000
