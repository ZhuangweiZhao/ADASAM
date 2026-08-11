from __future__ import annotations

import pytest
import torch

from adasam.losses import MagnitudeTeacherDistillationLoss


def test_magnitude_teacher_loss_is_finite_and_differentiable() -> None:
    logits = torch.zeros(2, 2, 8, 8, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[:, :, :2] = 1.0
    loss = MagnitudeTeacherDistillationLoss()(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_magnitude_teacher_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share shape"):
        MagnitudeTeacherDistillationLoss()(
            torch.zeros(1, 2, 8, 8), torch.zeros(1, 1, 8, 8)
        )
