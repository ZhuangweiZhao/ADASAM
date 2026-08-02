import torch

from adasam.losses import PrototypeCompactnessLoss
from adasam.models.prototype import DefectPrototypeMemory


def test_dpm_updates_memory_and_preserves_shape() -> None:
    module = DefectPrototypeMemory(num_classes=4, feature_dim=16, momentum=0.9).train()
    feature = torch.randn(4, 16, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (4, 32, 32))
    enhanced, auxiliary = module(feature, target)
    assert enhanced.shape == feature.shape
    assert auxiliary["similarity"].shape == (4, 4, 8, 8)
    assert module.initialized.all()
    enhanced.mean().backward()
    assert feature.grad is not None
    assert module.prior_projection.weight.grad is not None
    assert module.alpha.grad is not None


def test_dpm_eval_does_not_update_memory() -> None:
    module = DefectPrototypeMemory(num_classes=4, feature_dim=16).eval()
    before = module.prototypes.clone()
    module(torch.randn(2, 16, 8, 8))
    assert torch.equal(before, module.prototypes)


def test_prototype_compactness_backward() -> None:
    feature = torch.randn(4, 16, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (4, 32, 32))
    prototypes = torch.randn(4, 16)
    loss = PrototypeCompactnessLoss()(feature, target, prototypes, torch.ones(4, dtype=torch.bool))
    loss.backward()
    assert loss.ndim == 0
    assert feature.grad is not None
