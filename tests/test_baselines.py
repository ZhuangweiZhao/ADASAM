"""Tests for the standard baselines (DeepLabV3+ / SegFormer)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adasam.models import (
    DeepLabV3PlusBaseline,
    SegFormerBaseline,
    build_baseline,
)
from adasam.models.baselines import DWConv

ROOT = Path(__file__).resolve().parents[1]
SEGFORMER_WEIGHT = ROOT / "weights" / "mit_b0.pth"

smplib = pytest.importorskip("segmentation_models_pytorch", reason="SMP not installed")


def test_deeplabv3plus_forward_shape_and_counts() -> None:
    model = DeepLabV3PlusBaseline(num_classes=4, pretrained=False)
    model.eval()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4, 128, 128)
    counts = model.parameter_counts()
    assert counts["total"] == counts["trainable"] > 0
    assert counts["frozen"] == 0


def test_deeplabv3plus_forward_with_auxiliary() -> None:
    model = DeepLabV3PlusBaseline(num_classes=4, pretrained=False)
    model.eval()
    logits, prompts, auxiliary = model.forward_with_auxiliary(torch.rand(1, 3, 64, 64))
    assert logits.shape == (1, 4, 64, 64)
    assert prompts is None and auxiliary is None


def test_deeplabv3plus_odd_input_size_padding() -> None:
    """NEU tiles are 200x200 (not divisible by 16); forward must pad and crop back."""
    model = DeepLabV3PlusBaseline(num_classes=4, pretrained=False)
    model.eval()
    x = torch.rand(1, 3, 200, 200)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4, 200, 200)


def test_deeplabv3plus_backward_step() -> None:
    model = DeepLabV3PlusBaseline(num_classes=4, pretrained=False)
    model.train()
    x = torch.rand(2, 3, 64, 64)
    target = torch.randint(0, 4, (2, 64, 64))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_deeplabv3plus_batch_one_training_step() -> None:
    model = DeepLabV3PlusBaseline(num_classes=4, pretrained=False).train()
    logits = model(torch.rand(1, 3, 64, 64))
    logits.mean().backward()
    assert logits.shape == (1, 4, 64, 64)


def test_deeplabv3plus_pretrained_raises_when_local_weights_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="project-local path"):
        DeepLabV3PlusBaseline(
            num_classes=4, encoder_name="resnet50", pretrained=True,
            weights_root=tmp_path, allow_download=False,
        )


def test_deeplabv3plus_loads_project_local_weights(tmp_path) -> None:
    from safetensors.torch import save_file

    source = DeepLabV3PlusBaseline(num_classes=4, pretrained=False)
    path = tmp_path / "resnet50_imagenet.safetensors"
    save_file(source.model.encoder.state_dict(), str(path))
    loaded = DeepLabV3PlusBaseline(
        num_classes=4, pretrained=True, weights_root=tmp_path
    )
    assert torch.equal(
        loaded.model.encoder.conv1.weight, source.model.encoder.conv1.weight
    )


def test_segformer_dwconv_preserves_two_dimensional_layout() -> None:
    layer = DWConv(dim=2)
    image = torch.randn(1, 2, 3, 5)
    tokens = image.flatten(2).transpose(1, 2)
    expected = layer.dwconv(image).flatten(2).transpose(1, 2)
    actual = layer(tokens, 3, 5)
    assert torch.allclose(actual, expected)


def test_segformer_forward_shape_and_counts() -> None:
    model = SegFormerBaseline(num_classes=4, variant="b0", pretrained=False)
    model.eval()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4, 128, 128)
    counts = model.parameter_counts()
    assert counts["total"] == counts["trainable"] > 0
    assert counts["frozen"] == 0


def test_segformer_all_variants_forward() -> None:
    for variant in ("b0", "b1", "b2"):
        model = SegFormerBaseline(num_classes=7, variant=variant, pretrained=False)
        with torch.no_grad():
            out = model(torch.rand(1, 3, 128, 128))
        assert out.shape == (1, 7, 128, 128), variant


def test_segformer_rectangular_input() -> None:
    model = SegFormerBaseline(num_classes=4, variant="b0", pretrained=False).eval()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 96, 160))
    assert output.shape == (1, 4, 96, 160)


def test_segformer_pretrained_loads_when_weights_present() -> None:
    if not SEGFORMER_WEIGHT.exists():
        pytest.skip("weights/mit_b0.pth not present")
    model = SegFormerBaseline(num_classes=4, variant="b0", pretrained=True)
    assert model.backbone is not None


def test_segformer_pretrained_raises_when_weights_missing(tmp_path) -> None:
    model_dir = tmp_path / "no_weights"
    model_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        SegFormerBaseline(
            num_classes=4, variant="b0", pretrained=True, weights_root=model_dir
        )


def test_build_baseline_factory() -> None:
    model = build_baseline("segformer", num_classes=3, pretrained=False,
                           segformer_variant="b1")
    assert isinstance(model, SegFormerBaseline)
    with pytest.raises(ValueError):
        build_baseline("nope", num_classes=3)
