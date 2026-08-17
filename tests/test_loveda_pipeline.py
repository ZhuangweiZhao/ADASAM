from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from adasam.datasets.industrial import LoveDASemanticDataset, fixed_validation_split_indices
from adasam.losses import LabelEfficientSegmentationLoss
from tools.train_loveda import (
    collect_routing_statistics,
    confusion_aware_region_prototype_loss,
    parse_args,
)


def make_loveda_sample(root: Path, split: str = "Train") -> None:
    image_dir = root / split / split / "Rural" / "images_png"
    mask_dir = root / split / split / "Rural" / "masks_png"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 1] = 128
    mask = np.ones((16, 16), dtype=np.uint8)
    mask[:4, :4] = 0
    mask[4:8, 4:8] = 7
    cv2.imwrite(str(image_dir / "0.png"), image)
    cv2.imwrite(str(mask_dir / "0.png"), mask)


def test_loveda_dataset_maps_labels_and_resizes(tmp_path: Path) -> None:
    make_loveda_sample(tmp_path)
    dataset = LoveDASemanticDataset(tmp_path, "train", image_size=32)
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (32, 32)
    assert set(torch.unique(sample["mask"]).tolist()) == {0, 6, 255}
    assert sample["id"] == "Rural_0"


def test_loveda_loss_ignores_unlabeled_pixels() -> None:
    prediction = torch.randn(2, 7, 16, 16, requires_grad=True)
    target = torch.randint(0, 7, (2, 16, 16))
    target[:, :4, :4] = 255
    loss = LabelEfficientSegmentationLoss(ignore_index=255)(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None


def test_loveda_fixed_split_is_nested() -> None:
    train_5, validation_5, pool = fixed_validation_split_indices(2522, 5, 42, 0.2, 42)
    train_10, validation_10, _ = fixed_validation_split_indices(2522, 10, 42, 0.2, 42)
    assert len(validation_5) == 504
    assert len(pool) == 2018
    assert len(train_5) == 101
    assert len(train_10) == 202
    assert validation_5 == validation_10
    assert set(train_5).issubset(train_10)


def test_loveda_accepts_magnitude_teacher_survival_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_loveda.py", "--model", "mobilesam", "--label-ratio", "10",
            "--fusion-version", "semantic_budget",
            "--spatial-policy", "distilled_magnitude",
            "--feature-retention-ratio", "0.25",
            "--magnitude-distill-weight", "1.0",
        ],
    )
    args = parse_args()
    assert args.spatial_policy == "distilled_magnitude"
    assert args.feature_retention_ratio == 0.25
    assert args.magnitude_distill_weight == 1.0


def test_loveda_accepts_rural_domain_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_loveda.py", "--model", "mobilesam_finetune", "--label-ratio", "100",
            "--augmentation", "remote_strong", "--rural-sampling-multiplier", "1.5",
        ],
    )
    args = parse_args()
    assert args.augmentation == "remote_strong"
    assert args.rural_sampling_multiplier == 1.5


def test_loveda_accepts_lora_peft_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_loveda.py", "--model", "mobilesam", "--label-ratio", "10",
            "--adapter", "none", "--lora-rank", "4", "--lora-alpha", "8",
            "--lora-targets", "qkv", "proj",
        ],
    )
    args = parse_args()
    assert args.lora_rank == 4
    assert args.lora_alpha == 8.0
    assert args.lora_targets == ["qkv", "proj"]


def test_loveda_accepts_fixed_selection_manifest(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_loveda.py", "--model", "mobilesam", "--label-ratio", "5",
            "--selection-manifest", "/runs/manifests/hrcs_ratio5.json",
        ],
    )
    args = parse_args()
    assert args.selection_manifest == "/runs/manifests/hrcs_ratio5.json"


def test_loveda_accepts_lovasz_region_loss_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_loveda.py", "--model", "mobilesam_finetune", "--label-ratio", "100",
            "--class-balanced-ce", "--lovasz-weight", "0.5",
            "--lovasz-class-weights", "1", "1", "1", "2", "1", "2", "1",
        ],
    )
    args = parse_args()
    assert args.class_balanced_ce
    assert args.lovasz_weight == 0.5
    assert args.lovasz_class_weights == [1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0]


def test_loveda_accepts_semantic_progressive_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_loveda.py", "--model", "mobilesam_finetune", "--label-ratio", "100",
            "--fusion-version", "semantic_progressive",
            "--progressive-aux-weight", "0.2",
        ],
    )
    args = parse_args()
    assert args.fusion_version == "semantic_progressive"
    assert args.progressive_aux_weight == 0.2


def test_loveda_accepts_regional_semantic_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_loveda.py", "--model", "mobilesam_finetune", "--label-ratio", "100",
            "--fusion-version", "regional_semantic", "--progressive-aux-weight", "0.2",
            "--regional-contrast-weight", "0.1",
            "--regional-contrast-temperature", "0.2",
        ],
    )
    args = parse_args()
    assert args.fusion_version == "regional_semantic"
    assert args.progressive_aux_weight == 0.2
    assert args.regional_contrast_weight == 0.1
    assert args.regional_contrast_temperature == 0.2


def test_confusion_aware_region_prototype_loss_rewards_correct_similarity() -> None:
    target = torch.tensor([[[3, 0, 2], [5, 6, 255]]])
    correct = torch.full((1, 7, 2, 3), -2.0)
    for row, column, class_id in ((0, 0, 3), (0, 1, 0), (0, 2, 2), (1, 0, 5), (1, 1, 6)):
        correct[0, class_id, row, column] = 2.0
    correct.requires_grad_()
    wrong = -correct
    correct_loss = confusion_aware_region_prototype_loss(
        {"prototype_similarity_logits": correct}, target, temperature=0.2
    )
    wrong_loss = confusion_aware_region_prototype_loss(
        {"prototype_similarity_logits": wrong}, target, temperature=0.2
    )
    assert correct_loss is not None and wrong_loss is not None
    assert correct_loss < wrong_loss
    correct_loss.backward()
    assert correct.grad is not None
    assert torch.isfinite(correct.grad).all()


def test_loveda_accepts_semantic_progressive_v2_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_loveda.py", "--model", "mobilesam_finetune", "--label-ratio", "100",
            "--fusion-version", "semantic_progressive_v2",
            "--progressive-aux-weight", "0.05",
        ],
    )
    args = parse_args()
    assert args.fusion_version == "semantic_progressive_v2"
    assert args.progressive_aux_weight == 0.05


def test_loveda_accepts_semantic_progressive_v3_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_loveda.py", "--model", "mobilesam_finetune", "--label-ratio", "100",
            "--fusion-version", "semantic_progressive_v3",
            "--progressive-aux-weight", "0.0",
            "--utility-gate-weight", "0.1",
            "--utility-gate-temperature", "0.25",
        ],
    )
    args = parse_args()
    assert args.fusion_version == "semantic_progressive_v3"
    assert args.progressive_aux_weight == 0.0
    assert args.utility_gate_weight == 0.1
    assert args.utility_gate_temperature == 0.25


def test_routing_statistics_accepts_no_ignore_index() -> None:
    class Decoder:
        fusion_version = "scsr"
        representation_budget = 3
        last_routing = {
            "weights": torch.full((1, 3, 2, 2), 1.0 / 3.0),
            "entropy": torch.zeros(1, 2, 2),
        }

    class Model:
        decoder = Decoder()

        def eval(self):
            return self

        def __call__(self, image):
            return image

    loader = [
        {
            "image": torch.zeros(1, 3, 4, 4),
            "mask": torch.tensor(
                [[[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]]
            ),
        }
    ]

    statistics = collect_routing_statistics(
        Model(), loader, torch.device("cpu"), num_classes=2, ignore_index=None
    )

    assert statistics is not None
    assert statistics["pixels"] == 4
    assert set(statistics["class_mean_weights"]) == {"0", "1"}


def test_task_routing_statistics_reports_oracle_agreement() -> None:
    class Decoder:
        fusion_version = "scsr_task"
        representation_budget = 3
        last_routing = None

    class Model:
        decoder = Decoder()

        def eval(self):
            return self

        def __call__(self, image):
            self.decoder.last_routing = {
                "weights": torch.full((1, 3, 2, 2), 1.0 / 3.0),
                "entropy": torch.zeros(1, 2, 2),
                "scale_logits": (
                    torch.zeros(1, 2, 2, 2),
                    torch.zeros(1, 2, 2, 2),
                    torch.zeros(1, 2, 2, 2),
                ),
            }
            return image

    statistics = collect_routing_statistics(
        Model(),
        [{"image": torch.zeros(1, 3, 4, 4), "mask": torch.zeros(1, 4, 4, dtype=torch.long)}],
        torch.device("cpu"),
        num_classes=2,
        ignore_index=None,
    )

    assert statistics is not None
    assert statistics["oracle_route_agreement"] == 1.0
    assert sum(statistics["oracle_scale_fraction"].values()) == 1.0
