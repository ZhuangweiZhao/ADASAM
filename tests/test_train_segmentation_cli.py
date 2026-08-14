from __future__ import annotations

import sys

from tools.train_segmentation import parse_args


def test_neuseg_accepts_hierarchical_fusion_options(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_segmentation.py",
            "--label_ratio", "5",
            "--feature-scales", "p3_p4_embedding",
            "--fusion-version", "semantic_budget",
            "--representation-budget", "2",
            "--spatial-policy", "magnitude",
            "--feature-retention-ratio", "0.5",
        ],
    )

    args = parse_args()

    assert args.feature_scales == "p3_p4_embedding"
    assert args.fusion_version == "semantic_budget"
    assert args.representation_budget == 2
    assert args.spatial_policy == "magnitude"
    assert args.feature_retention_ratio == 0.5


def test_neuseg_accepts_scsr_v2(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_segmentation.py", "--label_ratio", "5", "--fusion-version", "scsr_v2"],
    )

    args = parse_args()

    assert args.fusion_version == "scsr_v2"


def test_neuseg_accepts_task_supervised_scsr(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_segmentation.py",
            "--label_ratio", "5",
            "--fusion-version", "scsr_task",
            "--routing-loss-weight", "0.2",
            "--routing-warmup-epochs", "10",
            "--routing-hard",
        ],
    )

    args = parse_args()

    assert args.fusion_version == "scsr_task"
    assert args.routing_loss_weight == 0.2
    assert args.routing_warmup_epochs == 10
    assert args.routing_hard is True


def test_neuseg_preserves_existing_fusion_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_segmentation.py", "--label_ratio", "5"])

    args = parse_args()

    assert args.feature_scales == "p3_p4_embedding"
    assert args.fusion_version == "hierarchical"
    assert args.representation_budget == 3
    assert args.spatial_policy == "adaptive"
    assert args.feature_retention_ratio == 1.0


def test_neuseg_accepts_baseline_models(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_segmentation.py",
            "--label_ratio", "5",
            "--model", "deeplabv3plus",
            "--baseline-encoder", "resnet101",
            "--pretrained",
        ],
    )

    args = parse_args()

    assert args.model == "deeplabv3plus"
    assert args.baseline_encoder == "resnet101"
    assert args.pretrained is True


def test_neuseg_accepts_segformer_variant(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_segmentation.py",
            "--label_ratio", "5",
            "--model", "segformer",
            "--segformer-variant", "b2",
            "--no-pretrained",
        ],
    )

    args = parse_args()

    assert args.model == "segformer"
    assert args.segformer_variant == "b2"
    assert args.pretrained is False
