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
        ],
    )

    args = parse_args()

    assert args.feature_scales == "p3_p4_embedding"
    assert args.fusion_version == "semantic_budget"
    assert args.representation_budget == 2


def test_neuseg_preserves_existing_fusion_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_segmentation.py", "--label_ratio", "5"])

    args = parse_args()

    assert args.feature_scales == "p3_p4_embedding"
    assert args.fusion_version == "hierarchical"
    assert args.representation_budget == 3
