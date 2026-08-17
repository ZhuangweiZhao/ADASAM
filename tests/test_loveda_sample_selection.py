from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from adasam.datasets.selection import load_selection_manifest, sample_id_fingerprint
from tools.run_loveda_selection_survival import command_for
from tools.select_loveda_coreset import (
    coverage_statistics,
    deterministic_kcenter,
    pca_normalize,
)


def test_pca_normalize_is_deterministic_and_unit_length() -> None:
    features = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0]]
    )
    first = pca_normalize(features, output_dim=2)
    second = pca_normalize(features, output_dim=2)
    assert torch.allclose(first, second)
    assert torch.allclose(first.norm(dim=1), torch.ones(4))


def test_kcenter_is_deterministic_and_improves_coverage() -> None:
    descriptor = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]), dim=1
    )
    descriptors = {"embedding": descriptor}
    selected = deterministic_kcenter(descriptors, count=2, levels=("embedding",))
    assert selected == deterministic_kcenter(descriptors, count=2, levels=("embedding",))
    one = coverage_statistics(descriptors, selected[:1], ("embedding",))
    two = coverage_statistics(descriptors, selected, ("embedding",))
    assert two["max_nearest_distance"] <= one["max_nearest_distance"]
    assert two["mean_nearest_distance"] <= one["mean_nearest_distance"]


def manifest_for(sample_ids: list[str], selected: list[int]) -> dict:
    pool = [0, 1, 2, 3]
    return {
        "schema_version": 1,
        "dataset": "LoveDA",
        "dataset_fingerprint": sample_id_fingerprint(sample_ids),
        "training_pool_fingerprint": sample_id_fingerprint([sample_ids[index] for index in pool]),
        "validation_seed": 42,
        "label_ratio": 50,
        "strategy": "hrcs",
        "selected_indices": selected,
        "selected_sample_ids": [sample_ids[index] for index in selected],
    }


def test_manifest_loader_accepts_pool_subset(tmp_path: Path) -> None:
    sample_ids = ["Rural_0", "Rural_1", "Urban_2", "Urban_3", "Urban_val"]
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(manifest_for(sample_ids, [0, 2])), encoding="utf-8")
    selected, metadata = load_selection_manifest(
        path,
        dataset_sample_ids=sample_ids,
        training_pool=[0, 1, 2, 3],
        label_ratio=50,
        validation_seed=42,
    )
    assert selected == [0, 2]
    assert metadata["strategy"] == "hrcs"


def test_manifest_loader_rejects_validation_index(tmp_path: Path) -> None:
    sample_ids = ["Rural_0", "Rural_1", "Urban_2", "Urban_3", "Urban_val"]
    manifest = manifest_for(sample_ids, [0, 4])
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="validation or unknown"):
        load_selection_manifest(
            path,
            dataset_sample_ids=sample_ids,
            training_pool=[0, 1, 2, 3],
            label_ratio=50,
            validation_seed=42,
        )


def test_survival_command_uses_fixed_lora_sum_manifest() -> None:
    args = Namespace(
        data_root="/data/LoveDA",
        checkpoint="/weights/mobile_sam.pt",
        manifest_dir="/runs/manifests",
        output_dir="/runs/selection",
        epochs=100,
        batch_size=8,
        num_workers=4,
        device="cuda",
    )
    command = command_for(args, "hrcs")
    assert command[command.index("--fusion-version") + 1] == "sum"
    assert command[command.index("--lora-rank") + 1] == "4"
    assert command[command.index("--label-ratio") + 1] == "5"
    assert command[command.index("--selection-manifest") + 1].endswith("hrcs_ratio5.json")
