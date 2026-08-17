"""Auditable sample-selection manifests for label-efficient experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def sample_id_fingerprint(sample_ids: Sequence[str]) -> str:
    """Return an order-independent fingerprint for a set of sample IDs."""
    payload = "\n".join(sorted(str(sample_id) for sample_id in sample_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_selection_manifest(
    path: str | Path,
    *,
    dataset_sample_ids: Sequence[str],
    training_pool: Sequence[int],
    label_ratio: int,
    validation_seed: int,
) -> tuple[list[int], dict[str, Any]]:
    """Load and validate a manifest without allowing validation-pool leakage."""
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("selection manifest must use schema_version=1")
    if str(manifest.get("dataset", "")).lower() != "loveda":
        raise ValueError("selection manifest dataset must be LoveDA")
    if manifest.get("label_ratio") != label_ratio:
        raise ValueError(
            f"selection manifest ratio {manifest.get('label_ratio')} does not match "
            f"--label-ratio {label_ratio}"
        )
    if manifest.get("validation_seed") != validation_seed:
        raise ValueError(
            f"selection manifest validation_seed {manifest.get('validation_seed')} does not "
            f"match --validation-seed {validation_seed}"
        )
    if manifest.get("dataset_fingerprint") != sample_id_fingerprint(dataset_sample_ids):
        raise ValueError("selection manifest was generated for a different LoveDA dataset")

    pool = list(training_pool)
    pool_ids = [dataset_sample_ids[index] for index in pool]
    if manifest.get("training_pool_fingerprint") != sample_id_fingerprint(pool_ids):
        raise ValueError("selection manifest does not match the fixed training pool")

    selected = manifest.get("selected_indices")
    selected_ids = manifest.get("selected_sample_ids")
    if not isinstance(selected, list) or not all(isinstance(index, int) for index in selected):
        raise ValueError("selection manifest selected_indices must be a list of integers")
    expected_count = (
        len(pool) if label_ratio == 100 else max(1, round(len(pool) * label_ratio / 100))
    )
    if len(selected) != expected_count:
        raise ValueError(
            f"selection manifest contains {len(selected)} samples; expected {expected_count}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("selection manifest contains duplicate indices")
    if not set(selected).issubset(set(pool)):
        raise ValueError("selection manifest contains validation or unknown dataset indices")
    resolved_ids = [dataset_sample_ids[index] for index in selected]
    if selected_ids != resolved_ids:
        raise ValueError("selection manifest sample IDs do not match selected_indices")
    return selected, manifest
