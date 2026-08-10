"""Utilities shared by spatial-budget training and analysis entry points."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_static_importance_map(path: str | Path | None) -> torch.Tensor | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    if resolved.suffix.lower() == ".npy":
        value = torch.from_numpy(np.load(resolved))
    else:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        value = payload.get("mean_importance", payload) if isinstance(payload, dict) else payload
    if not isinstance(value, torch.Tensor):
        raise TypeError("static importance artifact must contain a tensor")
    return value.float()
