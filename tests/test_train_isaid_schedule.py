from __future__ import annotations

import pytest

from tools.train_isaid import training_epoch_limit


def test_epoch_limit_keeps_epoch_mode_unchanged() -> None:
    assert training_epoch_limit(num_batches=209, epochs=100, max_iterations=None) == 100


def test_epoch_limit_covers_iteration_target() -> None:
    assert training_epoch_limit(num_batches=209, epochs=100, max_iterations=80_000) == 383


def test_epoch_limit_rejects_empty_loader() -> None:
    with pytest.raises(ValueError, match="at least one batch"):
        training_epoch_limit(num_batches=0, epochs=100, max_iterations=80_000)
