"""Tests for preprocessing helpers."""

import numpy as np
import pytest

from src.data.preprocess import apply_hu_windowing, normalize_to_range


def test_apply_hu_windowing_clips_to_window() -> None:
    volume = np.array([-2000, -600, 0, 500], dtype=np.float32)
    windowed = apply_hu_windowing(volume, window_center=-600, window_width=1500)
    assert np.isclose(windowed.min(), -1350.0)
    assert np.isclose(windowed.max(), 150.0)


def test_normalize_to_range_outputs_target_interval() -> None:
    volume = np.array([0.0, 5.0, 10.0], dtype=np.float32)
    normalized = normalize_to_range(volume, target_min=-1.0, target_max=1.0)
    assert np.isclose(normalized.min(), -1.0)
    assert np.isclose(normalized.max(), 1.0)


def test_normalize_to_range_handles_constant_volume() -> None:
    volume = np.full((3,), 7.0, dtype=np.float32)
    normalized = normalize_to_range(volume)
    assert np.allclose(normalized, 0.0)


def test_normalize_to_range_rejects_invalid_target_range() -> None:
    volume = np.array([0.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError):
        normalize_to_range(volume, target_min=1.0, target_max=0.0)
