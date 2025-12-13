"""Tests for intensity normalization utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.intensity import (
    HistogramMethod,
    NormalizationStats,
    clip_hounsfield,
    normalize_intensity,
)


def test_clip_hounsfield_clamps_values() -> None:
    volume = np.array([[-1200.0, -500.0], [200.0, 600.0]], dtype=np.float32)
    clipped = clip_hounsfield(volume, window_min=-1000.0, window_max=400.0)
    assert np.isclose(clipped.min(), -1000.0)
    assert np.isclose(clipped.max(), 400.0)
    assert clipped.dtype == np.float32


def test_normalize_intensity_scales_to_unit_interval() -> None:
    volume = np.array([[-1000.0, 0.0], [200.0, 400.0]], dtype=np.float32)
    normalized, stats = normalize_intensity(
        volume,
        window=(-1000.0, 400.0),
        target_range=(0.0, 1.0),
        histogram_method="none",
        return_stats=True,
    )

    assert isinstance(stats, NormalizationStats)
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)
    assert stats.window_min == -1000.0
    assert stats.window_max == 400.0
    assert 0.0 <= stats.clip_fraction <= 1.0


def test_normalize_intensity_handles_constant_values() -> None:
    volume = np.full((2, 2), 800.0, dtype=np.float32)
    normalized = normalize_intensity(volume, window=(-1000.0, 400.0))
    assert np.allclose(normalized, 1.0)


def test_histogram_equalization_alters_distribution() -> None:
    volume = np.array(
        [
            [-1000.0, -500.0, -100.0, 0.0],
            [50.0, 100.0, 200.0, 400.0],
        ],
        dtype=np.float32,
    )

    baseline = normalize_intensity(volume, histogram_method="none")
    equalised = normalize_intensity(volume, histogram_method="global")

    assert not np.allclose(equalised, baseline)
    assert equalised.min() >= 0.0
    assert equalised.max() <= 1.0


@pytest.mark.parametrize("method", ["none", "global", "adaptive"])
def test_histogram_method_accepts_supported_values(method: HistogramMethod) -> None:
    volume = np.array([[-1000.0, 400.0]], dtype=np.float32)
    result = normalize_intensity(volume, histogram_method=method)
    assert result.shape == volume.shape


def test_normalize_intensity_rejects_empty_volume() -> None:
    with pytest.raises(ValueError):
        normalize_intensity(np.array([], dtype=np.float32))


def test_clip_hounsfield_validates_window() -> None:
    with pytest.raises(ValueError):
        clip_hounsfield(np.zeros((2, 2), dtype=np.float32), window_min=10.0, window_max=5.0)
