"""Tests for center heatmap ground truth generation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.data.heatmap import HeatmapConfig, generate_center_heatmap


def test_generate_center_heatmap_single_peak() -> None:
    heatmap = generate_center_heatmap((32, 32, 32), [(16.0, 16.0, 16.0)])

    assert heatmap.shape == (32, 32, 32)
    assert heatmap.dtype == np.float32
    peak_position = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    assert peak_position == (16, 16, 16)
    assert heatmap[peak_position] == pytest.approx(1.0)


def test_generate_center_heatmap_respects_spacing() -> None:
    config = HeatmapConfig(spacing=(2.0, 1.0, 0.5), sigma_mm=4.0)
    heatmap = generate_center_heatmap((21, 21, 21), [(10.0, 10.0, 10.0)], config=config)

    sigma_vox_z = config.sigma_mm / config.spacing[0]
    expected = math.exp(-(1.0**2) / (2.0 * sigma_vox_z**2))
    assert heatmap[11, 10, 10] == pytest.approx(expected, rel=1e-3)


def test_generate_center_heatmap_overlapping_centers_normalized() -> None:
    heatmap = generate_center_heatmap((16, 16, 16), [(8.0, 8.0, 8.0), (8.0, 9.0, 8.0)])

    assert heatmap.max() == pytest.approx(1.0)
    assert heatmap.min() >= 0.0


def test_generate_center_heatmap_ignores_out_of_bounds_centers() -> None:
    heatmap = generate_center_heatmap((8, 8, 8), [(100.0, 100.0, 100.0)])

    assert np.count_nonzero(heatmap) == 0


def test_generate_center_heatmap_sum_mode_normalizes() -> None:
    config = HeatmapConfig(mode="sum")
    heatmap = generate_center_heatmap(
        (16, 16, 16), [(8.0, 8.0, 8.0), (8.0, 8.0, 8.0)], config=config
    )

    assert heatmap.max() == pytest.approx(1.0)


def test_generate_center_heatmap_validates_shape_dimension_count() -> None:
    with pytest.raises(ValueError, match="three-dimensional"):
        generate_center_heatmap((32, 32), [(10.0, 10.0, 10.0)])


def test_generate_center_heatmap_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma_mm must be positive"):
        generate_center_heatmap((16, 16, 16), [(8.0, 8.0, 8.0)], config=HeatmapConfig(sigma_mm=0.0))


def test_generate_center_heatmap_rejects_invalid_center_triplets() -> None:
    with pytest.raises(ValueError, match="triplets"):
        generate_center_heatmap((16, 16, 16), [(8.0, 8.0)])


def test_generate_center_heatmap_rejects_invalid_spacing() -> None:
    with pytest.raises(ValueError, match="spacing values must be positive"):
        generate_center_heatmap(
            (16, 16, 16), [(8.0, 8.0, 8.0)], config=HeatmapConfig(spacing=(1.0, 0.0, 1.0))
        )


def test_generate_center_heatmap_rejects_non_positive_truncate() -> None:
    with pytest.raises(ValueError, match="truncate must be positive"):
        generate_center_heatmap((16, 16, 16), [(8.0, 8.0, 8.0)], config=HeatmapConfig(truncate=0.0))


def test_generate_center_heatmap_requires_three_spacing_values() -> None:
    with pytest.raises(ValueError, match="spacing must define three values"):
        generate_center_heatmap(
            (16, 16, 16), [(8.0, 8.0, 8.0)], config=HeatmapConfig(spacing=(1.0, 1.0))
        )
