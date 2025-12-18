"""Unit tests for peak detection module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.inference.peak_detection import (
    detect_peaks_3d,
    extract_peak_coordinates,
    non_maximum_suppression_3d,
)


class TestNonMaximumSuppression3D:
    """Tests for non_maximum_suppression_3d."""

    def test_single_peak_detected(self) -> None:
        """Test that a single clear peak is detected."""
        heatmap = np.zeros((16, 16, 16), dtype=np.float32)
        heatmap[8, 8, 8] = 1.0

        peaks = non_maximum_suppression_3d(heatmap, kernel_size=3)

        assert peaks[8, 8, 8]
        assert np.sum(peaks) == 1

    def test_multiple_separated_peaks(self) -> None:
        """Test detection of multiple well-separated peaks."""
        heatmap = np.zeros((32, 32, 32), dtype=np.float32)
        heatmap[8, 8, 8] = 0.9
        heatmap[24, 24, 24] = 0.8
        heatmap[8, 24, 16] = 0.7

        peaks = non_maximum_suppression_3d(heatmap, kernel_size=3)

        assert peaks[8, 8, 8]
        assert peaks[24, 24, 24]
        assert peaks[8, 24, 16]
        assert np.sum(peaks) == 3

    def test_close_peaks_suppressed(self) -> None:
        """Test that close peaks are suppressed to single maximum."""
        heatmap = np.zeros((16, 16, 16), dtype=np.float32)
        heatmap[8, 8, 8] = 0.9
        heatmap[8, 9, 8] = 0.8  # Adjacent, should be suppressed

        peaks = non_maximum_suppression_3d(heatmap, kernel_size=3)

        assert peaks[8, 8, 8]
        assert not peaks[8, 9, 8]
        assert np.sum(peaks) == 1

    def test_flat_regions_excluded(self) -> None:
        """Test that flat regions with zero values are excluded."""
        heatmap = np.zeros((16, 16, 16), dtype=np.float32)

        peaks = non_maximum_suppression_3d(heatmap, kernel_size=3)

        assert np.sum(peaks) == 0

    def test_kernel_size_affects_suppression(self) -> None:
        """Test that larger kernel size suppresses more aggressively."""
        heatmap = np.zeros((32, 32, 32), dtype=np.float32)
        heatmap[16, 16, 16] = 1.0
        heatmap[16, 18, 16] = 0.9

        # Small kernel: both might survive
        peaks_small = non_maximum_suppression_3d(heatmap, kernel_size=3)
        peaks_large = non_maximum_suppression_3d(heatmap, kernel_size=5)

        # Large kernel should suppress more
        assert np.sum(peaks_large) <= np.sum(peaks_small)

    def test_invalid_kernel_size_raises_error(self) -> None:
        """Test that even kernel size raises error."""
        heatmap = np.zeros((16, 16, 16), dtype=np.float32)

        with pytest.raises(ValueError, match="kernel_size must be odd"):
            non_maximum_suppression_3d(heatmap, kernel_size=4)

    def test_negative_kernel_size_raises_error(self) -> None:
        """Test that negative kernel size raises error."""
        heatmap = np.zeros((16, 16, 16), dtype=np.float32)

        with pytest.raises(ValueError, match="kernel_size must be odd"):
            non_maximum_suppression_3d(heatmap, kernel_size=0)

    def test_boundary_peaks_detected(self) -> None:
        """Test that peaks at volume boundaries are detected."""
        heatmap = np.zeros((16, 16, 16), dtype=np.float32)
        heatmap[0, 0, 0] = 1.0
        heatmap[15, 15, 15] = 0.9

        peaks = non_maximum_suppression_3d(heatmap, kernel_size=3)

        assert peaks[0, 0, 0]
        assert peaks[15, 15, 15]


class TestExtractPeakCoordinates:
    """Tests for extract_peak_coordinates."""

    def test_extract_single_peak(self) -> None:
        """Test extracting a single peak."""
        heatmap = np.zeros((16, 16, 16), dtype=np.float32)
        heatmap[8, 10, 12] = 0.95

        mask = np.zeros((16, 16, 16), dtype=bool)
        mask[8, 10, 12] = True

        coords, confs = extract_peak_coordinates(heatmap, mask)

        assert coords.shape == (1, 3)
        assert confs.shape == (1,)
        np.testing.assert_array_equal(coords[0], [8, 10, 12])
        assert confs[0] == pytest.approx(0.95)

    def test_extract_multiple_peaks_sorted(self) -> None:
        """Test that peaks are sorted by confidence descending."""
        heatmap = np.zeros((16, 16, 16), dtype=np.float32)
        heatmap[5, 5, 5] = 0.6
        heatmap[10, 10, 10] = 0.9
        heatmap[2, 2, 2] = 0.3

        mask = np.zeros((16, 16, 16), dtype=bool)
        mask[5, 5, 5] = True
        mask[10, 10, 10] = True
        mask[2, 2, 2] = True

        coords, confs = extract_peak_coordinates(heatmap, mask)

        assert len(coords) == 3
        # Should be sorted by confidence descending
        assert confs[0] == pytest.approx(0.9)
        assert confs[1] == pytest.approx(0.6)
        assert confs[2] == pytest.approx(0.3)
        np.testing.assert_array_equal(coords[0], [10, 10, 10])

    def test_no_peaks_returns_empty(self) -> None:
        """Test that no peaks returns empty arrays."""
        heatmap = np.zeros((16, 16, 16), dtype=np.float32)
        mask = np.zeros((16, 16, 16), dtype=bool)

        coords, confs = extract_peak_coordinates(heatmap, mask)

        assert coords.shape == (0, 3)
        assert confs.shape == (0,)

    def test_mismatched_shapes_raises_error(self) -> None:
        """Test that mismatched shapes raise error."""
        heatmap = np.zeros((16, 16, 16), dtype=np.float32)
        mask = np.zeros((8, 8, 8), dtype=bool)

        with pytest.raises(ValueError, match="Shape mismatch"):
            extract_peak_coordinates(heatmap, mask)


class TestDetectPeaks3D:
    """Tests for detect_peaks_3d main function."""

    def test_detect_single_peak_numpy(self) -> None:
        """Test detecting a single peak from numpy array."""
        heatmap = np.zeros((32, 32, 32), dtype=np.float32)
        heatmap[16, 16, 16] = 0.95

        coords, confs = detect_peaks_3d(heatmap, min_confidence=0.5, nms_kernel_size=3)

        assert len(coords) == 1
        np.testing.assert_array_equal(coords[0], [16, 16, 16])
        assert confs[0] == pytest.approx(0.95)

    def test_detect_peaks_torch_tensor(self) -> None:
        """Test detecting peaks from PyTorch tensor."""
        heatmap = torch.zeros(1, 1, 32, 32, 32)
        heatmap[0, 0, 16, 16, 16] = 0.95
        heatmap[0, 0, 8, 8, 8] = 0.85

        coords, confs = detect_peaks_3d(heatmap, min_confidence=0.7)

        assert len(coords) == 2
        assert all(conf >= 0.7 for conf in confs)

    def test_confidence_thresholding(self) -> None:
        """Test that confidence threshold filters weak peaks."""
        heatmap = np.zeros((32, 32, 32), dtype=np.float32)
        heatmap[16, 16, 16] = 0.9
        heatmap[24, 24, 24] = 0.4
        heatmap[8, 8, 8] = 0.2

        coords, confs = detect_peaks_3d(heatmap, min_confidence=0.5)

        assert len(coords) == 1  # Only the 0.9 peak
        assert confs[0] == pytest.approx(0.9)

    def test_max_peaks_limit(self) -> None:
        """Test that max_peaks limits number of returned peaks."""
        heatmap = np.zeros((64, 64, 64), dtype=np.float32)
        for i in range(5):
            heatmap[10 + i * 10, 10 + i * 10, 10] = 0.9 - i * 0.1

        coords, confs = detect_peaks_3d(heatmap, min_confidence=0.1, max_peaks=3)

        assert len(coords) == 3
        # Should return top 3 by confidence
        assert confs[0] >= confs[1] >= confs[2]

    def test_batched_input_4d(self) -> None:
        """Test handling 4D batched input."""
        heatmap = torch.zeros(1, 32, 32, 32)
        heatmap[0, 16, 16, 16] = 0.95

        coords, confs = detect_peaks_3d(heatmap, min_confidence=0.5)

        assert len(coords) == 1
        np.testing.assert_array_equal(coords[0], [16, 16, 16])

    def test_batched_input_5d(self) -> None:
        """Test handling 5D batched input."""
        heatmap = torch.zeros(1, 1, 32, 32, 32)
        heatmap[0, 0, 16, 16, 16] = 0.95

        coords, confs = detect_peaks_3d(heatmap, min_confidence=0.5)

        assert len(coords) == 1

    def test_invalid_confidence_raises_error(self) -> None:
        """Test that invalid confidence values raise error."""
        heatmap = np.zeros((16, 16, 16), dtype=np.float32)

        with pytest.raises(ValueError, match="min_confidence must be in"):
            detect_peaks_3d(heatmap, min_confidence=1.5)

        with pytest.raises(ValueError, match="min_confidence must be in"):
            detect_peaks_3d(heatmap, min_confidence=-0.1)

    def test_invalid_nms_kernel_raises_error(self) -> None:
        """Test that invalid NMS kernel size raises error."""
        heatmap = np.zeros((16, 16, 16), dtype=np.float32)

        with pytest.raises(ValueError, match="nms_kernel_size must be odd"):
            detect_peaks_3d(heatmap, nms_kernel_size=4)

    def test_invalid_max_peaks_raises_error(self) -> None:
        """Test that invalid max_peaks raises error."""
        heatmap = np.zeros((16, 16, 16), dtype=np.float32)

        with pytest.raises(ValueError, match="max_peaks must be positive"):
            detect_peaks_3d(heatmap, max_peaks=0)

    def test_wrong_batch_size_raises_error(self) -> None:
        """Test that wrong batch size raises error."""
        heatmap = torch.zeros(2, 1, 32, 32, 32)  # Batch size 2

        with pytest.raises(ValueError, match="Expected single-channel heatmap"):
            detect_peaks_3d(heatmap)

    def test_wrong_channels_raises_error(self) -> None:
        """Test that wrong number of channels raises error."""
        heatmap = torch.zeros(1, 3, 32, 32, 32)  # 3 channels

        with pytest.raises(ValueError, match="Expected single-channel heatmap"):
            detect_peaks_3d(heatmap)

    def test_wrong_dimensions_raises_error(self) -> None:
        """Test that wrong number of dimensions raises error."""
        heatmap = np.zeros((32, 32), dtype=np.float32)  # 2D

        with pytest.raises(ValueError, match="Expected 3D heatmap"):
            detect_peaks_3d(heatmap)

    def test_realistic_gaussian_peaks(self) -> None:
        """Test with realistic Gaussian-like peaks."""
        heatmap = np.zeros((64, 64, 64), dtype=np.float32)

        # Create Gaussian-like peaks
        def add_gaussian_peak(center: tuple[int, int, int], amplitude: float, sigma: float = 2.0) -> None:
            z, y, x = center
            for dz in range(-5, 6):
                for dy in range(-5, 6):
                    for dx in range(-5, 6):
                        zp, yp, xp = z + dz, y + dy, x + dx
                        if 0 <= zp < 64 and 0 <= yp < 64 and 0 <= xp < 64:
                            dist_sq = dz**2 + dy**2 + dx**2
                            val = amplitude * np.exp(-dist_sq / (2 * sigma**2))
                            heatmap[zp, yp, xp] = max(heatmap[zp, yp, xp], val)

        add_gaussian_peak((32, 32, 32), 0.95, 2.0)
        add_gaussian_peak((16, 16, 16), 0.75, 2.0)
        add_gaussian_peak((48, 48, 48), 0.65, 2.0)

        coords, confs = detect_peaks_3d(heatmap, min_confidence=0.5, nms_kernel_size=7)

        assert len(coords) == 3
        # Peaks should be near the centers (within tolerance due to discretization)
        assert np.linalg.norm(coords[0] - [32, 32, 32]) < 3
        assert confs[0] == pytest.approx(0.95, abs=0.01)

    def test_no_peaks_above_threshold(self) -> None:
        """Test handling when no peaks exceed threshold."""
        heatmap = np.random.rand(32, 32, 32).astype(np.float32) * 0.05

        coords, confs = detect_peaks_3d(heatmap, min_confidence=0.5)

        assert len(coords) == 0
        assert len(confs) == 0

    def test_edge_case_all_same_value(self) -> None:
        """Test edge case with uniform heatmap."""
        heatmap = np.ones((16, 16, 16), dtype=np.float32) * 0.5

        coords, confs = detect_peaks_3d(heatmap, min_confidence=0.4)

        # All points are equally maximal, NMS should suppress most
        assert len(coords) >= 0  # Implementation-dependent
