"""Tests for overlap blending."""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.overlap_blending import BlendMode, OverlapBlending


class TestBlendModeEnum:
    """Test BlendMode enumeration."""

    def test_blend_mode_values(self):
        """Test that all blend modes have string values."""
        assert BlendMode.AVERAGE.value == "average"
        assert BlendMode.GAUSSIAN.value == "gaussian"
        assert BlendMode.LINEAR.value == "linear"

    def test_blend_mode_from_string(self):
        """Test creating BlendMode from string."""
        mode = BlendMode("average")
        assert mode == BlendMode.AVERAGE

    def test_blend_mode_invalid_string(self):
        """Test invalid blend mode string."""
        with pytest.raises(ValueError):
            BlendMode("invalid_mode")


class TestOverlapBlendingInit:
    """Test overlap blending initialization."""

    def test_init_default_mode(self):
        """Test initialization with default mode."""
        blender = OverlapBlending(window_size=(64, 64, 64))

        assert blender.window_size == (64, 64, 64)
        assert blender.mode == BlendMode.AVERAGE

    def test_init_custom_mode(self):
        """Test initialization with custom mode."""
        blender = OverlapBlending(
            window_size=(64, 64, 64),
            mode=BlendMode.GAUSSIAN,
        )

        assert blender.mode == BlendMode.GAUSSIAN

    def test_init_mode_from_string(self):
        """Test initialization with mode as string."""
        blender = OverlapBlending(
            window_size=(64, 64, 64),
            mode="linear",
        )

        assert blender.mode == BlendMode.LINEAR

    def test_init_invalid_window_size_length(self):
        """Test initialization with invalid window size."""
        with pytest.raises(ValueError, match="window_size must be a 3-tuple"):
            OverlapBlending(window_size=(64, 64))

    def test_init_invalid_window_size_zero(self):
        """Test initialization with zero window size."""
        with pytest.raises(ValueError, match="window_size values must be positive"):
            OverlapBlending(window_size=(0, 64, 64))

    def test_init_invalid_mode(self):
        """Test initialization with invalid mode string."""
        with pytest.raises(ValueError, match="Unknown blending mode"):
            OverlapBlending(window_size=(64, 64, 64), mode="invalid")

    def test_weight_map_created(self):
        """Test that weight map is created on init."""
        blender = OverlapBlending(window_size=(64, 64, 64))

        assert blender._weight_map is not None
        assert blender._weight_map.shape == (64, 64, 64)


class TestWeightMapCreation:
    """Test weight map creation for different modes."""

    def test_average_weights(self):
        """Test average weight map."""
        blender = OverlapBlending(window_size=(32, 32, 32), mode=BlendMode.AVERAGE)

        weights = blender._weight_map

        assert weights.shape == (32, 32, 32)
        assert np.allclose(weights, 1.0)

    def test_gaussian_weights(self):
        """Test Gaussian weight map."""
        blender = OverlapBlending(window_size=(32, 32, 32), mode=BlendMode.GAUSSIAN)

        weights = blender._weight_map

        assert weights.shape == (32, 32, 32)
        # Gaussian should have max at center
        center = tuple(s // 2 for s in (32, 32, 32))
        assert weights[center] == pytest.approx(1.0, abs=0.01)
        # Gaussian should decay from center
        assert weights[0, 0, 0] < weights[16, 16, 16]

    def test_linear_weights(self):
        """Test linear weight map."""
        blender = OverlapBlending(window_size=(32, 32, 32), mode=BlendMode.LINEAR)

        weights = blender._weight_map

        assert weights.shape == (32, 32, 32)
        # Linear should have max at center
        center = tuple(s // 2 for s in (32, 32, 32))
        assert weights[center] == pytest.approx(1.0, abs=0.01)
        # Linear should decay from center
        assert weights[0, 0, 0] < weights[16, 16, 16]


class TestBlendMethod:
    """Test the blend method."""

    def test_blend_single_patch(self):
        """Test blending with single patch."""
        blender = OverlapBlending(window_size=(64, 64, 64), mode=BlendMode.AVERAGE)

        prediction = np.ones((1, 64, 64, 64), dtype=np.float32)
        predictions = [prediction]
        positions = [(0, 0, 0)]

        blended, weights = blender.blend(predictions, positions, (64, 64, 64))

        assert blended.shape == (1, 64, 64, 64)
        assert np.allclose(blended, 1.0)

    def test_blend_multiple_patches(self):
        """Test blending with multiple non-overlapping patches."""
        blender = OverlapBlending(window_size=(64, 64, 64), mode=BlendMode.AVERAGE)

        predictions = [
            np.ones((1, 64, 64, 64), dtype=np.float32) * 1.0,
            np.ones((1, 64, 64, 64), dtype=np.float32) * 2.0,
        ]
        positions = [(0, 0, 0), (64, 64, 64)]

        blended, weights = blender.blend(
            predictions,
            positions,
            (128, 128, 128),
        )

        assert blended.shape == (1, 128, 128, 128)
        # Non-overlapping regions should keep original values
        assert blended[0, 0, 0, 0] == pytest.approx(1.0)
        assert blended[0, 127, 127, 127] == pytest.approx(2.0)

    def test_blend_overlapping_patches_average(self):
        """Test blending overlapping patches with average mode."""
        blender = OverlapBlending(window_size=(64, 64, 64), mode=BlendMode.AVERAGE)

        predictions = [
            np.ones((1, 64, 64, 64), dtype=np.float32) * 1.0,
            np.ones((1, 64, 64, 64), dtype=np.float32) * 3.0,
        ]
        positions = [(0, 0, 0), (32, 32, 32)]

        blended, weights = blender.blend(
            predictions,
            positions,
            (96, 96, 96),
        )

        assert blended.shape == (1, 96, 96, 96)
        # Overlapping region should be average
        center = blended[0, 48, 48, 48]
        assert center == pytest.approx(2.0, abs=0.1)

    def test_blend_multi_channel(self):
        """Test blending multi-channel predictions."""
        blender = OverlapBlending(window_size=(64, 64, 64), mode=BlendMode.AVERAGE)

        predictions = [
            np.ones((4, 64, 64, 64), dtype=np.float32) * np.arange(4).reshape(4, 1, 1, 1),
        ]
        positions = [(0, 0, 0)]

        blended, weights = blender.blend(predictions, positions, (64, 64, 64))

        assert blended.shape == (4, 64, 64, 64)
        for c in range(4):
            assert np.allclose(blended[c], c)

    def test_blend_empty_predictions_raises_error(self):
        """Test blending with empty predictions list."""
        blender = OverlapBlending(window_size=(64, 64, 64))

        with pytest.raises(ValueError, match="predictions list cannot be empty"):
            blender.blend([], [], (64, 64, 64))

    def test_blend_mismatched_lengths_raises_error(self):
        """Test blending with mismatched predictions and positions."""
        blender = OverlapBlending(window_size=(64, 64, 64))

        predictions = [np.ones((1, 64, 64, 64), dtype=np.float32)]
        positions = [(0, 0, 0), (64, 64, 64)]

        with pytest.raises(ValueError, match="predictions and positions must have same length"):
            blender.blend(predictions, positions, (128, 128, 128))

    def test_blend_inconsistent_channels_raises_error(self):
        """Test blending with inconsistent channel counts."""
        blender = OverlapBlending(window_size=(64, 64, 64))

        predictions = [
            np.ones((1, 64, 64, 64), dtype=np.float32),
            np.ones((2, 64, 64, 64), dtype=np.float32),
        ]
        positions = [(0, 0, 0), (64, 64, 64)]

        with pytest.raises(ValueError, match="Inconsistent number of channels"):
            blender.blend(predictions, positions, (128, 128, 128))

    def test_blend_out_of_bounds_raises_error(self):
        """Test blending with patches exceeding output shape."""
        blender = OverlapBlending(window_size=(64, 64, 64))

        predictions = [np.ones((1, 64, 64, 64), dtype=np.float32)]
        positions = [(100, 0, 0)]  # Beyond output shape

        with pytest.raises(ValueError, match="exceeds output shape"):
            blender.blend(predictions, positions, (64, 64, 64))


class TestBlendWithCounts:
    """Test the blend_with_counts method."""

    def test_blend_with_counts_single_patch(self):
        """Test blending with counts for single patch."""
        blender = OverlapBlending(window_size=(64, 64, 64))

        assembled = np.ones((1, 64, 64, 64), dtype=np.float32)
        counts = np.ones((64, 64, 64), dtype=np.float32)

        blended = blender.blend_with_counts(assembled, counts)

        assert blended.shape == (1, 64, 64, 64)
        assert np.allclose(blended, 1.0)

    def test_blend_with_counts_varying_counts(self):
        """Test blending with varying counts."""
        blender = OverlapBlending(window_size=(64, 64, 64))

        assembled = np.ones((1, 64, 64, 64), dtype=np.float32) * 4.0
        counts = np.ones((64, 64, 64), dtype=np.float32) * 2.0

        blended = blender.blend_with_counts(assembled, counts)

        assert blended.shape == (1, 64, 64, 64)
        assert np.allclose(blended, 2.0)

    def test_blend_with_counts_invalid_assembled_shape(self):
        """Test blending with invalid assembled shape."""
        blender = OverlapBlending(window_size=(64, 64, 64))

        assembled = np.ones((64, 64, 64), dtype=np.float32)  # 3D instead of 4D
        counts = np.ones((64, 64, 64), dtype=np.float32)

        with pytest.raises(ValueError, match="assembled must be 4D"):
            blender.blend_with_counts(assembled, counts)

    def test_blend_with_counts_invalid_counts_shape(self):
        """Test blending with invalid counts shape."""
        blender = OverlapBlending(window_size=(64, 64, 64))

        assembled = np.ones((1, 64, 64, 64), dtype=np.float32)
        counts = np.ones((1, 64, 64, 64), dtype=np.float32)  # 4D instead of 3D

        with pytest.raises(ValueError, match="counts must be 3D"):
            blender.blend_with_counts(assembled, counts)

    def test_blend_with_counts_mismatched_spatial_dims(self):
        """Test blending with mismatched spatial dimensions."""
        blender = OverlapBlending(window_size=(64, 64, 64))

        assembled = np.ones((1, 64, 64, 64), dtype=np.float32)
        counts = np.ones((32, 32, 32), dtype=np.float32)

        with pytest.raises(ValueError, match="spatial dimensions must match"):
            blender.blend_with_counts(assembled, counts)


class TestDifferentWindowSizes:
    """Test blending with different window sizes."""

    def test_asymmetric_window_average(self):
        """Test blending with asymmetric window."""
        blender = OverlapBlending(
            window_size=(32, 64, 128),
            mode=BlendMode.AVERAGE,
        )

        assert blender._weight_map.shape == (32, 64, 128)

    def test_asymmetric_window_gaussian(self):
        """Test Gaussian weights with asymmetric window."""
        blender = OverlapBlending(
            window_size=(32, 64, 128),
            mode=BlendMode.GAUSSIAN,
        )

        weights = blender._weight_map

        assert weights.shape == (32, 64, 128)
        # Check center is max
        center = (16, 32, 64)
        assert weights[center] == pytest.approx(1.0, abs=0.01)

    def test_asymmetric_window_linear(self):
        """Test linear weights with asymmetric window."""
        blender = OverlapBlending(
            window_size=(32, 64, 128),
            mode=BlendMode.LINEAR,
        )

        weights = blender._weight_map

        assert weights.shape == (32, 64, 128)
        # Check center is max
        center = (16, 32, 64)
        assert weights[center] == pytest.approx(1.0, abs=0.01)
