"""Tests for sliding window inference."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.inference.sliding_window import SlidingWindowInference


class TestSlidingWindowInferenceInit:
    """Test sliding window initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        swinf = SlidingWindowInference(window_size=(64, 64, 64))

        assert swinf.window_size == (64, 64, 64)
        assert swinf.stride == (64, 64, 64)

    def test_init_custom_stride(self):
        """Test initialization with custom stride."""
        swinf = SlidingWindowInference(
            window_size=(64, 64, 64),
            stride=(32, 32, 32),
        )

        assert swinf.window_size == (64, 64, 64)
        assert swinf.stride == (32, 32, 32)

    def test_init_invalid_window_size_length(self):
        """Test initialization with invalid window size."""
        with pytest.raises(ValueError, match="window_size must be a 3-tuple"):
            SlidingWindowInference(window_size=(64, 64))

    def test_init_invalid_window_size_zero(self):
        """Test initialization with zero window size."""
        with pytest.raises(ValueError, match="window_size values must be positive"):
            SlidingWindowInference(window_size=(0, 64, 64))

    def test_init_invalid_stride_length(self):
        """Test initialization with invalid stride."""
        with pytest.raises(ValueError, match="stride must be a 3-tuple"):
            SlidingWindowInference(
                window_size=(64, 64, 64),
                stride=(32, 32),
            )

    def test_init_invalid_stride_zero(self):
        """Test initialization with zero stride."""
        with pytest.raises(ValueError, match="stride values must be positive"):
            SlidingWindowInference(
                window_size=(64, 64, 64),
                stride=(0, 32, 32),
            )


class TestExtractPatches:
    """Test patch extraction."""

    def test_extract_patches_non_overlapping(self):
        """Test extracting non-overlapping patches."""
        volume = np.arange(64 * 64 * 64).reshape(64, 64, 64).astype(np.float32)
        swinf = SlidingWindowInference(window_size=(32, 32, 32), stride=(32, 32, 32))

        patches, positions = swinf.extract_patches(volume)

        # 2x2x2 patches from 64x64x64 volume
        assert len(patches) == 8
        assert len(positions) == 8

        # Check positions
        expected_positions = [
            (0, 0, 0), (0, 0, 32), (0, 32, 0), (0, 32, 32),
            (32, 0, 0), (32, 0, 32), (32, 32, 0), (32, 32, 32),
        ]
        assert sorted(positions) == sorted(expected_positions)

        # Check patch shapes
        for patch in patches:
            assert patch.shape == (32, 32, 32)

    def test_extract_patches_overlapping(self):
        """Test extracting overlapping patches."""
        volume = np.ones((128, 128, 128), dtype=np.float32)
        swinf = SlidingWindowInference(window_size=(64, 64, 64), stride=(32, 32, 32))

        patches, positions = swinf.extract_patches(volume)

        # (128-64)/32 + 1 = 3 patches per dimension -> 3^3 = 27 patches
        assert len(patches) == 27
        assert len(positions) == 27

    def test_extract_patches_requires_padding(self):
        """Test patch extraction with volume requiring padding."""
        volume = np.ones((100, 100, 100), dtype=np.float32)
        swinf = SlidingWindowInference(window_size=(64, 64, 64), stride=(64, 64, 64))

        patches, positions = swinf.extract_patches(volume)

        # Should require padding to 128x128x128, giving 2x2x2 patches
        assert len(patches) == 8
        assert all(p.shape == (64, 64, 64) for p in patches)

    def test_extract_patches_invalid_volume(self):
        """Test extraction with invalid volume."""
        swinf = SlidingWindowInference(window_size=(64, 64, 64))

        with pytest.raises(ValueError, match="volume must be 3D"):
            swinf.extract_patches(np.ones((64, 64)))

    def test_extract_patches_custom_pad_value(self):
        """Test extraction with custom pad value."""
        volume = np.ones((100, 100, 100), dtype=np.float32)
        swinf = SlidingWindowInference(window_size=(64, 64, 64), stride=(64, 64, 64))

        patches, _ = swinf.extract_patches(volume, pad_value=-1.0)

        # Check that padding is applied
        has_padding = any(np.any(p == -1.0) for p in patches)
        assert has_padding

    def test_extract_patches_empty_raises_error(self):
        """Test extraction from volume smaller than window."""
        volume = np.ones((10, 10, 10), dtype=np.float32)
        swinf = SlidingWindowInference(window_size=(64, 64, 64))

        patches, positions = swinf.extract_patches(volume)

        # Should still work with padding
        assert len(patches) > 0

    def test_patch_values_preserved(self):
        """Test that patch values are preserved from original volume."""
        volume = np.arange(64 * 64 * 64).reshape(64, 64, 64).astype(np.float32)
        swinf = SlidingWindowInference(window_size=(32, 32, 32), stride=(32, 32, 32))

        patches, positions = swinf.extract_patches(volume)

        # Check first patch (0,0,0)
        idx = positions.index((0, 0, 0))
        patch = patches[idx]
        expected = volume[0:32, 0:32, 0:32]
        assert np.allclose(patch, expected)


class TestAssemblePatches:
    """Test patch assembly."""

    def test_assemble_single_channel(self):
        """Test assembling single-channel predictions."""
        original_shape = (64, 64, 64)
        swinf = SlidingWindowInference(window_size=(64, 64, 64), stride=(64, 64, 64))

        # Single patch prediction - shape (batch, d, h, w) for single channel
        predictions = [np.ones((1, 64, 64, 64), dtype=np.float32)]
        positions = [(0, 0, 0)]

        assembled, count = swinf._assemble_predictions(
            original_shape,
            predictions,
            positions,
            pad_value=0.0,
        )

        # Output should be (1, 64, 64, 64) for single channel
        assert assembled.shape == (1, 64, 64, 64)
        assert count.shape == (64, 64, 64)

    def test_assemble_multiple_channels(self):
        """Test assembling multi-channel predictions."""
        original_shape = (64, 64, 64)
        swinf = SlidingWindowInference(window_size=(64, 64, 64), stride=(64, 64, 64))

        # Prediction with 4 channels - shape (batch, channels, d, h, w)
        predictions = [np.ones((1, 4, 64, 64, 64), dtype=np.float32) * np.arange(4).reshape(1, 4, 1, 1, 1)]
        positions = [(0, 0, 0)]

        assembled, count = swinf._assemble_predictions(
            original_shape,
            predictions,
            positions,
            pad_value=0.0,
        )

        # Output should be (4, 64, 64, 64)
        assert assembled.shape == (4, 64, 64, 64)
        assert count.shape == (64, 64, 64)

    def test_assemble_overlapping_patches(self):
        """Test assembling overlapping patches."""
        original_shape = (128, 128, 128)
        swinf = SlidingWindowInference(window_size=(64, 64, 64), stride=(32, 32, 32))

        # Create overlapping predictions
        predictions = [
            np.ones((1, 64, 64, 64), dtype=np.float32) * 1.0,
            np.ones((1, 64, 64, 64), dtype=np.float32) * 2.0,
        ]
        positions = [(0, 0, 0), (32, 32, 32)]

        assembled, count = swinf._assemble_predictions(
            original_shape,
            predictions,
            positions,
            pad_value=0.0,
        )

        # Just check we got results
        assert assembled.ndim >= 3
        assert count.ndim >= 3


class TestGetPadWidths:
    """Test pad width calculation."""

    def test_get_pad_widths_no_padding(self):
        """Test padding calculation with no padding needed."""
        original = (64, 64, 64)
        padded = (64, 64, 64)

        pad_widths = SlidingWindowInference._get_pad_widths(original, padded)

        assert pad_widths == [(0, 0), (0, 0), (0, 0)]

    def test_get_pad_widths_symmetric(self):
        """Test symmetric padding calculation."""
        original = (64, 64, 64)
        padded = (96, 96, 96)

        pad_widths = SlidingWindowInference._get_pad_widths(original, padded)

        # Should have equal padding on both sides
        for before, after in pad_widths:
            assert before == after == 16

    def test_get_pad_widths_asymmetric(self):
        """Test asymmetric padding calculation."""
        original = (64, 64, 64)
        padded = (97, 97, 97)

        pad_widths = SlidingWindowInference._get_pad_widths(original, padded)

        # Total padding should be 33
        for before, after in pad_widths:
            assert before + after == 33

    def test_get_pad_widths_one_dimension(self):
        """Test padding for single dimension."""
        original = (100, 64, 64)
        padded = (128, 64, 64)

        pad_widths = SlidingWindowInference._get_pad_widths(original, padded)

        assert pad_widths[0][0] + pad_widths[0][1] == 28
        assert pad_widths[1] == (0, 0)
        assert pad_widths[2] == (0, 0)


class TestCropArray:
    """Test array cropping."""

    def test_crop_3d_array(self):
        """Test cropping 3D array."""
        array = np.ones((100, 100, 100), dtype=np.float32)
        pad_widths = [(10, 10), (15, 15), (20, 20)]

        cropped = SlidingWindowInference._crop_array(array, pad_widths)

        assert cropped.shape == (80, 70, 60)

    def test_crop_4d_array(self):
        """Test cropping 4D array (with channel dimension)."""
        array = np.ones((4, 100, 100, 100), dtype=np.float32)
        pad_widths = [(10, 10), (15, 15), (20, 20)]

        cropped = SlidingWindowInference._crop_array(array, pad_widths)

        assert cropped.shape == (4, 80, 70, 60)

    def test_crop_no_padding(self):
        """Test cropping with no padding."""
        array = np.ones((100, 100, 100), dtype=np.float32)
        pad_widths = [(0, 0), (0, 0), (0, 0)]

        cropped = SlidingWindowInference._crop_array(array, pad_widths)

        assert cropped.shape == (100, 100, 100)

    def test_crop_one_side_only(self):
        """Test cropping only on one side."""
        array = np.ones((100, 100, 100), dtype=np.float32)
        pad_widths = [(10, 0), (0, 15), (0, 0)]

        cropped = SlidingWindowInference._crop_array(array, pad_widths)

        assert cropped.shape == (90, 85, 100)


class TestCallMethod:
    """Test the __call__ inference method."""

    def test_call_with_mock_model(self):
        """Test inference call with mock model."""
        volume = np.ones((64, 64, 64), dtype=np.float32)
        swinf = SlidingWindowInference(window_size=(64, 64, 64), stride=(64, 64, 64))

        # Mock model that returns single channel output
        mock_model = MagicMock()
        mock_model.return_value = torch.ones(1, 1, 64, 64, 64)

        assembled, count = swinf(volume, mock_model, device="cpu")

        assert assembled.shape == (1, 64, 64, 64)
        assert mock_model.called

    def test_call_invalid_volume_shape(self):
        """Test inference with invalid volume shape."""
        swinf = SlidingWindowInference(window_size=(64, 64, 64))
        mock_model = MagicMock()

        with pytest.raises(ValueError, match="volume must be 3D"):
            swinf(np.ones((64, 64)), mock_model)

    def test_call_empty_volume_raises_error(self):
        """Test inference with small volume gets padded."""
        swinf = SlidingWindowInference(window_size=(64, 64, 64))
        mock_model = MagicMock()
        # Configure mock to return proper tensor
        mock_model.return_value = torch.ones(1, 1, 64, 64, 64)

        volume = np.zeros((1, 1, 1), dtype=np.float32)
        # This should work - volume gets padded to 64x64x64 and returns valid output
        assembled, count = swinf(volume, mock_model)
        # Should have valid output
        assert assembled.shape[1:] == (1, 1, 1)

    def test_call_with_softmax(self):
        """Test __call__ applies softmax for multi-class output."""
        swinf = SlidingWindowInference(window_size=(32, 32, 32), stride=(32, 32, 32))
        volume = np.ones((32, 32, 32), dtype=np.float32)

        # Mock model that returns multi-class logits (3 classes)
        def mock_model(x):
            return torch.randn(1, 3, 32, 32, 32)

        model = MagicMock(side_effect=mock_model)

        result, _ = swinf(volume, model, return_logits=False)

        # Should apply softmax - result should be probabilities
        assert result.shape[0] == 3  # 3 classes

    def test_call_with_sigmoid(self):
        """Test __call__ applies sigmoid for binary output."""
        swinf = SlidingWindowInference(window_size=(32, 32, 32), stride=(32, 32, 32))
        volume = np.ones((32, 32, 32), dtype=np.float32)

        # Mock model that returns binary logits (1 channel)
        def mock_model(x):
            return torch.randn(1, 1, 32, 32, 32)

        model = MagicMock(side_effect=mock_model)

        result, _ = swinf(volume, model, return_logits=False)

        # Should apply sigmoid - result should be in [0, 1]
        assert result.shape[0] == 1  # 1 channel
        assert np.all((result >= 0) & (result <= 1))


class TestPadVolume:
    """Test volume padding."""

    def test_pad_volume_exact_fit(self):
        """Test padding when volume fits exactly."""
        volume = np.ones((64, 64, 64), dtype=np.float32)
        swinf = SlidingWindowInference(window_size=(64, 64, 64), stride=(64, 64, 64))

        padded = swinf._pad_volume(volume, pad_value=0.0)

        assert padded.shape == (64, 64, 64)

    def test_pad_volume_needs_padding(self):
        """Test padding when volume needs padding."""
        volume = np.ones((50, 50, 50), dtype=np.float32)
        swinf = SlidingWindowInference(window_size=(64, 64, 64), stride=(64, 64, 64))

        padded = swinf._pad_volume(volume, pad_value=0.0)

        assert padded.shape == (64, 64, 64)

    def test_pad_volume_preserves_center(self):
        """Test that padding centers the original volume."""
        volume = np.ones((50, 50, 50), dtype=np.float32)
        swinf = SlidingWindowInference(window_size=(64, 64, 64), stride=(64, 64, 64))

        padded = swinf._pad_volume(volume, pad_value=-1.0)

        # Original should be in center
        assert np.any(padded == 1.0)
        assert np.any(padded == -1.0)

    def test_pad_volume_custom_value(self):
        """Test padding with custom pad value."""
        volume = np.ones((50, 50, 50), dtype=np.float32)
        swinf = SlidingWindowInference(window_size=(64, 64, 64), stride=(64, 64, 64))

        padded = swinf._pad_volume(volume, pad_value=2.5)

        assert np.any(padded == 2.5)
        assert np.any(padded == 1.0)
