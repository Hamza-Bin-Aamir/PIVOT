"""Unit tests for isotropic resampling module."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.data.resampling import (
    calculate_isotropic_shape,
    calculate_resampling_factor,
    is_isotropic,
    resample_mask,
    resample_to_isotropic,
)


class TestResampleToIsotropic:
    """Tests for resample_to_isotropic function."""

    def test_basic_resampling(self):
        """Test basic isotropic resampling with known transformation."""
        volume = np.random.rand(60, 128, 128).astype(np.float32)
        original_spacing = (0.7, 0.7, 2.5)
        target_spacing = (1.0, 1.0, 1.0)

        resampled, actual_spacing = resample_to_isotropic(volume, original_spacing, target_spacing)

        assert all(abs(a - t) < 0.01 for a, t in zip(actual_spacing, target_spacing, strict=True))

        expected_shape = calculate_isotropic_shape(volume.shape, original_spacing, target_spacing)
        assert resampled.shape == expected_shape

        assert resampled.min() >= volume.min() - 1e-6
        assert resampled.max() <= volume.max() + 1e-6

    def test_already_isotropic(self):
        """Test that already isotropic volumes are not resampled."""
        volume = np.random.rand(32, 32, 32).astype(np.float32)
        original_spacing = (1.0, 1.0, 1.0)

        resampled, actual_spacing = resample_to_isotropic(
            volume, original_spacing, original_spacing
        )

        assert resampled.shape == volume.shape
        assert actual_spacing == original_spacing
        np.testing.assert_array_equal(resampled, volume)

    def test_anisotropic_to_isotropic(self):
        """Test conversion from highly anisotropic to isotropic."""
        volume = np.random.rand(24, 192, 192).astype(np.float32)
        original_spacing = (0.5, 0.5, 5.0)

        resampled, _ = resample_to_isotropic(volume, original_spacing)

        assert resampled.shape[0] > volume.shape[0] * 4
        assert resampled.shape[1] < volume.shape[1]
        assert resampled.shape[2] < volume.shape[2]

    def test_preserve_hu_values(self):
        """Test that HU values are preserved during resampling."""
        volume = np.random.uniform(-1000, 400, size=(24, 96, 96)).astype(np.float32)
        original_spacing = (0.7, 0.7, 2.5)

        resampled, _ = resample_to_isotropic(volume, original_spacing, preserve_range=True)

        assert resampled.min() >= volume.min() - 1e-6
        assert resampled.max() <= volume.max() + 1e-6

    def test_linear_interpolation(self):
        """Test linear interpolation mode."""
        volume = np.random.rand(24, 96, 96).astype(np.float32)
        original_spacing = (1.0, 1.0, 2.0)

        resampled, _ = resample_to_isotropic(volume, original_spacing, interpolation="linear")

        assert resampled.shape[0] > volume.shape[0]

    def test_nearest_interpolation(self):
        """Test nearest neighbor interpolation mode."""
        volume = np.random.randint(0, 5, size=(24, 96, 96))
        original_spacing = (2.0, 1.0, 1.0)

        resampled, _ = resample_to_isotropic(volume, original_spacing, interpolation="nearest")

        assert set(np.unique(resampled)).issubset(set(np.unique(volume)))

    def test_bspline_interpolation(self):
        """Test B-spline interpolation mode."""
        volume = np.random.rand(24, 96, 96).astype(np.float32)
        original_spacing = (1.0, 1.0, 2.0)

        resampled, _ = resample_to_isotropic(volume, original_spacing, interpolation="bspline")

        assert resampled.shape[0] > volume.shape[0]

    def test_custom_target_spacing(self):
        """Test resampling to custom (non-1mm) isotropic spacing."""
        volume = np.random.rand(60, 128, 128).astype(np.float32)
        original_spacing = (0.7, 0.7, 2.5)
        target_spacing = (2.0, 2.0, 2.0)

        resampled, actual_spacing = resample_to_isotropic(volume, original_spacing, target_spacing)

        assert all(abs(a - 2.0) < 0.01 for a in actual_spacing)

    def test_invalid_volume_dimensions(self):
        """Test error handling for non-3D volumes."""
        volume_2d = np.random.rand(64, 64)

        with pytest.raises(ValueError, match="Expected 3D volume"):
            resample_to_isotropic(volume_2d, (1.0, 1.0, 1.0))

    def test_invalid_spacing_dimensions(self):
        """Test error handling for invalid spacing dimensions."""
        volume = np.random.rand(32, 32, 32)

        with pytest.raises(ValueError, match="Expected 3D spacing"):
            resample_to_isotropic(volume, (1.0, 1.0))

    def test_negative_spacing(self):
        """Test error handling for negative spacing values."""
        volume = np.random.rand(32, 32, 32)

        with pytest.raises(ValueError, match="must be positive"):
            resample_to_isotropic(volume, (-1.0, 1.0, 1.0))

    def test_zero_spacing(self):
        """Test error handling for zero spacing values."""
        volume = np.random.rand(32, 32, 32)

        with pytest.raises(ValueError, match="must be positive"):
            resample_to_isotropic(volume, (0.0, 1.0, 1.0))

    def test_negative_target_spacing(self):
        """Test error handling for negative target spacing."""
        volume = np.random.rand(32, 32, 32)

        with pytest.raises(ValueError, match="must be positive"):
            resample_to_isotropic(volume, (1.0, 1.0, 1.0), (1.0, -1.0, 1.0))

    def test_physical_size_preservation(self):
        """Test that physical size is preserved during resampling."""
        volume = np.random.rand(60, 128, 128).astype(np.float32)
        original_spacing = (0.7, 0.7, 2.5)
        target_spacing = (1.0, 1.0, 1.0)

        original_size = tuple(
            s * sp for s, sp in zip(volume.shape[::-1], original_spacing, strict=True)
        )

        resampled, actual_spacing = resample_to_isotropic(volume, original_spacing, target_spacing)

        new_size = tuple(
            s * sp for s, sp in zip(resampled.shape[::-1], actual_spacing, strict=True)
        )

        for orig, new in zip(original_size, new_size, strict=True):
            assert abs(orig - new) / orig < 0.05

    def test_upsampling(self):
        """Test upsampling (increasing resolution)."""
        volume = np.ones((24, 48, 48), dtype=np.float32)
        resampled, _ = resample_to_isotropic(volume, (2.0, 2.0, 2.0))

        assert all(r > o * 1.8 for r, o in zip(resampled.shape, volume.shape, strict=True))

    def test_downsampling(self):
        """Test downsampling (decreasing resolution)."""
        volume = np.ones((96, 160, 160), dtype=np.float32)
        resampled, _ = resample_to_isotropic(volume, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))

        assert all(r < o * 0.6 for r, o in zip(resampled.shape, volume.shape, strict=True))

    def test_mixed_updown_sampling(self):
        """Test mixed upsampling and downsampling."""
        volume = np.random.rand(24, 192, 192).astype(np.float32)
        original_spacing = (0.5, 0.5, 5.0)

        resampled, _ = resample_to_isotropic(volume, original_spacing)

        assert resampled.shape[0] > volume.shape[0]
        assert resampled.shape[1] < volume.shape[1]
        assert resampled.shape[2] < volume.shape[2]

    def test_resampler_execute_failure(self):
        """Resampler exceptions should surface as RuntimeError."""
        volume = np.random.rand(8, 16, 16).astype(np.float32)
        original_spacing = (0.7, 0.7, 2.5)

        with (
            patch(
                "src.data.resampling.sitk.ResampleImageFilter.Execute",
                side_effect=RuntimeError("filter error"),
                autospec=True,
            ),
            pytest.raises(RuntimeError, match="Resampling failed"),
        ):
            resample_to_isotropic(volume, original_spacing)


class TestResampleMask:
    """Tests for resample_mask function."""

    def test_binary_preservation(self):
        """Test that binary values are preserved in mask resampling."""
        mask = np.random.randint(0, 2, size=(48, 128, 128))
        resampled_mask, _ = resample_mask(mask, (0.7, 0.7, 2.5))

        assert set(np.unique(resampled_mask)).issubset({0, 1})

    def test_multi_label_preservation(self):
        """Test that discrete labels are preserved."""
        mask = np.random.randint(0, 5, size=(32, 96, 96))
        resampled_mask, _ = resample_mask(mask, (2.0, 1.0, 1.0))

        assert set(np.unique(resampled_mask)).issubset(set(np.unique(mask)))

    def test_mask_shape_calculation(self):
        """Test that mask resampling produces correct shape."""
        mask = np.random.randint(0, 2, size=(48, 128, 128))
        original_spacing = (0.7, 0.7, 2.5)
        target_spacing = (1.0, 1.0, 1.0)

        resampled_mask, actual_spacing = resample_mask(mask, original_spacing, target_spacing)

        expected_shape = calculate_isotropic_shape(mask.shape, original_spacing, target_spacing)
        assert resampled_mask.shape == expected_shape
        assert actual_spacing == target_spacing


class TestCalculateIsotropicShape:
    """Tests for calculate_isotropic_shape function."""

    def test_basic_shape_calculation(self):
        """Test basic shape calculation."""
        original_shape = (60, 128, 128)
        original_spacing = (0.7, 0.7, 2.5)

        new_shape = calculate_isotropic_shape(original_shape, original_spacing, (1.0, 1.0, 1.0))

        assert new_shape[0] > original_shape[0]
        assert new_shape[1] < original_shape[1]
        assert new_shape[2] < original_shape[2]

    def test_shape_physical_size(self):
        """Test that physical size is preserved in shape calculation."""
        original_shape = (60, 128, 128)
        original_spacing = (0.7, 0.7, 2.5)

        new_shape = calculate_isotropic_shape(original_shape, original_spacing, (1.0, 1.0, 1.0))

        original_size = tuple(
            s * sp for s, sp in zip(original_shape[::-1], original_spacing, strict=True)
        )
        new_size = tuple(s * sp for s, sp in zip(new_shape[::-1], (1.0, 1.0, 1.0), strict=True))

        for orig, new in zip(original_size, new_size, strict=True):
            assert abs(orig - new) / orig < 0.02

    def test_invalid_shape_dimensions(self):
        """Test error handling for invalid shape dimensions."""
        with pytest.raises(ValueError, match="Expected 3D shape"):
            calculate_isotropic_shape((128, 128), (1.0, 1.0, 1.0))

    def test_invalid_spacing_in_calculation(self):
        """Test error handling for invalid spacing in shape calculation."""
        with pytest.raises(ValueError, match="Expected 3D spacing"):
            calculate_isotropic_shape((60, 128, 128), (1.0, 1.0))

    def test_invalid_target_spacing(self):
        """Test error handling for invalid target spacing."""
        with pytest.raises(ValueError, match="Expected 3D target spacing"):
            calculate_isotropic_shape((60, 128, 128), (0.7, 0.7, 2.5), (1.0, 1.0))


class TestIsIsotropic:
    """Tests for is_isotropic function."""

    def test_perfectly_isotropic(self):
        """Test detection of perfectly isotropic spacing."""
        assert is_isotropic((1.0, 1.0, 1.0)) is True

    def test_anisotropic(self):
        """Test detection of anisotropic spacing."""
        assert is_isotropic((2.5, 0.7, 0.7)) is False

    def test_nearly_isotropic_within_tolerance(self):
        """Test nearly isotropic spacing within tolerance."""
        assert is_isotropic((1.0, 1.005, 0.995), tolerance=0.01) is True

    def test_nearly_isotropic_outside_tolerance(self):
        """Test nearly isotropic spacing outside tolerance."""
        assert is_isotropic((1.0, 1.05, 0.95), tolerance=0.01) is False

    def test_custom_tolerance(self):
        """Test custom tolerance values."""
        assert is_isotropic((1.0, 1.1, 0.9), tolerance=0.05) is False
        assert is_isotropic((1.0, 1.1, 0.9), tolerance=0.15) is True

    def test_different_isotropic_values(self):
        """Test isotropic spacing with different values."""
        for spacing in ((2.0, 2.0, 2.0), (0.5, 0.5, 0.5), (3.14, 3.14, 3.14)):
            assert is_isotropic(spacing) is True

    def test_isotropic_invalid_length(self):
        """Non-3D spacing should raise ValueError."""
        with pytest.raises(ValueError, match="Expected 3D spacing"):
            is_isotropic((1.0, 1.0))

    def test_isotropic_non_positive(self):
        """Zero or negative spacing should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            is_isotropic((0.0, 1.0, 1.0))


class TestCalculateResamplingFactor:
    """Tests for calculate_resampling_factor function."""

    def test_upsampling_factor(self):
        """Test calculation of upsampling factor."""
        factors = calculate_resampling_factor((2.0, 2.0, 2.0), (1.0, 1.0, 1.0))
        assert all(f == 2.0 for f in factors)

    def test_downsampling_factor(self):
        """Test calculation of downsampling factor."""
        factors = calculate_resampling_factor((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
        assert all(f == 0.5 for f in factors)

    def test_mixed_factors(self):
        """Test mixed upsampling and downsampling factors."""
        factors = calculate_resampling_factor((5.0, 0.5, 0.5), (1.0, 1.0, 1.0))
        assert factors == (5.0, 0.5, 0.5)

    def test_no_resampling(self):
        """Test factor when no resampling needed."""
        factors = calculate_resampling_factor((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        assert all(f == 1.0 for f in factors)


class TestIntegration:
    """Integration tests for resampling workflow."""

    def test_resample_and_verify_roundtrip(self):
        """Test that roundtrip resampling approximately recovers original."""
        volume = np.zeros((32, 64, 64), dtype=np.float32)
        volume[16, 32, 32] = 1.0

        original_spacing = (2.0, 1.0, 1.0)
        target_spacing = (1.0, 1.0, 1.0)

        resampled, _ = resample_to_isotropic(volume, original_spacing, target_spacing)
        recovered, _ = resample_to_isotropic(resampled, target_spacing, original_spacing)

        assert recovered.shape == volume.shape

    def test_volume_with_realistic_ct_data(self):
        """Test with realistic CT HU values."""
        volume = np.random.uniform(-1000, 400, size=(48, 192, 192)).astype(np.float32)
        volume[20:30, 80:90, 80:90] = np.random.uniform(50, 200, size=(10, 10, 10))

        resampled, _ = resample_to_isotropic(volume, (0.7, 0.7, 2.5), preserve_range=True)

        assert -1000 <= resampled.min() <= -990
        assert 390 <= resampled.max() <= 400

    def test_mask_and_volume_consistent_resampling(self):
        """Test that mask and volume resample to same shape."""
        volume = np.random.rand(48, 128, 128).astype(np.float32)
        mask = np.random.randint(0, 2, size=(48, 128, 128))
        original_spacing = (0.7, 0.7, 2.5)

        resampled_volume, spacing1 = resample_to_isotropic(volume, original_spacing)
        resampled_mask, spacing2 = resample_mask(mask, original_spacing)

        assert resampled_volume.shape == resampled_mask.shape
        assert spacing1 == spacing2

    def test_large_volume_handling(self):
        """Test handling of moderately large volumes."""
        volume = np.random.rand(64, 256, 256).astype(np.float32)
        resampled, _ = resample_to_isotropic(volume, (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))

        assert resampled.nbytes < volume.nbytes

    def test_resampling_guard_prevents_huge_volume(self):
        """Safety guard should raise when output would exceed limit."""
        volume = np.random.rand(16, 32, 32).astype(np.float32)
        target_spacing = (0.2, 0.2, 0.2)

        with pytest.raises(ValueError, match="voxels"):
            resample_to_isotropic(
                volume,
                (1.0, 1.0, 1.0),
                target_spacing,
                max_voxels=volume.size * 4,
            )

    def test_resampling_guard_override(self):
        """Safety guard can be disabled when caller accepts the cost."""
        volume = np.random.rand(16, 32, 32).astype(np.float32)
        target_spacing = (0.2, 0.2, 0.2)

        resampled, _ = resample_to_isotropic(
            volume,
            (1.0, 1.0, 1.0),
            target_spacing,
            max_voxels=None,
        )

        expected_shape = calculate_isotropic_shape(volume.shape, (1.0, 1.0, 1.0), target_spacing)
        assert resampled.shape == expected_shape
