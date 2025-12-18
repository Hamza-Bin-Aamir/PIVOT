"""Tests for size estimation accuracy evaluation."""

import numpy as np
import pytest

from src.eval.size_accuracy import (
    calculate_mae,
    calculate_rmse,
    calculate_size_metrics,
    compute_diameter_error,
    compute_diameter_from_mask,
    compute_volume_error,
    compute_volume_from_mask,
)


class TestComputeVolumeFromMask:
    """Test compute_volume_from_mask function."""

    def test_empty_mask(self) -> None:
        """Test volume of empty mask."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        volume = compute_volume_from_mask(mask)
        assert volume == 0.0

    def test_full_mask_unit_spacing(self) -> None:
        """Test volume with unit spacing."""
        mask = np.ones((5, 5, 5), dtype=bool)
        volume = compute_volume_from_mask(mask)
        assert volume == 125.0  # 5 * 5 * 5 = 125 voxels

    def test_partial_mask(self) -> None:
        """Test volume of partially filled mask."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[0:5, 0:5, 0:5] = True  # 125 voxels
        volume = compute_volume_from_mask(mask)
        assert volume == 125.0

    def test_anisotropic_spacing(self) -> None:
        """Test volume with anisotropic spacing."""
        mask = np.ones((5, 5, 5), dtype=bool)
        spacing = np.array([2.0, 2.0, 3.0])
        volume = compute_volume_from_mask(mask, spacing)
        # 125 voxels * (2.0 * 2.0 * 3.0) = 125 * 12 = 1500
        assert volume == 1500.0

    def test_2d_mask(self) -> None:
        """Test volume calculation for 2D mask."""
        mask = np.ones((10, 10), dtype=bool)
        spacing = np.array([0.5, 0.5])
        volume = compute_volume_from_mask(mask, spacing)
        # 100 voxels * (0.5 * 0.5) = 25
        assert volume == 25.0

    def test_invalid_spacing_dimension(self) -> None:
        """Test error for mismatched spacing dimension."""
        mask = np.ones((5, 5, 5), dtype=bool)
        spacing = np.array([1.0, 1.0])  # 2D spacing for 3D mask
        with pytest.raises(ValueError, match="Spacing dimension .* must match mask dimension"):
            compute_volume_from_mask(mask, spacing)


class TestComputeDiameterFromMask:
    """Test compute_diameter_from_mask function."""

    def test_empty_mask(self) -> None:
        """Test diameter of empty mask."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        diameter = compute_diameter_from_mask(mask)
        assert diameter == 0.0

    def test_single_voxel_unit_spacing(self) -> None:
        """Test diameter of single voxel."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True
        diameter = compute_diameter_from_mask(mask)
        # Volume = 1, diameter = (6*1/pi)^(1/3) ≈ 1.24
        assert abs(diameter - 1.24) < 0.01

    def test_sphere_approximation(self) -> None:
        """Test diameter calculation approximates sphere."""
        # Create a roughly spherical mask
        mask = np.zeros((20, 20, 20), dtype=bool)
        center = np.array([10, 10, 10])
        radius = 5

        # Fill voxels within radius
        for i in range(20):
            for j in range(20):
                for k in range(20):
                    if np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2) <= radius:
                        mask[i, j, k] = True

        diameter = compute_diameter_from_mask(mask)
        # Expected diameter ≈ 2 * radius = 10
        # Allow some tolerance due to discretization
        assert abs(diameter - 10.0) < 1.0

    def test_anisotropic_spacing(self) -> None:
        """Test diameter with anisotropic spacing."""
        mask = np.ones((5, 5, 5), dtype=bool)
        spacing = np.array([2.0, 2.0, 2.0])
        diameter = compute_diameter_from_mask(mask, spacing)
        # Volume = 125 * 8 = 1000
        # Diameter = (6*1000/pi)^(1/3) ≈ 12.41
        assert abs(diameter - 12.41) < 0.01


class TestComputeVolumeError:
    """Test compute_volume_error function."""

    def test_perfect_prediction(self) -> None:
        """Test error for perfect predictions."""
        predicted = np.array([10.0, 20.0, 30.0])
        ground_truth = np.array([10.0, 20.0, 30.0])
        errors = compute_volume_error(predicted, ground_truth)
        np.testing.assert_array_equal(errors, np.zeros(3))

    def test_absolute_error(self) -> None:
        """Test absolute error calculation."""
        predicted = np.array([15.0, 25.0, 35.0])
        ground_truth = np.array([10.0, 20.0, 30.0])
        errors = compute_volume_error(predicted, ground_truth, relative=False)
        expected = np.array([5.0, 5.0, 5.0])
        np.testing.assert_array_equal(errors, expected)

    def test_relative_error(self) -> None:
        """Test relative error calculation."""
        predicted = np.array([15.0, 30.0, 45.0])
        ground_truth = np.array([10.0, 20.0, 30.0])
        errors = compute_volume_error(predicted, ground_truth, relative=True)
        expected = np.array([0.5, 0.5, 0.5])  # 50% overestimation
        np.testing.assert_array_almost_equal(errors, expected)

    def test_underestimation(self) -> None:
        """Test negative errors for underestimation."""
        predicted = np.array([5.0, 10.0, 15.0])
        ground_truth = np.array([10.0, 20.0, 30.0])
        errors = compute_volume_error(predicted, ground_truth, relative=False)
        expected = np.array([-5.0, -10.0, -15.0])
        np.testing.assert_array_equal(errors, expected)

    def test_zero_ground_truth_relative(self) -> None:
        """Test relative error with zero ground truth."""
        predicted = np.array([10.0, 0.0])
        ground_truth = np.array([0.0, 0.0])
        errors = compute_volume_error(predicted, ground_truth, relative=True)
        # Should be inf where ground_truth is zero (np.where sets it to inf)
        assert errors[0] == np.inf
        assert errors[1] == np.inf  # 0/0 case also set to inf by np.where

    def test_shape_mismatch(self) -> None:
        """Test error for mismatched shapes."""
        predicted = np.array([10.0, 20.0])
        ground_truth = np.array([10.0, 20.0, 30.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_volume_error(predicted, ground_truth)


class TestComputeDiameterError:
    """Test compute_diameter_error function."""

    def test_perfect_prediction(self) -> None:
        """Test error for perfect predictions."""
        predicted = np.array([5.0, 10.0, 15.0])
        ground_truth = np.array([5.0, 10.0, 15.0])
        errors = compute_diameter_error(predicted, ground_truth)
        np.testing.assert_array_equal(errors, np.zeros(3))

    def test_absolute_error(self) -> None:
        """Test absolute error calculation."""
        predicted = np.array([6.0, 11.0, 16.0])
        ground_truth = np.array([5.0, 10.0, 15.0])
        errors = compute_diameter_error(predicted, ground_truth, relative=False)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(errors, expected)

    def test_relative_error(self) -> None:
        """Test relative error calculation."""
        predicted = np.array([6.0, 12.0, 18.0])
        ground_truth = np.array([5.0, 10.0, 15.0])
        errors = compute_diameter_error(predicted, ground_truth, relative=True)
        expected = np.array([0.2, 0.2, 0.2])  # 20% overestimation
        np.testing.assert_array_almost_equal(errors, expected)

    def test_shape_mismatch(self) -> None:
        """Test error for mismatched shapes."""
        predicted = np.array([5.0, 10.0])
        ground_truth = np.array([5.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_diameter_error(predicted, ground_truth)


class TestCalculateMAE:
    """Test calculate_mae function."""

    def test_zero_errors(self) -> None:
        """Test MAE with zero errors."""
        errors = np.array([0.0, 0.0, 0.0])
        mae = calculate_mae(errors)
        assert mae == 0.0

    def test_positive_errors(self) -> None:
        """Test MAE with positive errors."""
        errors = np.array([1.0, 2.0, 3.0])
        mae = calculate_mae(errors)
        assert mae == 2.0

    def test_mixed_errors(self) -> None:
        """Test MAE with mixed positive/negative errors."""
        errors = np.array([-1.0, 2.0, -3.0, 4.0])
        mae = calculate_mae(errors)
        assert mae == 2.5  # (1 + 2 + 3 + 4) / 4

    def test_empty_array(self) -> None:
        """Test MAE with empty array."""
        errors = np.array([])
        mae = calculate_mae(errors)
        assert np.isnan(mae)

    def test_infinite_values(self) -> None:
        """Test MAE filters out infinite values."""
        errors = np.array([1.0, 2.0, np.inf, -np.inf])
        mae = calculate_mae(errors)
        assert mae == 1.5  # (1 + 2) / 2

    def test_nan_values(self) -> None:
        """Test MAE filters out nan values."""
        errors = np.array([1.0, 2.0, np.nan, 3.0])
        mae = calculate_mae(errors)
        assert mae == 2.0  # (1 + 2 + 3) / 3

    def test_all_infinite(self) -> None:
        """Test MAE with all infinite values."""
        errors = np.array([np.inf, -np.inf, np.nan])
        mae = calculate_mae(errors)
        assert np.isnan(mae)


class TestCalculateRMSE:
    """Test calculate_rmse function."""

    def test_zero_errors(self) -> None:
        """Test RMSE with zero errors."""
        errors = np.array([0.0, 0.0, 0.0])
        rmse = calculate_rmse(errors)
        assert rmse == 0.0

    def test_positive_errors(self) -> None:
        """Test RMSE with positive errors."""
        errors = np.array([1.0, 2.0, 3.0])
        rmse = calculate_rmse(errors)
        expected = np.sqrt((1 + 4 + 9) / 3)  # sqrt(14/3) ≈ 2.16
        assert abs(rmse - expected) < 0.01

    def test_mixed_errors(self) -> None:
        """Test RMSE with mixed positive/negative errors."""
        errors = np.array([-2.0, 2.0])
        rmse = calculate_rmse(errors)
        expected = np.sqrt((4 + 4) / 2)  # sqrt(4) = 2.0
        assert rmse == expected

    def test_empty_array(self) -> None:
        """Test RMSE with empty array."""
        errors = np.array([])
        rmse = calculate_rmse(errors)
        assert np.isnan(rmse)

    def test_filters_infinite(self) -> None:
        """Test RMSE filters out infinite values."""
        errors = np.array([1.0, 2.0, np.inf])
        rmse = calculate_rmse(errors)
        expected = np.sqrt((1 + 4) / 2)  # sqrt(2.5) ≈ 1.58
        assert abs(rmse - expected) < 0.01


class TestCalculateSizeMetrics:
    """Test calculate_size_metrics function."""

    def test_perfect_predictions(self) -> None:
        """Test metrics for perfect predictions."""
        predicted = np.array([10.0, 20.0, 30.0])
        ground_truth = np.array([10.0, 20.0, 30.0])
        metrics = calculate_size_metrics(predicted, ground_truth)

        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mean_error"] == 0.0
        assert metrics["median_error"] == 0.0
        assert metrics["std_error"] == 0.0
        assert metrics["mean_abs_relative_error"] == 0.0

    def test_consistent_overestimation(self) -> None:
        """Test metrics for consistent overestimation."""
        predicted = np.array([15.0, 25.0, 35.0])
        ground_truth = np.array([10.0, 20.0, 30.0])
        metrics = calculate_size_metrics(predicted, ground_truth)

        assert metrics["mae"] == 5.0
        assert metrics["rmse"] == 5.0
        assert metrics["mean_error"] == 5.0  # Positive bias
        assert metrics["median_error"] == 5.0
        assert metrics["std_error"] == 0.0  # No variation in errors

    def test_mixed_errors(self) -> None:
        """Test metrics with mixed over/underestimation."""
        predicted = np.array([8.0, 22.0])
        ground_truth = np.array([10.0, 20.0])
        metrics = calculate_size_metrics(predicted, ground_truth)

        assert metrics["mae"] == 2.0  # (2 + 2) / 2
        assert metrics["rmse"] == 2.0  # sqrt((4 + 4) / 2)
        assert metrics["mean_error"] == 0.0  # (-2 + 2) / 2 = 0
        assert metrics["median_error"] == 0.0
        assert abs(metrics["std_error"] - np.sqrt(4)) < 0.01  # std of [-2, 2]

    def test_relative_errors(self) -> None:
        """Test mean absolute relative error calculation."""
        predicted = np.array([11.0, 24.0])
        ground_truth = np.array([10.0, 20.0])
        metrics = calculate_size_metrics(predicted, ground_truth)

        # Relative errors: 0.1 and 0.2, mean = 0.15
        assert abs(metrics["mean_abs_relative_error"] - 0.15) < 0.01

    def test_shape_mismatch(self) -> None:
        """Test error for mismatched shapes."""
        predicted = np.array([10.0, 20.0])
        ground_truth = np.array([10.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            calculate_size_metrics(predicted, ground_truth)

    def test_all_nan(self) -> None:
        """Test metrics with all nan errors."""
        predicted = np.array([np.nan, np.nan])
        ground_truth = np.array([10.0, 20.0])
        metrics = calculate_size_metrics(predicted, ground_truth)

        assert np.isnan(metrics["mae"])
        assert np.isnan(metrics["rmse"])
        assert np.isnan(metrics["mean_error"])

    def test_zero_ground_truth(self) -> None:
        """Test metrics when ground truth contains zeros."""
        predicted = np.array([10.0, 20.0, 5.0])
        ground_truth = np.array([10.0, 20.0, 0.0])
        metrics = calculate_size_metrics(predicted, ground_truth)

        # Third value contributes to absolute error metrics (5 - 0 = 5)
        # MAE = (0 + 0 + 5) / 3 = 1.667
        assert abs(metrics["mae"] - 1.667) < 0.01
        # RMSE = sqrt((0 + 0 + 25) / 3) = 2.887
        assert abs(metrics["rmse"] - 2.887) < 0.01
        # Relative error for third element should be filtered out (inf)
        assert metrics["mean_abs_relative_error"] == 0.0


class TestIntegrationSizeAccuracy:
    """Integration tests for size accuracy evaluation."""

    def test_mask_to_metrics_workflow(self) -> None:
        """Test complete workflow from masks to metrics."""
        # Create predicted and ground truth masks
        pred_mask = np.zeros((10, 10, 10), dtype=bool)
        pred_mask[0:6, 0:6, 0:6] = True  # 216 voxels

        gt_mask = np.zeros((10, 10, 10), dtype=bool)
        gt_mask[0:5, 0:5, 0:5] = True  # 125 voxels

        spacing = np.array([1.0, 1.0, 1.0])

        # Compute volumes
        pred_vol = compute_volume_from_mask(pred_mask, spacing)
        gt_vol = compute_volume_from_mask(gt_mask, spacing)

        # Compute errors
        volumes_pred = np.array([pred_vol])
        volumes_gt = np.array([gt_vol])
        errors = compute_volume_error(volumes_pred, volumes_gt)

        # Check error
        assert errors[0] == 216 - 125  # 91

        # Calculate metrics
        metrics = calculate_size_metrics(volumes_pred, volumes_gt)
        assert metrics["mae"] == 91.0

    def test_diameter_workflow(self) -> None:
        """Test diameter calculation workflow."""
        # Create a spherical-ish mask
        mask = np.zeros((20, 20, 20), dtype=bool)
        center = np.array([10, 10, 10])
        radius = 5

        for i in range(20):
            for j in range(20):
                for k in range(20):
                    if np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2) <= radius:
                        mask[i, j, k] = True

        diameter = compute_diameter_from_mask(mask)
        # Expected diameter ≈ 10
        assert abs(diameter - 10.0) < 1.5

    def test_multiple_nodules(self) -> None:
        """Test metrics for multiple nodules."""
        # Simulate 5 nodules with varying accuracy
        predicted = np.array([10.0, 20.5, 30.0, 15.2, 25.0])
        ground_truth = np.array([10.0, 20.0, 32.0, 15.0, 24.0])

        metrics = calculate_size_metrics(predicted, ground_truth)

        # All metrics should be computed
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert not np.isnan(metrics["mean_error"])
        assert not np.isnan(metrics["median_error"])
        assert metrics["std_error"] >= 0
