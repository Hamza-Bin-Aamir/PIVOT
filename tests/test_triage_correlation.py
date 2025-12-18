"""Tests for triage score correlation evaluation."""

import numpy as np
import pytest

from src.eval.triage_correlation import (
    calculate_expected_calibration_error,
    compute_auc_roc,
    compute_calibration_curve,
    compute_pearson_correlation,
    compute_spearman_correlation,
)


class TestComputePearsonCorrelation:
    """Test compute_pearson_correlation function."""

    def test_perfect_correlation(self) -> None:
        """Test perfect positive correlation."""
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr, p_value = compute_pearson_correlation(predicted, actual)
        assert abs(corr - 1.0) < 0.001
        assert p_value < 0.01  # Highly significant

    def test_perfect_negative_correlation(self) -> None:
        """Test perfect negative correlation."""
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        corr, p_value = compute_pearson_correlation(predicted, actual)
        assert abs(corr + 1.0) < 0.001  # Close to -1
        assert p_value < 0.01

    def test_no_correlation(self) -> None:
        """Test uncorrelated data."""
        predicted = np.array([1.0, 2.0, 1.0, 2.0])
        actual = np.array([1.0, 1.0, 2.0, 2.0])
        corr, _ = compute_pearson_correlation(predicted, actual)
        assert abs(corr) < 0.5  # Low correlation

    def test_insufficient_data(self) -> None:
        """Test with insufficient data points."""
        predicted = np.array([1.0])
        actual = np.array([2.0])
        corr, p_value = compute_pearson_correlation(predicted, actual)
        assert np.isnan(corr)
        assert np.isnan(p_value)

    def test_zero_variance(self) -> None:
        """Test with zero variance in one variable."""
        predicted = np.array([1.0, 1.0, 1.0])
        actual = np.array([1.0, 2.0, 3.0])
        corr, p_value = compute_pearson_correlation(predicted, actual)
        assert np.isnan(corr)
        assert np.isnan(p_value)

    def test_with_nan_values(self) -> None:
        """Test filtering of NaN values."""
        predicted = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr, _ = compute_pearson_correlation(predicted, actual)
        # Should compute on 4 valid points
        assert abs(corr - 1.0) < 0.001

    def test_shape_mismatch(self) -> None:
        """Test error for mismatched shapes."""
        predicted = np.array([1.0, 2.0])
        actual = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_pearson_correlation(predicted, actual)


class TestComputeSpearmanCorrelation:
    """Test compute_spearman_correlation function."""

    def test_perfect_monotonic(self) -> None:
        """Test perfect monotonic relationship."""
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # Squares (monotonic but not linear)
        corr, _ = compute_spearman_correlation(predicted, actual)
        assert abs(corr - 1.0) < 0.001

    def test_ranks_matter_not_values(self) -> None:
        """Test that Spearman uses ranks, not raw values."""
        predicted = np.array([1.0, 2.0, 3.0, 4.0])
        actual = np.array([10.0, 20.0, 30.0, 40.0])
        corr_spearman, _ = compute_spearman_correlation(predicted, actual)
        corr_pearson, _ = compute_pearson_correlation(predicted, actual)
        # Both should be 1.0 for this data
        assert abs(corr_spearman - corr_pearson) < 0.001

    def test_handles_ties(self) -> None:
        """Test handling of tied ranks."""
        predicted = np.array([1.0, 2.0, 2.0, 3.0])
        actual = np.array([1.0, 2.0, 2.0, 3.0])
        corr, _ = compute_spearman_correlation(predicted, actual)
        assert abs(corr - 1.0) < 0.001

    def test_insufficient_data(self) -> None:
        """Test with insufficient data."""
        predicted = np.array([1.0])
        actual = np.array([2.0])
        corr, p_value = compute_spearman_correlation(predicted, actual)
        assert np.isnan(corr)
        assert np.isnan(p_value)

    def test_shape_mismatch(self) -> None:
        """Test error for mismatched shapes."""
        predicted = np.array([1.0, 2.0])
        actual = np.array([1.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_spearman_correlation(predicted, actual)


class TestComputeAUCROC:
    """Test compute_auc_roc function."""

    def test_perfect_classifier(self) -> None:
        """Test AUC for perfect classification."""
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        labels = np.array([True, True, True, False, False, False])
        auc = compute_auc_roc(scores, labels)
        assert auc == 1.0

    def test_reverse_classifier(self) -> None:
        """Test AUC for reversed predictions."""
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = np.array([True, True, True, False, False, False])
        auc = compute_auc_roc(scores, labels)
        assert auc == 0.0

    def test_random_classifier(self) -> None:
        """Test AUC for random-like performance."""
        scores = np.array([0.5, 0.6, 0.4, 0.5, 0.6, 0.4])
        labels = np.array([True, False, True, False, True, False])
        auc = compute_auc_roc(scores, labels)
        # Should be around 0.5 for random performance
        assert 0.3 < auc < 0.7

    def test_partial_separation(self) -> None:
        """Test AUC with partial class separation."""
        scores = np.array([0.8, 0.7, 0.6, 0.4, 0.3, 0.2])
        labels = np.array([True, True, False, True, False, False])
        auc = compute_auc_roc(scores, labels)
        # Should be between 0.5 and 1.0
        assert 0.5 < auc < 1.0

    def test_handles_ties(self) -> None:
        """Test handling of tied scores."""
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([True, True, False, False])
        auc = compute_auc_roc(scores, labels)
        # All scores equal, should give 0.5
        assert abs(auc - 0.5) < 0.001

    def test_single_class(self) -> None:
        """Test with only one class present."""
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([True, True, True])  # Only positive class
        auc = compute_auc_roc(scores, labels)
        assert np.isnan(auc)

    def test_insufficient_data(self) -> None:
        """Test with insufficient data."""
        scores = np.array([0.5])
        labels = np.array([True])
        auc = compute_auc_roc(scores, labels)
        assert np.isnan(auc)

    def test_with_nan_scores(self) -> None:
        """Test filtering of NaN scores."""
        scores = np.array([0.9, np.nan, 0.7, 0.3, 0.2, 0.1])
        labels = np.array([True, True, True, False, False, False])
        auc = compute_auc_roc(scores, labels)
        # Should compute on 5 valid points (2 positive, 3 negative)
        assert 0.5 < auc <= 1.0

    def test_shape_mismatch(self) -> None:
        """Test error for mismatched shapes."""
        scores = np.array([0.1, 0.2])
        labels = np.array([True])
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_auc_roc(scores, labels)


class TestComputeCalibrationCurve:
    """Test compute_calibration_curve function."""

    def test_perfect_calibration(self) -> None:
        """Test perfectly calibrated predictions."""
        # Create data where predicted probability matches actual frequency
        probs = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
        labels = np.array([False, False, False, True, True, True])
        # Bin 1: [0.1, 0.1] -> 0% positive (actual: 0.0)
        # Bin 5: [0.5, 0.5] -> 50% positive (actual: 0.5)
        # Bin 9: [0.9, 0.9] -> 100% positive (actual: 1.0)

        mean_pred, actual_freq, bin_counts = compute_calibration_curve(probs, labels, n_bins=10)

        # Check that populated bins have matching predictions and frequencies
        for i in range(len(bin_counts)):
            if bin_counts[i] > 0:
                # Allow some tolerance due to binning
                assert abs(mean_pred[i] - actual_freq[i]) < 0.5

    def test_overconfident_predictions(self) -> None:
        """Test overconfident predictions (predicted > actual)."""
        probs = np.array([0.9, 0.9, 0.9, 0.9])  # High confidence
        labels = np.array([True, False, False, False])  # Only 25% positive
        mean_pred, actual_freq, _ = compute_calibration_curve(probs, labels, n_bins=10)

        # Find the bin with data
        valid_bins = ~np.isnan(mean_pred)
        if np.any(valid_bins):
            # Predicted should be higher than actual
            assert mean_pred[valid_bins][0] > actual_freq[valid_bins][0]

    def test_empty_bins(self) -> None:
        """Test that empty bins have NaN values."""
        probs = np.array([0.1, 0.9])  # Only use extreme bins
        labels = np.array([False, True])
        mean_pred, actual_freq, bin_counts = compute_calibration_curve(probs, labels, n_bins=10)

        # Most bins should be empty (count = 0, values = NaN)
        empty_bins = bin_counts == 0
        assert np.all(np.isnan(mean_pred[empty_bins]))
        assert np.all(np.isnan(actual_freq[empty_bins]))

    def test_single_bin(self) -> None:
        """Test with single bin."""
        probs = np.array([0.2, 0.4, 0.6, 0.8])
        labels = np.array([False, True, True, True])
        mean_pred, actual_freq, bin_counts = compute_calibration_curve(probs, labels, n_bins=1)

        assert len(mean_pred) == 1
        assert len(actual_freq) == 1
        assert bin_counts[0] == 4
        assert abs(actual_freq[0] - 0.75) < 0.01  # 3/4 positive

    def test_empty_input(self) -> None:
        """Test with empty arrays."""
        probs = np.array([])
        labels = np.array([], dtype=bool)
        mean_pred, actual_freq, bin_counts = compute_calibration_curve(probs, labels, n_bins=10)

        assert len(mean_pred) == 0
        assert len(actual_freq) == 0
        assert len(bin_counts) == 0

    def test_invalid_n_bins(self) -> None:
        """Test error for invalid n_bins."""
        probs = np.array([0.5])
        labels = np.array([True])
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            compute_calibration_curve(probs, labels, n_bins=0)

    def test_shape_mismatch(self) -> None:
        """Test error for mismatched shapes."""
        probs = np.array([0.1, 0.2])
        labels = np.array([True])
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_calibration_curve(probs, labels)


class TestCalculateExpectedCalibrationError:
    """Test calculate_expected_calibration_error function."""

    def test_perfect_calibration(self) -> None:
        """Test ECE for perfectly calibrated predictions."""
        # Create perfectly calibrated data
        probs = np.concatenate([np.full(10, 0.1), np.full(10, 0.5), np.full(10, 0.9)])
        labels = np.concatenate([
            np.array([True] * 1 + [False] * 9),  # 10% positive for 0.1
            np.array([True] * 5 + [False] * 5),  # 50% positive for 0.5
            np.array([True] * 9 + [False] * 1),  # 90% positive for 0.9
        ])
        ece = calculate_expected_calibration_error(probs, labels, n_bins=10)

        # Should be very close to 0
        assert ece < 0.05

    def test_poor_calibration(self) -> None:
        """Test ECE for poorly calibrated predictions."""
        # Overconfident predictions
        probs = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        labels = np.array([True, False, False, False, False])  # Only 20% positive
        ece = calculate_expected_calibration_error(probs, labels, n_bins=10)

        # Should have high ECE (predicted 0.9, actual 0.2)
        assert ece > 0.5

    def test_empty_input(self) -> None:
        """Test ECE with empty input."""
        probs = np.array([])
        labels = np.array([], dtype=bool)
        ece = calculate_expected_calibration_error(probs, labels)
        assert np.isnan(ece)

    def test_with_nan_values(self) -> None:
        """Test ECE filters NaN values."""
        probs = np.array([0.5, np.nan, 0.5, 0.5])
        labels = np.array([True, False, True, False])
        ece = calculate_expected_calibration_error(probs, labels)
        # Should compute on 3 valid values
        assert np.isfinite(ece)

    def test_shape_mismatch(self) -> None:
        """Test error for mismatched shapes."""
        probs = np.array([0.1, 0.2])
        labels = np.array([True])
        with pytest.raises(ValueError, match="Shape mismatch"):
            calculate_expected_calibration_error(probs, labels)


class TestIntegrationTriageCorrelation:
    """Integration tests for triage correlation evaluation."""

    def test_realistic_triage_scenario(self) -> None:
        """Test realistic triage score evaluation."""
        # Simulate triage scores (0-1) and actual severity (0-10)
        predicted_triage = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2])
        actual_severity = np.array([9.0, 8.0, 6.0, 4.0, 3.0, 1.0])

        # Should have high Pearson correlation
        corr_pearson, _ = compute_pearson_correlation(predicted_triage, actual_severity)
        assert corr_pearson > 0.9

        # Should have high Spearman correlation (perfect monotonic)
        corr_spearman, _ = compute_spearman_correlation(predicted_triage, actual_severity)
        assert corr_spearman > 0.95

    def test_binary_triage_classification(self) -> None:
        """Test binary triage (urgent vs non-urgent)."""
        # Predicted probabilities of being urgent
        predicted_probs = np.array([0.95, 0.85, 0.75, 0.25, 0.15, 0.05])
        # Actual urgency labels
        is_urgent = np.array([True, True, True, False, False, False])

        # Should have perfect AUC
        auc = compute_auc_roc(predicted_probs, is_urgent)
        assert auc == 1.0

        # Should have good calibration
        ece = calculate_expected_calibration_error(predicted_probs, is_urgent)
        assert ece < 0.3

    def test_correlation_vs_calibration(self) -> None:
        """Test that correlation and calibration measure different things."""
        # High correlation but poor calibration (systematic bias)
        predicted = np.array([0.6, 0.7, 0.8, 0.9, 1.0])  # Shifted up by 0.5
        actual_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        actual_labels = np.array([False, False, False, True, True])

        # Should have perfect correlation (same ordering)
        corr, _ = compute_pearson_correlation(predicted, actual_probs)
        assert abs(corr - 1.0) < 0.01

        # But poor calibration (predictions too high)
        ece = calculate_expected_calibration_error(predicted, actual_labels)
        assert ece > 0.3  # Significant calibration error

    def test_mixed_quality_predictions(self) -> None:
        """Test evaluation with mixed quality predictions."""
        # Some good, some bad predictions
        predicted = np.array([0.9, 0.8, 0.5, 0.2, 0.6, 0.3, 0.7, 0.4])
        actual = np.array([0.95, 0.85, 0.55, 0.25, 0.4, 0.5, 0.6, 0.3])

        corr_pearson, p_value = compute_pearson_correlation(predicted, actual)
        # Should have moderate to high correlation
        assert corr_pearson > 0.5
        assert p_value < 0.05  # Significant

        corr_spearman, _ = compute_spearman_correlation(predicted, actual)
        # Spearman should also show correlation
        assert corr_spearman > 0.5
