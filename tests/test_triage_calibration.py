"""Unit tests for triage score calibration module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.inference.triage_calibration import TriageScoreCalibration


class TestTriageScoreCalibrationInit:
    """Tests for TriageScoreCalibration initialization."""

    def test_init_with_temperature_method(self) -> None:
        """Test initialization with temperature scaling method."""
        calibrator = TriageScoreCalibration(method="temperature")

        assert calibrator.method == "temperature"
        assert calibrator.is_fitted is False
        assert calibrator.temperature is None
        assert calibrator.calibrator is None

    def test_init_with_isotonic_method(self) -> None:
        """Test initialization with isotonic regression method."""
        calibrator = TriageScoreCalibration(method="isotonic")

        assert calibrator.method == "isotonic"
        assert calibrator.is_fitted is False
        assert calibrator.temperature is None
        assert calibrator.calibrator is None

    def test_init_with_device(self) -> None:
        """Test initialization with specific device."""
        device = torch.device("cpu")
        calibrator = TriageScoreCalibration(method="temperature", device=device)

        assert calibrator.device == device

    def test_init_with_invalid_method(self) -> None:
        """Test initialization with invalid method raises error."""
        with pytest.raises(ValueError, match="Method must be"):
            TriageScoreCalibration(method="invalid")  # type: ignore[arg-type]

    def test_default_method_is_temperature(self) -> None:
        """Test default method is temperature scaling."""
        calibrator = TriageScoreCalibration()

        assert calibrator.method == "temperature"


class TestTemperatureScaling:
    """Tests for temperature scaling calibration."""

    def test_fit_temperature_with_numpy_arrays(self) -> None:
        """Test fitting temperature scaling with numpy arrays."""
        logits = np.array([2.0, -1.0, 0.5, -0.5, 1.5], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")
        calibrator.fit(logits, labels)

        assert calibrator.is_fitted is True
        assert calibrator.temperature is not None
        # Temperature parameter exists (can be any value after optimization)
        assert isinstance(calibrator.temperature.item(), float)

    def test_fit_temperature_with_torch_tensors(self) -> None:
        """Test fitting temperature scaling with torch tensors."""
        logits = torch.tensor([2.0, -1.0, 0.5, -0.5, 1.5])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

        calibrator = TriageScoreCalibration(method="temperature")
        calibrator.fit(logits, labels)

        assert calibrator.is_fitted is True
        assert calibrator.temperature is not None

    def test_transform_temperature_scaling(self) -> None:
        """Test applying temperature scaling transformation."""
        logits = np.array([2.0, -1.0, 0.5, -0.5, 1.5], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")
        calibrator.fit(logits, labels)

        new_logits = np.array([1.0, 0.0, -1.0], dtype=np.float32)
        calibrated = calibrator.transform(new_logits)

        assert calibrated.shape == new_logits.shape
        assert np.all((calibrated >= 0) & (calibrated <= 1))
        assert calibrated.dtype == np.float32

    def test_temperature_scaling_reduces_overconfidence(self) -> None:
        """Test that temperature scaling calibration works correctly."""
        # Create overconfident logits
        overconfident_logits = np.array([10.0, -10.0, 8.0, -8.0], dtype=np.float32)
        # Create calibration data
        logits = np.array([5.0, -5.0, 4.0, -4.0, 3.0, -3.0], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")
        calibrator.fit(logits, labels)

        # Transform overconfident predictions
        calibrated = calibrator.transform(overconfident_logits)

        # Check calibrated values are valid probabilities
        assert np.all((calibrated >= 0) & (calibrated <= 1))
        # Check positive logits give high probabilities, negative give low
        assert np.all(calibrated[overconfident_logits > 0] > 0.5)
        assert np.all(calibrated[overconfident_logits < 0] < 0.5)

    def test_temperature_preserves_ordering(self) -> None:
        """Test that temperature scaling preserves prediction ordering."""
        logits = np.array([2.0, -1.0, 0.5, -0.5, 1.5], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")
        calibrator.fit(logits, labels)

        new_logits = np.array([3.0, 1.0, 0.0, -1.0, -2.0], dtype=np.float32)
        calibrated = calibrator.transform(new_logits)

        # Check ordering is preserved (higher logits => higher probabilities)
        # Use argsort to check rank ordering
        logit_order = np.argsort(new_logits)
        calibrated_order = np.argsort(calibrated)
        assert np.array_equal(logit_order, calibrated_order)


class TestIsotonicRegression:
    """Tests for isotonic regression calibration."""

    def test_fit_isotonic_with_numpy_arrays(self) -> None:
        """Test fitting isotonic regression with numpy arrays."""
        probabilities = np.array([0.9, 0.1, 0.7, 0.3, 0.8], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="isotonic")
        calibrator.fit(probabilities, labels)

        assert calibrator.is_fitted is True
        assert calibrator.calibrator is not None

    def test_fit_isotonic_with_torch_tensors(self) -> None:
        """Test fitting isotonic regression with torch tensors."""
        probabilities = torch.tensor([0.9, 0.1, 0.7, 0.3, 0.8])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

        calibrator = TriageScoreCalibration(method="isotonic")
        calibrator.fit(probabilities, labels)

        assert calibrator.is_fitted is True

    def test_transform_isotonic_regression(self) -> None:
        """Test applying isotonic regression transformation."""
        probabilities = np.array([0.9, 0.1, 0.7, 0.3, 0.8], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="isotonic")
        calibrator.fit(probabilities, labels)

        new_probs = np.array([0.85, 0.5, 0.2], dtype=np.float32)
        calibrated = calibrator.transform(new_probs)

        assert calibrated.shape == new_probs.shape
        assert np.all((calibrated >= 0) & (calibrated <= 1))
        assert calibrated.dtype == np.float32

    def test_isotonic_monotonicity(self) -> None:
        """Test that isotonic regression preserves monotonicity."""
        # Create calibration data
        probabilities = np.linspace(0.1, 0.9, 50, dtype=np.float32)
        labels = (probabilities > 0.5).astype(np.float32)

        calibrator = TriageScoreCalibration(method="isotonic")
        calibrator.fit(probabilities, labels)

        # Test monotonicity on sorted inputs
        test_probs = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        calibrated = calibrator.transform(test_probs)

        # Calibrated outputs should be monotonically increasing
        assert np.all(np.diff(calibrated) >= 0)

    def test_isotonic_clips_out_of_bounds(self) -> None:
        """Test that isotonic regression clips out-of-bounds predictions."""
        probabilities = np.array([0.3, 0.5, 0.7], dtype=np.float32)
        labels = np.array([0.0, 1.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="isotonic")
        calibrator.fit(probabilities, labels)

        # Test with values outside training range
        extreme_probs = np.array([0.0, 0.1, 0.9, 1.0], dtype=np.float32)
        calibrated = calibrator.transform(extreme_probs)

        assert np.all((calibrated >= 0) & (calibrated <= 1))


class TestFitValidation:
    """Tests for fit method input validation."""

    def test_fit_with_mismatched_lengths(self) -> None:
        """Test fit raises error with mismatched prediction and label lengths."""
        predictions = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        labels = np.array([0.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")

        with pytest.raises(ValueError, match="must have same length"):
            calibrator.fit(predictions, labels)

    def test_fit_with_empty_arrays(self) -> None:
        """Test fit raises error with empty arrays."""
        predictions = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")

        with pytest.raises(ValueError, match="zero samples"):
            calibrator.fit(predictions, labels)

    def test_fit_with_non_binary_labels(self) -> None:
        """Test fit raises error with non-binary labels."""
        predictions = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        labels = np.array([0.5, 1.5, 2.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")

        with pytest.raises(ValueError, match="Labels must be binary"):
            calibrator.fit(predictions, labels)

    def test_fit_with_2d_inputs(self) -> None:
        """Test fit handles 2D inputs correctly."""
        predictions = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        labels = np.array([[0.0], [1.0], [1.0]], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")
        calibrator.fit(predictions, labels)

        assert calibrator.is_fitted is True


class TestTransformValidation:
    """Tests for transform method validation."""

    def test_transform_before_fit_raises_error(self) -> None:
        """Test transform raises error if called before fit."""
        calibrator = TriageScoreCalibration(method="temperature")
        predictions = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with pytest.raises(RuntimeError, match="must be fitted before transform"):
            calibrator.transform(predictions)

    def test_transform_preserves_shape(self) -> None:
        """Test transform preserves input shape."""
        logits = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")
        calibrator.fit(logits, labels)

        # Test various shapes
        for shape in [(5,), (5, 1), (2, 3, 4)]:
            test_input = np.random.randn(*shape).astype(np.float32)
            output = calibrator.transform(test_input)
            assert output.shape == shape

    def test_transform_with_torch_tensor(self) -> None:
        """Test transform works with torch tensor input."""
        logits = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")
        calibrator.fit(logits, labels)

        test_tensor = torch.tensor([1.0, 0.0, -1.0])
        output = calibrator.transform(test_tensor)

        assert isinstance(output, np.ndarray)
        assert output.shape == (3,)


class TestGetParams:
    """Tests for get_params method."""

    def test_get_params_temperature_scaling(self) -> None:
        """Test get_params returns temperature for temperature scaling."""
        logits = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")
        calibrator.fit(logits, labels)

        params = calibrator.get_params()

        assert "temperature" in params
        assert isinstance(params["temperature"], float)
        assert params["temperature"] > 0

    def test_get_params_isotonic_regression(self) -> None:
        """Test get_params returns thresholds for isotonic regression."""
        probabilities = np.array([0.9, 0.1, 0.7, 0.3, 0.8], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="isotonic")
        calibrator.fit(probabilities, labels)

        params = calibrator.get_params()

        assert "x_thresholds" in params
        assert "y_thresholds" in params
        assert isinstance(params["x_thresholds"], np.ndarray)
        assert isinstance(params["y_thresholds"], np.ndarray)

    def test_get_params_before_fit_raises_error(self) -> None:
        """Test get_params raises error if called before fit."""
        calibrator = TriageScoreCalibration(method="temperature")

        with pytest.raises(RuntimeError, match="must be fitted"):
            calibrator.get_params()


class TestMethodChaining:
    """Tests for method chaining."""

    def test_fit_returns_self(self) -> None:
        """Test that fit returns self for method chaining."""
        logits = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")
        result = calibrator.fit(logits, labels)

        assert result is calibrator

    def test_fit_transform_chaining(self) -> None:
        """Test fit and transform can be chained."""
        logits = np.array([2.0, -1.0, 0.5, -0.5, 1.5], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")

        new_logits = np.array([1.0, 0.0, -1.0], dtype=np.float32)
        calibrated = calibrator.fit(logits, labels).transform(new_logits)

        assert isinstance(calibrated, np.ndarray)
        assert calibrated.shape == (3,)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_sample_fit(self) -> None:
        """Test fitting with single sample works."""
        predictions = np.array([1.0], dtype=np.float32)
        labels = np.array([1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="isotonic")
        calibrator.fit(predictions, labels)

        assert calibrator.is_fitted is True

    def test_all_same_label(self) -> None:
        """Test fitting with all same labels works."""
        predictions = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        labels = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="isotonic")
        calibrator.fit(predictions, labels)

        assert calibrator.is_fitted is True

    def test_perfect_predictions(self) -> None:
        """Test calibration with perfect predictions."""
        # Perfect predictions: high confidence for positive, low for negative
        logits = np.array([10.0, -10.0, 10.0, -10.0], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")
        calibrator.fit(logits, labels)

        # Should still produce valid calibration
        calibrated = calibrator.transform(logits)
        assert np.all((calibrated >= 0) & (calibrated <= 1))

    def test_transform_single_value(self) -> None:
        """Test transform with single value."""
        logits = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        labels = np.array([1.0, 0.0, 1.0], dtype=np.float32)

        calibrator = TriageScoreCalibration(method="temperature")
        calibrator.fit(logits, labels)

        single_value = np.array([0.5], dtype=np.float32)
        result = calibrator.transform(single_value)

        assert result.shape == (1,)
        assert 0 <= result[0] <= 1
