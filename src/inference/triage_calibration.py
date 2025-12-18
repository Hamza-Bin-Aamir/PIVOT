"""Triage score calibration for converting raw model outputs to calibrated probabilities.

This module provides calibration methods to transform uncalibrated model outputs
into well-calibrated probability scores. Two primary methods are supported:

1. Temperature Scaling: A single scalar parameter that rescales logits before
   applying sigmoid/softmax. Simple and effective for neural networks.

2. Isotonic Regression: Non-parametric calibration that learns a monotonic
   mapping from uncalibrated to calibrated probabilities.

Both methods require a validation set with ground truth labels to fit the
calibration parameters.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression

__all__ = [
    "CalibrationMethod",
    "TriageScoreCalibration",
]

CalibrationMethod = Literal["temperature", "isotonic"]


class TriageScoreCalibration:
    """Calibrate raw triage score predictions to proper probabilities.

    This class provides two calibration methods:
    - Temperature scaling: Rescales logits by a learned temperature parameter
    - Isotonic regression: Learns a monotonic mapping using scikit-learn

    The calibration is fitted on a validation set and then applied to new
    predictions during inference.

    Args:
        method: Calibration method to use ("temperature" or "isotonic")
        device: PyTorch device for temperature scaling computations

    Attributes:
        method: The calibration method being used
        device: PyTorch device for computations
        is_fitted: Whether calibration has been fitted to data
        temperature: Learned temperature parameter (for temperature scaling)
        calibrator: Fitted isotonic regression model (for isotonic method)

    Examples:
        >>> # Temperature scaling
        >>> calibrator = TriageScoreCalibration(method="temperature")
        >>> calibrator.fit(logits, labels)
        >>> calibrated = calibrator.transform(new_logits)
        >>>
        >>> # Isotonic regression
        >>> calibrator = TriageScoreCalibration(method="isotonic")
        >>> calibrator.fit(probabilities, labels)
        >>> calibrated = calibrator.transform(new_probabilities)
    """

    def __init__(
        self,
        method: CalibrationMethod = "temperature",
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize calibration with specified method.

        Args:
            method: Calibration method ("temperature" or "isotonic")
            device: Device for torch computations (None uses default)

        Raises:
            ValueError: If method is not recognized
        """
        if method not in {"temperature", "isotonic"}:
            msg = f"Method must be 'temperature' or 'isotonic', got '{method}'"
            raise ValueError(msg)

        self.method = method
        self.device = device if device is not None else torch.device("cpu")
        self.is_fitted = False

        # Method-specific attributes
        self.temperature: torch.nn.Parameter | None = None
        self.calibrator: IsotonicRegression | None = None

    def fit(
        self,
        predictions: NDArray[np.float32] | torch.Tensor,
        labels: NDArray[np.float32] | torch.Tensor,
        *,
        learning_rate: float = 0.01,
        max_iter: int = 50,
    ) -> TriageScoreCalibration:
        """Fit calibration parameters on validation data.

        Args:
            predictions: Raw model outputs (logits for temperature, probabilities for isotonic)
                        Shape: (N,) or (N, 1)
            labels: Ground truth binary labels (0 or 1)
                   Shape: (N,) or (N, 1)
            learning_rate: Learning rate for temperature scaling optimization
            max_iter: Maximum iterations for temperature scaling

        Returns:
            Self for method chaining

        Raises:
            ValueError: If predictions and labels have different lengths
            ValueError: If labels are not binary (0 or 1)
            ValueError: If no valid samples provided
        """
        # Convert to numpy arrays and flatten
        pred_array = self._to_numpy(predictions).flatten()
        label_array = self._to_numpy(labels).flatten()

        # Validate inputs
        if len(pred_array) != len(label_array):
            msg = f"Predictions ({len(pred_array)}) and labels ({len(label_array)}) must have same length"
            raise ValueError(msg)

        if len(pred_array) == 0:
            msg = "Cannot fit calibration with zero samples"
            raise ValueError(msg)

        # Check labels are binary
        unique_labels = np.unique(label_array)
        if not np.all(np.isin(unique_labels, [0, 1])):
            msg = f"Labels must be binary (0 or 1), got unique values: {unique_labels}"
            raise ValueError(msg)

        # Fit based on method
        if self.method == "temperature":
            self._fit_temperature(pred_array, label_array, learning_rate, max_iter)
        else:  # isotonic
            self._fit_isotonic(pred_array, label_array)

        self.is_fitted = True
        return self

    def transform(
        self,
        predictions: NDArray[np.float32] | torch.Tensor,
    ) -> NDArray[np.float32]:
        """Apply calibration to new predictions.

        Args:
            predictions: Raw model outputs to calibrate
                        Shape: (N,) or (N, 1) or (B, 1, D, H, W)

        Returns:
            Calibrated probabilities with same shape as input

        Raises:
            RuntimeError: If calibration has not been fitted
        """
        if not self.is_fitted:
            msg = "Calibration must be fitted before transform. Call fit() first."
            raise RuntimeError(msg)

        original_shape = predictions.shape
        pred_array = self._to_numpy(predictions).flatten()

        # Apply calibration
        if self.method == "temperature":
            calibrated = self._apply_temperature(pred_array)
        else:  # isotonic
            calibrated = self._apply_isotonic(pred_array)

        # Restore original shape
        return calibrated.reshape(original_shape)

    def _fit_temperature(
        self,
        logits: NDArray[np.float32],
        labels: NDArray[np.float32],
        lr: float,
        max_iter: int,
    ) -> None:
        """Fit temperature scaling parameter using gradient descent.

        Temperature scaling rescales logits as: sigmoid(logits / T)
        We optimize T to minimize negative log likelihood on validation set.

        Args:
            logits: Raw model logits (uncalibrated)
            labels: Binary ground truth labels
            lr: Learning rate
            max_iter: Maximum optimization iterations
        """
        # Convert to torch tensors
        logits_t = torch.from_numpy(logits).to(self.device)
        labels_t = torch.from_numpy(labels).to(self.device)

        # Initialize temperature parameter
        self.temperature = torch.nn.Parameter(
            torch.ones(1, device=self.device, dtype=torch.float32)
        )

        # Optimize temperature using NLL loss
        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=lr,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            # Scale logits by temperature and compute probabilities
            # Use absolute value to ensure positive scaling
            assert self.temperature is not None  # Type narrowing for mypy
            scaled_logits = logits_t / self.temperature.abs().clamp(min=1e-4)
            probs = torch.sigmoid(scaled_logits)
            # Binary cross-entropy loss
            loss = torch.nn.functional.binary_cross_entropy(
                probs, labels_t, reduction="mean"
            )
            loss.backward()
            return loss

        optimizer.step(closure)

    def _fit_isotonic(
        self,
        probabilities: NDArray[np.float32],
        labels: NDArray[np.float32],
    ) -> None:
        """Fit isotonic regression calibrator.

        Isotonic regression learns a monotonic mapping from uncalibrated
        probabilities to calibrated probabilities.

        Args:
            probabilities: Uncalibrated probability predictions
            labels: Binary ground truth labels
        """
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(probabilities, labels)

    def _apply_temperature(self, logits: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply temperature scaling to logits.

        Args:
            logits: Raw model logits

        Returns:
            Calibrated probabilities
        """
        if self.temperature is None:
            msg = "Temperature not fitted"
            raise RuntimeError(msg)

        # Convert to torch, scale, and convert back
        logits_t = torch.from_numpy(logits).to(self.device)
        # Use absolute value of temperature to ensure positive scaling
        # Assert for type checker - we already checked for None above
        assert self.temperature is not None
        temp_abs = self.temperature.abs().clamp(min=1e-4)
        scaled = logits_t / temp_abs
        probs = torch.sigmoid(scaled)
        result: NDArray[np.float32] = probs.detach().cpu().numpy().astype(np.float32)
        return result

    def _apply_isotonic(self, probabilities: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply isotonic regression calibration.

        Args:
            probabilities: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            msg = "Isotonic calibrator not fitted"
            raise RuntimeError(msg)

        calibrated = self.calibrator.transform(probabilities)
        result: NDArray[np.float32] = calibrated.astype(np.float32)
        return result

    def _to_numpy(self, array: NDArray[np.float32] | torch.Tensor) -> NDArray[np.float32]:
        """Convert tensor or array to numpy array.

        Args:
            array: Input tensor or array

        Returns:
            NumPy array
        """
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy().astype(np.float32)
        return array.astype(np.float32)

    def get_params(self) -> dict[str, float | NDArray[np.float32]]:
        """Get calibration parameters for inspection or serialization.

        Returns:
            Dictionary containing calibration parameters:
            - For temperature: {"temperature": float}
            - For isotonic: {"x_thresholds": array, "y_thresholds": array}

        Raises:
            RuntimeError: If calibration has not been fitted
        """
        if not self.is_fitted:
            msg = "Calibration must be fitted before getting parameters"
            raise RuntimeError(msg)

        if self.method == "temperature":
            if self.temperature is None:
                msg = "Temperature parameter not initialized"
                raise RuntimeError(msg)
            # Return absolute value of temperature
            return {"temperature": float(self.temperature.abs().item())}
        else:  # isotonic
            if self.calibrator is None:
                msg = "Isotonic calibrator not initialized"
                raise RuntimeError(msg)
            return {
                "x_thresholds": self.calibrator.X_thresholds_.astype(np.float32),
                "y_thresholds": self.calibrator.y_thresholds_.astype(np.float32),
            }
