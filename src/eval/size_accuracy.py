"""Size estimation accuracy evaluation for 3D detection and segmentation tasks.

This module provides functions to evaluate the accuracy of predicted sizes (volumes and diameters)
compared to ground truth measurements.
"""

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_volume_from_mask(
    mask: NDArray[np.bool_],
    spacing: NDArray[np.float64] | None = None,
) -> float:
    """Calculate volume from a binary segmentation mask.

    Args:
        mask: Binary segmentation mask (any shape).
        spacing: Physical spacing for each dimension. If None, assumes unit spacing.

    Returns:
        Volume in physical units (product of voxel count and voxel volume).
    """
    if spacing is None:
        spacing = np.ones(mask.ndim)
    elif spacing.shape[0] != mask.ndim:
        raise ValueError(f"Spacing dimension {spacing.shape[0]} must match mask dimension {mask.ndim}")

    voxel_volume = np.prod(spacing)
    num_voxels = np.sum(mask)
    return float(num_voxels * voxel_volume)


def compute_diameter_from_mask(
    mask: NDArray[np.bool_],
    spacing: NDArray[np.float64] | None = None,
) -> float:
    """Calculate equivalent spherical diameter from a binary segmentation mask.

    Computes the diameter of a sphere with the same volume as the mask.

    Args:
        mask: Binary segmentation mask (any shape).
        spacing: Physical spacing for each dimension. If None, assumes unit spacing.

    Returns:
        Equivalent spherical diameter in physical units.
        Returns 0.0 if mask is empty.
    """
    volume = compute_volume_from_mask(mask, spacing)

    if volume <= 0:
        return 0.0

    # Volume of sphere = (4/3) * pi * r^3
    # Solving for diameter d = 2*r: d = (6*V/pi)^(1/3)
    diameter = (6 * volume / np.pi) ** (1 / 3)
    return float(diameter)


def compute_volume_error(
    predicted_volumes: NDArray[np.float64],
    ground_truth_volumes: NDArray[np.float64],
    relative: bool = False,
) -> NDArray[np.float64]:
    """Calculate volume estimation errors.

    Args:
        predicted_volumes: Array of predicted volumes.
        ground_truth_volumes: Array of ground truth volumes.
        relative: If True, compute relative error (error / ground_truth).
                 If False, compute absolute error (predicted - ground_truth).

    Returns:
        Array of volume errors (same shape as inputs).

    Raises:
        ValueError: If input shapes don't match.
    """
    if predicted_volumes.shape != ground_truth_volumes.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted_volumes.shape} vs ground_truth {ground_truth_volumes.shape}"
        )

    if relative:
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            errors = (predicted_volumes - ground_truth_volumes) / ground_truth_volumes
            # Set errors to inf where ground_truth is zero
            errors = np.where(ground_truth_volumes == 0, np.inf, errors)
        return errors
    else:
        return predicted_volumes - ground_truth_volumes


def compute_diameter_error(
    predicted_diameters: NDArray[np.float64],
    ground_truth_diameters: NDArray[np.float64],
    relative: bool = False,
) -> NDArray[np.float64]:
    """Calculate diameter estimation errors.

    Args:
        predicted_diameters: Array of predicted diameters.
        ground_truth_diameters: Array of ground truth diameters.
        relative: If True, compute relative error (error / ground_truth).
                 If False, compute absolute error (predicted - ground_truth).

    Returns:
        Array of diameter errors (same shape as inputs).

    Raises:
        ValueError: If input shapes don't match.
    """
    if predicted_diameters.shape != ground_truth_diameters.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted_diameters.shape} vs ground_truth {ground_truth_diameters.shape}"
        )

    if relative:
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            errors = (predicted_diameters - ground_truth_diameters) / ground_truth_diameters
            # Set errors to inf where ground_truth is zero
            errors = np.where(ground_truth_diameters == 0, np.inf, errors)
        return errors
    else:
        return predicted_diameters - ground_truth_diameters


def calculate_mae(errors: NDArray[np.float64]) -> float:
    """Calculate Mean Absolute Error.

    Args:
        errors: Array of errors (can be from volume or diameter).

    Returns:
        Mean absolute error. Returns nan if errors is empty or contains only inf/nan.
    """
    if len(errors) == 0:
        return np.nan

    # Filter out inf and nan values
    finite_errors = errors[np.isfinite(errors)]

    if len(finite_errors) == 0:
        return np.nan

    return float(np.mean(np.abs(finite_errors)))


def calculate_rmse(errors: NDArray[np.float64]) -> float:
    """Calculate Root Mean Square Error.

    Args:
        errors: Array of errors (can be from volume or diameter).

    Returns:
        Root mean square error. Returns nan if errors is empty or contains only inf/nan.
    """
    if len(errors) == 0:
        return np.nan

    # Filter out inf and nan values
    finite_errors = errors[np.isfinite(errors)]

    if len(finite_errors) == 0:
        return np.nan

    return float(np.sqrt(np.mean(finite_errors**2)))


def calculate_size_metrics(
    predicted_sizes: NDArray[np.float64],
    ground_truth_sizes: NDArray[np.float64],
) -> dict[str, float]:
    """Calculate comprehensive size estimation metrics.

    Args:
        predicted_sizes: Array of predicted sizes (volumes or diameters).
        ground_truth_sizes: Array of ground truth sizes.

    Returns:
        Dictionary containing:
        - mae: Mean Absolute Error
        - rmse: Root Mean Square Error
        - mean_error: Mean signed error (bias)
        - median_error: Median signed error
        - std_error: Standard deviation of errors
        - mean_abs_relative_error: Mean absolute relative error

    Raises:
        ValueError: If input shapes don't match.
    """
    if predicted_sizes.shape != ground_truth_sizes.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted_sizes.shape} vs ground_truth {ground_truth_sizes.shape}")

    # Compute absolute errors
    abs_errors = predicted_sizes - ground_truth_sizes

    # Filter finite values
    finite_mask = np.isfinite(abs_errors)
    if not np.any(finite_mask):
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mean_error": np.nan,
            "median_error": np.nan,
            "std_error": np.nan,
            "mean_abs_relative_error": np.nan,
        }

    finite_abs_errors = abs_errors[finite_mask]
    finite_ground_truth = ground_truth_sizes[finite_mask]

    mae = float(np.mean(np.abs(finite_abs_errors)))
    rmse = float(np.sqrt(np.mean(finite_abs_errors**2)))
    mean_error = float(np.mean(finite_abs_errors))
    median_error = float(np.median(finite_abs_errors))
    std_error = float(np.std(finite_abs_errors))

    # Calculate mean absolute relative error (even if relative=False, this is useful)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_errors = np.abs(finite_abs_errors) / finite_ground_truth
        rel_errors = rel_errors[np.isfinite(rel_errors)]
        mean_abs_rel_error = float(np.mean(rel_errors)) if len(rel_errors) > 0 else np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "mean_error": mean_error,
        "median_error": median_error,
        "std_error": std_error,
        "mean_abs_relative_error": mean_abs_rel_error,
    }
