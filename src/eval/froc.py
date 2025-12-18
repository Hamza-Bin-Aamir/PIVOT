"""FROC (Free-Response Receiver Operating Characteristic) curve calculation.

This module provides functions for computing FROC curves, which are used to evaluate
detection performance by plotting sensitivity vs. false positives per image.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FROCPoint:
    """A point on the FROC curve."""

    threshold: float  # Confidence threshold
    sensitivity: float  # True positive rate (recall)
    fppi: float  # False positives per image
    num_tp: int  # Number of true positives
    num_fp: int  # Number of false positives
    num_fn: int  # Number of false negatives


@dataclass
class Detection:
    """A detection with position and confidence."""

    center: tuple[float, float, float]  # (z, y, x) coordinates
    confidence: float  # Detection confidence score


def compute_distance_3d(
    point1: tuple[float, float, float],
    point2: tuple[float, float, float],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute Euclidean distance between two 3D points.

    Args:
        point1: First point (z, y, x).
        point2: Second point (z, y, x).
        spacing: Voxel spacing (z, y, x) in mm. Defaults to (1.0, 1.0, 1.0).

    Returns:
        Euclidean distance in mm.

    Examples:
        >>> dist = compute_distance_3d((0, 0, 0), (3, 4, 0))
        >>> assert dist == 5.0
    """
    dz = (point1[0] - point2[0]) * spacing[0]
    dy = (point1[1] - point2[1]) * spacing[1]
    dx = (point1[2] - point2[2]) * spacing[2]
    return float(np.sqrt(dz**2 + dy**2 + dx**2))


def match_detections_to_ground_truth(
    detections: list[Detection],
    ground_truth: list[tuple[float, float, float]],
    max_distance: float = 10.0,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[list[bool], list[int]]:
    """Match detections to ground truth nodules using distance threshold.

    Each detection is matched to the nearest ground truth nodule within max_distance.
    Each ground truth can only be matched once (to highest confidence detection).

    Args:
        detections: List of detections sorted by confidence (descending).
        ground_truth: List of ground truth nodule centers (z, y, x).
        max_distance: Maximum distance in mm for a match. Defaults to 10.0.
        spacing: Voxel spacing (z, y, x) in mm. Defaults to (1.0, 1.0, 1.0).

    Returns:
        Tuple of (is_true_positive, matched_gt_indices):
            - is_true_positive: Boolean list indicating if each detection is TP
            - matched_gt_indices: Indices of matched ground truth (-1 if FP)

    Examples:
        >>> detections = [Detection((5, 5, 5), 0.9), Detection((20, 20, 20), 0.7)]
        >>> ground_truth = [(5, 5, 5), (10, 10, 10)]
        >>> is_tp, matched = match_detections_to_ground_truth(detections, ground_truth)
        >>> assert is_tp[0] == True  # First detection matches GT
        >>> assert is_tp[1] == False  # Second detection is FP
    """
    is_true_positive: list[bool] = []
    matched_gt_indices: list[int] = []
    matched_gt_set = set()

    for detection in detections:
        best_distance = float("inf")
        best_gt_idx = -1

        # Find nearest unmatched ground truth
        for gt_idx, gt_center in enumerate(ground_truth):
            if gt_idx in matched_gt_set:
                continue

            distance = compute_distance_3d(detection.center, gt_center, spacing)
            if distance < best_distance and distance <= max_distance:
                best_distance = distance
                best_gt_idx = gt_idx

        # Mark as TP if matched, FP otherwise
        if best_gt_idx >= 0:
            is_true_positive.append(True)
            matched_gt_indices.append(best_gt_idx)
            matched_gt_set.add(best_gt_idx)
        else:
            is_true_positive.append(False)
            matched_gt_indices.append(-1)

    return is_true_positive, matched_gt_indices


def compute_froc_curve(
    detections: list[Detection],
    ground_truth: list[tuple[float, float, float]],
    num_images: int = 1,
    max_distance: float = 10.0,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    thresholds: list[float] | None = None,
) -> list[FROCPoint]:
    """Compute FROC curve for detection evaluation.

    FROC plots sensitivity (recall) vs. false positives per image (FPPI) at
    different confidence thresholds.

    Args:
        detections: List of detections (automatically sorted by confidence).
        ground_truth: List of ground truth nodule centers (z, y, x).
        num_images: Number of images in the dataset. Defaults to 1.
        max_distance: Maximum distance in mm for a match. Defaults to 10.0.
        spacing: Voxel spacing (z, y, x) in mm. Defaults to (1.0, 1.0, 1.0).
        thresholds: Optional list of thresholds to evaluate. If None, uses
            confidence values from detections.

    Returns:
        List of FROCPoint objects, sorted by descending threshold.

    Examples:
        >>> detections = [
        ...     Detection((5, 5, 5), 0.9),
        ...     Detection((10, 10, 10), 0.7),
        ... ]
        >>> ground_truth = [(5, 5, 5), (10, 10, 10)]
        >>> froc = compute_froc_curve(detections, ground_truth)
        >>> assert froc[-1].sensitivity == 1.0  # All detected at low threshold
    """
    if not ground_truth:
        logger.warning("No ground truth nodules provided")
        return []

    # Sort detections by confidence (descending)
    sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

    # Match all detections to ground truth
    is_tp, _ = match_detections_to_ground_truth(
        sorted_detections, ground_truth, max_distance, spacing
    )

    # Determine thresholds to evaluate
    if thresholds is None:
        if sorted_detections:
            # Use unique confidence values from detections
            thresholds = sorted({d.confidence for d in sorted_detections}, reverse=True)
            # Add threshold slightly below minimum to capture all detections
            min_conf = min(d.confidence for d in sorted_detections)
            thresholds.append(min_conf - 0.001)
        else:
            # No detections, use single threshold
            thresholds = [0.0]

    # Compute FROC points
    froc_points: list[FROCPoint] = []
    num_gt = len(ground_truth)

    for threshold in thresholds:
        num_tp = 0
        num_fp = 0

        # Count TP and FP above this threshold
        for i, detection in enumerate(sorted_detections):
            if detection.confidence >= threshold:
                if is_tp[i]:
                    num_tp += 1
                else:
                    num_fp += 1

        num_fn = num_gt - num_tp
        sensitivity = num_tp / num_gt if num_gt > 0 else 0.0
        fppi = num_fp / num_images if num_images > 0 else 0.0

        froc_points.append(
            FROCPoint(
                threshold=threshold,
                sensitivity=sensitivity,
                fppi=fppi,
                num_tp=num_tp,
                num_fp=num_fp,
                num_fn=num_fn,
            )
        )

    logger.debug(f"Computed FROC curve with {len(froc_points)} points")
    return froc_points


def compute_sensitivity_at_fppi(
    froc_points: list[FROCPoint], target_fppi: float
) -> float:
    """Compute sensitivity at a specific FPPI value.

    Uses linear interpolation if exact FPPI is not in the curve.

    Args:
        froc_points: FROC curve points.
        target_fppi: Target false positives per image.

    Returns:
        Sensitivity at the target FPPI.

    Examples:
        >>> froc = [
        ...     FROCPoint(0.9, 0.5, 0.0, 5, 0, 5),
        ...     FROCPoint(0.5, 1.0, 2.0, 10, 20, 0),
        ... ]
        >>> sens = compute_sensitivity_at_fppi(froc, 1.0)
        >>> assert 0.5 <= sens <= 1.0  # Interpolated between points
    """
    if not froc_points:
        logger.warning("No FROC points provided")
        return 0.0

    # Sort by FPPI (ascending)
    sorted_points = sorted(froc_points, key=lambda p: p.fppi)

    # Check if target is before first point
    if target_fppi <= sorted_points[0].fppi:
        return sorted_points[0].sensitivity

    # Check if target is after last point
    if target_fppi >= sorted_points[-1].fppi:
        return sorted_points[-1].sensitivity

    # Find surrounding points and interpolate
    for i in range(len(sorted_points) - 1):
        if sorted_points[i].fppi <= target_fppi <= sorted_points[i + 1].fppi:
            # Linear interpolation
            x0, y0 = sorted_points[i].fppi, sorted_points[i].sensitivity
            x1, y1 = sorted_points[i + 1].fppi, sorted_points[i + 1].sensitivity

            if x1 - x0 == 0:
                return y0

            return y0 + (y1 - y0) * (target_fppi - x0) / (x1 - x0)

    # Should not reach here
    logger.warning(f"Could not interpolate sensitivity at FPPI={target_fppi}")
    return 0.0


def compute_average_sensitivity(
    froc_points: list[FROCPoint], fppi_range: tuple[float, float] = (0.125, 8.0)
) -> float:
    """Compute average sensitivity over a range of FPPI values.

    This is commonly used as a summary metric for FROC curves.

    Args:
        froc_points: FROC curve points.
        fppi_range: Tuple of (min_fppi, max_fppi) for averaging. Defaults to
            (0.125, 8.0) which is standard for LUNA16 challenge.

    Returns:
        Average sensitivity over the FPPI range.

    Examples:
        >>> froc = [
        ...     FROCPoint(0.9, 0.5, 0.0, 5, 0, 5),
        ...     FROCPoint(0.5, 1.0, 10.0, 10, 100, 0),
        ... ]
        >>> avg_sens = compute_average_sensitivity(froc, (0.0, 5.0))
        >>> assert 0.5 <= avg_sens <= 1.0
    """
    # Standard FPPI values for LUNA16 challenge
    fppi_values = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    # Filter to requested range
    fppi_values = [f for f in fppi_values if fppi_range[0] <= f <= fppi_range[1]]

    if not fppi_values:
        logger.warning(f"No standard FPPI values in range {fppi_range}")
        return 0.0

    sensitivities = [
        compute_sensitivity_at_fppi(froc_points, fppi) for fppi in fppi_values
    ]

    avg = np.mean(sensitivities)
    logger.info(
        f"Average sensitivity: {avg:.4f} at FPPI range {fppi_range} "
        f"({len(fppi_values)} points)"
    )

    return float(avg)
