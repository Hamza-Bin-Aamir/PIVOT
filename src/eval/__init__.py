"""Evaluation metrics for nodule detection and segmentation."""

from .froc import (
    Detection,
    FROCPoint,
    compute_average_sensitivity,
    compute_distance_3d,
    compute_froc_curve,
    compute_sensitivity_at_fppi,
    match_detections_to_ground_truth,
)

__all__ = [
    "Detection",
    "FROCPoint",
    "compute_froc_curve",
    "compute_sensitivity_at_fppi",
    "compute_average_sensitivity",
    "match_detections_to_ground_truth",
    "compute_distance_3d",
]
