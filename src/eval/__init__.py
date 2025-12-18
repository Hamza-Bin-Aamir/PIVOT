"""Evaluation metrics for nodule detection and segmentation."""

from .center_accuracy import (
    CenterMatch,
    calculate_accuracy_metrics,
    compute_center_distance,
    match_predictions_to_ground_truth,
)
from .dice import (
    compute_batch_dice_scores,
    compute_batch_iou_scores,
    compute_dice_score,
    compute_iou,
    dice_to_iou,
    iou_to_dice,
)
from .froc import (
    Detection,
    FROCPoint,
    compute_average_sensitivity,
    compute_distance_3d,
    compute_froc_curve,
    compute_sensitivity_at_fppi,
    match_detections_to_ground_truth,
)
from .size_accuracy import (
    calculate_mae,
    calculate_rmse,
    calculate_size_metrics,
    compute_diameter_error,
    compute_diameter_from_mask,
    compute_volume_error,
    compute_volume_from_mask,
)
from .triage_correlation import (
    calculate_expected_calibration_error,
    compute_auc_roc,
    compute_calibration_curve,
    compute_pearson_correlation,
    compute_spearman_correlation,
)

__all__ = [
    # FROC
    "Detection",
    "FROCPoint",
    "compute_froc_curve",
    "compute_sensitivity_at_fppi",
    "compute_average_sensitivity",
    "match_detections_to_ground_truth",
    "compute_distance_3d",
    # Dice/IoU
    "compute_dice_score",
    "compute_iou",
    "compute_batch_dice_scores",
    "compute_batch_iou_scores",
    "dice_to_iou",
    "iou_to_dice",
    # Center Accuracy
    "CenterMatch",
    "compute_center_distance",
    "calculate_accuracy_metrics",
    # Size Accuracy
    "compute_volume_from_mask",
    "compute_diameter_from_mask",
    "compute_volume_error",
    "compute_diameter_error",
    "calculate_mae",
    "calculate_rmse",
    "calculate_size_metrics",
    # Triage Correlation
    "compute_pearson_correlation",
    "compute_spearman_correlation",
    "compute_auc_roc",
    "compute_calibration_curve",
    "calculate_expected_calibration_error",
]
