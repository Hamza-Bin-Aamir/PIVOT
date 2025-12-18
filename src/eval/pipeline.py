"""Evaluation pipeline for comprehensive model performance assessment.

This module provides a unified pipeline to evaluate detection and segmentation models
using all available metrics.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .center_accuracy import (
    calculate_accuracy_metrics as calculate_center_accuracy_metrics,
)
from .center_accuracy import match_predictions_to_ground_truth as match_centers
from .dice import compute_batch_dice_scores, compute_batch_iou_scores
from .froc import (
    Detection,
    compute_average_sensitivity,
    compute_froc_curve,
)
from .size_accuracy import (
    calculate_size_metrics,
    compute_diameter_from_mask,
    compute_volume_from_mask,
)
from .triage_correlation import (
    calculate_expected_calibration_error,
    compute_auc_roc,
    compute_pearson_correlation,
    compute_spearman_correlation,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectionPrediction:
    """Container for detection predictions from a single image.

    Attributes:
        centers: Predicted center coordinates, shape (N, 3).
        confidences: Confidence scores for each detection, shape (N,).
        sizes: Optional predicted sizes (volumes or diameters), shape (N,).
        segmentation_mask: Optional predicted segmentation mask, shape (D, H, W).
        triage_score: Optional triage/urgency score for the image.
    """

    centers: NDArray[np.float64]
    confidences: NDArray[np.float64]
    sizes: NDArray[np.float64] | None = None
    segmentation_mask: NDArray[np.bool_] | None = None
    triage_score: float | None = None


@dataclass
class GroundTruth:
    """Container for ground truth annotations from a single image.

    Attributes:
        centers: Ground truth center coordinates, shape (N, 3).
        sizes: Optional ground truth sizes (volumes or diameters), shape (N,).
        segmentation_mask: Optional ground truth segmentation mask, shape (D, H, W).
        triage_score: Optional ground truth triage/urgency score.
        is_urgent: Optional binary urgency label.
    """

    centers: NDArray[np.float64]
    sizes: NDArray[np.float64] | None = None
    segmentation_mask: NDArray[np.bool_] | None = None
    triage_score: float | None = None
    is_urgent: bool | None = None


@dataclass
class EvaluationResults:
    """Container for all evaluation metrics.

    Attributes:
        froc_metrics: FROC curve metrics (sensitivity, FPPI, average sensitivity).
        detection_metrics: Detection accuracy metrics (precision, recall, center distance).
        segmentation_metrics: Segmentation metrics (Dice, IoU).
        size_metrics: Size estimation metrics (MAE, RMSE).
        triage_metrics: Triage score correlation metrics.
        num_images: Number of images evaluated.
        num_predictions: Total number of predictions.
        num_ground_truths: Total number of ground truth objects.
    """

    froc_metrics: dict[str, Any]
    detection_metrics: dict[str, Any]
    segmentation_metrics: dict[str, Any]
    size_metrics: dict[str, Any]
    triage_metrics: dict[str, Any]
    num_images: int
    num_predictions: int
    num_ground_truths: int


class EvaluationPipeline:
    """Pipeline for comprehensive model evaluation.

    This class orchestrates the evaluation of detection and segmentation models
    across multiple metrics.
    """

    def __init__(
        self,
        spacing: NDArray[np.float64] | None = None,
        detection_distance_threshold: float = 10.0,
        froc_thresholds: list[float] | None = None,
        froc_fppi_values: list[float] | None = None,
    ) -> None:
        """Initialize evaluation pipeline.

        Args:
            spacing: Physical spacing for each dimension (x, y, z). If None, assumes isotropic.
            detection_distance_threshold: Maximum distance for matching detections to ground truth.
            froc_thresholds: Confidence thresholds for FROC curve. If None, uses auto thresholds.
            froc_fppi_values: FPPI values for FROC curve evaluation. If None, uses LUNA16 standard.
        """
        self.spacing = spacing if spacing is not None else np.ones(3)
        self.detection_distance_threshold = detection_distance_threshold
        self.froc_thresholds = froc_thresholds
        self.froc_fppi_values = froc_fppi_values

        # Accumulators for metrics
        self.predictions: list[DetectionPrediction] = []
        self.ground_truths: list[GroundTruth] = []

    def add_case(
        self,
        prediction: DetectionPrediction,
        ground_truth: GroundTruth,
    ) -> None:
        """Add a single case (image) to the evaluation pipeline.

        Args:
            prediction: Predictions for this case.
            ground_truth: Ground truth annotations for this case.
        """
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)

    def evaluate(self) -> EvaluationResults:
        """Run comprehensive evaluation on all accumulated cases.

        Returns:
            EvaluationResults containing all computed metrics.

        Raises:
            ValueError: If no cases have been added.
        """
        if not self.predictions or not self.ground_truths:
            raise ValueError("No cases added to pipeline. Use add_case() first.")

        logger.info(f"Evaluating {len(self.predictions)} cases")

        # Evaluate FROC and detection metrics
        froc_metrics, detection_metrics = self._evaluate_detection()

        # Evaluate segmentation metrics
        segmentation_metrics = self._evaluate_segmentation()

        # Evaluate size estimation metrics
        size_metrics = self._evaluate_size()

        # Evaluate triage correlation metrics
        triage_metrics = self._evaluate_triage()

        # Count statistics
        num_predictions = sum(len(pred.centers) for pred in self.predictions)
        num_ground_truths = sum(len(gt.centers) for gt in self.ground_truths)

        results = EvaluationResults(
            froc_metrics=froc_metrics,
            detection_metrics=detection_metrics,
            segmentation_metrics=segmentation_metrics,
            size_metrics=size_metrics,
            triage_metrics=triage_metrics,
            num_images=len(self.predictions),
            num_predictions=num_predictions,
            num_ground_truths=num_ground_truths,
        )

        logger.info("Evaluation complete")
        return results

    def _evaluate_detection(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Evaluate detection performance using FROC curve and center accuracy.

        Returns:
            Tuple of (froc_metrics, detection_metrics).
        """
        # Flatten all detections and ground truths for FROC curve
        all_detections: list[Detection] = []
        all_ground_truths: list[tuple[float, float, float]] = []

        for pred, gt in zip(self.predictions, self.ground_truths, strict=True):
            # Convert to Detection objects
            for i in range(len(pred.centers)):
                detection = Detection(center=tuple(pred.centers[i]), confidence=pred.confidences[i])
                all_detections.append(detection)

            # Convert ground truths to tuples
            for center in gt.centers:
                all_ground_truths.append(tuple(center))

        # Compute FROC curve
        froc_curve = compute_froc_curve(
            detections=all_detections,
            ground_truth=all_ground_truths,
            num_images=len(self.predictions),
            max_distance=self.detection_distance_threshold,
            spacing=tuple(self.spacing),
            thresholds=self.froc_thresholds,
        )

        # Compute average sensitivity
        if self.froc_fppi_values and len(self.froc_fppi_values) >= 2:
            fppi_range = (min(self.froc_fppi_values), max(self.froc_fppi_values))
            avg_sensitivity = compute_average_sensitivity(froc_curve, fppi_range=fppi_range)
        else:
            avg_sensitivity = compute_average_sensitivity(froc_curve)

        froc_metrics = {
            "average_sensitivity": avg_sensitivity,
            "num_froc_points": len(froc_curve),
            "froc_curve": froc_curve,  # Include full curve for plotting
        }

        # Compute center accuracy metrics by matching all predictions
        all_matches = []
        for pred, gt in zip(self.predictions, self.ground_truths, strict=True):
            matches = match_centers(
                predictions=list(pred.centers),
                ground_truths=list(gt.centers),
                max_distance=self.detection_distance_threshold,
                spacing=self.spacing,
            )
            all_matches.extend(matches)

        detection_metrics = calculate_center_accuracy_metrics(all_matches)

        return froc_metrics, detection_metrics

    def _evaluate_segmentation(self) -> dict[str, Any]:
        """Evaluate segmentation performance using Dice and IoU.

        Returns:
            Dictionary of segmentation metrics.
        """
        pred_masks = []
        gt_masks = []

        for pred, gt in zip(self.predictions, self.ground_truths, strict=True):
            if pred.segmentation_mask is not None and gt.segmentation_mask is not None:
                pred_masks.append(pred.segmentation_mask)
                gt_masks.append(gt.segmentation_mask)

        if not pred_masks:
            logger.warning("No segmentation masks available for evaluation")
            return {
                "mean_dice": np.nan,
                "std_dice": np.nan,
                "mean_iou": np.nan,
                "std_iou": np.nan,
                "num_masks": 0,
            }

        # Stack masks into batch
        pred_batch = np.stack(pred_masks, axis=0)
        gt_batch = np.stack(gt_masks, axis=0)

        # Compute batch metrics
        dice_scores = compute_batch_dice_scores(pred_batch, gt_batch)
        iou_scores = compute_batch_iou_scores(pred_batch, gt_batch)

        return {
            "mean_dice": float(np.mean(dice_scores)),
            "std_dice": float(np.std(dice_scores)),
            "mean_iou": float(np.mean(iou_scores)),
            "std_iou": float(np.std(iou_scores)),
            "num_masks": len(pred_masks),
            "dice_scores": dice_scores,  # Individual scores for analysis
            "iou_scores": iou_scores,
        }

    def _evaluate_size(self) -> dict[str, Any]:
        """Evaluate size estimation accuracy.

        Returns:
            Dictionary of size estimation metrics.
        """
        # Collect sizes from detections
        pred_sizes: list[float] = []
        gt_sizes: list[float] = []

        for pred, gt in zip(self.predictions, self.ground_truths, strict=True):
            if pred.sizes is not None and gt.sizes is not None:
                # Only include matched detections for fair comparison
                # This is a simplified approach; could be improved with matching
                min_len = min(len(pred.sizes), len(gt.sizes))
                pred_sizes.extend(pred.sizes[:min_len])
                gt_sizes.extend(gt.sizes[:min_len])

        # Also compute sizes from segmentation masks if available
        mask_pred_sizes = []
        mask_gt_sizes = []

        for pred, gt in zip(self.predictions, self.ground_truths, strict=True):
            if pred.segmentation_mask is not None and gt.segmentation_mask is not None:
                pred_vol = compute_volume_from_mask(pred.segmentation_mask, self.spacing)
                gt_vol = compute_volume_from_mask(gt.segmentation_mask, self.spacing)
                mask_pred_sizes.append(pred_vol)
                mask_gt_sizes.append(gt_vol)

                pred_diam = compute_diameter_from_mask(pred.segmentation_mask, self.spacing)
                gt_diam = compute_diameter_from_mask(gt.segmentation_mask, self.spacing)
                mask_pred_sizes.append(pred_diam)
                mask_gt_sizes.append(gt_diam)

        if not pred_sizes and not mask_pred_sizes:
            logger.warning("No size information available for evaluation")
            return {
                "detection_size_metrics": {},
                "mask_size_metrics": {},
                "num_detections": 0,
                "num_masks": 0,
            }

        results: dict[str, Any] = {}

        # Evaluate detection sizes
        if pred_sizes:
            detection_metrics = calculate_size_metrics(np.array(pred_sizes), np.array(gt_sizes))
            results["detection_size_metrics"] = detection_metrics
            results["num_detections"] = len(pred_sizes)
        else:
            results["detection_size_metrics"] = {}
            results["num_detections"] = 0

        # Evaluate mask-based sizes
        if mask_pred_sizes:
            mask_metrics = calculate_size_metrics(np.array(mask_pred_sizes), np.array(mask_gt_sizes))
            results["mask_size_metrics"] = mask_metrics
            results["num_masks"] = len(mask_pred_sizes) // 2  # Volume and diameter per mask
        else:
            results["mask_size_metrics"] = {}
            results["num_masks"] = 0

        return results

    def _evaluate_triage(self) -> dict[str, Any]:
        """Evaluate triage score correlation and calibration.

        Returns:
            Dictionary of triage metrics.
        """
        pred_scores = []
        gt_scores = []
        urgency_labels = []

        for pred, gt in zip(self.predictions, self.ground_truths, strict=True):
            if pred.triage_score is not None and gt.triage_score is not None:
                pred_scores.append(pred.triage_score)
                gt_scores.append(gt.triage_score)

            if pred.triage_score is not None and gt.is_urgent is not None:
                urgency_labels.append((pred.triage_score, gt.is_urgent))

        if not pred_scores and not urgency_labels:
            logger.warning("No triage information available for evaluation")
            return {
                "correlation_metrics": {},
                "classification_metrics": {},
                "num_scores": 0,
                "num_binary_labels": 0,
            }

        results: dict[str, Any] = {}

        # Evaluate correlation
        if pred_scores:
            pred_arr = np.array(pred_scores)
            gt_arr = np.array(gt_scores)

            pearson_corr, pearson_p = compute_pearson_correlation(pred_arr, gt_arr)
            spearman_corr, spearman_p = compute_spearman_correlation(pred_arr, gt_arr)

            results["correlation_metrics"] = {
                "pearson_correlation": pearson_corr,
                "pearson_p_value": pearson_p,
                "spearman_correlation": spearman_corr,
                "spearman_p_value": spearman_p,
            }
            results["num_scores"] = len(pred_scores)
        else:
            results["correlation_metrics"] = {}
            results["num_scores"] = 0

        # Evaluate binary classification (urgent vs non-urgent)
        if urgency_labels:
            pred_probs = np.array([score for score, _ in urgency_labels])
            gt_labels = np.array([label for _, label in urgency_labels])

            auc = compute_auc_roc(pred_probs, gt_labels)
            ece = calculate_expected_calibration_error(pred_probs, gt_labels)

            results["classification_metrics"] = {
                "auc_roc": auc,
                "expected_calibration_error": ece,
            }
            results["num_binary_labels"] = len(urgency_labels)
        else:
            results["classification_metrics"] = {}
            results["num_binary_labels"] = 0

        return results

    def reset(self) -> None:
        """Clear all accumulated predictions and ground truths."""
        self.predictions.clear()
        self.ground_truths.clear()
        logger.info("Pipeline reset")
