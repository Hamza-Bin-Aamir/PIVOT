"""Center point accuracy evaluation for 3D detection tasks.

This module provides functions to evaluate the accuracy of predicted center points
compared to ground truth centers in 3D medical imaging.
"""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class CenterMatch:
    """Represents a match between prediction and ground truth center.

    Attributes:
        pred_idx: Index of the matched prediction (-1 if unmatched GT).
        gt_idx: Index of the matched ground truth (-1 if unmatched prediction).
        distance: Distance between centers (np.inf if unmatched).
    """

    pred_idx: int
    gt_idx: int
    distance: float


def compute_center_distance(
    center1: NDArray[np.float64],
    center2: NDArray[np.float64],
    spacing: NDArray[np.float64] | None = None,
) -> float:
    """Calculate Euclidean distance between two 3D centers.

    Args:
        center1: First center coordinates (x, y, z).
        center2: Second center coordinates (x, y, z).
        spacing: Physical spacing for each dimension (x, y, z). If None, assumes isotropic spacing.

    Returns:
        Euclidean distance between the two centers in physical units.

    Raises:
        ValueError: If centers have different shapes or are not 3D.
    """
    if center1.shape != (3,) or center2.shape != (3,):
        raise ValueError(f"Centers must be 3D points, got shapes {center1.shape} and {center2.shape}")

    if spacing is None:
        spacing = np.ones(3)
    elif spacing.shape != (3,):
        raise ValueError(f"Spacing must be 3D, got shape {spacing.shape}")

    diff = (center1 - center2) * spacing
    return float(np.sqrt(np.sum(diff**2)))


def match_predictions_to_ground_truth(
    predictions: list[NDArray[np.float64]],
    ground_truths: list[NDArray[np.float64]],
    max_distance: float,
    spacing: NDArray[np.float64] | None = None,
) -> list[CenterMatch]:
    """Match predictions to ground truth centers using greedy nearest-neighbor matching.

    Each prediction is matched to at most one ground truth, and vice versa.
    Matches are found greedily by distance, starting with the closest pairs.

    Args:
        predictions: List of predicted center coordinates (each is (3,) array).
        ground_truths: List of ground truth center coordinates (each is (3,) array).
        max_distance: Maximum allowed distance for a valid match.
        spacing: Physical spacing for each dimension (x, y, z). If None, assumes isotropic spacing.

    Returns:
        List of CenterMatch objects representing matched and unmatched centers.
        Includes: matched pairs, unmatched predictions (gt_idx=-1), unmatched GTs (pred_idx=-1).
    """
    if not predictions and not ground_truths:
        return []

    matches: list[CenterMatch] = []

    if not predictions:
        # All ground truths are unmatched
        for gt_idx in range(len(ground_truths)):
            matches.append(CenterMatch(pred_idx=-1, gt_idx=gt_idx, distance=np.inf))
        return matches

    if not ground_truths:
        # All predictions are unmatched
        for pred_idx in range(len(predictions)):
            matches.append(CenterMatch(pred_idx=pred_idx, gt_idx=-1, distance=np.inf))
        return matches

    # Compute all pairwise distances
    distances = np.zeros((len(predictions), len(ground_truths)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truths):
            distances[i, j] = compute_center_distance(pred, gt, spacing)

    # Greedy matching: find closest pairs first
    matched_preds = set()
    matched_gts = set()

    # Get all valid matches (distance <= max_distance)
    valid_matches = []
    for i in range(len(predictions)):
        for j in range(len(ground_truths)):
            if distances[i, j] <= max_distance:
                valid_matches.append((distances[i, j], i, j))

    # Sort by distance (closest first)
    valid_matches.sort()

    # Greedily assign matches
    for dist, pred_idx, gt_idx in valid_matches:
        if pred_idx not in matched_preds and gt_idx not in matched_gts:
            matches.append(CenterMatch(pred_idx=pred_idx, gt_idx=gt_idx, distance=dist))
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)

    # Add unmatched predictions
    for pred_idx in range(len(predictions)):
        if pred_idx not in matched_preds:
            matches.append(CenterMatch(pred_idx=pred_idx, gt_idx=-1, distance=np.inf))

    # Add unmatched ground truths
    for gt_idx in range(len(ground_truths)):
        if gt_idx not in matched_gts:
            matches.append(CenterMatch(pred_idx=-1, gt_idx=gt_idx, distance=np.inf))

    return matches


def calculate_accuracy_metrics(
    matches: list[CenterMatch],
) -> dict[str, float]:
    """Calculate center point accuracy metrics from matches.

    Args:
        matches: List of CenterMatch objects from match_predictions_to_ground_truth.

    Returns:
        Dictionary containing:
        - mean_distance: Mean distance of matched pairs (nan if no matches).
        - median_distance: Median distance of matched pairs (nan if no matches).
        - p95_distance: 95th percentile distance of matched pairs (nan if no matches).
        - max_distance: Maximum distance of matched pairs (nan if no matches).
        - num_matched: Number of successfully matched pairs.
        - num_unmatched_preds: Number of predictions without a match.
        - num_unmatched_gts: Number of ground truths without a match.
        - precision: Proportion of predictions that were matched (nan if no predictions).
        - recall: Proportion of ground truths that were matched (nan if no GTs).
    """
    if not matches:
        return {
            "mean_distance": np.nan,
            "median_distance": np.nan,
            "p95_distance": np.nan,
            "max_distance": np.nan,
            "num_matched": 0,
            "num_unmatched_preds": 0,
            "num_unmatched_gts": 0,
            "precision": np.nan,
            "recall": np.nan,
        }

    # Separate matched and unmatched
    matched_pairs = [m for m in matches if m.pred_idx != -1 and m.gt_idx != -1]
    unmatched_preds = [m for m in matches if m.pred_idx != -1 and m.gt_idx == -1]
    unmatched_gts = [m for m in matches if m.pred_idx == -1 and m.gt_idx != -1]

    # Calculate distance statistics
    if matched_pairs:
        distances = np.array([m.distance for m in matched_pairs])
        mean_dist = float(np.mean(distances))
        median_dist = float(np.median(distances))
        p95_dist = float(np.percentile(distances, 95))
        max_dist = float(np.max(distances))
    else:
        mean_dist = median_dist = p95_dist = max_dist = np.nan

    # Calculate counts
    num_matched = len(matched_pairs)
    num_unmatched_preds = len(unmatched_preds)
    num_unmatched_gts = len(unmatched_gts)

    # Calculate precision and recall
    total_preds = num_matched + num_unmatched_preds
    total_gts = num_matched + num_unmatched_gts

    precision = num_matched / total_preds if total_preds > 0 else np.nan
    recall = num_matched / total_gts if total_gts > 0 else np.nan

    return {
        "mean_distance": mean_dist,
        "median_distance": median_dist,
        "p95_distance": p95_dist,
        "max_distance": max_dist,
        "num_matched": num_matched,
        "num_unmatched_preds": num_unmatched_preds,
        "num_unmatched_gts": num_unmatched_gts,
        "precision": precision,
        "recall": recall,
    }
