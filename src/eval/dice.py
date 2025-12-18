"""Dice coefficient and IoU metrics for segmentation evaluation.

This module provides functions for computing Dice coefficient and Intersection over Union (IoU)
for evaluating 3D segmentation quality.
"""

import logging

import numpy as np
import torch
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_dice_score(
    prediction: NDArray[np.bool_] | torch.Tensor,
    ground_truth: NDArray[np.bool_] | torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    """Compute Dice coefficient between prediction and ground truth masks.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        prediction: Predicted binary mask.
        ground_truth: Ground truth binary mask.
        smooth: Smoothing factor to avoid division by zero. Defaults to 1e-6.

    Returns:
        Dice coefficient in range [0, 1], where 1 is perfect overlap.

    Examples:
        >>> pred = np.ones((10, 10, 10), dtype=bool)
        >>> gt = np.ones((10, 10, 10), dtype=bool)
        >>> dice = compute_dice_score(pred, gt)
        >>> assert dice == 1.0  # Perfect overlap
    """
    # Convert to numpy if torch tensor
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    # Convert to boolean
    pred_bool = prediction.astype(bool)
    gt_bool = ground_truth.astype(bool)

    # Compute intersection and union
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    pred_sum = pred_bool.sum()
    gt_sum = gt_bool.sum()

    # Compute Dice
    dice = (2.0 * intersection + smooth) / (pred_sum + gt_sum + smooth)

    logger.debug(
        f"Dice score: {dice:.4f} (intersection={intersection}, "
        f"pred={pred_sum}, gt={gt_sum})"
    )

    return float(dice)


def compute_iou(
    prediction: NDArray[np.bool_] | torch.Tensor,
    ground_truth: NDArray[np.bool_] | torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    """Compute Intersection over Union (IoU) between prediction and ground truth.

    IoU = |A ∩ B| / |A ∪ B|

    Args:
        prediction: Predicted binary mask.
        ground_truth: Ground truth binary mask.
        smooth: Smoothing factor to avoid division by zero. Defaults to 1e-6.

    Returns:
        IoU score in range [0, 1], where 1 is perfect overlap.

    Examples:
        >>> pred = np.ones((10, 10, 10), dtype=bool)
        >>> gt = np.ones((10, 10, 10), dtype=bool)
        >>> iou = compute_iou(pred, gt)
        >>> assert iou == 1.0  # Perfect overlap
    """
    # Convert to numpy if torch tensor
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    # Convert to boolean
    pred_bool = prediction.astype(bool)
    gt_bool = ground_truth.astype(bool)

    # Compute intersection and union
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)

    logger.debug(
        f"IoU score: {iou:.4f} (intersection={intersection}, union={union})"
    )

    return float(iou)


def compute_batch_dice_scores(
    predictions: NDArray[np.bool_] | torch.Tensor,
    ground_truths: NDArray[np.bool_] | torch.Tensor,
    smooth: float = 1e-6,
) -> NDArray[np.float64]:
    """Compute Dice scores for a batch of predictions.

    Args:
        predictions: Batch of predicted masks (N, D, H, W).
        ground_truths: Batch of ground truth masks (N, D, H, W).
        smooth: Smoothing factor. Defaults to 1e-6.

    Returns:
        Array of Dice scores for each sample in the batch.

    Examples:
        >>> preds = np.ones((5, 10, 10, 10), dtype=bool)
        >>> gts = np.ones((5, 10, 10, 10), dtype=bool)
        >>> scores = compute_batch_dice_scores(preds, gts)
        >>> assert len(scores) == 5
        >>> assert all(s == 1.0 for s in scores)
    """
    # Convert to numpy if torch tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(ground_truths, torch.Tensor):
        ground_truths = ground_truths.detach().cpu().numpy()

    if predictions.shape != ground_truths.shape:
        msg = (
            f"Shape mismatch: predictions {predictions.shape} "
            f"vs ground_truths {ground_truths.shape}"
        )
        raise ValueError(msg)

    batch_size = predictions.shape[0]
    scores = np.zeros(batch_size, dtype=np.float64)

    for i in range(batch_size):
        scores[i] = compute_dice_score(predictions[i], ground_truths[i], smooth)

    logger.debug(
        f"Computed {batch_size} Dice scores: mean={scores.mean():.4f}, "
        f"std={scores.std():.4f}"
    )

    return scores


def compute_batch_iou_scores(
    predictions: NDArray[np.bool_] | torch.Tensor,
    ground_truths: NDArray[np.bool_] | torch.Tensor,
    smooth: float = 1e-6,
) -> NDArray[np.float64]:
    """Compute IoU scores for a batch of predictions.

    Args:
        predictions: Batch of predicted masks (N, D, H, W).
        ground_truths: Batch of ground truth masks (N, D, H, W).
        smooth: Smoothing factor. Defaults to 1e-6.

    Returns:
        Array of IoU scores for each sample in the batch.

    Examples:
        >>> preds = np.ones((5, 10, 10, 10), dtype=bool)
        >>> gts = np.ones((5, 10, 10, 10), dtype=bool)
        >>> scores = compute_batch_iou_scores(preds, gts)
        >>> assert len(scores) == 5
        >>> assert all(s == 1.0 for s in scores)
    """
    # Convert to numpy if torch tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(ground_truths, torch.Tensor):
        ground_truths = ground_truths.detach().cpu().numpy()

    if predictions.shape != ground_truths.shape:
        msg = (
            f"Shape mismatch: predictions {predictions.shape} "
            f"vs ground_truths {ground_truths.shape}"
        )
        raise ValueError(msg)

    batch_size = predictions.shape[0]
    scores = np.zeros(batch_size, dtype=np.float64)

    for i in range(batch_size):
        scores[i] = compute_iou(predictions[i], ground_truths[i], smooth)

    logger.debug(
        f"Computed {batch_size} IoU scores: mean={scores.mean():.4f}, "
        f"std={scores.std():.4f}"
    )

    return scores


def dice_to_iou(dice: float) -> float:
    """Convert Dice coefficient to IoU.

    IoU = Dice / (2 - Dice)

    Args:
        dice: Dice coefficient in range [0, 1].

    Returns:
        IoU score in range [0, 1].

    Examples:
        >>> iou = dice_to_iou(0.8)
        >>> assert abs(iou - 0.6667) < 0.001
    """
    if dice == 0.0:
        return 0.0
    return dice / (2.0 - dice)


def iou_to_dice(iou: float) -> float:
    """Convert IoU to Dice coefficient.

    Dice = 2 * IoU / (1 + IoU)

    Args:
        iou: IoU score in range [0, 1].

    Returns:
        Dice coefficient in range [0, 1].

    Examples:
        >>> dice = iou_to_dice(0.6667)
        >>> assert abs(dice - 0.8) < 0.001
    """
    if iou == 0.0:
        return 0.0
    return (2.0 * iou) / (1.0 + iou)
