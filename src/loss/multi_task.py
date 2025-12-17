"""Multi-Task Loss Aggregator for nodule detection tasks.

This module implements a loss aggregator that combines losses from all four
tasks in the multi-task nodule detection pipeline: segmentation, center detection,
size regression, and malignancy triage.
"""

import torch
import torch.nn as nn

from .dice import DiceLoss
from .focal import FocalLoss
from .smooth_l1 import SmoothL1Loss
from .weighted_bce import WeightedBCELoss


class MultiTaskLoss(nn.Module):
    """Multi-Task Loss Aggregator for nodule detection.

    This loss function combines losses from all four tasks in the multi-task
    nodule detection pipeline:
    1. Segmentation: DiceLoss for binary segmentation
    2. Center Detection: FocalLoss for center point detection
    3. Size Regression: SmoothL1Loss for 3D size prediction
    4. Malignancy Triage: WeightedBCELoss for malignancy classification

    The final loss is a weighted combination:
        total_loss = w_seg * L_seg + w_center * L_center + w_size * L_size + w_triage * L_triage

    Args:
        seg_weight: Weight for segmentation loss. Default: 1.0
        center_weight: Weight for center detection loss. Default: 1.0
        size_weight: Weight for size regression loss. Default: 1.0
        triage_weight: Weight for malignancy triage loss. Default: 1.0
        seg_loss_kwargs: Keyword arguments for DiceLoss. Default: {}
        center_loss_kwargs: Keyword arguments for FocalLoss. Default: {}
        size_loss_kwargs: Keyword arguments for SmoothL1Loss. Default: {}
        triage_loss_kwargs: Keyword arguments for WeightedBCELoss. Default: {}
        reduction: How to reduce the final aggregated loss: 'mean' | 'sum' | 'none'.
            Default: 'mean'

    Shape:
        - predictions: Dictionary with keys:
            - 'segmentation': (B, 1, D, H, W)
            - 'center': (B, 1, D, H, W)
            - 'size': (B, 3, 1, 1, 1) - global pooled
            - 'triage': (B, 1, 1, 1, 1) - global pooled
        - targets: Dictionary with same keys and shapes as predictions
        - Output: scalar loss value

    Examples:
        >>> # Default equal weighting
        >>> loss_fn = MultiTaskLoss()
        >>> predictions = {
        ...     'segmentation': torch.randn(2, 1, 16, 16, 16),
        ...     'center': torch.randn(2, 1, 16, 16, 16),
        ...     'size': torch.randn(2, 3, 1, 1, 1),
        ...     'triage': torch.randn(2, 1, 1, 1, 1)
        ... }
        >>> targets = {
        ...     'segmentation': torch.randint(0, 2, (2, 1, 16, 16, 16)).float(),
        ...     'center': torch.randint(0, 2, (2, 1, 16, 16, 16)).float(),
        ...     'size': torch.randn(2, 3, 1, 1, 1),
        ...     'triage': torch.randint(0, 2, (2, 1, 1, 1, 1)).float()
        ... }
        >>> loss = loss_fn(predictions, targets)

        >>> # Custom task weights (emphasize triage and center detection)
        >>> loss_fn = MultiTaskLoss(
        ...     seg_weight=1.0,
        ...     center_weight=2.0,
        ...     size_weight=0.5,
        ...     triage_weight=3.0
        ... )

        >>> # Custom loss configurations
        >>> loss_fn = MultiTaskLoss(
        ...     seg_loss_kwargs={'smooth': 1.0},
        ...     center_loss_kwargs={'alpha': 0.25, 'gamma': 2.0},
        ...     size_loss_kwargs={'beta': 1.0},
        ...     triage_loss_kwargs={'pos_weight': 3.0, 'neg_weight': 1.0}
        ... )
    """

    def __init__(
        self,
        seg_weight: float = 1.0,
        center_weight: float = 1.0,
        size_weight: float = 1.0,
        triage_weight: float = 1.0,
        seg_loss_kwargs: dict | None = None,
        center_loss_kwargs: dict | None = None,
        size_loss_kwargs: dict | None = None,
        triage_loss_kwargs: dict | None = None,
        reduction: str = "mean",
    ):
        super().__init__()

        # Validate reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")

        # Store task weights
        self.seg_weight = seg_weight
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.triage_weight = triage_weight
        self.reduction = reduction

        # Initialize individual loss functions
        seg_kwargs = seg_loss_kwargs or {}
        center_kwargs = center_loss_kwargs or {}
        size_kwargs = size_loss_kwargs or {}
        triage_kwargs = triage_loss_kwargs or {}

        # Ensure individual losses use 'mean' reduction for proper weighting
        seg_kwargs.setdefault("reduction", "mean")
        center_kwargs.setdefault("reduction", "mean")
        size_kwargs.setdefault("reduction", "mean")
        triage_kwargs.setdefault("reduction", "mean")

        self.seg_loss = DiceLoss(**seg_kwargs)
        self.center_loss = FocalLoss(**center_kwargs)
        self.size_loss = SmoothL1Loss(**size_kwargs)
        self.triage_loss = WeightedBCELoss(**triage_kwargs)

    def forward(
        self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute aggregated multi-task loss.

        Args:
            predictions: Dictionary of predictions from MultiTaskUNet3D with keys:
                - 'segmentation': Segmentation predictions
                - 'center': Center detection predictions
                - 'size': Size regression predictions (global pooled)
                - 'triage': Malignancy triage predictions (global pooled)
            targets: Dictionary of ground truth with same keys as predictions.

        Returns:
            If reduction='none': Dictionary with individual losses and 'total'
            Otherwise: Scalar total loss value

        Raises:
            ValueError: If required keys are missing from predictions or targets.
        """
        # Validate required keys
        required_keys = ["segmentation", "center", "size", "triage"]
        for key in required_keys:
            if key not in predictions:
                raise ValueError(f"Missing '{key}' in predictions")
            if key not in targets:
                raise ValueError(f"Missing '{key}' in targets")

        # Compute individual losses
        seg_loss = self.seg_loss(predictions["segmentation"], targets["segmentation"])
        center_loss = self.center_loss(predictions["center"], targets["center"])
        size_loss = self.size_loss(predictions["size"], targets["size"])
        triage_loss = self.triage_loss(predictions["triage"], targets["triage"])

        # Apply task weights
        weighted_seg = self.seg_weight * seg_loss
        weighted_center = self.center_weight * center_loss
        weighted_size = self.size_weight * size_loss
        weighted_triage = self.triage_weight * triage_loss

        # Aggregate
        total_loss = weighted_seg + weighted_center + weighted_size + weighted_triage

        # Return based on reduction mode
        if self.reduction == "none":
            return {
                "total": total_loss,
                "segmentation": weighted_seg,
                "center": weighted_center,
                "size": weighted_size,
                "triage": weighted_triage,
                "segmentation_unweighted": seg_loss,
                "center_unweighted": center_loss,
                "size_unweighted": size_loss,
                "triage_unweighted": triage_loss,
            }
        elif self.reduction == "sum":
            # Already summed across tasks
            return total_loss
        else:  # mean
            # Already mean across tasks (since individual losses use mean)
            return total_loss

    def get_task_weights(self) -> dict[str, float]:
        """Get current task weights.

        Returns:
            Dictionary mapping task names to their weights.
        """
        return {
            "segmentation": self.seg_weight,
            "center": self.center_weight,
            "size": self.size_weight,
            "triage": self.triage_weight,
        }

    def set_task_weights(
        self,
        seg_weight: float | None = None,
        center_weight: float | None = None,
        size_weight: float | None = None,
        triage_weight: float | None = None,
    ) -> None:
        """Update task weights dynamically.

        This allows adjusting the importance of different tasks during training,
        e.g., curriculum learning or dynamic task weighting strategies.

        Args:
            seg_weight: New weight for segmentation task.
            center_weight: New weight for center detection task.
            size_weight: New weight for size regression task.
            triage_weight: New weight for malignancy triage task.
        """
        if seg_weight is not None:
            self.seg_weight = seg_weight
        if center_weight is not None:
            self.center_weight = center_weight
        if size_weight is not None:
            self.size_weight = size_weight
        if triage_weight is not None:
            self.triage_weight = triage_weight
