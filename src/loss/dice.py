"""Dice loss for segmentation tasks."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice loss for binary and multi-class segmentation.

    Computes the Dice coefficient (F1 score) between predictions and targets,
    then returns 1 - Dice as the loss. Includes smoothing to prevent division
    by zero and supports both binary and multi-class segmentation.

    The Dice coefficient is defined as:
        Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)

    Loss is then: 1 - Dice

    Args:
        smooth (float): Smoothing constant to prevent division by zero.
                        Default: 1.0 (standard Dice)
        from_logits (bool): If True, applies sigmoid/softmax to predictions.
                            If False, assumes predictions are probabilities.
                            Default: True
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'.
                         Default: 'mean'

    Attributes:
        smooth (float): Smoothing constant
        from_logits (bool): Whether to apply activation
        reduction (str): Reduction method

    Example:
        >>> loss_fn = DiceLoss(smooth=1.0, from_logits=True)
        >>> predictions = torch.randn(2, 1, 64, 64, 64)  # Logits
        >>> targets = torch.randint(0, 2, (2, 1, 64, 64, 64)).float()
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        smooth: float = 1.0,
        from_logits: bool = True,
        reduction: str = "mean",
    ) -> None:
        """Initialize Dice loss.

        Args:
            smooth: Smoothing constant
            from_logits: Whether predictions are logits
            reduction: Reduction method ('none', 'mean', 'sum')

        Raises:
            ValueError: If reduction is not one of 'none', 'mean', 'sum'
        """
        super().__init__()

        if reduction not in ("none", "mean", "sum"):
            msg = f"Invalid reduction: {reduction}. Must be 'none', 'mean', or 'sum'"
            raise ValueError(msg)

        self.smooth = smooth
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            predictions: Predicted segmentation masks
                        Shape: (B, C, D, H, W) or (B, C, H, W)
                        If from_logits=True, these are logits
                        If from_logits=False, these are probabilities [0, 1]
            targets: Ground truth segmentation masks
                    Shape: (B, C, D, H, W) or (B, C, H, W)
                    Values should be in [0, 1] for binary or one-hot encoded

        Returns:
            Dice loss value:
            - Scalar if reduction='mean' or 'sum'
            - Tensor of shape (B,) if reduction='none'

        Raises:
            ValueError: If predictions and targets have different shapes
        """
        if predictions.shape != targets.shape:
            msg = f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            raise ValueError(msg)

        # Apply activation if working with logits
        if self.from_logits:
            if predictions.shape[1] == 1:
                # Binary segmentation: use sigmoid
                predictions = torch.sigmoid(predictions)
            else:
                # Multi-class segmentation: use softmax
                predictions = torch.softmax(predictions, dim=1)

        # Flatten spatial dimensions while keeping batch and channel separate
        # (B, C, D, H, W) -> (B, C, D*H*W)
        batch_size = predictions.shape[0]
        num_classes = predictions.shape[1]

        predictions_flat = predictions.view(batch_size, num_classes, -1)
        targets_flat = targets.view(batch_size, num_classes, -1)

        # Compute Dice coefficient per class per batch
        # Intersection: sum over spatial dimension
        intersection = (predictions_flat * targets_flat).sum(dim=2)

        # Union: sum of predictions + targets
        predictions_sum = predictions_flat.sum(dim=2)
        targets_sum = targets_flat.sum(dim=2)

        # Dice coefficient: 2*|X∩Y| / (|X| + |Y|)
        dice_coeff = (2.0 * intersection + self.smooth) / (
            predictions_sum + targets_sum + self.smooth
        )

        # Dice loss: 1 - Dice coefficient
        # Average over classes: (B, C) -> (B,)
        dice_loss = 1.0 - dice_coeff.mean(dim=1)

        # Apply reduction
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:  # 'none'
            return dice_loss
