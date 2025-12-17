"""Smooth L1 Loss for size regression in nodule detection tasks.

This module implements Smooth L1 Loss (Huber Loss variant), designed for robust
regression that is less sensitive to outliers than L2 loss while maintaining
differentiability at zero unlike L1 loss.
"""

import torch
import torch.nn as nn


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss for robust regression tasks like size prediction.

    Smooth L1 Loss (also known as Huber Loss) combines the advantages of L1 and L2 loss.
    It is quadratic for small errors (like L2) and linear for large errors (like L1),
    making it more robust to outliers than pure L2 loss while being differentiable
    everywhere unlike L1 loss.

    The loss function is defined as:
        loss = 0.5 * (x - y)^2 / beta,     if |x - y| < beta
        loss = |x - y| - 0.5 * beta,       otherwise

    This is particularly useful for size regression where some predictions may have
    large errors (outliers) that shouldn't dominate the training.

    Reference:
        Girshick, R. (2015). "Fast R-CNN." In Proceedings of the IEEE international
        conference on computer vision (pp. 1440-1448).

    Args:
        beta: Threshold at which to change between L1 and L2 loss. Lower values
            make the loss more robust to outliers. Must be positive.
            Default: 1.0
        reduction: Specifies reduction: 'none' | 'mean' | 'sum'.
            - 'none': no reduction, returns per-element loss
            - 'mean': returns mean of loss
            - 'sum': returns sum of loss
            Default: 'mean'

    Shape:
        - Input: (B, C, *) where * means any number of additional dimensions
        - Target: (B, C, *) same shape as input
        - Output: scalar if reduction in ['mean', 'sum'], else same shape as input

    Examples:
        >>> # Size regression with default beta
        >>> loss_fn = SmoothL1Loss(beta=1.0)
        >>> predictions = torch.randn(2, 3, 16, 16, 16)  # 3D size predictions (x,y,z)
        >>> targets = torch.randn(2, 3, 16, 16, 16)
        >>> loss = loss_fn(predictions, targets)

        >>> # More robust to outliers with smaller beta
        >>> loss_fn = SmoothL1Loss(beta=0.5, reduction='sum')
        >>> predictions = torch.randn(2, 3, 32, 32, 32)
        >>> targets = torch.randn(2, 3, 32, 32, 32)
        >>> loss = loss_fn(predictions, targets)

        >>> # Per-element loss
        >>> loss_fn = SmoothL1Loss(beta=1.0, reduction='none')
        >>> predictions = torch.randn(2, 3, 8, 8, 8)
        >>> targets = torch.randn(2, 3, 8, 8, 8)
        >>> loss = loss_fn(predictions, targets)  # Same shape as input
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()

        # Validate reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")

        # Validate beta
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

        self.beta = beta
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute smooth L1 loss.

        Args:
            predictions: Predicted values (e.g., size in mm).
                Shape: (B, C, *) where C is typically 3 for 3D sizes.
            targets: Ground truth values.
                Shape: (B, C, *) same as predictions.

        Returns:
            Smooth L1 loss value. Scalar if reduction in ['mean', 'sum'],
            else per-element.

        Raises:
            ValueError: If shapes are incompatible.
        """
        # Validate shapes
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

        # Compute absolute difference
        diff = torch.abs(predictions - targets)

        # Smooth L1 loss formula:
        # loss = 0.5 * diff^2 / beta,     if diff < beta
        # loss = diff - 0.5 * beta,       otherwise
        loss = torch.where(
            diff < self.beta,
            0.5 * diff.pow(2) / self.beta,
            diff - 0.5 * self.beta,
        )

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # none
            return loss
