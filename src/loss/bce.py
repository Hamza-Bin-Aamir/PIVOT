"""Binary cross-entropy loss for classification tasks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """Binary cross-entropy loss with logits support.

    Computes the binary cross-entropy loss between predictions and targets.
    Supports class weighting for imbalanced datasets via pos_weight parameter.

    When from_logits=True, uses BCEWithLogitsLoss which combines sigmoid and BCE
    for numerical stability. When from_logits=False, uses standard BCE.

    Args:
        pos_weight (torch.Tensor | None): Weight for positive class to handle
                                           class imbalance. Shape: (C,) where C
                                           is number of classes. Default: None
        from_logits (bool): If True, applies sigmoid to predictions.
                            If False, assumes predictions are probabilities.
                            Default: True
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'.
                         Default: 'mean'

    Attributes:
        pos_weight (torch.Tensor | None): Positive class weight
        from_logits (bool): Whether to apply sigmoid activation
        reduction (str): Reduction method

    Example:
        >>> # Binary segmentation with class imbalance
        >>> pos_weight = torch.tensor([2.0])  # Positive class is 2x important
        >>> loss_fn = BCELoss(pos_weight=pos_weight, from_logits=True)
        >>> predictions = torch.randn(2, 1, 64, 64, 64)  # Logits
        >>> targets = torch.randint(0, 2, (2, 1, 64, 64, 64)).float()
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        from_logits: bool = True,
        reduction: str = "mean",
    ) -> None:
        """Initialize BCE loss.

        Args:
            pos_weight: Weight for positive class
            from_logits: Whether predictions are logits
            reduction: Reduction method ('none', 'mean', 'sum')

        Raises:
            ValueError: If reduction is not one of 'none', 'mean', 'sum'
        """
        super().__init__()

        if reduction not in ("none", "mean", "sum"):
            msg = f"Invalid reduction: {reduction}. Must be 'none', 'mean', or 'sum'"
            raise ValueError(msg)

        self.register_buffer("pos_weight", pos_weight)
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss.

        Args:
            predictions: Predicted values
                        Shape: (B, C, ...) where ... can be any spatial dims
                        If from_logits=True, these are logits
                        If from_logits=False, these are probabilities [0, 1]
            targets: Ground truth binary labels
                    Shape: (B, C, ...)
                    Values should be 0 or 1

        Returns:
            BCE loss value:
            - Scalar if reduction='mean' or 'sum'
            - Tensor of same shape as input if reduction='none'

        Raises:
            ValueError: If predictions and targets have different shapes
        """
        if predictions.shape != targets.shape:
            msg = f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            raise ValueError(msg)

        if self.from_logits:
            # Use BCEWithLogitsLoss for numerical stability
            pos_weight_tensor = (
                self.pos_weight if isinstance(self.pos_weight, torch.Tensor) else None
            )
            loss = F.binary_cross_entropy_with_logits(
                predictions,
                targets,
                pos_weight=pos_weight_tensor,
                reduction=self.reduction,
            )
        else:
            # Standard BCE (predictions should be in [0, 1])
            if self.pos_weight is not None:
                # Manual pos_weight application for BCE
                # BCE = -[y*log(p) + (1-y)*log(1-p)]
                # With pos_weight: -[w*y*log(p) + (1-y)*log(1-p)]
                pos_weight_value = (
                    self.pos_weight
                    if isinstance(self.pos_weight, torch.Tensor)
                    else torch.tensor(self.pos_weight)
                )
                loss = -(
                    pos_weight_value * targets * torch.log(predictions + 1e-7)
                    + (1 - targets) * torch.log(1 - predictions + 1e-7)
                )
            else:
                loss = F.binary_cross_entropy(
                    predictions,
                    targets,
                    reduction="none",
                )

            # Apply reduction manually if pos_weight was used
            if self.pos_weight is not None or self.reduction != "none":
                if self.reduction == "mean":
                    loss = loss.mean()
                elif self.reduction == "sum":
                    loss = loss.sum()

        return loss
