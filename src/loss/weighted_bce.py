"""Weighted Binary Cross-Entropy Loss for malignancy triage in nodule detection.

This module implements a weighted variant of Binary Cross-Entropy Loss specifically
designed for malignancy triage tasks where class imbalance is common (e.g., more
benign nodules than malignant ones).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy Loss for malignancy triage classification.

    This loss function is specifically designed for the malignancy triage task in
    nodule detection, where there is often significant class imbalance between
    benign and malignant cases. It extends standard BCE with configurable class
    weights to handle imbalanced datasets.

    The weighted BCE loss is computed as:
        loss = -[w_pos * y * log(p) + w_neg * (1-y) * log(1-p)]

    where w_pos and w_neg are the weights for positive (malignant) and negative
    (benign) classes respectively.

    Args:
        pos_weight: Weight for the positive class (malignant). Can be a scalar or
            tensor. Higher values increase importance of malignant cases.
            Default: None (no weighting, equivalent to standard BCE)
        neg_weight: Weight for the negative class (benign). Can be a scalar or
            tensor. Higher values increase importance of benign cases.
            Default: 1.0
        from_logits: If True, expects raw logits as input. If False, expects
            probabilities (after sigmoid). Using logits is recommended for
            numerical stability.
            Default: True
        reduction: Specifies reduction: 'none' | 'mean' | 'sum'.
            - 'none': no reduction, returns per-element loss
            - 'mean': returns mean of loss
            - 'sum': returns sum of loss
            Default: 'mean'

    Shape:
        - Input: (B, C, *) where * means any number of additional dimensions
        - Target: (B, C, *) same shape as input, with values in [0, 1]
        - Output: scalar if reduction in ['mean', 'sum'], else same shape as input

    Examples:
        >>> # Malignancy triage with higher weight on malignant cases
        >>> loss_fn = WeightedBCELoss(pos_weight=3.0, neg_weight=1.0)
        >>> logits = torch.randn(2, 1, 16, 16, 16)
        >>> targets = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()
        >>> loss = loss_fn(logits, targets)

        >>> # Using tensor weights for multi-output triage
        >>> pos_weight = torch.tensor([2.0, 3.0, 4.0])
        >>> loss_fn = WeightedBCELoss(pos_weight=pos_weight)
        >>> logits = torch.randn(2, 3, 8, 8, 8)
        >>> targets = torch.randint(0, 2, (2, 3, 8, 8, 8)).float()
        >>> loss = loss_fn(logits, targets)

        >>> # With probabilities instead of logits
        >>> loss_fn = WeightedBCELoss(pos_weight=2.0, from_logits=False)
        >>> probs = torch.sigmoid(torch.randn(2, 1, 8, 8, 8))
        >>> targets = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()
        >>> loss = loss_fn(probs, targets)
    """

    def __init__(
        self,
        pos_weight: float | torch.Tensor | None = None,
        neg_weight: float | torch.Tensor = 1.0,
        from_logits: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        # Validate reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")

        # Store pos_weight as buffer if tensor, otherwise as attribute
        if isinstance(pos_weight, torch.Tensor):
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = pos_weight

        # Store neg_weight as buffer if tensor, otherwise as attribute
        if isinstance(neg_weight, torch.Tensor):
            self.register_buffer("neg_weight", neg_weight)
        else:
            self.neg_weight = neg_weight

        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss.

        Args:
            predictions: Predicted logits or probabilities for malignancy.
                Shape: (B, C, *) where C is number of triage outputs.
            targets: Ground truth labels (0 for benign, 1 for malignant).
                Shape: (B, C, *) same as predictions.

        Returns:
            Weighted BCE loss value. Scalar if reduction in ['mean', 'sum'],
            else per-element.

        Raises:
            ValueError: If shapes are incompatible.
        """
        # Validate shapes
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

        # Compute BCE loss
        if self.from_logits:
            # Use BCEWithLogitsLoss for numerical stability
            # Note: PyTorch's pos_weight is only for positive class
            # We need to manually apply both pos_weight and neg_weight
            bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        else:
            # Use standard BCE
            bce_loss = F.binary_cross_entropy(predictions, targets, reduction="none")

        # Apply class weights
        # Separate positive and negative losses
        pos_mask = targets == 1
        neg_mask = targets == 0

        weighted_loss = torch.zeros_like(bce_loss)

        # Apply positive weight
        if self.pos_weight is not None:
            if pos_mask.any():
                weighted_loss[pos_mask] = self.pos_weight * bce_loss[pos_mask]
        else:
            # No positive weighting, use original loss
            if pos_mask.any():
                weighted_loss[pos_mask] = bce_loss[pos_mask]

        # Apply negative weight
        if neg_mask.any():
            weighted_loss[neg_mask] = self.neg_weight * bce_loss[neg_mask]

        # Apply reduction
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:  # none
            return weighted_loss
