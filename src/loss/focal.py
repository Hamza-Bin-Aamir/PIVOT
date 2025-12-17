"""Focal Loss for center detection in nodule detection tasks.

This module implements Focal Loss (Lin et al., 2017), designed to address
class imbalance by down-weighting easy examples and focusing on hard negatives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in binary/multi-class classification.

    Focal Loss applies a modulating factor (1 - p_t)^gamma to the standard cross-entropy
    loss, reducing the relative loss for well-classified examples and focusing training
    on hard negatives. This is particularly useful for center detection where the
    vast majority of voxels are background.

    Reference:
        Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        "Focal loss for dense object detection." In Proceedings of the IEEE
        international conference on computer vision (pp. 2980-2988).

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples,
            or tensor of weights for each class. If float, alpha is applied to
            class 1 and (1-alpha) to class 0. Default: 0.25
        gamma: Focusing parameter gamma >= 0. Higher gamma increases focus on
            hard examples. gamma=0 is equivalent to standard cross-entropy.
            Default: 2.0
        from_logits: If True, expects raw logits. If False, expects probabilities.
            Default: True
        reduction: Specifies reduction: 'none' | 'mean' | 'sum'.
            - 'none': no reduction, returns per-element loss
            - 'mean': returns mean of loss
            - 'sum': returns sum of loss
            Default: 'mean'

    Shape:
        - Input: (B, C, *) where * means any number of additional dimensions
        - Target: (B, C, *) same shape as input for binary, or (B, *) with class indices
        - Output: scalar if reduction in ['mean', 'sum'], else same shape as input

    Examples:
        >>> # Binary classification with logits
        >>> focal_loss = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)
        >>> logits = torch.randn(2, 1, 64, 64, 64)
        >>> targets = torch.randint(0, 2, (2, 1, 64, 64, 64)).float()
        >>> loss = focal_loss(logits, targets)

        >>> # Multi-class with probabilities
        >>> focal_loss = FocalLoss(alpha=0.5, gamma=2.0, from_logits=False)
        >>> probs = torch.softmax(torch.randn(2, 3, 64, 64, 64), dim=1)
        >>> targets = torch.randint(0, 3, (2, 64, 64, 64)).long()
        >>> loss = focal_loss(probs, targets)

        >>> # With class weights
        >>> alpha = torch.tensor([0.25, 0.5, 0.25])
        >>> focal_loss = FocalLoss(alpha=alpha, gamma=2.0)
        >>> logits = torch.randn(2, 3, 32, 32, 32)
        >>> targets = torch.randint(0, 3, (2, 32, 32, 32)).long()
        >>> loss = focal_loss(logits, targets)
    """

    def __init__(
        self,
        alpha: float | torch.Tensor = 0.25,
        gamma: float = 2.0,
        from_logits: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        # Validate reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")

        # Validate gamma
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")

        # Store alpha as buffer if tensor, otherwise as attribute
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha)
        else:
            if not 0 <= alpha <= 1:
                raise ValueError(f"alpha must be in [0, 1], got {alpha}")
            self.alpha = alpha

        self.gamma = gamma
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            predictions: Predicted logits or probabilities.
                Shape: (B, C, *) for multi-class or (B, 1, *) for binary.
            targets: Ground truth labels.
                Shape: (B, C, *) for binary or (B, *) with class indices for multi-class.

        Returns:
            Focal loss value. Scalar if reduction in ['mean', 'sum'], else per-element.

        Raises:
            ValueError: If shapes are incompatible.
        """
        # Handle binary case (C=1) vs multi-class (C>1)
        num_classes = predictions.shape[1]

        if num_classes == 1:
            # Binary classification
            return self._focal_loss_binary(predictions, targets)
        else:
            # Multi-class classification
            return self._focal_loss_multiclass(predictions, targets)

    def _focal_loss_binary(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for binary classification.

        Args:
            predictions: Predicted logits or probabilities. Shape: (B, 1, *)
            targets: Ground truth labels. Shape: (B, 1, *)

        Returns:
            Focal loss value.
        """
        # Validate shapes
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

        # Get probabilities
        if self.from_logits:
            probs = torch.sigmoid(predictions)
            # Compute BCE with logits for numerical stability
            bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        else:
            probs = predictions
            # Clamp probabilities to avoid log(0)
            probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)
            bce_loss = F.binary_cross_entropy(probs, targets, reduction="none")

        # Compute p_t (probability of true class)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        if isinstance(self.alpha, torch.Tensor):
            # alpha is a tensor, use first element for binary
            alpha_t = self.alpha[0] * targets + (1 - self.alpha[0]) * (1 - targets)
        else:
            # alpha is a float
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_loss = alpha_t * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # none
            return focal_loss

    def _focal_loss_multiclass(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss for multi-class classification.

        Args:
            predictions: Predicted logits or probabilities. Shape: (B, C, *)
            targets: Ground truth class indices. Shape: (B, *)

        Returns:
            Focal loss value.
        """
        # Get probabilities
        if self.from_logits:
            probs = F.softmax(predictions, dim=1)
            # Compute log probabilities for numerical stability
            log_probs = F.log_softmax(predictions, dim=1)
        else:
            probs = predictions
            # Clamp and compute log
            probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)
            log_probs = torch.log(probs)

        # Get probability of true class (p_t)
        # Reshape targets to (B, 1, *) for gathering
        targets_expanded = targets.unsqueeze(1)

        # Gather probabilities and log probabilities for true class
        p_t = torch.gather(probs, 1, targets_expanded).squeeze(1)
        log_p_t = torch.gather(log_probs, 1, targets_expanded).squeeze(1)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_t = self.alpha[targets] if isinstance(self.alpha, torch.Tensor) else self.alpha

        # Compute focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_loss = -alpha_t * focal_weight * log_p_t

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # none
            return focal_loss
