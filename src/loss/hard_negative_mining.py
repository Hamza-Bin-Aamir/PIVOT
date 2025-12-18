"""Hard Negative Mining for Online Hard Example Mining (OHEM).

This module implements hard negative mining to address extreme class imbalance
in nodule detection by selecting the hardest negative examples during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HardNegativeMiningLoss(nn.Module):
    """Hard Negative Mining wrapper for loss functions.

    This class wraps any loss function and performs Online Hard Example Mining (OHEM)
    by selecting the top-k hardest examples (highest loss values) for backpropagation.
    This is particularly useful for center detection where background voxels vastly
    outnumber nodule centers.

    The algorithm:
    1. Compute per-element loss for all examples
    2. Separate positive and negative examples
    3. Keep all positive examples
    4. Select top-k hardest negatives based on loss magnitude
    5. Compute final loss using selected examples only

    Reference:
        Shrivastava, A., Gupta, A., & Girshick, R. (2016). "Training region-based
        object detectors with online hard example mining." In CVPR (pp. 761-769).

    Args:
        base_loss: The underlying loss function to wrap. Must accept 'reduction="none"'
            and return per-element losses.
        hard_negative_ratio: Ratio of hard negatives to positives. For example,
            if ratio=3.0 and there are 100 positive examples, select top 300
            hardest negative examples. Default: 3.0
        min_negative_samples: Minimum number of negative samples to mine, regardless
            of positive count. Ensures learning even with very few positives.
            Default: 100
        reduction: Final reduction to apply after mining: 'mean' | 'sum'.
            Default: 'mean'

    Shape:
        - Input: Same as base_loss input
        - Target: Same as base_loss target (should be binary 0/1 for pos/neg separation)
        - Output: Scalar loss value

    Examples:
        >>> # Wrap Focal Loss with hard negative mining
        >>> from src.loss.focal import FocalLoss
        >>> base_loss = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)
        >>> loss_fn = HardNegativeMiningLoss(base_loss, hard_negative_ratio=3.0)
        >>> predictions = torch.randn(2, 1, 64, 64, 64)
        >>> targets = torch.randint(0, 2, (2, 1, 64, 64, 64)).float()
        >>> loss = loss_fn(predictions, targets)

        >>> # With custom ratio
        >>> loss_fn = HardNegativeMiningLoss(
        ...     base_loss,
        ...     hard_negative_ratio=5.0,
        ...     min_negative_samples=500
        ... )
        >>> loss = loss_fn(predictions, targets)

        >>> # Works with any loss function
        >>> from src.loss.bce import BCELoss
        >>> base_loss = BCELoss(from_logits=True)
        >>> loss_fn = HardNegativeMiningLoss(base_loss, hard_negative_ratio=4.0)
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        hard_negative_ratio: float = 3.0,
        min_negative_samples: int = 100,
        reduction: str = "mean",
    ) -> None:
        """Initialize hard negative mining loss.

        Args:
            base_loss: Base loss function to wrap
            hard_negative_ratio: Ratio of hard negatives to positives
            min_negative_samples: Minimum negative samples to mine
            reduction: Final reduction method ('mean' or 'sum')

        Raises:
            ValueError: If hard_negative_ratio <= 0 or min_negative_samples < 0
            ValueError: If reduction not in ['mean', 'sum']
        """
        super().__init__()

        if hard_negative_ratio <= 0:
            raise ValueError(
                f"hard_negative_ratio must be positive, got {hard_negative_ratio}"
            )
        if min_negative_samples < 0:
            raise ValueError(
                f"min_negative_samples must be >= 0, got {min_negative_samples}"
            )
        if reduction not in ["mean", "sum"]:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")

        self.base_loss = base_loss
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.min_negative_samples = int(min_negative_samples)
        self.reduction = reduction

        # Verify base loss supports reduction='none'
        if not hasattr(base_loss, "reduction"):
            raise ValueError("base_loss must have a 'reduction' attribute")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute hard negative mining loss.

        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels (binary 0/1 values)

        Returns:
            Scalar loss value after hard negative mining

        Raises:
            ValueError: If predictions and targets have different shapes
        """
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

        # Store original reduction setting
        original_reduction: str = self.base_loss.reduction  # type: ignore[assignment]

        # Compute per-element loss
        self.base_loss.reduction = "none"  # type: ignore[assignment]
        per_element_loss = self.base_loss(predictions, targets)
        self.base_loss.reduction = original_reduction  # type: ignore[assignment]

        # Flatten tensors for easier indexing
        flat_loss = per_element_loss.view(-1)
        flat_targets = targets.view(-1)

        # Separate positive and negative examples
        positive_mask = flat_targets > 0.5  # Assume binary 0/1 targets
        negative_mask = ~positive_mask

        positive_indices = torch.nonzero(positive_mask, as_tuple=True)[0]
        negative_indices = torch.nonzero(negative_mask, as_tuple=True)[0]

        num_positives = positive_indices.numel()
        num_negatives = negative_indices.numel()

        # Edge case: no positives or negatives
        if num_positives == 0 and num_negatives == 0:
            # Return zero loss
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

        # Keep all positive examples
        selected_indices = positive_indices

        # Select hard negatives
        if num_negatives > 0:
            # Calculate number of negatives to mine
            num_hard_negatives = max(
                int(num_positives * self.hard_negative_ratio),
                self.min_negative_samples,
            )
            num_hard_negatives = min(num_hard_negatives, num_negatives)

            if num_hard_negatives > 0:
                # Get losses for negative examples
                negative_losses = flat_loss[negative_indices]

                # Select top-k hardest negatives (highest loss)
                if num_hard_negatives < num_negatives:
                    # Need to select subset
                    _, hard_negative_idx = torch.topk(
                        negative_losses, num_hard_negatives, largest=True, sorted=False
                    )
                    selected_negative_indices = negative_indices[hard_negative_idx]
                else:
                    # Use all negatives
                    selected_negative_indices = negative_indices

                # Combine positive and hard negative indices
                selected_indices = torch.cat([selected_indices, selected_negative_indices])

        # Compute final loss using only selected examples
        if selected_indices.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

        selected_losses = flat_loss[selected_indices]

        if self.reduction == "mean":
            return selected_losses.mean()  # type: ignore[no-any-return]
        else:  # sum
            return selected_losses.sum()  # type: ignore[no-any-return]

    def get_statistics(self, targets: torch.Tensor) -> dict[str, int | float]:
        """Get statistics about mined samples (for debugging/monitoring).

        Args:
            targets: Ground truth labels

        Returns:
            Dictionary with counts:
                - 'num_positives': Number of positive examples
                - 'num_negatives': Total number of negative examples
                - 'num_hard_negatives': Number of selected hard negatives
                - 'num_selected': Total number of selected examples
        """
        flat_targets = targets.view(-1)

        positive_mask = flat_targets > 0.5
        negative_mask = ~positive_mask

        num_positives = positive_mask.sum().item()
        num_negatives = negative_mask.sum().item()

        num_hard_negatives: int | float = max(
            int(num_positives * self.hard_negative_ratio),
            self.min_negative_samples,
        )
        num_hard_negatives = min(num_hard_negatives, num_negatives)

        num_selected: int | float = num_positives + num_hard_negatives

        return {
            "num_positives": num_positives,
            "num_negatives": num_negatives,
            "num_hard_negatives": num_hard_negatives,
            "num_selected": num_selected,
        }


__all__ = ["HardNegativeMiningLoss"]
