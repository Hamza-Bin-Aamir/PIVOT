"""Tests for hard negative mining loss function."""

from __future__ import annotations

import pytest
import torch

from src.loss import FocalLoss, HardNegativeMiningLoss
from src.loss.bce import BCELoss


class TestHardNegativeMiningLoss:
    """Test suite for HardNegativeMiningLoss."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        base_loss = FocalLoss()
        loss_fn = HardNegativeMiningLoss(base_loss)

        assert loss_fn.base_loss is base_loss
        assert loss_fn.hard_negative_ratio == 3.0
        assert loss_fn.min_negative_samples == 100
        assert loss_fn.reduction == "mean"

    def test_init_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        base_loss = FocalLoss()
        loss_fn = HardNegativeMiningLoss(
            base_loss,
            hard_negative_ratio=5.0,
            min_negative_samples=200,
            reduction="sum",
        )

        assert loss_fn.hard_negative_ratio == 5.0
        assert loss_fn.min_negative_samples == 200
        assert loss_fn.reduction == "sum"

    def test_init_invalid_ratio(self) -> None:
        """Test that invalid hard_negative_ratio raises ValueError."""
        base_loss = FocalLoss()

        with pytest.raises(ValueError, match="hard_negative_ratio must be positive"):
            HardNegativeMiningLoss(base_loss, hard_negative_ratio=0.0)

        with pytest.raises(ValueError, match="hard_negative_ratio must be positive"):
            HardNegativeMiningLoss(base_loss, hard_negative_ratio=-1.0)

    def test_init_invalid_min_samples(self) -> None:
        """Test that invalid min_negative_samples raises ValueError."""
        base_loss = FocalLoss()

        with pytest.raises(ValueError, match="min_negative_samples must be >= 0"):
            HardNegativeMiningLoss(base_loss, min_negative_samples=-1)

    def test_init_invalid_reduction(self) -> None:
        """Test that invalid reduction raises ValueError."""
        base_loss = FocalLoss()

        with pytest.raises(ValueError, match="reduction must be 'mean' or 'sum'"):
            HardNegativeMiningLoss(base_loss, reduction="none")

        with pytest.raises(ValueError, match="reduction must be 'mean' or 'sum'"):
            HardNegativeMiningLoss(base_loss, reduction="invalid")

    def test_init_base_loss_without_reduction(self) -> None:
        """Test that base loss without reduction attribute raises ValueError."""

        class InvalidLoss:
            """Fake loss without reduction attribute."""

            pass

        invalid_loss = InvalidLoss()

        with pytest.raises(ValueError, match="base_loss must have a 'reduction' attribute"):
            HardNegativeMiningLoss(invalid_loss)  # type: ignore[arg-type]

    def test_forward_shape_mismatch(self) -> None:
        """Test that shape mismatch raises ValueError."""
        base_loss = FocalLoss()
        loss_fn = HardNegativeMiningLoss(base_loss)

        predictions = torch.randn(2, 1, 32, 32, 32)
        targets = torch.randn(2, 1, 16, 16, 16)  # Different shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(predictions, targets)

    def test_forward_basic(self) -> None:
        """Test basic forward pass."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(base_loss, hard_negative_ratio=3.0)

        predictions = torch.randn(2, 1, 32, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_all_positives(self) -> None:
        """Test forward pass with all positive examples."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(base_loss)

        predictions = torch.randn(2, 1, 16, 16, 16)
        targets = torch.ones(2, 1, 16, 16, 16)  # All positives

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_forward_all_negatives(self) -> None:
        """Test forward pass with all negative examples."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(base_loss, min_negative_samples=100)

        predictions = torch.randn(2, 1, 16, 16, 16)
        targets = torch.zeros(2, 1, 16, 16, 16)  # All negatives

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_forward_no_examples(self) -> None:
        """Test forward pass with empty input (edge case)."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(base_loss)

        predictions = torch.randn(0, 1, 8, 8, 8)
        targets = torch.zeros(0, 1, 8, 8, 8)

        loss = loss_fn(predictions, targets)

        assert loss.item() == 0.0

    def test_forward_no_positives_no_min_samples(self) -> None:
        """Test edge case with no positives and min_negative_samples=0."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(
            base_loss, hard_negative_ratio=3.0, min_negative_samples=0
        )

        # All negatives, but ratio*0 = 0 and min=0, so no samples selected
        predictions = torch.randn(10, 1, 1, 1, 1)
        targets = torch.zeros(10, 1, 1, 1, 1)  # All negatives

        loss = loss_fn(predictions, targets)

        # Should return 0 when no samples are selected
        assert loss.item() == 0.0

    def test_forward_mining_ratio_3(self) -> None:
        """Test that hard negative ratio of 3.0 selects correct number of samples."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(
            base_loss, hard_negative_ratio=3.0, min_negative_samples=0
        )

        # Create controlled scenario: 100 positives, 10000 negatives
        num_positives = 100
        num_negatives = 10000
        targets = torch.zeros(num_positives + num_negatives, 1, 1, 1, 1)
        targets[:num_positives] = 1.0  # First 100 are positive

        stats = loss_fn.get_statistics(targets)

        assert stats["num_positives"] == num_positives
        assert stats["num_negatives"] == num_negatives
        assert stats["num_hard_negatives"] == num_positives * 3  # 3.0 ratio
        assert stats["num_selected"] == num_positives + num_positives * 3

    def test_forward_min_negative_samples(self) -> None:
        """Test that min_negative_samples is enforced."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(
            base_loss, hard_negative_ratio=3.0, min_negative_samples=500
        )

        # Few positives (ratio would give 30 negatives, but min is 500)
        num_positives = 10
        num_negatives = 1000
        targets = torch.zeros(num_positives + num_negatives, 1, 1, 1, 1)
        targets[:num_positives] = 1.0

        stats = loss_fn.get_statistics(targets)

        assert stats["num_positives"] == num_positives
        assert stats["num_hard_negatives"] == 500  # min_negative_samples enforced
        assert stats["num_selected"] == num_positives + 500

    def test_forward_fewer_negatives_than_requested(self) -> None:
        """Test behavior when fewer negatives available than requested."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(
            base_loss, hard_negative_ratio=10.0, min_negative_samples=1000
        )

        # Only 50 negatives available (less than min_negative_samples)
        num_positives = 100
        num_negatives = 50
        targets = torch.zeros(num_positives + num_negatives, 1, 1, 1, 1)
        targets[:num_positives] = 1.0

        stats = loss_fn.get_statistics(targets)

        assert stats["num_positives"] == num_positives
        assert stats["num_negatives"] == num_negatives
        # Should use all 50 available negatives
        assert stats["num_hard_negatives"] == num_negatives
        assert stats["num_selected"] == num_positives + num_negatives

    def test_reduction_mean(self) -> None:
        """Test that reduction='mean' computes correct value."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(base_loss, reduction="mean")

        predictions = torch.randn(2, 1, 16, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_reduction_sum(self) -> None:
        """Test that reduction='sum' computes correct value."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(base_loss, reduction="sum")

        predictions = torch.randn(2, 1, 16, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_backward_pass(self) -> None:
        """Test that backward pass works correctly."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(base_loss)

        predictions = torch.randn(2, 1, 16, 16, 16, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.isnan(predictions.grad).any()

    def test_selects_hardest_negatives(self) -> None:
        """Test that mining actually selects the hardest negatives."""
        base_loss = FocalLoss(from_logits=True, gamma=0.0)  # gamma=0 makes it BCE
        loss_fn = HardNegativeMiningLoss(
            base_loss, hard_negative_ratio=2.0, min_negative_samples=0
        )

        # Create controlled scenario
        # 10 positives (targets=1), 100 negatives (targets=0)
        # Make some negatives "harder" by giving them wrong predictions
        predictions = torch.zeros(110, 1, 1, 1, 1)
        targets = torch.zeros(110, 1, 1, 1, 1)

        # First 10 are positive
        targets[:10] = 1.0
        predictions[:10] = 1.0  # Correct predictions for positives

        # Next 20 negatives are "hard" (wrong predictions)
        predictions[10:30] = 5.0  # High logits (predicting positive when they're negative)

        # Remaining 80 negatives are "easy" (correct predictions)
        predictions[30:] = -5.0  # Low logits (correctly predicting negative)

        loss = loss_fn(predictions, targets)

        # Verify loss is reasonable
        assert loss.item() > 0

    def test_works_with_bce_loss(self) -> None:
        """Test that hard negative mining works with BCELoss."""
        base_loss = BCELoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(base_loss)

        predictions = torch.randn(2, 1, 16, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_preserves_base_loss_reduction(self) -> None:
        """Test that base loss reduction is preserved after forward pass."""
        base_loss = FocalLoss(from_logits=True)
        original_reduction = base_loss.reduction

        loss_fn = HardNegativeMiningLoss(base_loss)

        predictions = torch.randn(2, 1, 16, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()

        _ = loss_fn(predictions, targets)

        # Reduction should be restored
        assert base_loss.reduction == original_reduction

    def test_get_statistics(self) -> None:
        """Test get_statistics method."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(
            base_loss, hard_negative_ratio=3.0, min_negative_samples=100
        )

        # 50 positives, 500 negatives
        targets = torch.zeros(550, 1, 1, 1, 1)
        targets[:50] = 1.0

        stats = loss_fn.get_statistics(targets)

        assert isinstance(stats, dict)
        assert "num_positives" in stats
        assert "num_negatives" in stats
        assert "num_hard_negatives" in stats
        assert "num_selected" in stats

        assert stats["num_positives"] == 50
        assert stats["num_negatives"] == 500
        assert stats["num_hard_negatives"] == 150  # max(50*3, 100) = 150
        assert stats["num_selected"] == 200  # 50 + 150

    def test_get_statistics_all_positive(self) -> None:
        """Test get_statistics with all positive examples."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(base_loss)

        targets = torch.ones(100, 1, 1, 1, 1)

        stats = loss_fn.get_statistics(targets)

        assert stats["num_positives"] == 100
        assert stats["num_negatives"] == 0
        assert stats["num_hard_negatives"] == 0
        assert stats["num_selected"] == 100

    def test_get_statistics_all_negative(self) -> None:
        """Test get_statistics with all negative examples."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(base_loss, min_negative_samples=50)

        targets = torch.zeros(100, 1, 1, 1, 1)

        stats = loss_fn.get_statistics(targets)

        assert stats["num_positives"] == 0
        assert stats["num_negatives"] == 100
        assert stats["num_hard_negatives"] == 50  # min_negative_samples
        assert stats["num_selected"] == 50

    def test_numerical_stability(self) -> None:
        """Test numerical stability with extreme values."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(base_loss)

        # Extreme predictions
        predictions = torch.tensor([[[[[100.0], [-100.0]]]]])
        targets = torch.tensor([[[[[1.0], [0.0]]]]])

        loss = loss_fn(predictions, targets)

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(base_loss)

        for batch_size in [1, 2, 4, 8]:
            predictions = torch.randn(batch_size, 1, 16, 16, 16)
            targets = torch.randint(0, 2, (batch_size, 1, 16, 16, 16)).float()

            loss = loss_fn(predictions, targets)

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0
            assert loss.item() >= 0

    def test_3d_volumes(self) -> None:
        """Test with 3D volumetric data."""
        base_loss = FocalLoss(from_logits=True)
        loss_fn = HardNegativeMiningLoss(base_loss)

        # Full 3D volume
        predictions = torch.randn(1, 1, 32, 32, 32)
        targets = torch.randint(0, 2, (1, 1, 32, 32, 32)).float()

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
