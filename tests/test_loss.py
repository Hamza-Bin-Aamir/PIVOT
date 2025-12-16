"""Tests for loss functions."""

from __future__ import annotations

import pytest
import torch

from src.loss.bce import BCELoss
from src.loss.dice import DiceLoss
from src.loss.focal import FocalLoss
from src.loss.smooth_l1 import SmoothL1Loss


class TestDiceLoss:
    """Test suite for DiceLoss."""

    def test_init_default(self):
        """Test default initialization."""
        loss_fn = DiceLoss()

        assert loss_fn.smooth == 1.0
        assert loss_fn.from_logits is True
        assert loss_fn.reduction == "mean"

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        loss_fn = DiceLoss(smooth=0.5, from_logits=False, reduction="sum")

        assert loss_fn.smooth == 0.5
        assert loss_fn.from_logits is False
        assert loss_fn.reduction == "sum"

    def test_init_invalid_reduction(self):
        """Test initialization with invalid reduction."""
        with pytest.raises(ValueError, match="Invalid reduction"):
            DiceLoss(reduction="invalid")

    def test_binary_segmentation_perfect_match(self):
        """Test binary segmentation with perfect predictions."""
        loss_fn = DiceLoss(from_logits=False)

        # Perfect match: predictions == targets
        predictions = torch.ones(2, 1, 8, 8, 8)
        targets = torch.ones(2, 1, 8, 8, 8)

        loss = loss_fn(predictions, targets)

        # Perfect Dice = 1, so loss = 1 - 1 = 0
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_binary_segmentation_no_overlap(self):
        """Test binary segmentation with no overlap."""
        loss_fn = DiceLoss(from_logits=False, smooth=0.0)

        # No overlap
        predictions = torch.ones(2, 1, 8, 8, 8)
        targets = torch.zeros(2, 1, 8, 8, 8)

        loss = loss_fn(predictions, targets)

        # Dice = 0, so loss = 1 - 0 = 1
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)

    def test_binary_segmentation_partial_overlap(self):
        """Test binary segmentation with partial overlap."""
        loss_fn = DiceLoss(from_logits=False, smooth=1.0)

        # Create partial overlap
        predictions = torch.zeros(1, 1, 4, 4, 4)
        predictions[0, 0, :2, :2, :2] = 1.0  # 8 voxels

        targets = torch.zeros(1, 1, 4, 4, 4)
        targets[0, 0, 1:3, 1:3, 1:3] = 1.0  # 8 voxels, 1 voxel overlap

        loss = loss_fn(predictions, targets)

        # Intersection = 1, Union = 8 + 8 = 16
        # Dice = (2*1 + 1) / (16 + 1) = 3/17
        # Loss = 1 - 3/17 ≈ 0.824
        expected_dice = (2.0 * 1.0 + 1.0) / (8.0 + 8.0 + 1.0)
        expected_loss = 1.0 - expected_dice

        assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-4)

    def test_binary_segmentation_with_logits(self):
        """Test binary segmentation with logits."""
        loss_fn = DiceLoss(from_logits=True)

        # High positive logits -> sigmoid ≈ 1
        predictions = torch.ones(2, 1, 4, 4, 4) * 10.0
        targets = torch.ones(2, 1, 4, 4, 4)

        loss = loss_fn(predictions, targets)

        # Should be close to 0 (perfect match after sigmoid)
        assert loss < 0.01

    def test_multi_class_segmentation(self):
        """Test multi-class segmentation."""
        loss_fn = DiceLoss(from_logits=False)

        # 3 classes
        predictions = torch.zeros(1, 3, 4, 4, 4)
        predictions[0, 0, :2, :, :] = 1.0  # Class 0
        predictions[0, 1, 2:, :, :] = 1.0  # Class 1
        # Class 2 all zeros

        targets = torch.zeros(1, 3, 4, 4, 4)
        targets[0, 0, :2, :, :] = 1.0  # Class 0 matches
        targets[0, 1, 2:, :, :] = 1.0  # Class 1 matches

        loss = loss_fn(predictions, targets)

        # Classes 0 and 1 have perfect Dice, class 2 has Dice=smooth/(2*smooth)
        # Average Dice across classes should be high
        assert loss < 0.5

    def test_reduction_none(self):
        """Test reduction='none' returns per-sample losses."""
        loss_fn = DiceLoss(from_logits=False, reduction="none")

        predictions = torch.randn(4, 1, 4, 4, 4).sigmoid()
        targets = torch.randint(0, 2, (4, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)

        assert loss.shape == (4,)  # One loss per sample
        assert (loss >= 0).all() and (loss <= 1).all()

    def test_reduction_sum(self):
        """Test reduction='sum' returns sum of losses."""
        loss_fn_sum = DiceLoss(from_logits=False, reduction="sum")
        loss_fn_none = DiceLoss(from_logits=False, reduction="none")

        predictions = torch.randn(4, 1, 4, 4, 4).sigmoid()
        targets = torch.randint(0, 2, (4, 1, 4, 4, 4)).float()

        loss_sum = loss_fn_sum(predictions, targets)
        loss_none = loss_fn_none(predictions, targets)

        expected_sum = loss_none.sum()
        assert torch.isclose(loss_sum, expected_sum)

    def test_reduction_mean(self):
        """Test reduction='mean' returns mean of losses."""
        loss_fn_mean = DiceLoss(from_logits=False, reduction="mean")
        loss_fn_none = DiceLoss(from_logits=False, reduction="none")

        predictions = torch.randn(4, 1, 4, 4, 4).sigmoid()
        targets = torch.randint(0, 2, (4, 1, 4, 4, 4)).float()

        loss_mean = loss_fn_mean(predictions, targets)
        loss_none = loss_fn_none(predictions, targets)

        expected_mean = loss_none.mean()
        assert torch.isclose(loss_mean, expected_mean)

    def test_shape_mismatch_error(self):
        """Test error on shape mismatch."""
        loss_fn = DiceLoss()

        predictions = torch.randn(2, 1, 8, 8, 8)
        targets = torch.randn(2, 1, 4, 4, 4)  # Wrong shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(predictions, targets)

    def test_gradient_flow(self):
        """Test gradient flow through loss."""
        loss_fn = DiceLoss(from_logits=True)

        predictions = torch.randn(2, 1, 4, 4, 4, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.all(predictions.grad == 0)

    def test_2d_images(self):
        """Test with 2D images (B, C, H, W)."""
        loss_fn = DiceLoss(from_logits=False)

        predictions = torch.randn(2, 1, 64, 64).sigmoid()
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0 and loss <= 1

    def test_3d_volumes(self):
        """Test with 3D volumes (B, C, D, H, W)."""
        loss_fn = DiceLoss(from_logits=False)

        predictions = torch.randn(2, 1, 16, 16, 16).sigmoid()
        targets = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0 and loss <= 1

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        loss_fn = DiceLoss(from_logits=False)

        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            predictions = torch.randn(batch_size, 1, 8, 8, 8).sigmoid()
            targets = torch.randint(0, 2, (batch_size, 1, 8, 8, 8)).float()

            loss = loss_fn(predictions, targets)
            assert loss.ndim == 0  # Always returns scalar with mean reduction

    def test_smoothing_prevents_nan(self):
        """Test that smoothing prevents NaN with empty predictions/targets."""
        loss_fn = DiceLoss(from_logits=False, smooth=1.0)

        # Both predictions and targets are zeros
        predictions = torch.zeros(2, 1, 4, 4, 4)
        targets = torch.zeros(2, 1, 4, 4, 4)

        loss = loss_fn(predictions, targets)

        # With smoothing, Dice = (0 + 1) / (0 + 1) = 1, loss = 0
        assert not torch.isnan(loss)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_no_smoothing_edge_case(self):
        """Test behavior without smoothing on edge case."""
        loss_fn = DiceLoss(from_logits=False, smooth=0.0)

        # Both empty
        predictions = torch.zeros(1, 1, 4, 4, 4)
        targets = torch.zeros(1, 1, 4, 4, 4)

        loss = loss_fn(predictions, targets)

        # 0/0 = nan without smoothing
        assert torch.isnan(loss)

    def test_multi_class_with_logits(self):
        """Test multi-class segmentation with logits."""
        loss_fn = DiceLoss(from_logits=True)

        # 3 classes, use softmax
        logits = torch.randn(2, 3, 8, 8, 8)
        targets = torch.zeros(2, 3, 8, 8, 8)
        targets[:, 0, :4, :, :] = 1.0  # First half is class 0
        targets[:, 1, 4:, :, :] = 1.0  # Second half is class 1

        loss = loss_fn(logits, targets)

        assert loss >= 0 and loss <= 1

    def test_deterministic_output(self):
        """Test deterministic behavior."""
        loss_fn = DiceLoss(from_logits=False)

        predictions = torch.randn(2, 1, 8, 8, 8).sigmoid()
        targets = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()

        loss1 = loss_fn(predictions, targets)
        loss2 = loss_fn(predictions, targets)

        assert torch.equal(loss1, loss2)

    def test_loss_range(self):
        """Test that loss is always in [0, 1]."""
        loss_fn = DiceLoss(from_logits=False)

        # Test with random data
        for _ in range(10):
            predictions = torch.randn(2, 1, 8, 8, 8).sigmoid()
            targets = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()

            loss = loss_fn(predictions, targets)

            assert loss >= 0 and loss <= 1

    def test_single_class_binary(self):
        """Test single-class binary segmentation."""
        loss_fn = DiceLoss(from_logits=True)

        # Single channel (binary)
        predictions = torch.randn(2, 1, 8, 8, 8)
        targets = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()

        loss = loss_fn(predictions, targets)

        assert loss >= 0 and loss <= 1

    def test_integration_with_model(self):
        """Test integration with segmentation head output."""
        from src.model.unet import SegmentationHead

        loss_fn = DiceLoss(from_logits=True)
        head = SegmentationHead(in_channels=32, apply_sigmoid=False)

        features = torch.randn(2, 32, 16, 16, 16)
        predictions = head(features)  # Logits
        targets = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()

        loss = loss_fn(predictions, targets)

        assert loss >= 0 and loss <= 1
        assert not torch.isnan(loss)

    def test_backward_pass_with_model(self):
        """Test backward pass through model and loss."""
        from src.model.unet import SegmentationHead

        loss_fn = DiceLoss(from_logits=True)
        head = SegmentationHead(in_channels=32, apply_sigmoid=False)

        features = torch.randn(2, 32, 8, 8, 8, requires_grad=True)
        predictions = head(features)
        targets = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()

        loss = loss_fn(predictions, targets)
        loss.backward()

        # Check gradients exist
        assert features.grad is not None
        for param in head.parameters():
            assert param.grad is not None


class TestBCELoss:
    """Test suite for BCELoss."""

    def test_init_default(self):
        """Test default initialization."""
        loss_fn = BCELoss()

        assert loss_fn.pos_weight is None
        assert loss_fn.from_logits is True
        assert loss_fn.reduction == "mean"

    def test_init_with_pos_weight(self):
        """Test initialization with positive weight."""
        pos_weight = torch.tensor([2.0])
        loss_fn = BCELoss(pos_weight=pos_weight)

        assert torch.equal(loss_fn.pos_weight, pos_weight)
        assert loss_fn.from_logits is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        loss_fn = BCELoss(from_logits=False, reduction="sum")

        assert loss_fn.from_logits is False
        assert loss_fn.reduction == "sum"

    def test_init_invalid_reduction(self):
        """Test initialization with invalid reduction."""
        with pytest.raises(ValueError, match="Invalid reduction"):
            BCELoss(reduction="invalid")

    def test_binary_perfect_predictions(self):
        """Test with perfect binary predictions."""
        loss_fn = BCELoss(from_logits=False)

        # Perfect predictions
        predictions = torch.ones(2, 1, 8, 8, 8)
        targets = torch.ones(2, 1, 8, 8, 8)

        loss = loss_fn(predictions, targets)

        # Perfect match should give loss ≈ 0
        assert loss < 0.01

    def test_binary_worst_predictions(self):
        """Test with worst binary predictions."""
        loss_fn = BCELoss(from_logits=False)

        # Worst predictions (opposite of targets)
        predictions = torch.ones(2, 1, 8, 8, 8) * 0.999  # Close to 1
        targets = torch.zeros(2, 1, 8, 8, 8)  # All zeros

        loss = loss_fn(predictions, targets)

        # Should have high loss
        assert loss > 5.0

    def test_with_logits(self):
        """Test BCE with logits."""
        loss_fn = BCELoss(from_logits=True)

        # High positive logits -> sigmoid ≈ 1
        predictions = torch.ones(2, 1, 4, 4, 4) * 10.0
        targets = torch.ones(2, 1, 4, 4, 4)

        loss = loss_fn(predictions, targets)

        # Should be close to 0 (perfect match after sigmoid)
        assert loss < 0.01

    def test_with_pos_weight_logits(self):
        """Test BCE with positive weight and logits."""
        pos_weight = torch.tensor([2.0])
        loss_fn = BCELoss(pos_weight=pos_weight, from_logits=True)

        predictions = torch.randn(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)

        # Should be finite and positive
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_with_pos_weight_probabilities(self):
        """Test BCE with positive weight and probabilities."""
        pos_weight = torch.tensor([2.0])
        loss_fn = BCELoss(pos_weight=pos_weight, from_logits=False)

        predictions = torch.rand(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_reduction_none(self):
        """Test reduction='none' returns per-element losses."""
        loss_fn = BCELoss(from_logits=False, reduction="none")

        predictions = torch.rand(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)

        # Should return same shape as input
        assert loss.shape == predictions.shape
        assert (loss >= 0).all()

    def test_reduction_sum(self):
        """Test reduction='sum' returns sum of losses."""
        loss_fn_sum = BCELoss(from_logits=False, reduction="sum")
        loss_fn_none = BCELoss(from_logits=False, reduction="none")

        predictions = torch.rand(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss_sum = loss_fn_sum(predictions, targets)
        loss_none = loss_fn_none(predictions, targets)

        expected_sum = loss_none.sum()
        assert torch.isclose(loss_sum, expected_sum, rtol=1e-4)

    def test_reduction_mean(self):
        """Test reduction='mean' returns mean of losses."""
        loss_fn_mean = BCELoss(from_logits=False, reduction="mean")
        loss_fn_none = BCELoss(from_logits=False, reduction="none")

        predictions = torch.rand(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss_mean = loss_fn_mean(predictions, targets)
        loss_none = loss_fn_none(predictions, targets)

        expected_mean = loss_none.mean()
        assert torch.isclose(loss_mean, expected_mean, rtol=1e-4)

    def test_shape_mismatch_error(self):
        """Test error on shape mismatch."""
        loss_fn = BCELoss()

        predictions = torch.randn(2, 1, 8, 8, 8)
        targets = torch.randn(2, 1, 4, 4, 4)  # Wrong shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(predictions, targets)

    def test_gradient_flow(self):
        """Test gradient flow through loss."""
        loss_fn = BCELoss(from_logits=True)

        predictions = torch.randn(2, 1, 4, 4, 4, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.all(predictions.grad == 0)

    def test_2d_images(self):
        """Test with 2D images (B, C, H, W)."""
        loss_fn = BCELoss(from_logits=False)

        predictions = torch.rand(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_3d_volumes(self):
        """Test with 3D volumes (B, C, D, H, W)."""
        loss_fn = BCELoss(from_logits=False)

        predictions = torch.rand(2, 1, 16, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_multi_channel(self):
        """Test with multi-channel predictions."""
        loss_fn = BCELoss(from_logits=False)

        # 3 channels (e.g., 3 independent binary classifications)
        predictions = torch.rand(2, 3, 8, 8, 8)
        targets = torch.randint(0, 2, (2, 3, 8, 8, 8)).float()

        loss = loss_fn(predictions, targets)

        assert loss >= 0

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        loss_fn = BCELoss(from_logits=False)

        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            predictions = torch.rand(batch_size, 1, 8, 8, 8)
            targets = torch.randint(0, 2, (batch_size, 1, 8, 8, 8)).float()

            loss = loss_fn(predictions, targets)
            assert loss.ndim == 0  # Scalar with mean reduction

    def test_numerical_stability_logits(self):
        """Test numerical stability with extreme logits."""
        loss_fn = BCELoss(from_logits=True)

        # Very large positive and negative logits
        predictions = torch.randn(2, 1, 4, 4, 4) * 100
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_numerical_stability_probabilities(self):
        """Test numerical stability with edge probabilities."""
        loss_fn = BCELoss(from_logits=False)

        # Probabilities very close to 0 and 1
        predictions = torch.rand(2, 1, 4, 4, 4)
        predictions[0] = 0.0001
        predictions[1] = 0.9999
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_pos_weight_effect(self):
        """Test that pos_weight affects loss magnitude."""
        loss_fn_no_weight = BCELoss(from_logits=True, pos_weight=None)
        loss_fn_with_weight = BCELoss(from_logits=True, pos_weight=torch.tensor([3.0]))

        # Predictions and targets with positive samples
        predictions = torch.randn(2, 1, 4, 4, 4)
        targets = torch.ones(2, 1, 4, 4, 4)  # All positive

        loss_no_weight = loss_fn_no_weight(predictions, targets)
        loss_with_weight = loss_fn_with_weight(predictions, targets)

        # With positive samples and pos_weight > 1, loss should be higher
        # (or at least different)
        assert not torch.isclose(loss_no_weight, loss_with_weight)

    def test_deterministic_output(self):
        """Test deterministic behavior."""
        loss_fn = BCELoss(from_logits=False)

        predictions = torch.rand(2, 1, 8, 8, 8)
        targets = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()

        loss1 = loss_fn(predictions, targets)
        loss2 = loss_fn(predictions, targets)

        assert torch.equal(loss1, loss2)

    def test_integration_with_segmentation_head(self):
        """Test integration with segmentation head."""
        from src.model.unet import SegmentationHead

        loss_fn = BCELoss(from_logits=True)
        head = SegmentationHead(in_channels=32, apply_sigmoid=False)

        features = torch.randn(2, 32, 16, 16, 16)
        predictions = head(features)  # Logits
        targets = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()

        loss = loss_fn(predictions, targets)

        assert loss >= 0
        assert not torch.isnan(loss)

    def test_backward_pass_with_model(self):
        """Test backward pass through model and loss."""
        from src.model.unet import SegmentationHead

        loss_fn = BCELoss(from_logits=True)
        head = SegmentationHead(in_channels=32, apply_sigmoid=False)

        features = torch.randn(2, 32, 8, 8, 8, requires_grad=True)
        predictions = head(features)
        targets = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()

        loss = loss_fn(predictions, targets)
        loss.backward()

        # Check gradients exist
        assert features.grad is not None
        for param in head.parameters():
            assert param.grad is not None

    def test_comparison_with_pytorch_bce(self):
        """Test that our implementation matches PyTorch's BCE."""
        import torch.nn as nn

        loss_fn_ours = BCELoss(from_logits=True, reduction="mean")
        loss_fn_pytorch = nn.BCEWithLogitsLoss(reduction="mean")

        predictions = torch.randn(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss_ours = loss_fn_ours(predictions, targets)
        loss_pytorch = loss_fn_pytorch(predictions, targets)

        # Should be virtually identical
        assert torch.isclose(loss_ours, loss_pytorch, rtol=1e-5)

    def test_comparison_with_pytorch_bce_pos_weight(self):
        """Test pos_weight matches PyTorch implementation."""
        import torch.nn as nn

        pos_weight = torch.tensor([2.5])
        loss_fn_ours = BCELoss(from_logits=True, pos_weight=pos_weight, reduction="mean")
        loss_fn_pytorch = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

        predictions = torch.randn(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss_ours = loss_fn_ours(predictions, targets)
        loss_pytorch = loss_fn_pytorch(predictions, targets)

        # Should match PyTorch implementation
        assert torch.isclose(loss_ours, loss_pytorch, rtol=1e-5)


class TestFocalLoss:
    """Test suite for FocalLoss."""

    def test_init_default(self):
        """Test default initialization."""
        loss_fn = FocalLoss()

        assert loss_fn.alpha == 0.25
        assert loss_fn.gamma == 2.0
        assert loss_fn.from_logits is True
        assert loss_fn.reduction == "mean"

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        loss_fn = FocalLoss(alpha=0.5, gamma=3.0, from_logits=False, reduction="sum")

        assert loss_fn.alpha == 0.5
        assert loss_fn.gamma == 3.0
        assert loss_fn.from_logits is False
        assert loss_fn.reduction == "sum"

    def test_init_with_tensor_alpha(self):
        """Test initialization with tensor alpha."""
        alpha = torch.tensor([0.25, 0.5, 0.25])
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)

        assert torch.equal(loss_fn.alpha, alpha)
        assert loss_fn.gamma == 2.0

    def test_init_invalid_reduction(self):
        """Test initialization with invalid reduction."""
        with pytest.raises(ValueError, match="Invalid reduction"):
            FocalLoss(reduction="invalid")

    def test_init_invalid_gamma(self):
        """Test initialization with invalid gamma."""
        with pytest.raises(ValueError, match="gamma must be >= 0"):
            FocalLoss(gamma=-1.0)

    def test_init_invalid_alpha(self):
        """Test initialization with invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            FocalLoss(alpha=1.5)

    def test_binary_perfect_predictions(self):
        """Test with perfect binary predictions."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=False)

        # Perfect predictions (probability = target)
        predictions = torch.ones(2, 1, 8, 8, 8)
        targets = torch.ones(2, 1, 8, 8, 8)

        loss = loss_fn(predictions, targets)

        # Perfect match should give very low loss
        assert loss < 0.01

    def test_binary_with_logits(self):
        """Test binary focal loss with logits."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)

        predictions = torch.randn(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_binary_gamma_effect(self):
        """Test that gamma affects loss magnitude."""
        loss_fn_gamma0 = FocalLoss(alpha=0.25, gamma=0.0, from_logits=False)
        loss_fn_gamma2 = FocalLoss(alpha=0.25, gamma=2.0, from_logits=False)

        # Use somewhat confident but not perfect predictions
        predictions = torch.ones(2, 1, 4, 4, 4) * 0.8
        targets = torch.ones(2, 1, 4, 4, 4)

        loss_gamma0 = loss_fn_gamma0(predictions, targets)
        loss_gamma2 = loss_fn_gamma2(predictions, targets)

        # Higher gamma should reduce loss for easy examples
        assert loss_gamma2 < loss_gamma0

    def test_binary_alpha_effect(self):
        """Test that alpha affects loss magnitude."""
        loss_fn_alpha025 = FocalLoss(alpha=0.25, gamma=2.0, from_logits=False)
        loss_fn_alpha075 = FocalLoss(alpha=0.75, gamma=2.0, from_logits=False)

        predictions = torch.rand(2, 1, 4, 4, 4)
        targets = torch.ones(2, 1, 4, 4, 4)  # All positive

        loss_alpha025 = loss_fn_alpha025(predictions, targets)
        loss_alpha075 = loss_fn_alpha075(predictions, targets)

        # Different alpha should give different loss
        assert not torch.isclose(loss_alpha025, loss_alpha075)

    def test_multiclass_with_logits(self):
        """Test multi-class focal loss with logits."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)

        predictions = torch.randn(2, 3, 8, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8, 8)).long()

        loss = loss_fn(predictions, targets)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_multiclass_with_probabilities(self):
        """Test multi-class focal loss with probabilities."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=False)

        # Create valid probability distribution
        logits = torch.randn(2, 3, 8, 8, 8)
        predictions = torch.softmax(logits, dim=1)
        targets = torch.randint(0, 3, (2, 8, 8, 8)).long()

        loss = loss_fn(predictions, targets)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_multiclass_tensor_alpha(self):
        """Test multi-class with tensor alpha (per-class weights)."""
        alpha = torch.tensor([0.25, 0.5, 0.25])
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0, from_logits=True)

        predictions = torch.randn(2, 3, 4, 4, 4)
        targets = torch.randint(0, 3, (2, 4, 4, 4)).long()

        loss = loss_fn(predictions, targets)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_reduction_none_binary(self):
        """Test reduction='none' for binary classification."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")

        predictions = torch.randn(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)

        # Should return same shape as input
        assert loss.shape == predictions.shape
        assert (loss >= 0).all()

    def test_reduction_none_multiclass(self):
        """Test reduction='none' for multi-class classification."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")

        predictions = torch.randn(2, 3, 4, 4, 4)
        targets = torch.randint(0, 3, (2, 4, 4, 4)).long()

        loss = loss_fn(predictions, targets)

        # Should return shape (B, *) without class dimension
        expected_shape = (2, 4, 4, 4)
        assert loss.shape == expected_shape
        assert (loss >= 0).all()

    def test_reduction_sum_binary(self):
        """Test reduction='sum' for binary classification."""
        loss_fn_sum = FocalLoss(alpha=0.25, gamma=2.0, reduction="sum")
        loss_fn_none = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")

        predictions = torch.randn(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss_sum = loss_fn_sum(predictions, targets)
        loss_none = loss_fn_none(predictions, targets)

        expected_sum = loss_none.sum()
        assert torch.isclose(loss_sum, expected_sum, rtol=1e-4)

    def test_reduction_mean_binary(self):
        """Test reduction='mean' for binary classification."""
        loss_fn_mean = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        loss_fn_none = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")

        predictions = torch.randn(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss_mean = loss_fn_mean(predictions, targets)
        loss_none = loss_fn_none(predictions, targets)

        expected_mean = loss_none.mean()
        assert torch.isclose(loss_mean, expected_mean, rtol=1e-4)

    def test_shape_mismatch_binary(self):
        """Test error on shape mismatch for binary."""
        loss_fn = FocalLoss()

        predictions = torch.randn(2, 1, 8, 8, 8)
        targets = torch.randn(2, 1, 4, 4, 4)  # Wrong shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(predictions, targets)

    def test_gradient_flow_binary(self):
        """Test gradient flow for binary classification."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)

        predictions = torch.randn(2, 1, 4, 4, 4, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.all(predictions.grad == 0)

    def test_gradient_flow_multiclass(self):
        """Test gradient flow for multi-class classification."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)

        predictions = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
        targets = torch.randint(0, 3, (2, 4, 4, 4)).long()

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.all(predictions.grad == 0)

    def test_2d_images_binary(self):
        """Test with 2D images for binary classification."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=False)

        predictions = torch.rand(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_3d_volumes_binary(self):
        """Test with 3D volumes for binary classification."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=False)

        predictions = torch.rand(2, 1, 16, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_3d_volumes_multiclass(self):
        """Test with 3D volumes for multi-class classification."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)

        predictions = torch.randn(2, 4, 16, 16, 16)
        targets = torch.randint(0, 4, (2, 16, 16, 16)).long()

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)

        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            predictions = torch.randn(batch_size, 1, 8, 8, 8)
            targets = torch.randint(0, 2, (batch_size, 1, 8, 8, 8)).float()

            loss = loss_fn(predictions, targets)
            assert loss.ndim == 0  # Scalar with mean reduction

    def test_numerical_stability_binary_logits(self):
        """Test numerical stability with extreme logits for binary."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)

        # Very large positive and negative logits
        predictions = torch.randn(2, 1, 4, 4, 4) * 100
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_numerical_stability_binary_probabilities(self):
        """Test numerical stability with edge probabilities for binary."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=False)

        # Probabilities very close to 0 and 1
        predictions = torch.rand(2, 1, 4, 4, 4)
        predictions[0] = 0.0001
        predictions[1] = 0.9999
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_numerical_stability_multiclass(self):
        """Test numerical stability for multi-class."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)

        # Very large logits
        predictions = torch.randn(2, 3, 4, 4, 4) * 100
        targets = torch.randint(0, 3, (2, 4, 4, 4)).long()

        loss = loss_fn(predictions, targets)

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_gamma_zero_equivalence_to_ce(self):
        """Test that gamma=0 is equivalent to standard cross-entropy."""
        import torch.nn as nn

        # Binary case
        focal_loss = FocalLoss(alpha=0.5, gamma=0.0, from_logits=True)
        bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

        predictions = torch.randn(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss_focal = focal_loss(predictions, targets)
        loss_bce = bce_loss(predictions, targets)

        # With gamma=0 and alpha=0.5, focal loss should be close to BCE
        # (within alpha scaling factor of 0.5)
        assert torch.isclose(loss_focal, loss_bce * 0.5, rtol=1e-3)

    def test_deterministic_output_binary(self):
        """Test deterministic behavior for binary."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)

        predictions = torch.randn(2, 1, 8, 8, 8)
        targets = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()

        loss1 = loss_fn(predictions, targets)
        loss2 = loss_fn(predictions, targets)

        assert torch.equal(loss1, loss2)

    def test_deterministic_output_multiclass(self):
        """Test deterministic behavior for multi-class."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)

        predictions = torch.randn(2, 3, 8, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8, 8)).long()

        loss1 = loss_fn(predictions, targets)
        loss2 = loss_fn(predictions, targets)

        assert torch.equal(loss1, loss2)

    def test_integration_with_center_detection_head(self):
        """Test integration with center detection head."""
        from src.model.unet import CenterDetectionHead

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)
        head = CenterDetectionHead(in_channels=32, apply_sigmoid=False)

        features = torch.randn(2, 32, 16, 16, 16)
        predictions = head(features)  # Logits
        targets = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()

        loss = loss_fn(predictions, targets)

        assert loss >= 0
        assert not torch.isnan(loss)

    def test_backward_pass_with_model(self):
        """Test backward pass through model and loss."""
        from src.model.unet import CenterDetectionHead

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)
        head = CenterDetectionHead(in_channels=32, apply_sigmoid=False)

        features = torch.randn(2, 32, 8, 8, 8, requires_grad=True)
        predictions = head(features)
        targets = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()

        loss = loss_fn(predictions, targets)
        loss.backward()

        # Check gradients exist
        assert features.grad is not None
        for param in head.parameters():
            assert param.grad is not None

    def test_high_gamma_focuses_on_hard_examples(self):
        """Test that high gamma focuses on hard examples."""
        loss_fn = FocalLoss(alpha=0.5, gamma=5.0, from_logits=False)

        # Easy example: high confidence correct prediction
        easy_pred = torch.tensor([[[[0.99]]]])
        easy_target = torch.tensor([[[[1.0]]]])

        # Hard example: low confidence correct prediction
        hard_pred = torch.tensor([[[[0.51]]]])
        hard_target = torch.tensor([[[[1.0]]]])

        loss_easy = loss_fn(easy_pred, easy_target)
        loss_hard = loss_fn(hard_pred, hard_target)

        # Hard example should have much higher loss due to high gamma
        assert loss_hard > loss_easy * 10

    def test_multiclass_different_num_classes(self):
        """Test multi-class with different number of classes."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)

        num_classes_list = [2, 3, 5, 10]
        for num_classes in num_classes_list:
            predictions = torch.randn(2, num_classes, 8, 8, 8)
            targets = torch.randint(0, num_classes, (2, 8, 8, 8)).long()

            loss = loss_fn(predictions, targets)
            assert torch.isfinite(loss)
            assert loss >= 0

    def test_binary_with_tensor_alpha(self):
        """Test binary classification with tensor alpha."""
        alpha = torch.tensor([0.75])
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0, from_logits=True)

        predictions = torch.randn(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()

        loss = loss_fn(predictions, targets)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_reduction_sum_multiclass(self):
        """Test reduction='sum' for multi-class classification."""
        loss_fn_sum = FocalLoss(alpha=0.25, gamma=2.0, reduction="sum")
        loss_fn_none = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")

        predictions = torch.randn(2, 3, 4, 4, 4)
        targets = torch.randint(0, 3, (2, 4, 4, 4)).long()

        loss_sum = loss_fn_sum(predictions, targets)
        loss_none = loss_fn_none(predictions, targets)

        expected_sum = loss_none.sum()
        assert torch.isclose(loss_sum, expected_sum, rtol=1e-4)


class TestSmoothL1Loss:
    """Test suite for SmoothL1Loss."""

    def test_init_default(self):
        """Test default initialization."""
        loss_fn = SmoothL1Loss()

        assert loss_fn.beta == 1.0
        assert loss_fn.reduction == "mean"

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        loss_fn = SmoothL1Loss(beta=0.5, reduction="sum")

        assert loss_fn.beta == 0.5
        assert loss_fn.reduction == "sum"

    def test_init_invalid_reduction(self):
        """Test initialization with invalid reduction."""
        with pytest.raises(ValueError, match="Invalid reduction"):
            SmoothL1Loss(reduction="invalid")

    def test_init_invalid_beta(self):
        """Test initialization with invalid beta."""
        with pytest.raises(ValueError, match="beta must be positive"):
            SmoothL1Loss(beta=0.0)

        with pytest.raises(ValueError, match="beta must be positive"):
            SmoothL1Loss(beta=-1.0)

    def test_perfect_predictions(self):
        """Test with perfect predictions (zero error)."""
        loss_fn = SmoothL1Loss(beta=1.0)

        predictions = torch.randn(2, 3, 8, 8, 8)
        targets = predictions.clone()

        loss = loss_fn(predictions, targets)

        # Perfect match should give zero loss
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_small_errors_quadratic(self):
        """Test that small errors (< beta) use quadratic loss."""
        beta = 1.0
        loss_fn = SmoothL1Loss(beta=beta, reduction="none")

        # Small errors (< beta)
        predictions = torch.tensor([[[[0.3]]]])
        targets = torch.tensor([[[[0.0]]]])

        loss = loss_fn(predictions, targets)
        expected = 0.5 * (0.3**2) / beta

        assert torch.isclose(loss, torch.tensor(expected), rtol=1e-5)

    def test_large_errors_linear(self):
        """Test that large errors (>= beta) use linear loss."""
        beta = 1.0
        loss_fn = SmoothL1Loss(beta=beta, reduction="none")

        # Large errors (>= beta)
        predictions = torch.tensor([[[[2.0]]]])
        targets = torch.tensor([[[[0.0]]]])

        loss = loss_fn(predictions, targets)
        expected = 2.0 - 0.5 * beta

        assert torch.isclose(loss, torch.tensor(expected), rtol=1e-5)

    def test_beta_threshold(self):
        """Test behavior exactly at beta threshold."""
        beta = 1.0
        loss_fn = SmoothL1Loss(beta=beta, reduction="none")

        # Error exactly at beta
        predictions = torch.tensor([[[[1.0]]]])
        targets = torch.tensor([[[[0.0]]]])

        loss = loss_fn(predictions, targets)

        # At threshold, linear formula applies: |diff| - 0.5 * beta
        expected = 1.0 - 0.5 * beta

        assert torch.isclose(loss, torch.tensor(expected), rtol=1e-5)

    def test_beta_effect_on_robustness(self):
        """Test that smaller beta transitions to linear regime earlier."""
        loss_fn_beta1 = SmoothL1Loss(beta=1.0, reduction="none")
        loss_fn_beta01 = SmoothL1Loss(beta=0.1, reduction="none")

        # Medium error that's in quadratic regime for beta=1.0 but linear for beta=0.1
        predictions = torch.tensor([[[[0.5]]]])
        targets = torch.tensor([[[[0.0]]]])

        loss_beta1 = loss_fn_beta1(predictions, targets)
        loss_beta01 = loss_fn_beta01(predictions, targets)

        # For diff=0.5: beta=1.0 is quadratic (0.125), beta=0.1 is linear (0.45)
        # Linear regime has lower growth rate for large errors
        assert loss_beta1 < loss_beta01  # Quadratic is smaller for medium errors

    def test_reduction_none(self):
        """Test reduction='none' returns per-element losses."""
        loss_fn = SmoothL1Loss(beta=1.0, reduction="none")

        predictions = torch.randn(2, 3, 4, 4, 4)
        targets = torch.randn(2, 3, 4, 4, 4)

        loss = loss_fn(predictions, targets)

        # Should return same shape as input
        assert loss.shape == predictions.shape
        assert (loss >= 0).all()

    def test_reduction_sum(self):
        """Test reduction='sum' returns sum of losses."""
        loss_fn_sum = SmoothL1Loss(beta=1.0, reduction="sum")
        loss_fn_none = SmoothL1Loss(beta=1.0, reduction="none")

        predictions = torch.randn(2, 3, 4, 4, 4)
        targets = torch.randn(2, 3, 4, 4, 4)

        loss_sum = loss_fn_sum(predictions, targets)
        loss_none = loss_fn_none(predictions, targets)

        expected_sum = loss_none.sum()
        assert torch.isclose(loss_sum, expected_sum, rtol=1e-5)

    def test_reduction_mean(self):
        """Test reduction='mean' returns mean of losses."""
        loss_fn_mean = SmoothL1Loss(beta=1.0, reduction="mean")
        loss_fn_none = SmoothL1Loss(beta=1.0, reduction="none")

        predictions = torch.randn(2, 3, 4, 4, 4)
        targets = torch.randn(2, 3, 4, 4, 4)

        loss_mean = loss_fn_mean(predictions, targets)
        loss_none = loss_fn_none(predictions, targets)

        expected_mean = loss_none.mean()
        assert torch.isclose(loss_mean, expected_mean, rtol=1e-5)

    def test_shape_mismatch_error(self):
        """Test error on shape mismatch."""
        loss_fn = SmoothL1Loss()

        predictions = torch.randn(2, 3, 8, 8, 8)
        targets = torch.randn(2, 3, 4, 4, 4)  # Wrong shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(predictions, targets)

    def test_gradient_flow(self):
        """Test gradient flow through loss."""
        loss_fn = SmoothL1Loss(beta=1.0)

        predictions = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
        targets = torch.randn(2, 3, 4, 4, 4)

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.all(predictions.grad == 0)

    def test_2d_images(self):
        """Test with 2D images (B, C, H, W)."""
        loss_fn = SmoothL1Loss(beta=1.0)

        predictions = torch.randn(2, 3, 64, 64)
        targets = torch.randn(2, 3, 64, 64)

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_3d_volumes(self):
        """Test with 3D volumes (B, C, D, H, W)."""
        loss_fn = SmoothL1Loss(beta=1.0)

        predictions = torch.randn(2, 3, 16, 16, 16)
        targets = torch.randn(2, 3, 16, 16, 16)

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_single_channel(self):
        """Test with single channel (e.g., diameter)."""
        loss_fn = SmoothL1Loss(beta=1.0)

        predictions = torch.randn(2, 1, 8, 8, 8)
        targets = torch.randn(2, 1, 8, 8, 8)

        loss = loss_fn(predictions, targets)

        assert loss >= 0

    def test_multi_channel(self):
        """Test with multi-channel (e.g., 3D size)."""
        loss_fn = SmoothL1Loss(beta=1.0)

        # 3 channels for (width, height, depth)
        predictions = torch.randn(2, 3, 8, 8, 8)
        targets = torch.randn(2, 3, 8, 8, 8)

        loss = loss_fn(predictions, targets)

        assert loss >= 0

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        loss_fn = SmoothL1Loss(beta=1.0)

        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            predictions = torch.randn(batch_size, 3, 8, 8, 8)
            targets = torch.randn(batch_size, 3, 8, 8, 8)

            loss = loss_fn(predictions, targets)
            assert loss.ndim == 0  # Scalar with mean reduction

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        loss_fn = SmoothL1Loss(beta=1.0)

        # Very large values
        predictions = torch.randn(2, 3, 4, 4, 4) * 1000
        targets = torch.randn(2, 3, 4, 4, 4) * 1000

        loss = loss_fn(predictions, targets)

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_symmetry(self):
        """Test that loss is symmetric (same for swapped inputs)."""
        loss_fn = SmoothL1Loss(beta=1.0)

        predictions = torch.randn(2, 3, 4, 4, 4)
        targets = torch.randn(2, 3, 4, 4, 4)

        loss1 = loss_fn(predictions, targets)
        loss2 = loss_fn(targets, predictions)

        assert torch.isclose(loss1, loss2)

    def test_deterministic_output(self):
        """Test deterministic behavior."""
        loss_fn = SmoothL1Loss(beta=1.0)

        predictions = torch.randn(2, 3, 8, 8, 8)
        targets = torch.randn(2, 3, 8, 8, 8)

        loss1 = loss_fn(predictions, targets)
        loss2 = loss_fn(predictions, targets)

        assert torch.equal(loss1, loss2)

    def test_integration_with_size_regression_head(self):
        """Test integration with size regression head."""
        from src.model.unet import SizeRegressionHead

        loss_fn = SmoothL1Loss(beta=1.0)
        head = SizeRegressionHead(in_channels=32)

        features = torch.randn(2, 32, 16, 16, 16)
        predictions = head(features)  # 3D size predictions (B, 3, 1, 1, 1)
        targets = torch.randn(2, 3, 1, 1, 1)  # Match pooled shape

        loss = loss_fn(predictions, targets)

        assert loss >= 0
        assert not torch.isnan(loss)

    def test_backward_pass_with_model(self):
        """Test backward pass through model and loss."""
        from src.model.unet import SizeRegressionHead

        loss_fn = SmoothL1Loss(beta=1.0)
        head = SizeRegressionHead(in_channels=32)

        features = torch.randn(2, 32, 8, 8, 8, requires_grad=True)
        predictions = head(features)
        targets = torch.randn(2, 3, 1, 1, 1)  # Match pooled shape

        loss = loss_fn(predictions, targets)
        loss.backward()

        # Check gradients exist
        assert features.grad is not None
        for param in head.parameters():
            assert param.grad is not None

    def test_comparison_with_pytorch_smooth_l1(self):
        """Test that our implementation matches PyTorch's SmoothL1Loss."""
        import torch.nn as nn

        beta = 1.0
        loss_fn_ours = SmoothL1Loss(beta=beta, reduction="mean")
        # PyTorch uses beta parameter for the threshold
        loss_fn_pytorch = nn.SmoothL1Loss(beta=beta, reduction="mean")

        predictions = torch.randn(2, 3, 4, 4, 4)
        targets = torch.randn(2, 3, 4, 4, 4)

        loss_ours = loss_fn_ours(predictions, targets)
        loss_pytorch = loss_fn_pytorch(predictions, targets)

        # Should match PyTorch implementation
        assert torch.isclose(loss_ours, loss_pytorch, rtol=1e-5)

    def test_comparison_with_pytorch_different_beta(self):
        """Test beta parameter matches PyTorch implementation."""
        import torch.nn as nn

        beta = 0.5
        loss_fn_ours = SmoothL1Loss(beta=beta, reduction="sum")
        loss_fn_pytorch = nn.SmoothL1Loss(beta=beta, reduction="sum")

        predictions = torch.randn(2, 3, 4, 4, 4)
        targets = torch.randn(2, 3, 4, 4, 4)

        loss_ours = loss_fn_ours(predictions, targets)
        loss_pytorch = loss_fn_pytorch(predictions, targets)

        # Should match PyTorch implementation
        assert torch.isclose(loss_ours, loss_pytorch, rtol=1e-5)

    def test_less_sensitive_to_outliers_than_mse(self):
        """Test that Smooth L1 is less sensitive to outliers than MSE."""
        import torch.nn as nn

        smooth_l1_loss = SmoothL1Loss(beta=1.0, reduction="mean")
        mse_loss = nn.MSELoss(reduction="mean")

        # Most predictions are good, one is a large outlier
        predictions = torch.tensor(
            [[[[1.0, 1.0, 1.0, 100.0]]]]  # One outlier
        )
        targets = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]]]])

        loss_smooth = smooth_l1_loss(predictions, targets)
        loss_mse = mse_loss(predictions, targets)

        # Smooth L1 should be much less than MSE due to linear scaling
        # MSE will be dominated by (100-1)^2 = 9801, while Smooth L1 is ~99
        assert loss_smooth < loss_mse / 10
