"""Tests for loss functions."""

from __future__ import annotations

import pytest
import torch

from src.loss.bce import BCELoss
from src.loss.dice import DiceLoss


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
