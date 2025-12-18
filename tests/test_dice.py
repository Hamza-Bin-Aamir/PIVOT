"""Tests for Dice coefficient and IoU metrics."""

import numpy as np
import torch

from src.eval.dice import (
    compute_batch_dice_scores,
    compute_batch_iou_scores,
    compute_dice_score,
    compute_iou,
    dice_to_iou,
    iou_to_dice,
)


class TestComputeDiceScore:
    """Tests for Dice coefficient calculation."""

    def test_perfect_overlap(self) -> None:
        """Test Dice with perfect overlap."""
        pred = np.ones((10, 10, 10), dtype=bool)
        gt = np.ones((10, 10, 10), dtype=bool)

        dice = compute_dice_score(pred, gt)
        assert dice == 1.0

    def test_no_overlap(self) -> None:
        """Test Dice with no overlap."""
        pred = np.zeros((10, 10, 10), dtype=bool)
        pred[:5, :, :] = True
        gt = np.zeros((10, 10, 10), dtype=bool)
        gt[5:, :, :] = True

        dice = compute_dice_score(pred, gt)
        # Smoothing makes it very close to 0 but not exactly 0
        assert dice < 1e-8

    def test_partial_overlap(self) -> None:
        """Test Dice with partial overlap."""
        pred = np.zeros((10, 10, 10), dtype=bool)
        pred[:6, :, :] = True  # 600 voxels
        gt = np.zeros((10, 10, 10), dtype=bool)
        gt[4:, :, :] = True  # 600 voxels

        # Intersection: 2 slices = 200 voxels
        # Dice = 2*200 / (600+600) = 400/1200 = 1/3
        dice = compute_dice_score(pred, gt)
        assert abs(dice - 1.0 / 3.0) < 1e-6

    def test_empty_masks(self) -> None:
        """Test Dice with both masks empty."""
        pred = np.zeros((10, 10, 10), dtype=bool)
        gt = np.zeros((10, 10, 10), dtype=bool)

        # With smoothing, should be close to 1.0
        dice = compute_dice_score(pred, gt)
        assert 0.999 < dice < 1.001

    def test_one_empty_mask(self) -> None:
        """Test Dice when one mask is empty."""
        pred = np.ones((10, 10, 10), dtype=bool)
        gt = np.zeros((10, 10, 10), dtype=bool)

        dice = compute_dice_score(pred, gt)
        # Should be close to 0 (smooth/1000 is very small)
        assert dice < 0.001

    def test_torch_tensor_input(self) -> None:
        """Test Dice with PyTorch tensors."""
        pred = torch.ones((10, 10, 10), dtype=torch.bool)
        gt = torch.ones((10, 10, 10), dtype=torch.bool)

        dice = compute_dice_score(pred, gt)
        assert dice == 1.0

    def test_float_input(self) -> None:
        """Test Dice with float inputs (thresholded to bool)."""
        pred = np.random.rand(10, 10, 10) > 0.5
        gt = np.random.rand(10, 10, 10) > 0.5

        dice = compute_dice_score(pred, gt)
        assert 0.0 <= dice <= 1.0

    def test_smoothing_effect(self) -> None:
        """Test that smoothing prevents division by zero."""
        pred = np.zeros((10, 10, 10), dtype=bool)
        gt = np.zeros((10, 10, 10), dtype=bool)

        # With smoothing
        dice_with_smooth = compute_dice_score(pred, gt, smooth=1e-6)
        # Without smoothing (would be 0/0, but smooth prevents it)
        dice_no_smooth = compute_dice_score(pred, gt, smooth=0.0)

        assert dice_with_smooth > 0.9  # Close to 1
        assert np.isnan(dice_no_smooth)  # 0/0 = nan


class TestComputeIoU:
    """Tests for IoU calculation."""

    def test_perfect_overlap(self) -> None:
        """Test IoU with perfect overlap."""
        pred = np.ones((10, 10, 10), dtype=bool)
        gt = np.ones((10, 10, 10), dtype=bool)

        iou = compute_iou(pred, gt)
        assert iou == 1.0

    def test_no_overlap(self) -> None:
        """Test IoU with no overlap."""
        pred = np.zeros((10, 10, 10), dtype=bool)
        pred[:5, :, :] = True
        gt = np.zeros((10, 10, 10), dtype=bool)
        gt[5:, :, :] = True

        iou = compute_iou(pred, gt)
        # Smoothing makes it very close to 0 but not exactly 0
        assert iou < 1e-8
        assert iou < 1e-8

    def test_partial_overlap(self) -> None:
        """Test IoU with partial overlap."""
        pred = np.zeros((10, 10, 10), dtype=bool)
        pred[:6, :, :] = True  # 600 voxels
        gt = np.zeros((10, 10, 10), dtype=bool)
        gt[4:, :, :] = True  # 600 voxels

        # Intersection: 2 slices = 200 voxels
        # Union: 10 slices = 1000 voxels
        # IoU = 200/1000 = 0.2
        iou = compute_iou(pred, gt)
        assert abs(iou - 0.2) < 1e-6

    def test_empty_masks(self) -> None:
        """Test IoU with both masks empty."""
        pred = np.zeros((10, 10, 10), dtype=bool)
        gt = np.zeros((10, 10, 10), dtype=bool)

        # With smoothing, should be close to 1.0
        iou = compute_iou(pred, gt)
        assert 0.999 < iou < 1.001

    def test_one_empty_mask(self) -> None:
        """Test IoU when one mask is empty."""
        pred = np.ones((10, 10, 10), dtype=bool)
        gt = np.zeros((10, 10, 10), dtype=bool)

        iou = compute_iou(pred, gt)
        # Should be close to 0
        assert iou < 0.001

    def test_torch_tensor_input(self) -> None:
        """Test IoU with PyTorch tensors."""
        pred = torch.ones((10, 10, 10), dtype=torch.bool)
        gt = torch.ones((10, 10, 10), dtype=torch.bool)

        iou = compute_iou(pred, gt)
        assert iou == 1.0

    def test_contained_mask(self) -> None:
        """Test IoU when prediction is contained in ground truth."""
        pred = np.zeros((10, 10, 10), dtype=bool)
        pred[2:8, 2:8, 2:8] = True  # 6^3 = 216 voxels
        gt = np.ones((10, 10, 10), dtype=bool)  # 1000 voxels

        # Intersection: 216, Union: 1000
        iou = compute_iou(pred, gt)
        assert abs(iou - 0.216) < 0.001


class TestComputeBatchDiceScores:
    """Tests for batch Dice score calculation."""

    def test_batch_perfect_overlap(self) -> None:
        """Test batch Dice with all perfect overlaps."""
        preds = np.ones((5, 10, 10, 10), dtype=bool)
        gts = np.ones((5, 10, 10, 10), dtype=bool)

        scores = compute_batch_dice_scores(preds, gts)
        assert len(scores) == 5
        assert all(s == 1.0 for s in scores)

    def test_batch_varying_scores(self) -> None:
        """Test batch Dice with varying overlaps."""
        preds = np.zeros((3, 10, 10, 10), dtype=bool)
        gts = np.zeros((3, 10, 10, 10), dtype=bool)

        # First: perfect overlap
        preds[0, :, :, :] = True
        gts[0, :, :, :] = True

        # Second: no overlap
        preds[1, :5, :, :] = True
        gts[1, 5:, :, :] = True

        # Third: partial overlap
        preds[2, :6, :, :] = True
        gts[2, 4:, :, :] = True

        scores = compute_batch_dice_scores(preds, gts)
        assert len(scores) == 3
        assert scores[0] == 1.0  # Perfect
        assert scores[1] < 1e-8  # No overlap (very small due to smoothing)
        assert 0.0 < scores[2] < 1.0  # Partial

    def test_batch_torch_tensors(self) -> None:
        """Test batch Dice with PyTorch tensors."""
        preds = torch.ones((5, 10, 10, 10), dtype=torch.bool)
        gts = torch.ones((5, 10, 10, 10), dtype=torch.bool)

        scores = compute_batch_dice_scores(preds, gts)
        assert len(scores) == 5
        assert all(s == 1.0 for s in scores)

    def test_batch_shape_mismatch(self) -> None:
        """Test that shape mismatch raises error."""
        preds = np.ones((5, 10, 10, 10), dtype=bool)
        gts = np.ones((5, 12, 12, 12), dtype=bool)

        try:
            compute_batch_dice_scores(preds, gts)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Shape mismatch" in str(e)

    def test_batch_empty(self) -> None:
        """Test batch with zero samples."""
        preds = np.zeros((0, 10, 10, 10), dtype=bool)
        gts = np.zeros((0, 10, 10, 10), dtype=bool)

        scores = compute_batch_dice_scores(preds, gts)
        assert len(scores) == 0


class TestComputeBatchIoUScores:
    """Tests for batch IoU score calculation."""

    def test_batch_perfect_overlap(self) -> None:
        """Test batch IoU with all perfect overlaps."""
        preds = np.ones((5, 10, 10, 10), dtype=bool)
        gts = np.ones((5, 10, 10, 10), dtype=bool)

        scores = compute_batch_iou_scores(preds, gts)
        assert len(scores) == 5
        assert all(s == 1.0 for s in scores)

    def test_batch_varying_scores(self) -> None:
        """Test batch IoU with varying overlaps."""
        preds = np.zeros((3, 10, 10, 10), dtype=bool)
        gts = np.zeros((3, 10, 10, 10), dtype=bool)

        # First: perfect overlap
        preds[0, :, :, :] = True
        gts[0, :, :, :] = True

        # Second: no overlap
        preds[1, :5, :, :] = True
        gts[1, 5:, :, :] = True

        # Third: partial overlap
        preds[2, :6, :, :] = True
        gts[2, 4:, :, :] = True

        scores = compute_batch_iou_scores(preds, gts)
        assert len(scores) == 3
        assert scores[0] == 1.0  # Perfect
        assert scores[1] < 1e-8  # No overlap (very small due to smoothing)
        assert 0.0 < scores[2] < 1.0  # Partial

    def test_batch_torch_tensors(self) -> None:
        """Test batch IoU with PyTorch tensors."""
        preds = torch.ones((5, 10, 10, 10), dtype=torch.bool)
        gts = torch.ones((5, 10, 10, 10), dtype=torch.bool)

        scores = compute_batch_iou_scores(preds, gts)
        assert len(scores) == 5
        assert all(s == 1.0 for s in scores)

    def test_batch_shape_mismatch(self) -> None:
        """Test that shape mismatch raises error."""
        preds = np.ones((5, 10, 10, 10), dtype=bool)
        gts = np.ones((5, 12, 12, 12), dtype=bool)

        try:
            compute_batch_iou_scores(preds, gts)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Shape mismatch" in str(e)


class TestDiceIoUConversion:
    """Tests for Dice-IoU conversion functions."""

    def test_dice_to_iou_perfect(self) -> None:
        """Test converting perfect Dice to IoU."""
        iou = dice_to_iou(1.0)
        assert iou == 1.0

    def test_dice_to_iou_zero(self) -> None:
        """Test converting zero Dice to IoU."""
        iou = dice_to_iou(0.0)
        assert iou == 0.0

    def test_dice_to_iou_half(self) -> None:
        """Test converting Dice=0.5 to IoU."""
        # Dice=0.5 -> IoU = 0.5/(2-0.5) = 0.5/1.5 = 1/3
        iou = dice_to_iou(0.5)
        assert abs(iou - 1.0 / 3.0) < 1e-6

    def test_dice_to_iou_08(self) -> None:
        """Test converting Dice=0.8 to IoU."""
        # Dice=0.8 -> IoU = 0.8/(2-0.8) = 0.8/1.2 = 2/3
        iou = dice_to_iou(0.8)
        assert abs(iou - 2.0 / 3.0) < 1e-6

    def test_iou_to_dice_perfect(self) -> None:
        """Test converting perfect IoU to Dice."""
        dice = iou_to_dice(1.0)
        assert dice == 1.0

    def test_iou_to_dice_zero(self) -> None:
        """Test converting zero IoU to Dice."""
        dice = iou_to_dice(0.0)
        assert dice == 0.0

    def test_iou_to_dice_half(self) -> None:
        """Test converting IoU=0.5 to Dice."""
        # IoU=0.5 -> Dice = 2*0.5/(1+0.5) = 1/1.5 = 2/3
        dice = iou_to_dice(0.5)
        assert abs(dice - 2.0 / 3.0) < 1e-6

    def test_iou_to_dice_two_thirds(self) -> None:
        """Test converting IoU=2/3 to Dice."""
        # IoU=2/3 -> Dice = 2*(2/3)/(1+2/3) = (4/3)/(5/3) = 4/5 = 0.8
        dice = iou_to_dice(2.0 / 3.0)
        assert abs(dice - 0.8) < 1e-6

    def test_round_trip_dice_iou_dice(self) -> None:
        """Test round-trip conversion Dice -> IoU -> Dice."""
        original = 0.75
        iou = dice_to_iou(original)
        recovered = iou_to_dice(iou)
        assert abs(recovered - original) < 1e-6

    def test_round_trip_iou_dice_iou(self) -> None:
        """Test round-trip conversion IoU -> Dice -> IoU."""
        original = 0.6
        dice = iou_to_dice(original)
        recovered = dice_to_iou(dice)
        assert abs(recovered - original) < 1e-6


class TestIntegrationDice:
    """Integration tests for Dice/IoU metrics."""

    def test_dice_iou_relationship(self) -> None:
        """Test that Dice and IoU maintain expected relationship."""
        pred = np.zeros((10, 10, 10), dtype=bool)
        pred[:6, :, :] = True
        gt = np.zeros((10, 10, 10), dtype=bool)
        gt[4:, :, :] = True

        dice = compute_dice_score(pred, gt)
        iou = compute_iou(pred, gt)

        # Verify conversion formulas
        iou_from_dice = dice_to_iou(dice)
        dice_from_iou = iou_to_dice(iou)

        assert abs(iou_from_dice - iou) < 1e-6
        assert abs(dice_from_iou - dice) < 1e-6

    def test_realistic_segmentation(self) -> None:
        """Test with realistic segmentation scenario."""
        # Create overlapping spherical regions
        pred = np.zeros((20, 20, 20), dtype=bool)
        gt = np.zeros((20, 20, 20), dtype=bool)

        # Prediction centered at (8, 10, 10)
        for z in range(20):
            for y in range(20):
                for x in range(20):
                    if (z - 8) ** 2 + (y - 10) ** 2 + (x - 10) ** 2 <= 25:
                        pred[z, y, x] = True

        # Ground truth centered at (10, 10, 10)
        for z in range(20):
            for y in range(20):
                for x in range(20):
                    if (z - 10) ** 2 + (y - 10) ** 2 + (x - 10) ** 2 <= 25:
                        gt[z, y, x] = True

        dice = compute_dice_score(pred, gt)
        iou = compute_iou(pred, gt)

        # Should have reasonable overlap
        assert 0.5 < dice < 0.95
        assert 0.3 < iou < 0.9
        # Dice should be higher than IoU for same data
        assert dice > iou

    def test_batch_statistics(self) -> None:
        """Test batch processing with statistics."""
        # Create batch with known distribution
        batch_size = 100
        preds = np.zeros((batch_size, 10, 10, 10), dtype=bool)
        gts = np.zeros((batch_size, 10, 10, 10), dtype=bool)

        # Half perfect overlap, half no overlap
        for i in range(batch_size // 2):
            preds[i, :, :, :] = True
            gts[i, :, :, :] = True

        for i in range(batch_size // 2, batch_size):
            preds[i, :5, :, :] = True
            gts[i, 5:, :, :] = True

        dice_scores = compute_batch_dice_scores(preds, gts)
        iou_scores = compute_batch_iou_scores(preds, gts)

        # Check statistics
        assert abs(dice_scores.mean() - 0.5) < 0.01
        assert abs(iou_scores.mean() - 0.5) < 0.01
        assert dice_scores.std() > 0.4  # High variance
        assert iou_scores.std() > 0.4
