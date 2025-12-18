"""Tests for center point accuracy evaluation."""

import numpy as np
import pytest

from src.eval.center_accuracy import (
    CenterMatch,
    calculate_accuracy_metrics,
    compute_center_distance,
    match_predictions_to_ground_truth,
)


class TestComputeCenterDistance:
    """Test compute_center_distance function."""

    def test_same_point(self) -> None:
        """Test distance between identical points."""
        center = np.array([1.0, 2.0, 3.0])
        assert compute_center_distance(center, center) == 0.0

    def test_unit_distance(self) -> None:
        """Test distance of 1.0 along x-axis."""
        center1 = np.array([0.0, 0.0, 0.0])
        center2 = np.array([1.0, 0.0, 0.0])
        assert compute_center_distance(center1, center2) == 1.0

    def test_pythagorean_3d(self) -> None:
        """Test 3D Pythagorean distance."""
        center1 = np.array([0.0, 0.0, 0.0])
        center2 = np.array([3.0, 4.0, 0.0])
        assert compute_center_distance(center1, center2) == 5.0

    def test_anisotropic_spacing(self) -> None:
        """Test distance with anisotropic voxel spacing."""
        center1 = np.array([0.0, 0.0, 0.0])
        center2 = np.array([1.0, 0.0, 0.0])
        spacing = np.array([2.0, 1.0, 1.0])
        # Distance = sqrt((1*2)^2) = 2.0
        assert compute_center_distance(center1, center2, spacing) == 2.0

    def test_invalid_dimension(self) -> None:
        """Test error for non-3D centers."""
        center1 = np.array([1.0, 2.0])  # 2D
        center2 = np.array([3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="Centers must be 3D"):
            compute_center_distance(center1, center2)

    def test_invalid_spacing(self) -> None:
        """Test error for invalid spacing."""
        center1 = np.array([0.0, 0.0, 0.0])
        center2 = np.array([1.0, 1.0, 1.0])
        spacing = np.array([1.0, 1.0])  # 2D spacing
        with pytest.raises(ValueError, match="Spacing must be 3D"):
            compute_center_distance(center1, center2, spacing)


class TestMatchPredictionsToGroundTruth:
    """Test match_predictions_to_ground_truth function."""

    def test_perfect_match(self) -> None:
        """Test perfect one-to-one matching."""
        preds = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])]
        gts = [np.array([0.1, 0.1, 0.1]), np.array([1.1, 1.1, 1.1])]
        max_distance = 1.0

        matches = match_predictions_to_ground_truth(preds, gts, max_distance)

        # Should have exactly 2 matches
        matched = [m for m in matches if m.pred_idx != -1 and m.gt_idx != -1]
        assert len(matched) == 2
        assert len(matches) == 2

        # Check distances are small
        for match in matched:
            assert match.distance < 0.2

    def test_no_matches(self) -> None:
        """Test when predictions are too far from ground truths."""
        preds = [np.array([0.0, 0.0, 0.0])]
        gts = [np.array([100.0, 100.0, 100.0])]
        max_distance = 1.0

        matches = match_predictions_to_ground_truth(preds, gts, max_distance)

        # Should have 1 unmatched prediction and 1 unmatched GT
        assert len(matches) == 2
        unmatched_preds = [m for m in matches if m.pred_idx != -1 and m.gt_idx == -1]
        unmatched_gts = [m for m in matches if m.pred_idx == -1 and m.gt_idx != -1]
        assert len(unmatched_preds) == 1
        assert len(unmatched_gts) == 1
        assert unmatched_preds[0].distance == np.inf
        assert unmatched_gts[0].distance == np.inf

    def test_one_to_one_matching(self) -> None:
        """Test that each prediction matches at most one GT."""
        # Two predictions close to one GT
        preds = [np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])]
        gts = [np.array([0.05, 0.0, 0.0])]
        max_distance = 1.0

        matches = match_predictions_to_ground_truth(preds, gts, max_distance)

        # Only one match should be made (the closer one)
        matched = [m for m in matches if m.pred_idx != -1 and m.gt_idx != -1]
        assert len(matched) == 1

        # One prediction should be unmatched
        unmatched_preds = [m for m in matches if m.pred_idx != -1 and m.gt_idx == -1]
        assert len(unmatched_preds) == 1

    def test_greedy_closest_first(self) -> None:
        """Test that greedy matching selects closest pairs first."""
        # Pred 0 closer to GT 0, Pred 1 closer to GT 1
        preds = [np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])]
        gts = [np.array([0.5, 0.0, 0.0]), np.array([10.5, 0.0, 0.0])]
        max_distance = 2.0

        matches = match_predictions_to_ground_truth(preds, gts, max_distance)

        matched = [m for m in matches if m.pred_idx != -1 and m.gt_idx != -1]
        assert len(matched) == 2

        # Check correct pairing
        match_dict = {m.pred_idx: m.gt_idx for m in matched}
        assert match_dict[0] == 0  # Pred 0 -> GT 0
        assert match_dict[1] == 1  # Pred 1 -> GT 1

    def test_empty_predictions(self) -> None:
        """Test with no predictions."""
        preds: list[np.ndarray] = []
        gts = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])]
        max_distance = 1.0

        matches = match_predictions_to_ground_truth(preds, gts, max_distance)

        # All GTs should be unmatched
        assert len(matches) == 2
        for match in matches:
            assert match.pred_idx == -1
            assert match.gt_idx != -1
            assert match.distance == np.inf

    def test_empty_ground_truths(self) -> None:
        """Test with no ground truths."""
        preds = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])]
        gts: list[np.ndarray] = []
        max_distance = 1.0

        matches = match_predictions_to_ground_truth(preds, gts, max_distance)

        # All predictions should be unmatched
        assert len(matches) == 2
        for match in matches:
            assert match.pred_idx != -1
            assert match.gt_idx == -1
            assert match.distance == np.inf

    def test_both_empty(self) -> None:
        """Test with no predictions and no ground truths."""
        preds: list[np.ndarray] = []
        gts: list[np.ndarray] = []
        max_distance = 1.0

        matches = match_predictions_to_ground_truth(preds, gts, max_distance)
        assert len(matches) == 0

    def test_anisotropic_matching(self) -> None:
        """Test matching with anisotropic spacing."""
        preds = [np.array([0.0, 0.0, 0.0])]
        gts = [np.array([1.0, 0.0, 0.0])]
        spacing = np.array([2.0, 1.0, 1.0])  # Distance = 2.0
        max_distance = 1.5

        matches = match_predictions_to_ground_truth(preds, gts, max_distance, spacing)

        # Distance is 2.0, max is 1.5, so no match
        assert len(matches) == 2
        matched = [m for m in matches if m.pred_idx != -1 and m.gt_idx != -1]
        assert len(matched) == 0

    def test_multiple_unmatched(self) -> None:
        """Test with multiple unmatched predictions and GTs."""
        preds = [np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0]), np.array([20.0, 0.0, 0.0])]
        gts = [np.array([0.5, 0.0, 0.0]), np.array([50.0, 0.0, 0.0])]
        max_distance = 1.0

        matches = match_predictions_to_ground_truth(preds, gts, max_distance)

        # Should have 1 match (pred 0 -> gt 0), 2 unmatched preds, 1 unmatched gt
        matched = [m for m in matches if m.pred_idx != -1 and m.gt_idx != -1]
        unmatched_preds = [m for m in matches if m.pred_idx != -1 and m.gt_idx == -1]
        unmatched_gts = [m for m in matches if m.pred_idx == -1 and m.gt_idx != -1]

        assert len(matched) == 1
        assert len(unmatched_preds) == 2
        assert len(unmatched_gts) == 1


class TestCalculateAccuracyMetrics:
    """Test calculate_accuracy_metrics function."""

    def test_perfect_matches(self) -> None:
        """Test metrics with all perfect matches."""
        matches = [
            CenterMatch(pred_idx=0, gt_idx=0, distance=0.5),
            CenterMatch(pred_idx=1, gt_idx=1, distance=0.8),
            CenterMatch(pred_idx=2, gt_idx=2, distance=1.0),
        ]

        metrics = calculate_accuracy_metrics(matches)

        assert metrics["num_matched"] == 3
        assert metrics["num_unmatched_preds"] == 0
        assert metrics["num_unmatched_gts"] == 0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert abs(metrics["mean_distance"] - 0.7667) < 0.001  # (0.5 + 0.8 + 1.0) / 3
        assert metrics["median_distance"] == 0.8
        assert metrics["max_distance"] == 1.0

    def test_some_unmatched(self) -> None:
        """Test metrics with some unmatched predictions and GTs."""
        matches = [
            CenterMatch(pred_idx=0, gt_idx=0, distance=0.5),
            CenterMatch(pred_idx=1, gt_idx=-1, distance=np.inf),  # Unmatched pred
            CenterMatch(pred_idx=-1, gt_idx=1, distance=np.inf),  # Unmatched GT
        ]

        metrics = calculate_accuracy_metrics(matches)

        assert metrics["num_matched"] == 1
        assert metrics["num_unmatched_preds"] == 1
        assert metrics["num_unmatched_gts"] == 1
        assert metrics["precision"] == 0.5  # 1/2 predictions matched
        assert metrics["recall"] == 0.5  # 1/2 GTs matched
        assert metrics["mean_distance"] == 0.5
        assert metrics["median_distance"] == 0.5

    def test_no_matches(self) -> None:
        """Test metrics with no matches."""
        matches = [
            CenterMatch(pred_idx=0, gt_idx=-1, distance=np.inf),
            CenterMatch(pred_idx=-1, gt_idx=0, distance=np.inf),
        ]

        metrics = calculate_accuracy_metrics(matches)

        assert metrics["num_matched"] == 0
        assert metrics["num_unmatched_preds"] == 1
        assert metrics["num_unmatched_gts"] == 1
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert np.isnan(metrics["mean_distance"])
        assert np.isnan(metrics["median_distance"])
        assert np.isnan(metrics["p95_distance"])
        assert np.isnan(metrics["max_distance"])

    def test_empty_matches(self) -> None:
        """Test metrics with empty match list."""
        matches: list[CenterMatch] = []

        metrics = calculate_accuracy_metrics(matches)

        assert metrics["num_matched"] == 0
        assert metrics["num_unmatched_preds"] == 0
        assert metrics["num_unmatched_gts"] == 0
        assert np.isnan(metrics["precision"])
        assert np.isnan(metrics["recall"])
        assert np.isnan(metrics["mean_distance"])

    def test_p95_distance(self) -> None:
        """Test 95th percentile distance calculation."""
        # Create 100 matches with distances 0.0 to 9.9
        matches = [CenterMatch(pred_idx=i, gt_idx=i, distance=i / 10.0) for i in range(100)]

        metrics = calculate_accuracy_metrics(matches)

        # 95th percentile of [0.0, 0.1, ..., 9.9] should be around 9.4-9.5
        assert 9.4 <= metrics["p95_distance"] <= 9.6

    def test_only_unmatched_predictions(self) -> None:
        """Test metrics with only unmatched predictions."""
        matches = [
            CenterMatch(pred_idx=0, gt_idx=-1, distance=np.inf),
            CenterMatch(pred_idx=1, gt_idx=-1, distance=np.inf),
        ]

        metrics = calculate_accuracy_metrics(matches)

        assert metrics["num_matched"] == 0
        assert metrics["num_unmatched_preds"] == 2
        assert metrics["num_unmatched_gts"] == 0
        assert metrics["precision"] == 0.0
        assert np.isnan(metrics["recall"])  # No GTs

    def test_only_unmatched_gts(self) -> None:
        """Test metrics with only unmatched ground truths."""
        matches = [
            CenterMatch(pred_idx=-1, gt_idx=0, distance=np.inf),
            CenterMatch(pred_idx=-1, gt_idx=1, distance=np.inf),
        ]

        metrics = calculate_accuracy_metrics(matches)

        assert metrics["num_matched"] == 0
        assert metrics["num_unmatched_preds"] == 0
        assert metrics["num_unmatched_gts"] == 2
        assert np.isnan(metrics["precision"])  # No predictions
        assert metrics["recall"] == 0.0


class TestIntegrationCenterAccuracy:
    """Integration tests for center accuracy evaluation."""

    def test_realistic_scenario(self) -> None:
        """Test realistic detection scenario."""
        # 3 predictions, 2 GTs, 1 perfect match, 1 close match, 1 FP
        preds = [
            np.array([10.0, 20.0, 30.0]),  # Close to GT 0
            np.array([50.2, 60.1, 70.3]),  # Close to GT 1
            np.array([100.0, 100.0, 100.0]),  # False positive
        ]
        gts = [
            np.array([10.5, 20.5, 30.5]),
            np.array([50.0, 60.0, 70.0]),
        ]
        max_distance = 2.0

        matches = match_predictions_to_ground_truth(preds, gts, max_distance)
        metrics = calculate_accuracy_metrics(matches)

        # Should have 2 matches (preds 0,1 -> gts 0,1), 1 FP
        assert metrics["num_matched"] == 2
        assert metrics["num_unmatched_preds"] == 1
        assert metrics["num_unmatched_gts"] == 0
        assert abs(metrics["precision"] - 2 / 3) < 0.001  # 2/3
        assert metrics["recall"] == 1.0  # 2/2
        assert metrics["mean_distance"] < 1.0  # Both matches are close

    def test_anisotropic_realistic(self) -> None:
        """Test with realistic anisotropic CT spacing."""
        spacing = np.array([0.7, 0.7, 2.5])  # Typical CT: 0.7mm in-plane, 2.5mm slice
        preds = [np.array([10.0, 10.0, 10.0])]
        gts = [np.array([10.0, 10.0, 11.0])]  # 1 slice apart in z

        # Physical distance = sqrt((0*0.7)^2 + (0*0.7)^2 + (1*2.5)^2) = 2.5mm
        matches = match_predictions_to_ground_truth(preds, gts, max_distance=3.0, spacing=spacing)
        metrics = calculate_accuracy_metrics(matches)

        assert metrics["num_matched"] == 1
        assert abs(metrics["mean_distance"] - 2.5) < 0.001

    def test_multiple_images(self) -> None:
        """Test combining results from multiple images."""
        # Image 1: 2 preds, 2 GTs, both match
        preds1 = [np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])]
        gts1 = [np.array([0.2, 0.0, 0.0]), np.array([10.2, 0.0, 0.0])]
        matches1 = match_predictions_to_ground_truth(preds1, gts1, max_distance=1.0)

        # Image 2: 1 pred, 2 GTs, 1 match
        preds2 = [np.array([0.0, 0.0, 0.0])]
        gts2 = [np.array([0.3, 0.0, 0.0]), np.array([100.0, 0.0, 0.0])]
        matches2 = match_predictions_to_ground_truth(preds2, gts2, max_distance=1.0)

        # Combine matches
        all_matches = matches1 + matches2
        metrics = calculate_accuracy_metrics(all_matches)

        assert metrics["num_matched"] == 3  # 2 from image 1, 1 from image 2
        assert metrics["num_unmatched_gts"] == 1  # 1 from image 2
        assert abs(metrics["recall"] - 3 / 4) < 0.001  # 3/4 GTs matched
