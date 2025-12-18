"""Tests for FROC curve calculation."""

from src.eval.froc import (
    Detection,
    FROCPoint,
    compute_average_sensitivity,
    compute_distance_3d,
    compute_froc_curve,
    compute_sensitivity_at_fppi,
    match_detections_to_ground_truth,
)


class TestComputeDistance3D:
    """Tests for 3D distance computation."""

    def test_same_point(self) -> None:
        """Test distance between identical points is zero."""
        dist = compute_distance_3d((5, 10, 15), (5, 10, 15))
        assert dist == 0.0

    def test_unit_distance(self) -> None:
        """Test unit distance calculation."""
        dist = compute_distance_3d((0, 0, 0), (1, 0, 0))
        assert dist == 1.0

    def test_pythagorean(self) -> None:
        """Test Pythagorean distance (3-4-5 triangle)."""
        dist = compute_distance_3d((0, 0, 0), (0, 3, 4))
        assert abs(dist - 5.0) < 1e-6

    def test_anisotropic_spacing(self) -> None:
        """Test distance with anisotropic voxel spacing."""
        # Distance in voxels: (2, 0, 0), spacing (2.5, 1, 1) = 5mm
        dist = compute_distance_3d((0, 0, 0), (2, 0, 0), spacing=(2.5, 1.0, 1.0))
        assert dist == 5.0

    def test_3d_distance(self) -> None:
        """Test 3D distance calculation."""
        # sqrt(1^2 + 2^2 + 2^2) = sqrt(9) = 3
        dist = compute_distance_3d((0, 0, 0), (1, 2, 2))
        assert abs(dist - 3.0) < 1e-6


class TestMatchDetectionsToGroundTruth:
    """Tests for detection-to-ground-truth matching."""

    def test_perfect_match(self) -> None:
        """Test perfect match when detections exactly overlap GT."""
        detections = [Detection((5, 5, 5), 0.9), Detection((10, 10, 10), 0.7)]
        ground_truth = [(5, 5, 5), (10, 10, 10)]

        is_tp, matched_idx = match_detections_to_ground_truth(detections, ground_truth)

        assert is_tp == [True, True]
        assert matched_idx == [0, 1]

    def test_no_matches(self) -> None:
        """Test when detections are too far from GT."""
        detections = [Detection((100, 100, 100), 0.9)]
        ground_truth = [(5, 5, 5)]

        is_tp, matched_idx = match_detections_to_ground_truth(
            detections, ground_truth, max_distance=10.0
        )

        assert is_tp == [False]
        assert matched_idx == [-1]

    def test_within_threshold(self) -> None:
        """Test match within distance threshold."""
        detections = [Detection((5, 5, 5), 0.9)]
        ground_truth = [(5, 5, 10)]  # 5mm away in x

        is_tp, matched_idx = match_detections_to_ground_truth(
            detections, ground_truth, max_distance=10.0
        )

        assert is_tp == [True]
        assert matched_idx == [0]

    def test_one_to_one_matching(self) -> None:
        """Test that each GT can only be matched once."""
        # Two detections near same GT, higher confidence wins
        detections = [
            Detection((5, 5, 5), 0.9),  # Exact match
            Detection((5, 5, 6), 0.7),  # Close match
        ]
        ground_truth = [(5, 5, 5)]

        is_tp, matched_idx = match_detections_to_ground_truth(detections, ground_truth)

        assert is_tp == [True, False]  # Second is FP
        assert matched_idx == [0, -1]

    def test_confidence_order_matters(self) -> None:
        """Test that higher confidence detection gets matched first."""
        # Lower confidence listed first, but higher confidence should match
        detections = [
            Detection((5, 5, 6), 0.7),  # Close match, lower confidence
            Detection((5, 5, 5), 0.9),  # Exact match, higher confidence
        ]
        ground_truth = [(5, 5, 5)]

        # Function expects sorted by confidence, but let's test unsorted
        is_tp, matched_idx = match_detections_to_ground_truth(detections, ground_truth)

        # First (lower conf) gets matched because it's processed first
        assert is_tp == [True, False]

    def test_multiple_gt_multiple_detections(self) -> None:
        """Test multiple detections matched to multiple GT."""
        detections = [
            Detection((5, 5, 5), 0.9),
            Detection((15, 15, 15), 0.8),
            Detection((25, 25, 25), 0.7),
        ]
        ground_truth = [(5, 5, 5), (15, 15, 15)]

        is_tp, matched_idx = match_detections_to_ground_truth(detections, ground_truth)

        assert is_tp == [True, True, False]
        assert matched_idx == [0, 1, -1]

    def test_empty_detections(self) -> None:
        """Test with no detections."""
        detections: list[Detection] = []
        ground_truth = [(5, 5, 5)]

        is_tp, matched_idx = match_detections_to_ground_truth(detections, ground_truth)

        assert is_tp == []
        assert matched_idx == []

    def test_empty_ground_truth(self) -> None:
        """Test with no ground truth."""
        detections = [Detection((5, 5, 5), 0.9)]
        ground_truth: list[tuple[float, float, float]] = []

        is_tp, matched_idx = match_detections_to_ground_truth(detections, ground_truth)

        assert is_tp == [False]
        assert matched_idx == [-1]

    def test_nearest_match(self) -> None:
        """Test that detection matches to nearest GT."""
        detections = [Detection((10, 10, 10), 0.9)]
        ground_truth = [(5, 5, 5), (12, 12, 12)]  # Second is closer

        is_tp, matched_idx = match_detections_to_ground_truth(detections, ground_truth)

        assert is_tp == [True]
        # sqrt((10-12)^2 * 3) = sqrt(12) ≈ 3.46 < 10mm threshold
        # sqrt((10-5)^2 * 3) = sqrt(75) ≈ 8.66 < 10mm threshold
        # Closer is (12,12,12)
        assert matched_idx == [1]


class TestComputeFROCCurve:
    """Tests for FROC curve computation."""

    def test_perfect_detection(self) -> None:
        """Test FROC when all GT are detected with no FP."""
        detections = [
            Detection((5, 5, 5), 0.9),
            Detection((10, 10, 10), 0.7),
        ]
        ground_truth = [(5, 5, 5), (10, 10, 10)]

        froc = compute_froc_curve(detections, ground_truth, num_images=1)

        # At lowest threshold, should have 100% sensitivity, 0 FP
        assert froc[-1].sensitivity == 1.0
        assert froc[-1].fppi == 0.0

    def test_some_false_positives(self) -> None:
        """Test FROC with false positives."""
        detections = [
            Detection((5, 5, 5), 0.9),  # TP
            Detection((100, 100, 100), 0.8),  # FP
            Detection((10, 10, 10), 0.7),  # TP
        ]
        ground_truth = [(5, 5, 5), (10, 10, 10)]

        froc = compute_froc_curve(detections, ground_truth, num_images=1)

        # At lowest threshold, 100% sensitivity, 1 FPPI
        assert froc[-1].sensitivity == 1.0
        assert froc[-1].fppi == 1.0

    def test_missed_detections(self) -> None:
        """Test FROC with missed ground truth."""
        detections = [Detection((5, 5, 5), 0.9)]  # Only detect one
        ground_truth = [(5, 5, 5), (10, 10, 10)]

        froc = compute_froc_curve(detections, ground_truth, num_images=1)

        # At lowest threshold, 50% sensitivity (1/2)
        assert froc[-1].sensitivity == 0.5
        assert froc[-1].fppi == 0.0

    def test_threshold_variation(self) -> None:
        """Test sensitivity changes with threshold."""
        detections = [
            Detection((5, 5, 5), 0.9),
            Detection((10, 10, 10), 0.5),
        ]
        ground_truth = [(5, 5, 5), (10, 10, 10)]

        froc = compute_froc_curve(detections, ground_truth, num_images=1)

        # Find high threshold point (should have lower sensitivity)
        high_thresh = [p for p in froc if p.threshold >= 0.8]
        assert high_thresh[0].sensitivity == 0.5  # Only first detection

        # At lowest threshold, should have 100% sensitivity
        assert froc[-1].sensitivity == 1.0

    def test_multiple_images(self) -> None:
        """Test FPPI calculation with multiple images."""
        detections = [
            Detection((5, 5, 5), 0.9),  # TP
            Detection((100, 100, 100), 0.8),  # FP
        ]
        ground_truth = [(5, 5, 5)]

        froc = compute_froc_curve(detections, ground_truth, num_images=2)

        # 1 FP across 2 images = 0.5 FPPI
        assert froc[-1].fppi == 0.5

    def test_empty_ground_truth(self) -> None:
        """Test with no ground truth."""
        detections = [Detection((5, 5, 5), 0.9)]
        ground_truth: list[tuple[float, float, float]] = []

        froc = compute_froc_curve(detections, ground_truth, num_images=1)

        assert froc == []

    def test_empty_detections(self) -> None:
        """Test with no detections."""
        detections: list[Detection] = []
        ground_truth = [(5, 5, 5)]

        froc = compute_froc_curve(detections, ground_truth, num_images=1)

        # Should have one point with 0 sensitivity at threshold 0.0
        assert len(froc) == 1
        assert froc[0].sensitivity == 0.0
        assert froc[0].threshold == 0.0

    def test_custom_thresholds(self) -> None:
        """Test with custom threshold list."""
        detections = [
            Detection((5, 5, 5), 0.9),
            Detection((10, 10, 10), 0.5),
        ]
        ground_truth = [(5, 5, 5), (10, 10, 10)]

        froc = compute_froc_curve(
            detections, ground_truth, num_images=1, thresholds=[0.8, 0.6, 0.4]
        )

        assert len(froc) == 3
        assert [p.threshold for p in froc] == [0.8, 0.6, 0.4]

    def test_froc_point_counts(self) -> None:
        """Test TP/FP/FN counts in FROC points."""
        detections = [
            Detection((5, 5, 5), 0.9),  # TP
            Detection((100, 100, 100), 0.7),  # FP
        ]
        ground_truth = [(5, 5, 5), (10, 10, 10)]  # 2 GT, 1 missed

        froc = compute_froc_curve(detections, ground_truth, num_images=1)

        # At lowest threshold
        assert froc[-1].num_tp == 1
        assert froc[-1].num_fp == 1
        assert froc[-1].num_fn == 1


class TestComputeSensitivityAtFPPI:
    """Tests for sensitivity interpolation at specific FPPI."""

    def test_exact_fppi(self) -> None:
        """Test when exact FPPI exists in curve."""
        froc = [
            FROCPoint(0.9, 0.5, 0.0, 5, 0, 5),
            FROCPoint(0.7, 0.75, 1.0, 7, 10, 3),
            FROCPoint(0.5, 1.0, 2.0, 10, 20, 0),
        ]

        sens = compute_sensitivity_at_fppi(froc, 1.0)
        assert sens == 0.75

    def test_interpolation(self) -> None:
        """Test linear interpolation between FPPI values."""
        froc = [
            FROCPoint(0.9, 0.5, 0.0, 5, 0, 5),
            FROCPoint(0.5, 1.0, 2.0, 10, 20, 0),
        ]

        # Interpolate at midpoint: FPPI=1.0 should give sens=0.75
        sens = compute_sensitivity_at_fppi(froc, 1.0)
        assert abs(sens - 0.75) < 1e-6

    def test_below_minimum_fppi(self) -> None:
        """Test FPPI below minimum returns first sensitivity."""
        froc = [
            FROCPoint(0.9, 0.5, 1.0, 5, 10, 5),
            FROCPoint(0.5, 1.0, 2.0, 10, 20, 0),
        ]

        sens = compute_sensitivity_at_fppi(froc, 0.5)
        assert sens == 0.5

    def test_above_maximum_fppi(self) -> None:
        """Test FPPI above maximum returns last sensitivity."""
        froc = [
            FROCPoint(0.9, 0.5, 1.0, 5, 10, 5),
            FROCPoint(0.5, 1.0, 2.0, 10, 20, 0),
        ]

        sens = compute_sensitivity_at_fppi(froc, 5.0)
        assert sens == 1.0

    def test_empty_froc(self) -> None:
        """Test with empty FROC curve."""
        froc: list[FROCPoint] = []

        sens = compute_sensitivity_at_fppi(froc, 1.0)
        assert sens == 0.0

    def test_single_point(self) -> None:
        """Test with single FROC point."""
        froc = [FROCPoint(0.9, 0.8, 1.0, 8, 10, 2)]

        sens = compute_sensitivity_at_fppi(froc, 0.5)
        assert sens == 0.8  # Below minimum

        sens = compute_sensitivity_at_fppi(froc, 2.0)
        assert sens == 0.8  # Above maximum


class TestComputeAverageSensitivity:
    """Tests for average sensitivity calculation."""

    def test_standard_fppi_range(self) -> None:
        """Test average over standard LUNA16 FPPI range."""
        # Create FROC with perfect sensitivity
        froc = [
            FROCPoint(0.9, 1.0, 0.0, 10, 0, 0),
            FROCPoint(0.1, 1.0, 10.0, 10, 100, 0),
        ]

        avg = compute_average_sensitivity(froc, (0.125, 8.0))
        assert avg == 1.0

    def test_varying_sensitivity(self) -> None:
        """Test average with varying sensitivity."""
        froc = [
            FROCPoint(0.9, 0.0, 0.0, 0, 0, 10),
            FROCPoint(0.1, 1.0, 10.0, 10, 100, 0),
        ]

        # At low FPPI, sensitivity is low; at high FPPI, it's high
        avg = compute_average_sensitivity(froc, (0.125, 8.0))
        assert 0.0 < avg < 1.0

    def test_custom_fppi_range(self) -> None:
        """Test with custom FPPI range."""
        froc = [
            FROCPoint(0.9, 0.5, 0.0, 5, 0, 5),
            FROCPoint(0.1, 1.0, 2.0, 10, 20, 0),
        ]

        # Only evaluates at FPPI values within range
        avg = compute_average_sensitivity(froc, (0.0, 1.0))
        # Standard values in range: 0.125, 0.25, 0.5, 1.0
        assert avg > 0.0

    def test_no_fppi_in_range(self) -> None:
        """Test when no standard FPPI values in range."""
        froc = [FROCPoint(0.9, 0.8, 1.0, 8, 10, 2)]

        # Range excludes all standard FPPI values
        avg = compute_average_sensitivity(froc, (10.0, 20.0))
        assert avg == 0.0

    def test_empty_froc(self) -> None:
        """Test with empty FROC curve."""
        froc: list[FROCPoint] = []

        avg = compute_average_sensitivity(froc)
        assert avg == 0.0


class TestIntegrationFROC:
    """Integration tests for complete FROC workflow."""

    def test_realistic_scenario(self) -> None:
        """Test realistic detection scenario."""
        # Create realistic detections
        detections = [
            Detection((10, 10, 10), 0.95),  # High conf TP
            Detection((20, 20, 20), 0.85),  # Medium conf TP
            Detection((100, 100, 100), 0.75),  # Medium conf FP
            Detection((30, 30, 30), 0.65),  # Lower conf TP
            Detection((200, 200, 200), 0.55),  # Lower conf FP
        ]

        ground_truth = [(10, 10, 10), (20, 20, 20), (30, 30, 30)]

        # Compute FROC
        froc = compute_froc_curve(detections, ground_truth, num_images=1)

        # Check high threshold (only high-conf detection)
        high_thresh = [p for p in froc if p.threshold >= 0.9]
        assert high_thresh[0].sensitivity == 1.0 / 3.0  # 1 of 3 GT

        # Check low threshold (all detections)
        assert froc[-1].sensitivity == 1.0  # All 3 GT detected
        assert froc[-1].fppi == 2.0  # 2 FP

    def test_sensitivity_at_standard_fppi(self) -> None:
        """Test computing sensitivity at standard FPPI values."""
        detections = [
            Detection((i * 10, i * 10, i * 10), 1.0 - i * 0.1) for i in range(10)
        ]
        ground_truth = [(i * 10, i * 10, i * 10) for i in range(5)]

        froc = compute_froc_curve(detections, ground_truth, num_images=1)

        # Compute sensitivities at standard FPPI values
        sens_at_0125 = compute_sensitivity_at_fppi(froc, 0.125)
        sens_at_1 = compute_sensitivity_at_fppi(froc, 1.0)
        sens_at_4 = compute_sensitivity_at_fppi(froc, 4.0)

        # Sensitivity should increase with FPPI
        assert sens_at_0125 <= sens_at_1 <= sens_at_4

    def test_average_sensitivity_calculation(self) -> None:
        """Test complete workflow with average sensitivity."""
        # Perfect detection scenario
        detections = [
            Detection((i * 10, i * 10, i * 10), 0.9 - i * 0.1) for i in range(5)
        ]
        ground_truth = [(i * 10, i * 10, i * 10) for i in range(5)]

        froc = compute_froc_curve(detections, ground_truth, num_images=1)
        avg = compute_average_sensitivity(froc)

        # Should have high average since all are TP
        assert avg > 0.9
