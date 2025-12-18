"""Tests for post-processing filters."""

from src.inference.nodule_properties import NoduleProperties
from src.inference.post_processing import (
    apply_all_filters,
    compute_iou_3d,
    filter_by_confidence,
    filter_by_overlap,
    filter_by_size,
)


class TestComputeIoU3D:
    """Tests for 3D IoU computation."""

    def test_identical_boxes(self) -> None:
        """Test IoU of identical boxes is 1.0."""
        bbox = ((0, 10), (0, 10), (0, 10))
        iou = compute_iou_3d(bbox, bbox)
        assert iou == 1.0

    def test_non_overlapping_boxes(self) -> None:
        """Test IoU of non-overlapping boxes is 0.0."""
        bbox1 = ((0, 5), (0, 5), (0, 5))
        bbox2 = ((10, 15), (10, 15), (10, 15))
        iou = compute_iou_3d(bbox1, bbox2)
        assert iou == 0.0

    def test_partial_overlap(self) -> None:
        """Test IoU computation for partially overlapping boxes."""
        bbox1 = ((0, 10), (0, 10), (0, 10))  # Volume 1000
        bbox2 = ((5, 15), (5, 15), (5, 15))  # Volume 1000, overlap 5x5x5=125
        iou = compute_iou_3d(bbox1, bbox2)
        # Intersection: 125, Union: 1000 + 1000 - 125 = 1875
        assert abs(iou - 125 / 1875) < 1e-6

    def test_contained_box(self) -> None:
        """Test IoU when one box is contained in another."""
        bbox1 = ((0, 20), (0, 20), (0, 20))  # Volume 8000
        bbox2 = ((5, 10), (5, 10), (5, 10))  # Volume 125, fully inside bbox1
        iou = compute_iou_3d(bbox1, bbox2)
        # Intersection: 125, Union: 8000
        assert abs(iou - 125 / 8000) < 1e-6

    def test_edge_touching(self) -> None:
        """Test IoU of boxes touching at edges (no overlap)."""
        bbox1 = ((0, 5), (0, 5), (0, 5))
        bbox2 = ((5, 10), (0, 5), (0, 5))  # Adjacent in z
        iou = compute_iou_3d(bbox1, bbox2)
        assert iou == 0.0


class TestFilterBySize:
    """Tests for size-based filtering."""

    def create_detection(
        self, volume_mm3: float, diameter_mm: float, confidence: float = 0.8
    ) -> NoduleProperties:
        """Helper to create a detection with specified properties."""
        return NoduleProperties(
            center=(10, 10, 10),
            volume_voxels=100,
            volume_mm3=volume_mm3,
            diameter_mm=(diameter_mm, diameter_mm, diameter_mm),
            bbox=((0, 20), (0, 20), (0, 20)),
            confidence=confidence,
        )

    def test_no_filters(self) -> None:
        """Test that no filters returns all detections."""
        detections = [
            self.create_detection(volume_mm3=10.0, diameter_mm=5.0),
            self.create_detection(volume_mm3=100.0, diameter_mm=10.0),
        ]
        filtered = filter_by_size(detections)
        assert len(filtered) == 2

    def test_min_volume_filter(self) -> None:
        """Test minimum volume filtering."""
        detections = [
            self.create_detection(volume_mm3=10.0, diameter_mm=5.0),
            self.create_detection(volume_mm3=100.0, diameter_mm=10.0),
            self.create_detection(volume_mm3=1000.0, diameter_mm=20.0),
        ]
        filtered = filter_by_size(detections, min_volume_mm3=50.0)
        assert len(filtered) == 2
        assert all(d.volume_mm3 >= 50.0 for d in filtered)

    def test_max_volume_filter(self) -> None:
        """Test maximum volume filtering."""
        detections = [
            self.create_detection(volume_mm3=10.0, diameter_mm=5.0),
            self.create_detection(volume_mm3=100.0, diameter_mm=10.0),
            self.create_detection(volume_mm3=1000.0, diameter_mm=20.0),
        ]
        filtered = filter_by_size(detections, max_volume_mm3=500.0)
        assert len(filtered) == 2
        assert all(d.volume_mm3 <= 500.0 for d in filtered)

    def test_min_diameter_filter(self) -> None:
        """Test minimum diameter filtering."""
        detections = [
            self.create_detection(volume_mm3=10.0, diameter_mm=3.0),
            self.create_detection(volume_mm3=100.0, diameter_mm=8.0),
            self.create_detection(volume_mm3=1000.0, diameter_mm=15.0),
        ]
        filtered = filter_by_size(detections, min_diameter_mm=5.0)
        assert len(filtered) == 2
        assert all(max(d.diameter_mm) >= 5.0 for d in filtered)

    def test_max_diameter_filter(self) -> None:
        """Test maximum diameter filtering."""
        detections = [
            self.create_detection(volume_mm3=10.0, diameter_mm=3.0),
            self.create_detection(volume_mm3=100.0, diameter_mm=8.0),
            self.create_detection(volume_mm3=1000.0, diameter_mm=15.0),
        ]
        filtered = filter_by_size(detections, max_diameter_mm=10.0)
        assert len(filtered) == 2
        assert all(max(d.diameter_mm) <= 10.0 for d in filtered)

    def test_combined_filters(self) -> None:
        """Test combination of volume and diameter filters."""
        detections = [
            self.create_detection(volume_mm3=10.0, diameter_mm=3.0),
            self.create_detection(volume_mm3=100.0, diameter_mm=8.0),
            self.create_detection(volume_mm3=1000.0, diameter_mm=15.0),
        ]
        filtered = filter_by_size(
            detections,
            min_volume_mm3=50.0,
            max_volume_mm3=500.0,
            min_diameter_mm=5.0,
            max_diameter_mm=10.0,
        )
        assert len(filtered) == 1
        assert filtered[0].volume_mm3 == 100.0

    def test_empty_list(self) -> None:
        """Test filtering empty list."""
        filtered = filter_by_size([], min_diameter_mm=5.0)
        assert len(filtered) == 0

    def test_all_filtered_out(self) -> None:
        """Test when all detections are filtered out."""
        detections = [
            self.create_detection(volume_mm3=10.0, diameter_mm=3.0),
            self.create_detection(volume_mm3=20.0, diameter_mm=4.0),
        ]
        filtered = filter_by_size(detections, min_diameter_mm=10.0)
        assert len(filtered) == 0


class TestFilterByConfidence:
    """Tests for confidence-based filtering."""

    def create_detection(
        self, confidence: float | None, diameter_mm: float = 10.0
    ) -> NoduleProperties:
        """Helper to create a detection with specified confidence."""
        return NoduleProperties(
            center=(10, 10, 10),
            volume_voxels=100,
            volume_mm3=100.0,
            diameter_mm=(diameter_mm, diameter_mm, diameter_mm),
            bbox=((0, 20), (0, 20), (0, 20)),
            confidence=confidence,
        )

    def test_basic_filtering(self) -> None:
        """Test basic confidence filtering."""
        detections = [
            self.create_detection(confidence=0.3),
            self.create_detection(confidence=0.6),
            self.create_detection(confidence=0.9),
        ]
        filtered = filter_by_confidence(detections, min_confidence=0.5)
        assert len(filtered) == 2
        assert all(d.confidence is not None and d.confidence >= 0.5 for d in filtered)

    def test_none_confidence_included(self) -> None:
        """Test that detections without confidence are included."""
        detections = [
            self.create_detection(confidence=0.3),
            self.create_detection(confidence=None),
            self.create_detection(confidence=0.9),
        ]
        filtered = filter_by_confidence(detections, min_confidence=0.5)
        assert len(filtered) == 2
        assert any(d.confidence is None for d in filtered)

    def test_high_threshold(self) -> None:
        """Test filtering with high confidence threshold."""
        detections = [
            self.create_detection(confidence=0.5),
            self.create_detection(confidence=0.7),
            self.create_detection(confidence=0.85),
        ]
        filtered = filter_by_confidence(detections, min_confidence=0.8)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.85

    def test_empty_list(self) -> None:
        """Test filtering empty list."""
        filtered = filter_by_confidence([], min_confidence=0.5)
        assert len(filtered) == 0


class TestFilterByOverlap:
    """Tests for overlap-based filtering (NMS)."""

    def create_detection(
        self,
        bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        confidence: float = 0.8,
    ) -> NoduleProperties:
        """Helper to create a detection with specified bbox and confidence."""
        center = (
            (bbox[0][0] + bbox[0][1]) // 2,
            (bbox[1][0] + bbox[1][1]) // 2,
            (bbox[2][0] + bbox[2][1]) // 2,
        )
        return NoduleProperties(
            center=center,
            volume_voxels=100,
            volume_mm3=100.0,
            diameter_mm=(10.0, 10.0, 10.0),
            bbox=bbox,
            confidence=confidence,
        )

    def test_no_overlap(self) -> None:
        """Test that non-overlapping detections are all kept."""
        detections = [
            self.create_detection(bbox=((0, 5), (0, 5), (0, 5)), confidence=0.9),
            self.create_detection(bbox=((10, 15), (10, 15), (10, 15)), confidence=0.8),
            self.create_detection(bbox=((20, 25), (20, 25), (20, 25)), confidence=0.7),
        ]
        filtered = filter_by_overlap(detections, iou_threshold=0.3)
        assert len(filtered) == 3

    def test_complete_overlap_keeps_higher_confidence(self) -> None:
        """Test that overlapping detections keep only the higher confidence one."""
        bbox = ((0, 10), (0, 10), (0, 10))
        detections = [
            self.create_detection(bbox=bbox, confidence=0.6),
            self.create_detection(bbox=bbox, confidence=0.9),
        ]
        filtered = filter_by_overlap(detections, iou_threshold=0.3)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9

    def test_partial_overlap_above_threshold(self) -> None:
        """Test filtering when IoU is above threshold."""
        detections = [
            self.create_detection(bbox=((0, 10), (0, 10), (0, 10)), confidence=0.9),
            self.create_detection(bbox=((5, 15), (5, 15), (5, 15)), confidence=0.7),
        ]
        # IoU = 125/1875 ≈ 0.067
        filtered = filter_by_overlap(detections, iou_threshold=0.05)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9

    def test_partial_overlap_below_threshold(self) -> None:
        """Test keeping both when IoU is below threshold."""
        detections = [
            self.create_detection(bbox=((0, 10), (0, 10), (0, 10)), confidence=0.9),
            self.create_detection(bbox=((5, 15), (5, 15), (5, 15)), confidence=0.7),
        ]
        # IoU = 125/1875 ≈ 0.067
        filtered = filter_by_overlap(detections, iou_threshold=0.1)
        assert len(filtered) == 2

    def test_multiple_overlaps(self) -> None:
        """Test NMS with multiple overlapping detections."""
        detections = [
            self.create_detection(bbox=((0, 10), (0, 10), (0, 10)), confidence=0.9),
            self.create_detection(bbox=((2, 12), (2, 12), (2, 12)), confidence=0.8),
            self.create_detection(bbox=((4, 14), (4, 14), (4, 14)), confidence=0.7),
            self.create_detection(bbox=((20, 30), (20, 30), (20, 30)), confidence=0.6),
        ]
        filtered = filter_by_overlap(detections, iou_threshold=0.3)
        # IoU(bbox1, bbox2) = 0.344 > 0.3, so bbox2 is filtered
        # IoU(bbox1, bbox3) = 0.121 < 0.3, so bbox3 is kept
        # IoU(bbox2, bbox3) would be checked but bbox2 already filtered
        # bbox4 is separate
        assert len(filtered) == 3
        assert filtered[0].confidence == 0.9
        assert filtered[1].confidence == 0.7
        assert filtered[2].confidence == 0.6

    def test_confidence_sorting(self) -> None:
        """Test that detections are processed in descending confidence order."""
        # Lower confidence detection listed first
        detections = [
            self.create_detection(bbox=((0, 10), (0, 10), (0, 10)), confidence=0.6),
            self.create_detection(bbox=((1, 11), (1, 11), (1, 11)), confidence=0.9),
        ]
        filtered = filter_by_overlap(detections, iou_threshold=0.3)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9

    def test_empty_list(self) -> None:
        """Test filtering empty list."""
        filtered = filter_by_overlap([], iou_threshold=0.3)
        assert len(filtered) == 0

    def test_none_confidence_handling(self) -> None:
        """Test handling of None confidence values."""
        detections = [
            self.create_detection(bbox=((0, 10), (0, 10), (0, 10)), confidence=0.9),
            NoduleProperties(
                center=(5, 5, 5),
                volume_voxels=100,
                volume_mm3=100.0,
                diameter_mm=(10.0, 10.0, 10.0),
                bbox=((1, 11), (1, 11), (1, 11)),
                confidence=None,
            ),
        ]
        # Should not crash, None treated as 0.0 for sorting
        filtered = filter_by_overlap(detections, iou_threshold=0.3)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9


class TestApplyAllFilters:
    """Tests for combined filter application."""

    def create_detection(
        self,
        volume_mm3: float,
        diameter_mm: float,
        bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        confidence: float,
    ) -> NoduleProperties:
        """Helper to create a detection with all properties."""
        center = (
            (bbox[0][0] + bbox[0][1]) // 2,
            (bbox[1][0] + bbox[1][1]) // 2,
            (bbox[2][0] + bbox[2][1]) // 2,
        )
        return NoduleProperties(
            center=center,
            volume_voxels=100,
            volume_mm3=volume_mm3,
            diameter_mm=(diameter_mm, diameter_mm, diameter_mm),
            bbox=bbox,
            confidence=confidence,
        )

    def test_no_filters(self) -> None:
        """Test that no filters returns all detections."""
        detections = [
            self.create_detection(
                volume_mm3=100.0,
                diameter_mm=10.0,
                bbox=((0, 10), (0, 10), (0, 10)),
                confidence=0.8,
            ),
            self.create_detection(
                volume_mm3=200.0,
                diameter_mm=15.0,
                bbox=((20, 30), (20, 30), (20, 30)),
                confidence=0.9,
            ),
        ]
        filtered = apply_all_filters(detections)
        assert len(filtered) == 2

    def test_size_only(self) -> None:
        """Test applying only size filters."""
        detections = [
            self.create_detection(
                volume_mm3=50.0,
                diameter_mm=5.0,
                bbox=((0, 10), (0, 10), (0, 10)),
                confidence=0.8,
            ),
            self.create_detection(
                volume_mm3=150.0,
                diameter_mm=12.0,
                bbox=((20, 30), (20, 30), (20, 30)),
                confidence=0.9,
            ),
        ]
        filtered = apply_all_filters(detections, min_diameter_mm=10.0)
        assert len(filtered) == 1
        assert max(filtered[0].diameter_mm) == 12.0

    def test_confidence_only(self) -> None:
        """Test applying only confidence filter."""
        detections = [
            self.create_detection(
                volume_mm3=100.0,
                diameter_mm=10.0,
                bbox=((0, 10), (0, 10), (0, 10)),
                confidence=0.4,
            ),
            self.create_detection(
                volume_mm3=100.0,
                diameter_mm=10.0,
                bbox=((20, 30), (20, 30), (20, 30)),
                confidence=0.9,
            ),
        ]
        filtered = apply_all_filters(detections, min_confidence=0.5)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9

    def test_overlap_only(self) -> None:
        """Test applying only overlap filter."""
        detections = [
            self.create_detection(
                volume_mm3=100.0,
                diameter_mm=10.0,
                bbox=((0, 10), (0, 10), (0, 10)),
                confidence=0.9,
            ),
            self.create_detection(
                volume_mm3=100.0,
                diameter_mm=10.0,
                bbox=((1, 11), (1, 11), (1, 11)),
                confidence=0.7,
            ),
        ]
        filtered = apply_all_filters(detections, iou_threshold=0.3)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9

    def test_all_filters_combined(self) -> None:
        """Test applying all filters together."""
        detections = [
            # Too small
            self.create_detection(
                volume_mm3=10.0,
                diameter_mm=3.0,
                bbox=((0, 5), (0, 5), (0, 5)),
                confidence=0.9,
            ),
            # Low confidence
            self.create_detection(
                volume_mm3=100.0,
                diameter_mm=10.0,
                bbox=((10, 20), (10, 20), (10, 20)),
                confidence=0.3,
            ),
            # Good detection
            self.create_detection(
                volume_mm3=100.0,
                diameter_mm=10.0,
                bbox=((30, 40), (30, 40), (30, 40)),
                confidence=0.9,
            ),
            # Overlaps with previous (lower confidence)
            self.create_detection(
                volume_mm3=100.0,
                diameter_mm=10.0,
                bbox=((32, 42), (32, 42), (32, 42)),
                confidence=0.7,
            ),
        ]
        filtered = apply_all_filters(
            detections,
            min_diameter_mm=5.0,
            min_confidence=0.5,
            iou_threshold=0.3,
        )
        assert len(filtered) == 1
        assert filtered[0].center == (35, 35, 35)
        assert filtered[0].confidence == 0.9

    def test_filter_order_matters(self) -> None:
        """Test that filter order (size -> confidence -> overlap) is correct."""
        detections = [
            # Large, high confidence, will be kept
            self.create_detection(
                volume_mm3=100.0,
                diameter_mm=10.0,
                bbox=((0, 10), (0, 10), (0, 10)),
                confidence=0.9,
            ),
            # Small, high confidence, filtered by size
            self.create_detection(
                volume_mm3=10.0,
                diameter_mm=3.0,
                bbox=((0, 10), (0, 10), (0, 10)),
                confidence=0.95,
            ),
        ]
        filtered = apply_all_filters(
            detections, min_diameter_mm=5.0, iou_threshold=0.3
        )
        # Small detection filtered before overlap check
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9

    def test_empty_list(self) -> None:
        """Test filtering empty list."""
        filtered = apply_all_filters(
            [], min_diameter_mm=5.0, min_confidence=0.5, iou_threshold=0.3
        )
        assert len(filtered) == 0
