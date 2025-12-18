"""Unit tests for nodule property extraction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.inference.nodule_properties import (
    compute_bounding_box,
    compute_diameter,
    compute_volume,
    extract_nodule_properties,
    extract_properties_from_mask,
)


class TestComputeVolume:
    """Tests for compute_volume function."""

    def test_unit_spacing(self) -> None:
        """Test volume computation with unit spacing."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:5, 3:7, 4:8] = True  # 3 * 4 * 4 = 48 voxels

        voxels, mm3 = compute_volume(mask, spacing=(1.0, 1.0, 1.0))

        assert voxels == 48
        assert mm3 == pytest.approx(48.0)

    def test_anisotropic_spacing(self) -> None:
        """Test volume computation with anisotropic spacing."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[0:2, 0:2, 0:2] = True  # 2 * 2 * 2 = 8 voxels

        voxels, mm3 = compute_volume(mask, spacing=(2.5, 1.0, 0.5))

        assert voxels == 8
        assert mm3 == pytest.approx(8 * 2.5 * 1.0 * 0.5)

    def test_empty_mask(self) -> None:
        """Test volume of empty mask."""
        mask = np.zeros((10, 10, 10), dtype=bool)

        voxels, mm3 = compute_volume(mask)

        assert voxels == 0
        assert mm3 == 0.0


class TestComputeDiameter:
    """Tests for compute_diameter function."""

    def test_cubic_region(self) -> None:
        """Test diameter of cubic region."""
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[5:15, 5:15, 5:15] = True  # 10x10x10 cube

        diameter = compute_diameter(mask, spacing=(1.0, 1.0, 1.0))

        assert diameter == pytest.approx((10.0, 10.0, 10.0))

    def test_elongated_region(self) -> None:
        """Test diameter of elongated region."""
        mask = np.zeros((30, 20, 20), dtype=bool)
        mask[5:25, 8:12, 8:12] = True  # 20x4x4

        diameter = compute_diameter(mask, spacing=(1.0, 1.0, 1.0))

        assert diameter == pytest.approx((20.0, 4.0, 4.0))

    def test_anisotropic_spacing(self) -> None:
        """Test diameter with anisotropic spacing."""
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[10:15, 10:15, 10:15] = True  # 5x5x5 voxels

        diameter = compute_diameter(mask, spacing=(2.0, 1.0, 0.5))

        assert diameter[0] == pytest.approx(5 * 2.0)  # z
        assert diameter[1] == pytest.approx(5 * 1.0)  # y
        assert diameter[2] == pytest.approx(5 * 0.5)  # x

    def test_single_voxel(self) -> None:
        """Test diameter of single voxel."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True

        diameter = compute_diameter(mask, spacing=(1.0, 1.0, 1.0))

        assert diameter == pytest.approx((1.0, 1.0, 1.0))


class TestComputeBoundingBox:
    """Tests for compute_bounding_box function."""

    def test_simple_box(self) -> None:
        """Test bounding box of simple region."""
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[5:10, 8:15, 3:18] = True

        bbox = compute_bounding_box(mask)

        assert bbox == ((5, 9), (8, 14), (3, 17))

    def test_single_voxel(self) -> None:
        """Test bounding box of single voxel."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[3, 5, 7] = True

        bbox = compute_bounding_box(mask)

        assert bbox == ((3, 3), (5, 5), (7, 7))

    def test_corner_region(self) -> None:
        """Test bounding box at volume corner."""
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[0:3, 0:3, 0:3] = True

        bbox = compute_bounding_box(mask)

        assert bbox == ((0, 2), (0, 2), (0, 2))

    def test_empty_mask_raises_error(self) -> None:
        """Test that empty mask raises error."""
        mask = np.zeros((10, 10, 10), dtype=bool)

        with pytest.raises(ValueError, match="Cannot compute bounding box of empty mask"):
            compute_bounding_box(mask)


class TestExtractPropertiesFromMask:
    """Tests for extract_properties_from_mask function."""

    def test_basic_extraction(self) -> None:
        """Test basic property extraction."""
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[5:10, 5:10, 5:10] = True  # 5x5x5 cube

        props = extract_properties_from_mask(mask, spacing=(1.0, 1.0, 1.0))

        assert props.volume_voxels == 125
        assert props.volume_mm3 == pytest.approx(125.0)
        assert props.diameter_mm == pytest.approx((5.0, 5.0, 5.0))
        assert props.bbox == ((5, 9), (5, 9), (5, 9))
        assert props.confidence is None

    def test_with_confidence(self) -> None:
        """Test extraction with confidence score."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[3:6, 3:6, 3:6] = True

        props = extract_properties_from_mask(mask, confidence=0.95)

        assert props.confidence == pytest.approx(0.95)

    def test_with_provided_center(self) -> None:
        """Test extraction with provided center."""
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[10:15, 10:15, 10:15] = True

        props = extract_properties_from_mask(mask, center=(12.5, 12.5, 12.5))

        assert props.center == (12.5, 12.5, 12.5)

    def test_computed_center(self) -> None:
        """Test that center is computed when not provided."""
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[5:10, 5:10, 5:10] = True

        props = extract_properties_from_mask(mask)

        # Center of mass of 5x5x5 cube from [5,10) is 7.0
        assert props.center[0] == pytest.approx(7.0)
        assert props.center[1] == pytest.approx(7.0)
        assert props.center[2] == pytest.approx(7.0)

    def test_anisotropic_spacing(self) -> None:
        """Test extraction with anisotropic spacing."""
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[5:10, 5:10, 5:10] = True  # 5x5x5 voxels

        props = extract_properties_from_mask(mask, spacing=(2.0, 1.0, 0.5))

        assert props.volume_mm3 == pytest.approx(125 * 2.0 * 1.0 * 0.5)
        assert props.diameter_mm[0] == pytest.approx(5 * 2.0)
        assert props.diameter_mm[1] == pytest.approx(5 * 1.0)
        assert props.diameter_mm[2] == pytest.approx(5 * 0.5)

    def test_empty_mask_raises_error(self) -> None:
        """Test that empty mask raises error."""
        mask = np.zeros((10, 10, 10), dtype=bool)

        with pytest.raises(ValueError, match="Mask is empty"):
            extract_properties_from_mask(mask)


class TestExtractNoduleProperties:
    """Tests for extract_nodule_properties function."""

    def test_simple_segmentation(self) -> None:
        """Test extraction from simple segmentation."""
        seg = np.zeros((30, 30, 30), dtype=np.float32)
        seg[10:20, 10:20, 10:20] = 1.0  # Solid cube

        center = (15.0, 15.0, 15.0)
        props = extract_nodule_properties(seg, center)

        assert props.volume_voxels == 1000  # 10x10x10
        assert props.center == center

    def test_torch_tensor_input(self) -> None:
        """Test with PyTorch tensor input."""
        seg = torch.zeros(1, 1, 30, 30, 30)
        seg[0, 0, 10:20, 10:20, 10:20] = 1.0

        center = (15.0, 15.0, 15.0)
        props = extract_nodule_properties(seg, center)

        assert props.volume_voxels == 1000

    def test_thresholding(self) -> None:
        """Test segmentation thresholding."""
        seg = np.zeros((30, 30, 30), dtype=np.float32)
        seg[10:20, 10:20, 10:20] = 0.8  # Above default threshold
        seg[5:8, 5:8, 5:8] = 0.3  # Below default threshold

        center = (15.0, 15.0, 15.0)
        props = extract_nodule_properties(seg, center, threshold=0.5)

        # Should only include the high-confidence region
        assert props.volume_voxels == 1000

    def test_connected_components(self) -> None:
        """Test that only the component containing center is used."""
        seg = np.zeros((40, 40, 40), dtype=np.float32)
        seg[10:15, 10:15, 10:15] = 1.0  # Component 1
        seg[25:30, 25:30, 25:30] = 1.0  # Component 2

        center = (12.0, 12.0, 12.0)  # In component 1
        props = extract_nodule_properties(seg, center)

        # Should only extract properties of component 1 (5x5x5 = 125)
        assert props.volume_voxels == 125
        assert props.bbox[0][0] >= 10 and props.bbox[0][1] <= 14

    def test_center_not_in_segmentation(self) -> None:
        """Test handling when center is not in a segmented region."""
        seg = np.zeros((30, 30, 30), dtype=np.float32)
        seg[10:15, 10:15, 10:15] = 1.0

        center = (20.0, 20.0, 20.0)  # Outside segmented region
        props = extract_nodule_properties(seg, center)

        # Should still extract properties (finds nearest component)
        assert props.volume_voxels > 0

    def test_confidence_passed_through(self) -> None:
        """Test that confidence is passed through."""
        seg = np.zeros((20, 20, 20), dtype=np.float32)
        seg[5:15, 5:15, 5:15] = 1.0

        center = (10.0, 10.0, 10.0)
        props = extract_nodule_properties(seg, center, confidence=0.92)

        assert props.confidence == pytest.approx(0.92)

    def test_anisotropic_spacing(self) -> None:
        """Test with anisotropic spacing."""
        seg = np.zeros((30, 30, 30), dtype=np.float32)
        seg[10:20, 10:20, 10:20] = 1.0

        center = (15.0, 15.0, 15.0)
        props = extract_nodule_properties(seg, center, spacing=(2.5, 1.0, 0.5))

        expected_volume = 1000 * 2.5 * 1.0 * 0.5
        assert props.volume_mm3 == pytest.approx(expected_volume)
        assert props.diameter_mm[0] == pytest.approx(10 * 2.5)
        assert props.diameter_mm[1] == pytest.approx(10 * 1.0)
        assert props.diameter_mm[2] == pytest.approx(10 * 0.5)

    def test_batched_4d_input(self) -> None:
        """Test with 4D batched input."""
        seg = torch.zeros(1, 30, 30, 30)
        seg[0, 10:20, 10:20, 10:20] = 1.0

        center = (15.0, 15.0, 15.0)
        props = extract_nodule_properties(seg, center)

        assert props.volume_voxels == 1000

    def test_batched_5d_input(self) -> None:
        """Test with 5D batched input."""
        seg = torch.zeros(1, 1, 30, 30, 30)
        seg[0, 0, 10:20, 10:20, 10:20] = 1.0

        center = (15.0, 15.0, 15.0)
        props = extract_nodule_properties(seg, center)

        assert props.volume_voxels == 1000

    def test_invalid_batch_size_raises_error(self) -> None:
        """Test that invalid batch size raises error."""
        seg = torch.zeros(2, 1, 30, 30, 30)  # Batch size 2

        with pytest.raises(ValueError, match="Expected single-channel"):
            extract_nodule_properties(seg, (15.0, 15.0, 15.0))

    def test_invalid_channels_raises_error(self) -> None:
        """Test that invalid number of channels raises error."""
        seg = torch.zeros(1, 3, 30, 30, 30)  # 3 channels

        with pytest.raises(ValueError, match="Expected single-channel"):
            extract_nodule_properties(seg, (15.0, 15.0, 15.0))

    def test_wrong_dimensions_raises_error(self) -> None:
        """Test that wrong dimensions raise error."""
        seg = np.zeros((30, 30), dtype=np.float32)  # 2D

        with pytest.raises(ValueError, match="Expected 3D segmentation"):
            extract_nodule_properties(seg, (15.0, 15.0))

    def test_center_out_of_bounds_raises_error(self) -> None:
        """Test that out-of-bounds center raises error."""
        seg = np.zeros((30, 30, 30), dtype=np.float32)

        with pytest.raises(ValueError, match="out of bounds"):
            extract_nodule_properties(seg, (50.0, 15.0, 15.0))

    def test_irregular_shape(self) -> None:
        """Test with irregular-shaped nodule."""
        seg = np.zeros((40, 40, 40), dtype=np.float32)
        # Create L-shaped nodule
        seg[10:20, 10:20, 10:15] = 1.0
        seg[10:15, 10:20, 15:25] = 1.0

        center = (12.0, 15.0, 12.0)
        props = extract_nodule_properties(seg, center)

        # Volume: 10*10*5 + 5*10*10 = 500 + 500 = 1000
        assert props.volume_voxels == 1000
        assert props.bbox[2] == (10, 24)  # x extends from 10 to 24

    def test_realistic_nodule_with_soft_boundaries(self) -> None:
        """Test with realistic soft-boundary segmentation."""
        seg = np.zeros((50, 50, 50), dtype=np.float32)
        # Create soft Gaussian-like nodule
        center_pos = np.array([25, 25, 25])
        for z in range(20, 31):
            for y in range(20, 31):
                for x in range(20, 31):
                    dist = np.linalg.norm(np.array([z, y, x]) - center_pos)
                    seg[z, y, x] = np.exp(-dist**2 / 10.0)

        center = (25.0, 25.0, 25.0)
        props = extract_nodule_properties(seg, center, threshold=0.3)

        # Should extract something reasonable
        assert props.volume_voxels > 0
        assert props.volume_mm3 > 0
        assert all(d > 0 for d in props.diameter_mm)
