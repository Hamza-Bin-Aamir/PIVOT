"""Nodule property extraction from segmentation masks and predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import ndimage

__all__ = [
    "NoduleProperties",
    "extract_nodule_properties",
    "extract_properties_from_mask",
    "compute_volume",
    "compute_diameter",
    "compute_bounding_box",
]

logger = logging.getLogger(__name__)


@dataclass
class NoduleProperties:
    """Properties of a detected nodule."""

    center: tuple[float, float, float]  # (z, y, x) center coordinates
    volume_voxels: int  # Volume in voxels
    volume_mm3: float  # Volume in cubic millimeters
    diameter_mm: tuple[float, float, float]  # Diameter in mm (z, y, x)
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]  # ((z_min, z_max), (y_min, y_max), (x_min, x_max))
    confidence: float | None = None  # Optional confidence score


def extract_nodule_properties(
    segmentation: NDArray[np.float32] | torch.Tensor,
    center: tuple[float, float, float],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    confidence: float | None = None,
    threshold: float = 0.5,
) -> NoduleProperties:
    """Extract properties of a nodule from segmentation mask around a center point.

    Args:
        segmentation: 3D segmentation mask of shape (D, H, W) or (1, 1, D, H, W)
        center: (z, y, x) center coordinates of the nodule
        spacing: Physical voxel spacing in mm (z, y, x)
        confidence: Optional confidence score for the detection
        threshold: Threshold for binarizing segmentation mask

    Returns:
        NoduleProperties dataclass with extracted properties

    Raises:
        ValueError: If segmentation has invalid shape or center is out of bounds
    """
    # Convert to numpy if needed
    seg_np = segmentation.detach().cpu().numpy() if isinstance(segmentation, torch.Tensor) else segmentation

    # Handle batched input
    if seg_np.ndim == 5:
        if seg_np.shape[0] != 1 or seg_np.shape[1] != 1:
            msg = f"Expected single-channel segmentation, got shape {seg_np.shape}"
            raise ValueError(msg)
        seg_np = seg_np[0, 0]
    elif seg_np.ndim == 4:
        if seg_np.shape[0] != 1:
            msg = f"Expected single-channel segmentation, got shape {seg_np.shape}"
            raise ValueError(msg)
        seg_np = seg_np[0]
    elif seg_np.ndim != 3:
        msg = f"Expected 3D segmentation, got {seg_np.ndim}D"
        raise ValueError(msg)

    # Validate center
    z, y, x = center
    if not (0 <= z < seg_np.shape[0] and 0 <= y < seg_np.shape[1] and 0 <= x < seg_np.shape[2]):
        msg = f"Center {center} is out of bounds for shape {seg_np.shape}"
        raise ValueError(msg)

    # Binarize segmentation
    binary_mask = seg_np >= threshold

    # Find connected component containing the center
    labeled, num_features = ndimage.label(binary_mask)
    center_int = (int(round(z)), int(round(y)), int(round(x)))
    nodule_label = labeled[center_int]

    if nodule_label == 0:
        # Center is not in a segmented region, use largest component near center
        logger.warning(f"Center {center} not in segmented region, using nearest component")
        # Find nearest labeled component
        distances = ndimage.distance_transform_edt(labeled == 0)
        nearest_idx = np.unravel_index(np.argmin(distances), labeled.shape)
        nodule_label = labeled[nearest_idx]

        if nodule_label == 0:
            # Still no component found, use entire mask
            logger.warning("No connected component found, using entire mask")
            nodule_mask = binary_mask
        else:
            nodule_mask = labeled == nodule_label
    else:
        nodule_mask = labeled == nodule_label

    return extract_properties_from_mask(nodule_mask, spacing, center, confidence)


def extract_properties_from_mask(
    mask: NDArray[np.bool_],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    center: tuple[float, float, float] | None = None,
    confidence: float | None = None,
) -> NoduleProperties:
    """Extract nodule properties from a binary mask.

    Args:
        mask: Binary mask of the nodule
        spacing: Physical voxel spacing in mm (z, y, x)
        center: Optional precomputed center (z, y, x), computed if None
        confidence: Optional confidence score

    Returns:
        NoduleProperties dataclass with extracted properties

    Raises:
        ValueError: If mask is empty
    """
    if not np.any(mask):
        msg = "Mask is empty"
        raise ValueError(msg)

    # Compute volume
    volume_voxels, volume_mm3 = compute_volume(mask, spacing)

    # Compute bounding box
    bbox = compute_bounding_box(mask)

    # Compute diameter
    diameter_mm = compute_diameter(mask, spacing)

    # Compute center if not provided
    if center is None:
        center_coords = ndimage.center_of_mass(mask)
        center = (float(center_coords[0]), float(center_coords[1]), float(center_coords[2]))

    return NoduleProperties(
        center=center,
        volume_voxels=volume_voxels,
        volume_mm3=volume_mm3,
        diameter_mm=diameter_mm,
        bbox=bbox,
        confidence=confidence,
    )


def compute_volume(
    mask: NDArray[np.bool_],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[int, float]:
    """Compute nodule volume from binary mask.

    Args:
        mask: Binary mask of the nodule
        spacing: Physical voxel spacing in mm (z, y, x)

    Returns:
        Tuple of (volume_voxels, volume_mm3)
    """
    volume_voxels = int(np.sum(mask))
    voxel_volume_mm3 = float(np.prod(spacing))
    volume_mm3 = volume_voxels * voxel_volume_mm3
    return volume_voxels, volume_mm3


def compute_diameter(
    mask: NDArray[np.bool_],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, float, float]:
    """Compute nodule diameter in each dimension.

    Args:
        mask: Binary mask of the nodule
        spacing: Physical voxel spacing in mm (z, y, x)

    Returns:
        Tuple of diameters in mm (z, y, x)
    """
    # Find extent in each dimension
    z_indices, y_indices, x_indices = np.where(mask)

    # Compute diameter as max extent in each dimension
    z_extent = (z_indices.max() - z_indices.min() + 1) if len(z_indices) > 0 else 0
    y_extent = (y_indices.max() - y_indices.min() + 1) if len(y_indices) > 0 else 0
    x_extent = (x_indices.max() - x_indices.min() + 1) if len(x_indices) > 0 else 0

    # Convert to physical units
    z_diameter = float(z_extent * spacing[0])
    y_diameter = float(y_extent * spacing[1])
    x_diameter = float(x_extent * spacing[2])

    return (z_diameter, y_diameter, x_diameter)


def compute_bounding_box(
    mask: NDArray[np.bool_],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Compute bounding box of a binary mask.

    Args:
        mask: Binary mask of the nodule

    Returns:
        Tuple of ((z_min, z_max), (y_min, y_max), (x_min, x_max))

    Raises:
        ValueError: If mask is empty
    """
    if not np.any(mask):
        msg = "Cannot compute bounding box of empty mask"
        raise ValueError(msg)

    z_indices, y_indices, x_indices = np.where(mask)

    z_min, z_max = int(z_indices.min()), int(z_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    x_min, x_max = int(x_indices.min()), int(x_indices.max())

    return ((z_min, z_max), (y_min, y_max), (x_min, x_max))
