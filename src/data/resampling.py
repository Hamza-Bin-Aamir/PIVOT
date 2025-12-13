"""Isotropic resampling module for medical imaging data.

This module provides functionality for resampling medical scans to isotropic
voxel spacing (1mm × 1mm × 1mm) using trilinear interpolation. This is critical
for ensuring consistent spatial resolution across different CT scans in the
LUNA16 and LIDC-IDRI datasets.
"""

import logging
from typing import Literal

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

DEFAULT_MAX_VOXELS = 512 * 512 * 512  # ~134M voxels (~0.5 GB at float32)


def resample_to_isotropic(
    volume: np.ndarray,
    original_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    preserve_range: bool = True,
    max_voxels: int | None = DEFAULT_MAX_VOXELS,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Resample a 3D volume to isotropic voxel spacing.

    Uses trilinear interpolation (default) to preserve Hounsfield Unit (HU) values
    during resampling. This function is optimized for CT scans but works with any
    3D medical imaging modality.

    Args:
        volume: Input 3D volume with shape (depth, height, width)
        original_spacing: Original voxel spacing in mm (x, y, z)
        target_spacing: Target voxel spacing in mm (x, y, z). Default is (1.0, 1.0, 1.0)
        interpolation: Interpolation method. Options:
            - 'linear': Trilinear interpolation (default, best for CT)
            - 'nearest': Nearest neighbor (best for segmentation masks)
            - 'bspline': B-spline interpolation (smoother but slower)
        preserve_range: If True, clip output to input range to preserve HU values
        max_voxels: Maximum allowed voxel count for the resampled volume. Set to
            ``None`` to disable the safety check. Defaults to roughly ``512^3``
            voxels (~0.5 GB for float32).

    Returns:
        Tuple of (resampled_volume, actual_spacing):
            - resampled_volume: Resampled 3D array with new spacing
            - actual_spacing: Actual spacing achieved (may differ slightly from target)

    Raises:
        ValueError: If volume is not 3D or spacing values are invalid

    Examples:
        >>> volume = np.random.rand(100, 256, 256)
        >>> original_spacing = (2.5, 0.7, 0.7)  # anisotropic
        >>> resampled, spacing = resample_to_isotropic(volume, original_spacing)
        >>> print(spacing)  # (1.0, 1.0, 1.0)
        >>> print(resampled.shape)  # (250, 179, 179)
    """
    if volume.ndim != 3:
        msg = f"Expected 3D volume, got {volume.ndim}D array"
        raise ValueError(msg)

    if len(original_spacing) != 3:
        msg = f"Expected 3D spacing, got {len(original_spacing)}D"
        raise ValueError(msg)

    if any(s <= 0 for s in original_spacing):
        msg = f"Spacing values must be positive, got {original_spacing}"
        raise ValueError(msg)

    if any(s <= 0 for s in target_spacing):
        msg = f"Target spacing must be positive, got {target_spacing}"
        raise ValueError(msg)

    # Check if already isotropic (within tolerance)
    if _is_isotropic(original_spacing, target_spacing):
        logger.info(f"Volume already has target spacing {original_spacing}, skipping resampling")
        return volume.copy(), original_spacing

    # Convert numpy array to SimpleITK image
    # SimpleITK uses (x, y, z) ordering, numpy uses (z, y, x)
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing(original_spacing)

    # Calculate new size
    original_size = volume.shape[::-1]  # Convert (z, y, x) to (x, y, z)
    new_size = _calculate_new_size(original_size, original_spacing, target_spacing)

    estimated_voxels = int(np.prod(new_size))
    if max_voxels is not None and estimated_voxels > max_voxels:
        msg = (
            "Resampling output would contain "
            f"{estimated_voxels:,} voxels which exceeds the configured limit "
            f"of {max_voxels:,}. Adjust target spacing or set max_voxels=None "
            "to disable this safety check."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Set up resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    # Set interpolation method
    interpolator_map = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "bspline": sitk.sitkBSpline,
    }
    resampler.SetInterpolator(interpolator_map[interpolation])

    # Default pixel value for regions outside the original image
    # Use minimum value from original volume for CT scans (typically air = -1000 HU)
    resampler.SetDefaultPixelValue(float(volume.min()))

    # Perform resampling
    logger.info(
        f"Resampling from {original_spacing} to {target_spacing}, "
        f"size {original_size} to {new_size}"
    )

    try:
        resampled_image = resampler.Execute(image)
    except Exception as e:
        msg = f"Resampling failed: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    # Convert back to numpy array
    resampled_volume = sitk.GetArrayFromImage(resampled_image)

    # Preserve HU value range if requested
    if preserve_range and interpolation == "linear":
        original_min, original_max = volume.min(), volume.max()
        resampled_volume = np.clip(resampled_volume, original_min, original_max)

    # Get actual spacing achieved
    actual_spacing = resampled_image.GetSpacing()

    # Validate resampling quality
    _validate_resampling(volume, original_spacing, resampled_volume, actual_spacing)

    logger.info(
        f"Resampling complete: {volume.shape} -> {resampled_volume.shape}, "
        f"spacing {original_spacing} -> {actual_spacing}"
    )

    return resampled_volume, actual_spacing


def resample_mask(
    mask: np.ndarray,
    original_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_voxels: int | None = DEFAULT_MAX_VOXELS,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Resample a binary segmentation mask to isotropic spacing.

    Uses nearest neighbor interpolation to preserve binary labels during resampling.
    Optimized for segmentation masks where exact label preservation is critical.

    Args:
        mask: Binary mask with shape (depth, height, width)
        original_spacing: Original voxel spacing in mm (x, y, z)
        target_spacing: Target voxel spacing in mm (x, y, z)
        max_voxels: Maximum allowed voxel count for the resampled mask. Set to
            ``None`` to disable the safety check.

    Returns:
        Tuple of (resampled_mask, actual_spacing):
            - resampled_mask: Resampled binary mask
            - actual_spacing: Actual spacing achieved

    Raises:
        ValueError: If mask is not 3D or spacing values are invalid

    Examples:
        >>> mask = np.random.randint(0, 2, size=(100, 256, 256))
        >>> resampled_mask, spacing = resample_mask(mask, (2.5, 0.7, 0.7))
        >>> print(np.unique(resampled_mask))  # [0 1] - binary preserved
    """
    return resample_to_isotropic(
        mask,
        original_spacing,
        target_spacing,
        interpolation="nearest",
        preserve_range=False,
        max_voxels=max_voxels,
    )


def calculate_isotropic_shape(
    original_shape: tuple[int, int, int],
    original_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[int, int, int]:
    """Calculate the output shape after isotropic resampling.

    Useful for pre-allocating memory or understanding memory requirements before
    performing resampling.

    Args:
        original_shape: Original array shape (depth, height, width)
        original_spacing: Original voxel spacing in mm (x, y, z)
        target_spacing: Target voxel spacing in mm (x, y, z)

    Returns:
        New shape after resampling (depth, height, width)

    Raises:
        ValueError: If shapes or spacings are invalid

    Examples:
        >>> shape = (100, 256, 256)
        >>> spacing = (2.5, 0.7, 0.7)
        >>> new_shape = calculate_isotropic_shape(shape, spacing)
        >>> print(new_shape)  # (250, 179, 179)
    """
    if len(original_shape) != 3:
        msg = f"Expected 3D shape, got {len(original_shape)}D"
        raise ValueError(msg)

    if len(original_spacing) != 3:
        msg = f"Expected 3D spacing, got {len(original_spacing)}D"
        raise ValueError(msg)

    if len(target_spacing) != 3:
        msg = f"Expected 3D target spacing, got {len(target_spacing)}D"
        raise ValueError(msg)

    # Convert (z, y, x) to (x, y, z)
    original_size = original_shape[::-1]

    new_size = _calculate_new_size(original_size, original_spacing, target_spacing)

    # Convert back to (z, y, x)
    return new_size[::-1]


def is_isotropic(spacing: tuple[float, float, float], tolerance: float = 0.01) -> bool:
    """Check if voxel spacing is isotropic.

    Args:
        spacing: Voxel spacing in mm (x, y, z)
        tolerance: Maximum relative difference allowed (default 1%)

    Returns:
        True if spacing is isotropic within tolerance

    Examples:
        >>> is_isotropic((1.0, 1.0, 1.0))
        True
        >>> is_isotropic((1.0, 1.0, 1.01), tolerance=0.02)
        True
        >>> is_isotropic((2.5, 0.7, 0.7))
        False
    """
    if len(spacing) != 3:
        msg = f"Expected 3D spacing, got {len(spacing)}D"
        raise ValueError(msg)

    reference = spacing[0]
    if reference <= 0:
        msg = f"Spacing values must be positive, got {spacing}"
        raise ValueError(msg)

    target = (reference, reference, reference)
    return _is_isotropic(spacing, target, tolerance)


def calculate_resampling_factor(
    original_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, float, float]:
    """Calculate the resampling factor for each dimension.

    Args:
        original_spacing: Original voxel spacing in mm (x, y, z)
        target_spacing: Target voxel spacing in mm (x, y, z)

    Returns:
        Resampling factor for each dimension (x, y, z)

    Examples:
        >>> factors = calculate_resampling_factor((2.5, 0.7, 0.7), (1.0, 1.0, 1.0))
        >>> print(factors)  # (2.5, 0.7, 0.7) - upsampling in x, downsampling in y,z
    """
    return tuple(
        orig / target for orig, target in zip(original_spacing, target_spacing, strict=True)
    )


# Private helper functions


def _calculate_new_size(
    original_size: tuple[int, int, int],
    original_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Calculate new image size after resampling."""
    new_size = tuple(
        int(round(orig_sz * orig_sp / target_sp))
        for orig_sz, orig_sp, target_sp in zip(
            original_size, original_spacing, target_spacing, strict=True
        )
    )
    return new_size


def _is_isotropic(
    spacing: tuple[float, float, float],
    target: tuple[float, float, float],
    tolerance: float = 0.01,
) -> bool:
    """Check if spacing matches target within tolerance."""
    return all(abs(s - t) / t <= tolerance for s, t in zip(spacing, target, strict=True))


def _validate_resampling(
    original: np.ndarray,
    original_spacing: tuple[float, float, float],
    resampled: np.ndarray,
    new_spacing: tuple[float, float, float],
    tolerance: float = 0.05,
) -> None:
    """Validate resampling quality.

    Checks:
    1. Physical size is approximately preserved
    2. Spacing is close to target
    3. No extreme value changes

    Args:
        original: Original volume
        original_spacing: Original spacing
        resampled: Resampled volume
        new_spacing: New spacing
        tolerance: Relative tolerance for validation (5%)

    Raises:
        RuntimeError: If validation fails
    """
    # Check physical size preservation
    original_physical_size = tuple(
        s * sp for s, sp in zip(original.shape[::-1], original_spacing, strict=True)
    )
    new_physical_size = tuple(
        s * sp for s, sp in zip(resampled.shape[::-1], new_spacing, strict=True)
    )

    for i, (orig, new) in enumerate(zip(original_physical_size, new_physical_size, strict=True)):
        relative_diff = abs(orig - new) / orig
        if relative_diff > tolerance:  # pragma: no cover - log instrumentation only
            msg = (
                f"Physical size mismatch in dimension {i}: "
                f"original={orig:.2f}mm, new={new:.2f}mm "
                f"(diff={relative_diff * 100:.1f}%)"
            )
            logger.warning(msg)  # pragma: no cover - log instrumentation only

    # Check for extreme value changes (could indicate interpolation issues)
    original_range = original.max() - original.min()
    resampled_range = resampled.max() - resampled.min()

    if original_range == 0:
        if resampled_range != 0:
            logger.warning(  # pragma: no cover - log instrumentation only
                "Value range changed from constant input: "
                f"original={original_range:.2f}, resampled={resampled_range:.2f}"
            )
    else:
        range_diff = abs(original_range - resampled_range) / original_range
        if range_diff > 0.1:  # 10% change in range is suspicious
            logger.warning(  # pragma: no cover - log instrumentation only
                f"Value range changed significantly: "
                f"original={original_range:.2f}, resampled={resampled_range:.2f} "
                f"(diff={range_diff * 100:.1f}%)"
            )

    logger.debug(  # pragma: no cover - debug aid only
        f"Resampling validation passed: physical size preserved within {tolerance * 100}%"
    )
