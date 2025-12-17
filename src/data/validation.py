"""Data validation and quality checks for CT scans and annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    message: str
    severity: str = "error"  # "error", "warning", "info"

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_valid


@dataclass
class VolumeQualityMetrics:
    """Quality metrics for a CT volume."""

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    intensity_range: tuple[float, float]
    intensity_mean: float
    intensity_std: float
    has_nan: bool
    has_inf: bool
    num_zero_slices: int
    is_isotropic: bool
    estimated_file_size_mb: float


def validate_volume_shape(
    volume: np.ndarray,
    min_size: tuple[int, int, int] = (32, 32, 32),
    max_size: tuple[int, int, int] | None = None,
) -> ValidationResult:
    """Validate volume dimensions are within acceptable ranges.

    Args:
        volume: 3D numpy array to validate
        min_size: Minimum acceptable dimensions (z, y, x)
        max_size: Maximum acceptable dimensions, or None for no limit

    Returns:
        ValidationResult indicating if volume shape is valid
    """
    if volume.ndim != 3:
        return ValidationResult(False, f"Expected 3D volume, got {volume.ndim}D array", "error")

    z, y, x = volume.shape

    if z < min_size[0] or y < min_size[1] or x < min_size[2]:
        return ValidationResult(
            False,
            f"Volume shape {volume.shape} is smaller than minimum {min_size}",
            "error",
        )

    if max_size is not None and (z > max_size[0] or y > max_size[1] or x > max_size[2]):
        return ValidationResult(
            False,
            f"Volume shape {volume.shape} exceeds maximum {max_size}",
            "error",
        )

    return ValidationResult(True, f"Volume shape {volume.shape} is valid", "info")


def validate_spacing(
    spacing: tuple[float, float, float],
    min_spacing: float = 0.1,
    max_spacing: float = 10.0,
    warn_anisotropic: bool = True,
    anisotropy_threshold: float = 0.5,
) -> list[ValidationResult]:
    """Validate voxel spacing is reasonable for CT scans.

    Args:
        spacing: Voxel spacing in mm (z, y, x)
        min_spacing: Minimum acceptable spacing in mm
        max_spacing: Maximum acceptable spacing in mm
        warn_anisotropic: Whether to warn about anisotropic spacing
        anisotropy_threshold: Maximum ratio difference for isotropy warning

    Returns:
        List of validation results
    """
    results = []

    # Check for valid spacing values
    if len(spacing) != 3:
        results.append(
            ValidationResult(False, f"Expected 3 spacing values, got {len(spacing)}", "error")
        )
        return results

    if any(s <= 0 for s in spacing):
        results.append(
            ValidationResult(False, f"Spacing values must be positive, got {spacing}", "error")
        )
        return results

    # Check spacing range
    if any(s < min_spacing for s in spacing):
        results.append(
            ValidationResult(
                False,
                f"Spacing {spacing} has values below minimum {min_spacing} mm",
                "error",
            )
        )

    if any(s > max_spacing for s in spacing):
        results.append(
            ValidationResult(
                False,
                f"Spacing {spacing} has values above maximum {max_spacing} mm",
                "error",
            )
        )

    # Check for anisotropic spacing
    if warn_anisotropic and len(results) == 0:
        min_s, max_s = min(spacing), max(spacing)
        if max_s > 0:
            ratio = max_s / min_s - 1
            if ratio > anisotropy_threshold:
                results.append(
                    ValidationResult(
                        True,
                        f"Anisotropic spacing detected: {spacing} (ratio: {ratio:.2f})",
                        "warning",
                    )
                )

    if not results:
        results.append(ValidationResult(True, f"Spacing {spacing} is valid", "info"))

    return results


def validate_intensity_range(
    volume: np.ndarray,
    min_hu: float = -2000,
    max_hu: float = 3000,
    warn_unusual: bool = True,
) -> list[ValidationResult]:
    """Validate intensity values are in expected Hounsfield Unit range.

    Args:
        volume: 3D numpy array with CT intensities
        min_hu: Minimum expected HU value
        max_hu: Maximum expected HU value
        warn_unusual: Warn about unusual intensity distributions

    Returns:
        List of validation results
    """
    results = []

    # Check for NaN or Inf
    if np.isnan(volume).any():
        results.append(
            ValidationResult(
                False,
                f"Volume contains {np.isnan(volume).sum()} NaN values",
                "error",
            )
        )

    if np.isinf(volume).any():
        results.append(
            ValidationResult(
                False,
                f"Volume contains {np.isinf(volume).sum()} infinite values",
                "error",
            )
        )

    if results:  # Don't continue if we have NaN/Inf
        return results

    # Check intensity range
    vol_min, vol_max = float(np.min(volume)), float(np.max(volume))

    if vol_min < min_hu:
        results.append(
            ValidationResult(
                False,
                f"Minimum intensity {vol_min:.1f} below expected range ({min_hu} HU)",
                "error",
            )
        )

    if vol_max > max_hu:
        results.append(
            ValidationResult(
                False,
                f"Maximum intensity {vol_max:.1f} above expected range ({max_hu} HU)",
                "error",
            )
        )

    # Warn about unusual distributions
    if warn_unusual and not results:
        vol_mean = float(np.mean(volume))
        vol_std = float(np.std(volume))

        # Typical lung CT should have mean around -500 HU (air + tissue)
        if vol_mean < -900 or vol_mean > 100:
            results.append(
                ValidationResult(
                    True,
                    f"Unusual mean intensity: {vol_mean:.1f} HU (expected -500 to 0)",
                    "warning",
                )
            )

        # Check for suspiciously low variance (constant image)
        if vol_std < 10:
            results.append(
                ValidationResult(
                    True,
                    f"Very low intensity variance: {vol_std:.1f} (possible constant image)",
                    "warning",
                )
            )

    if not results:
        results.append(
            ValidationResult(
                True, f"Intensity range [{vol_min:.1f}, {vol_max:.1f}] HU is valid", "info"
            )
        )

    return results


def validate_annotation_bounds(
    annotation_coords: np.ndarray,
    volume_shape: tuple[int, int, int],
    allow_margin: int = 0,
) -> ValidationResult:
    """Validate that annotation coordinates are within volume bounds.

    Args:
        annotation_coords: Nx3 array of (z, y, x) coordinates
        volume_shape: Shape of the volume (z, y, x)
        allow_margin: Allow coordinates this many voxels outside bounds

    Returns:
        ValidationResult indicating if all annotations are in bounds
    """
    if annotation_coords.size == 0:
        return ValidationResult(True, "No annotations to validate", "info")

    if annotation_coords.ndim != 2 or annotation_coords.shape[1] != 3:
        return ValidationResult(
            False,
            f"Expected Nx3 annotation array, got shape {annotation_coords.shape}",
            "error",
        )

    # Check each dimension
    z_max, y_max, x_max = volume_shape
    out_of_bounds = []

    for i, (z, y, x) in enumerate(annotation_coords):
        if not (-allow_margin <= z < z_max + allow_margin):
            out_of_bounds.append((i, "z", z, z_max))
        if not (-allow_margin <= y < y_max + allow_margin):
            out_of_bounds.append((i, "y", y, y_max))
        if not (-allow_margin <= x < x_max + allow_margin):
            out_of_bounds.append((i, "x", x, x_max))

    if out_of_bounds:
        examples = ", ".join(
            f"coord[{idx}].{dim}={val:.1f} (max={max_val})"
            for idx, dim, val, max_val in out_of_bounds[:5]
        )
        return ValidationResult(
            False,
            f"{len(out_of_bounds)} annotations out of bounds. Examples: {examples}",
            "error",
        )

    return ValidationResult(
        True,
        f"All {len(annotation_coords)} annotations within volume bounds",
        "info",
    )


def validate_mask_consistency(
    mask: np.ndarray,
    volume: np.ndarray,
    max_mask_ratio: float = 0.5,
) -> list[ValidationResult]:
    """Validate segmentation mask is consistent with volume.

    Args:
        mask: Binary or multi-class segmentation mask
        volume: Original CT volume
        max_mask_ratio: Maximum ratio of masked voxels (sanity check)

    Returns:
        List of validation results
    """
    results = []

    # Check shape consistency
    if mask.shape != volume.shape:
        results.append(
            ValidationResult(
                False,
                f"Mask shape {mask.shape} doesn't match volume shape {volume.shape}",
                "error",
            )
        )
        return results

    # Check mask values
    unique_vals = np.unique(mask)
    if not np.all(unique_vals >= 0):
        results.append(
            ValidationResult(
                False,
                f"Mask contains negative values: {unique_vals[unique_vals < 0]}",
                "error",
            )
        )

    # Check mask ratio
    if mask.dtype == bool or len(unique_vals) == 2:
        # Binary mask
        foreground_ratio = float(np.mean(mask > 0))
        if foreground_ratio > max_mask_ratio:
            results.append(
                ValidationResult(
                    True,
                    f"Mask covers {foreground_ratio:.1%} of volume (>{max_mask_ratio:.1%})",
                    "warning",
                )
            )
        elif foreground_ratio == 0:
            results.append(
                ValidationResult(True, "Mask is completely empty (no foreground)", "warning")
            )

    if not results:
        results.append(
            ValidationResult(True, f"Mask with {len(unique_vals)} classes is valid", "info")
        )

    return results


def check_zero_slices(
    volume: np.ndarray,
    axis: int = 0,
    threshold: float = 1e-6,
) -> tuple[int, list[int]]:
    """Detect slices that are entirely zero or near-zero.

    Args:
        volume: 3D volume to check
        axis: Axis along which to check slices (0=z, 1=y, 2=x)
        threshold: Values below this are considered zero

    Returns:
        Tuple of (count, indices) of zero slices
    """
    zero_indices = []

    for i in range(volume.shape[axis]):
        if axis == 0:
            slice_data = volume[i, :, :]
        elif axis == 1:
            slice_data = volume[:, i, :]
        else:
            slice_data = volume[:, :, i]

        if np.max(np.abs(slice_data)) < threshold:
            zero_indices.append(i)

    return len(zero_indices), zero_indices


def compute_quality_metrics(
    volume: np.ndarray,
    spacing: tuple[float, float, float] | None = None,
    origin: tuple[float, float, float] | None = None,
    direction: tuple[float, ...] | None = None,
) -> VolumeQualityMetrics:
    """Compute comprehensive quality metrics for a CT volume.

    Args:
        volume: 3D CT volume
        spacing: Voxel spacing in mm
        origin: Volume origin in physical coordinates
        direction: Volume orientation matrix

    Returns:
        VolumeQualityMetrics with detailed quality information
    """
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    if origin is None:
        origin = (0.0, 0.0, 0.0)
    if direction is None:
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # Check for problematic values
    has_nan = bool(np.isnan(volume).any())
    has_inf = bool(np.isinf(volume).any())

    # Basic statistics (ignore inf/nan to avoid runtime warnings)
    finite_mask = np.isfinite(volume)
    if finite_mask.any():
        finite_values = volume[finite_mask]
        with np.errstate(invalid="ignore"):
            intensity_range = (
                float(np.min(finite_values)),
                float(np.max(finite_values)),
            )
            intensity_mean = float(np.mean(finite_values))
            intensity_std = float(np.std(finite_values))
    else:
        intensity_range = (float("nan"), float("nan"))
        intensity_mean = float("nan")
        intensity_std = float("nan")

    # Count zero slices
    num_zero_slices, _ = check_zero_slices(volume, axis=0)

    # Check isotropy
    min_s, max_s = min(spacing), max(spacing)
    is_isotropic = (max_s / min_s - 1) < 0.1 if max_s > 0 else True

    # Estimate file size (rough approximation)
    num_voxels = np.prod(volume.shape)
    estimated_file_size_mb = num_voxels * volume.itemsize / (1024 * 1024)

    return VolumeQualityMetrics(
        shape=volume.shape,
        spacing=spacing,
        origin=origin,
        direction=direction,
        intensity_range=intensity_range,
        intensity_mean=intensity_mean,
        intensity_std=intensity_std,
        has_nan=has_nan,
        has_inf=has_inf,
        num_zero_slices=num_zero_slices,
        is_isotropic=is_isotropic,
        estimated_file_size_mb=estimated_file_size_mb,
    )


def validate_sitk_image(
    image: sitk.Image,
    min_size: tuple[int, int, int] = (32, 32, 32),
    max_size: tuple[int, int, int] | None = None,
) -> list[ValidationResult]:
    """Validate a SimpleITK image object.

    Args:
        image: SimpleITK image to validate
        min_size: Minimum acceptable dimensions
        max_size: Maximum acceptable dimensions

    Returns:
        List of validation results
    """
    results = []

    # Get volume as numpy array
    volume = sitk.GetArrayFromImage(image)

    # Validate shape
    results.append(validate_volume_shape(volume, min_size, max_size))

    # Validate spacing
    spacing = image.GetSpacing()[::-1]  # ITK is x,y,z; we want z,y,x
    results.extend(validate_spacing(spacing))

    # Validate intensity range
    results.extend(validate_intensity_range(volume))

    return results


def validate_file_exists(
    file_path: Path | str,
    extensions: list[str] | None = None,
) -> ValidationResult:
    """Validate that a file exists and has the correct extension.

    Args:
        file_path: Path to file
        extensions: List of valid extensions (e.g., ['.nii', '.nii.gz'])

    Returns:
        ValidationResult indicating if file is valid
    """
    path = Path(file_path)

    if not path.exists():
        return ValidationResult(False, f"File not found: {path}", "error")

    if not path.is_file():
        return ValidationResult(False, f"Path is not a file: {path}", "error")

    if extensions is not None and not any(str(path).endswith(ext) for ext in extensions):
        return ValidationResult(
            False,
            f"File {path.name} doesn't have valid extension. Expected: {extensions}",
            "error",
        )

    return ValidationResult(True, f"File {path.name} exists and is valid", "info")


def validate_dataset_structure(
    data_dir: Path | str,
    required_subdirs: list[str] | None = None,
    required_files: list[str] | None = None,
) -> list[ValidationResult]:
    """Validate dataset directory structure.

    Args:
        data_dir: Root directory of dataset
        required_subdirs: List of required subdirectories
        required_files: List of required files in root

    Returns:
        List of validation results
    """
    results = []
    data_path = Path(data_dir)

    # Check root directory exists
    if not data_path.exists():
        results.append(
            ValidationResult(False, f"Dataset directory not found: {data_path}", "error")
        )
        return results

    if not data_path.is_dir():
        results.append(ValidationResult(False, f"Path is not a directory: {data_path}", "error"))
        return results

    # Check required subdirectories
    if required_subdirs:
        for subdir in required_subdirs:
            subdir_path = data_path / subdir
            if not subdir_path.exists():
                results.append(
                    ValidationResult(False, f"Required subdirectory missing: {subdir}", "error")
                )
            elif not subdir_path.is_dir():
                results.append(
                    ValidationResult(False, f"Required path is not a directory: {subdir}", "error")
                )

    # Check required files
    if required_files:
        for file in required_files:
            file_path = data_path / file
            if not file_path.exists():
                results.append(ValidationResult(False, f"Required file missing: {file}", "error"))
            elif not file_path.is_file():
                results.append(
                    ValidationResult(False, f"Required path is not a file: {file}", "error")
                )

    if not results:
        results.append(ValidationResult(True, f"Dataset structure at {data_path} is valid", "info"))

    return results


def summarize_validation_results(
    results: list[ValidationResult],
) -> dict[str, Any]:
    """Summarize a list of validation results.

    Args:
        results: List of validation results

    Returns:
        Dictionary with summary statistics
    """
    errors = [r for r in results if r.severity == "error" and not r.is_valid]
    warnings = [r for r in results if r.severity == "warning"]

    return {
        "total_checks": len(results),
        "passed": len([r for r in results if r.is_valid]),
        "failed": len(errors),
        "warnings": len(warnings),
        "errors": [r.message for r in errors],
        "warning_messages": [r.message for r in warnings],
        "is_valid": len(errors) == 0,
    }
