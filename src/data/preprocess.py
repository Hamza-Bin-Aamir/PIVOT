"""3D preprocessing pipeline for CT scans."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .intensity import (
    HistogramMethod,
    NormalizationStats,
    clip_hounsfield,
    normalize_intensity,
)


def resample_to_isotropic(
    image: sitk.Image,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    """Resample CT scan to isotropic resolution."""

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, target_spacing, strict=True)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(interpolator)

    return resample.Execute(image)


def apply_hu_windowing(
    array: np.ndarray,
    window_center: int = -600,
    window_width: int = 1500,
) -> np.ndarray:
    """Apply HU windowing to CT data."""

    window_min = float(window_center) - float(window_width) / 2.0
    window_max = float(window_center) + float(window_width) / 2.0
    return clip_hounsfield(array, window_min, window_max)


def normalize_to_range(
    array: np.ndarray,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> np.ndarray:
    """Normalize array to target range."""

    if target_min >= target_max:
        msg = f"target_min must be < target_max, got ({target_min}, {target_max})"
        raise ValueError(msg)

    array = np.asarray(array, dtype=np.float32)
    array_min = float(array.min())
    value_range = float(array.max()) - array_min
    if np.isclose(value_range, 0.0):
        return np.full_like(array, target_min)

    scaled = (array - array_min) / value_range
    scaled = scaled * (target_max - target_min) + target_min
    return scaled.astype(np.float32, copy=False)


def preprocess_ct_scan(
    input_path: Path,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    window_center: int = -600,
    window_width: int = 1500,
    histogram_method: HistogramMethod = "none",
    adaptive_clip_limit: float = 0.01,
    adaptive_kernel_size: tuple[int, ...] | int | None = None,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, NormalizationStats]:
    """Complete preprocessing pipeline for a single CT scan.

    Args:
        input_path: Path to input volume (SimpleITK-compatible).
        target_spacing: Target voxel spacing in millimetres.
        window_center: HU window centre for clipping.
        window_width: HU window width for clipping.
        histogram_method: Optional histogram equalisation strategy.
        adaptive_clip_limit: Clip limit when using adaptive histogram equalisation.
        adaptive_kernel_size: Kernel size for adaptive histogram equalisation.
        return_stats: If True, also return a stats dictionary summarising the pass.

    Returns:
        Either the normalised volume or a tuple of (volume, stats) when
        ``return_stats`` is True.
    """

    image = sitk.ReadImage(str(input_path))
    resampled = resample_to_isotropic(image, target_spacing)
    array = sitk.GetArrayFromImage(resampled)

    window_min = float(window_center) - float(window_width) / 2.0
    window_max = float(window_center) + float(window_width) / 2.0

    kwargs = {
        "window": (window_min, window_max),
        "target_range": (0.0, 1.0),
        "histogram_method": histogram_method,
        "adaptive_clip_limit": adaptive_clip_limit,
        "adaptive_kernel_size": adaptive_kernel_size,
    }

    if return_stats:
        return normalize_intensity(array, return_stats=True, **kwargs)

    return normalize_intensity(array, return_stats=False, **kwargs)


def main():
    """Entry point for preprocessing script."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess CT scans")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()
    print(f"Preprocessing CT scans from {args.input_dir} to {args.output_dir}")
    # TODO: Implement batch processing logic


if __name__ == "__main__":
    main()
