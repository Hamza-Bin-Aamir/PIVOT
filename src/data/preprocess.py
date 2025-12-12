"""3D preprocessing pipeline for CT scans.

This module implements the preprocessing pipeline described in the README:
1. Resampling to isotropic resolution (1mm x 1mm x 1mm)
2. HU windowing (Level: -600 HU, Width: 1500 HU)
3. Normalization to [0, 1]
4. Patch generation (96, 96, 96)
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def resample_to_isotropic(
    image: sitk.Image,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    """Resample CT scan to isotropic resolution.

    Args:
        image: Input SimpleITK image
        target_spacing: Target spacing in mm (default: 1mm x 1mm x 1mm)
        interpolator: Interpolation method

    Returns:
        Resampled SimpleITK image
    """
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
    """Apply HU windowing to CT data.

    Standard lung window:
    - Center: -600 HU
    - Width: 1500 HU
    - Range: -1350 to +150 HU

    Args:
        array: Input array in HU
        window_center: Window center (level) in HU
        window_width: Window width in HU

    Returns:
        Windowed array
    """
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2

    windowed = np.clip(array, window_min, window_max)
    return windowed


def normalize_to_range(
    array: np.ndarray,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> np.ndarray:
    """Normalize array to target range.

    Args:
        array: Input array
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Normalized array
    """
    array_min = array.min()
    array_max = array.max()

    if array_max - array_min == 0:
        return np.full_like(array, target_min)

    normalized = (array - array_min) / (array_max - array_min)
    normalized = normalized * (target_max - target_min) + target_min

    return normalized


def preprocess_ct_scan(
    input_path: Path,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    window_center: int = -600,
    window_width: int = 1500,
) -> np.ndarray:
    """Complete preprocessing pipeline for a single CT scan.

    Args:
        input_path: Path to input file (.mhd or DICOM)
        target_spacing: Target isotropic spacing
        window_center: HU window center
        window_width: HU window width

    Returns:
        Preprocessed numpy array (normalized to [0, 1])
    """
    # Load image
    image = sitk.ReadImage(str(input_path))

    # Resample to isotropic
    resampled = resample_to_isotropic(image, target_spacing)

    # Convert to numpy array
    array = sitk.GetArrayFromImage(resampled)

    # Apply HU windowing
    windowed = apply_hu_windowing(array, window_center, window_width)

    # Normalize to [0, 1]
    normalized = normalize_to_range(windowed)

    return normalized


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
