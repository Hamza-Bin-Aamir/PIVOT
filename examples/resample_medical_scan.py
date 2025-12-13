#!/usr/bin/env python3
"""Example script demonstrating isotropic resampling for medical scans.

This script loads a medical volume using :class:`MedicalScanLoader`, reports the
original spacing, and resamples the data to an isotropic grid using the
`src.data.resampling` utilities. Optionally, a segmentation mask can be
resampled alongside the image to ensure voxel-wise alignment.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from src.data.loader import MedicalScanLoader
from src.data.resampling import (
    calculate_isotropic_shape,
    resample_mask,
    resample_to_isotropic,
)


def parse_spacing(values: Iterable[str]) -> tuple[float, float, float]:
    """Parse a sequence of three floats representing voxel spacing."""
    floats = tuple(float(v) for v in values)
    if len(floats) != 3:
        msg = f"Expected three spacing values, received {len(floats)}"
        raise argparse.ArgumentTypeError(msg)
    return floats  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample a medical scan to isotropic spacing")
    parser.add_argument("input", type=Path, help="Path to the scan (file or directory)")
    parser.add_argument(
        "--mask",
        type=Path,
        help="Optional path to a binary mask aligned with the input scan",
    )
    parser.add_argument(
        "--target-spacing",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
        help="Target isotropic spacing in millimetres (default: 1 1 1)",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=None,
        help="Optional safety cap on the number of voxels in the resampled volume",
    )

    args = parser.parse_args()

    print("Loading volume...")
    volume, metadata = MedicalScanLoader.load(args.input)
    original_spacing = metadata.spacing
    print(f"Original shape: {volume.shape}")
    print(f"Original spacing: {original_spacing} mm")

    target_spacing = tuple(args.target_spacing)
    estimated_shape = calculate_isotropic_shape(volume.shape, original_spacing, target_spacing)
    estimated_voxels = int(np.prod(estimated_shape))
    print(f"Target spacing: {target_spacing} mm")
    print(f"Estimated output shape: {estimated_shape} (~{estimated_voxels:,} voxels)")

    resampled_volume, new_spacing = resample_to_isotropic(
        volume,
        original_spacing,
        target_spacing,
        max_voxels=args.max_voxels,
    )
    print("Resampling completed")
    print(f"Resampled shape: {resampled_volume.shape}")
    print(f"Actual spacing: {new_spacing} mm")

    if args.mask:
        print("\nLoading mask...")
        mask, _ = MedicalScanLoader.load(args.mask)
        resampled_mask, _ = resample_mask(
            mask,
            original_spacing,
            target_spacing,
            max_voxels=args.max_voxels,
        )
        print("Mask resampled to match the volume")
        print(f"Mask shape: {resampled_mask.shape}")
        print("Unique labels:", np.unique(resampled_mask))

    print("\nDone. Use numpy.save or SimpleITK to persist the resampled arrays if needed.")


if __name__ == "__main__":
    main()
