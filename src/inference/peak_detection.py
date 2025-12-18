"""Peak detection for center heatmap post-processing."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import ndimage

__all__ = [
    "detect_peaks_3d",
    "non_maximum_suppression_3d",
    "extract_peak_coordinates",
]

logger = logging.getLogger(__name__)


def detect_peaks_3d(
    heatmap: NDArray[np.float32] | torch.Tensor,
    min_confidence: float = 0.1,
    nms_kernel_size: int = 3,
    max_peaks: int | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Detect peaks in a 3D center heatmap using non-maximum suppression.

    Args:
        heatmap: 3D probability heatmap of shape (D, H, W) or (1, 1, D, H, W)
        min_confidence: Minimum confidence threshold for peak detection
        nms_kernel_size: Kernel size for non-maximum suppression (must be odd)
        max_peaks: Maximum number of peaks to return (returns top-k by confidence)

    Returns:
        Tuple of (coordinates, confidences):
            - coordinates: (N, 3) array of (z, y, x) peak positions
            - confidences: (N,) array of confidence scores

    Raises:
        ValueError: If heatmap has invalid shape or parameters are invalid
    """
    # Convert to numpy if needed
    heatmap_np = heatmap.detach().cpu().numpy() if isinstance(heatmap, torch.Tensor) else heatmap

    # Handle batched input (1, 1, D, H, W) -> (D, H, W)
    if heatmap_np.ndim == 5:
        if heatmap_np.shape[0] != 1 or heatmap_np.shape[1] != 1:
            msg = f"Expected single-channel heatmap, got shape {heatmap_np.shape}"
            raise ValueError(msg)
        heatmap_np = heatmap_np[0, 0]
    elif heatmap_np.ndim == 4:
        if heatmap_np.shape[0] != 1:
            msg = f"Expected single-channel heatmap, got shape {heatmap_np.shape}"
            raise ValueError(msg)
        heatmap_np = heatmap_np[0]
    elif heatmap_np.ndim != 3:
        msg = f"Expected 3D heatmap, got {heatmap_np.ndim}D"
        raise ValueError(msg)

    # Validate parameters
    if not 0 <= min_confidence <= 1:
        msg = f"min_confidence must be in [0, 1], got {min_confidence}"
        raise ValueError(msg)

    if nms_kernel_size < 1 or nms_kernel_size % 2 == 0:
        msg = f"nms_kernel_size must be odd and positive, got {nms_kernel_size}"
        raise ValueError(msg)

    if max_peaks is not None and max_peaks < 1:
        msg = f"max_peaks must be positive, got {max_peaks}"
        raise ValueError(msg)

    # Apply non-maximum suppression
    peaks_mask = non_maximum_suppression_3d(heatmap_np, kernel_size=nms_kernel_size)

    # Apply confidence threshold
    peaks_mask &= heatmap_np >= min_confidence

    # Extract coordinates and confidences
    coordinates, confidences = extract_peak_coordinates(heatmap_np, peaks_mask)

    # Limit number of peaks if requested
    if max_peaks is not None and len(coordinates) > max_peaks:
        # Sort by confidence descending
        top_indices = np.argsort(confidences)[::-1][:max_peaks]
        coordinates = coordinates[top_indices]
        confidences = confidences[top_indices]

    logger.info(f"Detected {len(coordinates)} peaks (threshold={min_confidence:.3f})")
    return coordinates, confidences


def non_maximum_suppression_3d(
    heatmap: NDArray[np.float32],
    kernel_size: int = 3,
    mode: Literal["constant", "reflect", "nearest"] = "constant",
) -> NDArray[np.bool_]:
    """Apply 3D non-maximum suppression to find local maxima.

    Args:
        heatmap: 3D probability heatmap of shape (D, H, W)
        kernel_size: Size of the local maximum filter (must be odd)
        mode: Border mode for max filter

    Returns:
        Boolean mask where True indicates local maxima

    Raises:
        ValueError: If kernel_size is invalid
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        msg = f"kernel_size must be odd and positive, got {kernel_size}"
        raise ValueError(msg)

    # Apply maximum filter to find local maxima
    local_max = ndimage.maximum_filter(
        heatmap,
        size=kernel_size,
        mode=mode,
    )

    # A point is a peak if it equals the local maximum
    peaks = heatmap == local_max

    # Exclude points where the maximum is zero (flat regions)
    peaks &= local_max > 0

    return peaks  # type: ignore[no-any-return]


def extract_peak_coordinates(
    heatmap: NDArray[np.float32],
    peaks_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Extract coordinates and confidence values for detected peaks.

    Args:
        heatmap: 3D probability heatmap of shape (D, H, W)
        peaks_mask: Boolean mask of peak locations

    Returns:
        Tuple of (coordinates, confidences):
            - coordinates: (N, 3) array of (z, y, x) peak positions
            - confidences: (N,) array of confidence scores

    Raises:
        ValueError: If heatmap and peaks_mask have mismatched shapes
    """
    if heatmap.shape != peaks_mask.shape:
        msg = f"Shape mismatch: heatmap {heatmap.shape} vs mask {peaks_mask.shape}"
        raise ValueError(msg)

    # Find peak locations
    peak_indices = np.argwhere(peaks_mask)

    if len(peak_indices) == 0:
        # No peaks found
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

    # Extract confidence values
    confidences = heatmap[peaks_mask].astype(np.float32)

    # Convert to (z, y, x) coordinates
    coordinates = peak_indices.astype(np.float32)

    # Sort by confidence descending
    sort_indices = np.argsort(confidences)[::-1]
    coordinates = coordinates[sort_indices]
    confidences = confidences[sort_indices]

    return coordinates, confidences
