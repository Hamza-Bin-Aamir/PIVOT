"""Intensity normalization utilities for medical imaging volumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from skimage import exposure

HistogramMethod = Literal["none", "global", "adaptive"]


@dataclass(slots=True)
class NormalizationStats:
    """Summary of intensity normalization."""

    original_min: float
    original_max: float
    window_min: float
    window_max: float
    target_min: float
    target_max: float
    clip_fraction: float


def clip_hounsfield(
    volume: np.ndarray,
    window_min: float = -1000.0,
    window_max: float = 400.0,
) -> np.ndarray:
    """Clip a CT volume to the configured Hounsfield Unit window."""

    if window_min >= window_max:
        msg = f"window_min must be < window_max, got ({window_min}, {window_max})"
        raise ValueError(msg)

    array = np.asarray(volume, dtype=np.float32).copy()
    np.clip(array, window_min, window_max, out=array)
    return array  # type: ignore[no-any-return]


def _min_max_scale(
    volume: np.ndarray,
    source_min: float,
    source_max: float,
    target_min: float,
    target_max: float,
) -> np.ndarray:
    """Scale an array from a fixed source range to a target range."""

    if target_min >= target_max:
        msg = f"target_min must be < target_max, got ({target_min}, {target_max})"
        raise ValueError(msg)

    scale = source_max - source_min
    if np.isclose(scale, 0.0):
        return np.full_like(volume, target_min)

    normalized = (volume - source_min) / scale
    normalized = normalized * (target_max - target_min) + target_min
    return normalized.astype(np.float32, copy=False)


def _equalize_histogram(
    volume: np.ndarray,
    method: HistogramMethod,
    clip_limit: float,
    kernel_size: tuple[int, ...] | int | None,
) -> np.ndarray:
    """Apply histogram equalization to an array."""

    if method == "none":
        return volume

    if method == "global":
        equalized = exposure.equalize_hist(volume)
    elif method == "adaptive":
        equalized = exposure.equalize_adapthist(
            volume, clip_limit=clip_limit, kernel_size=kernel_size
        )
    else:  # pragma: no cover - safeguarded by typing
        msg = f"Unsupported histogram method '{method}'"
        raise ValueError(msg)

    return np.asarray(equalized, dtype=np.float32)


def normalize_intensity(
    volume: np.ndarray,
    window: tuple[float, float] = (-1000.0, 400.0),
    target_range: tuple[float, float] = (0.0, 1.0),
    histogram_method: HistogramMethod = "none",
    adaptive_clip_limit: float = 0.01,
    adaptive_kernel_size: tuple[int, ...] | int | None = None,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, NormalizationStats]:
    """Normalize intensities to a target range with optional histogram equalization."""

    if volume.size == 0:
        raise ValueError("volume must contain at least one element")

    window_min, window_max = window
    target_min, target_max = target_range

    clipped = clip_hounsfield(volume, window_min, window_max)

    scaled = _min_max_scale(clipped, window_min, window_max, target_min, target_max)
    equalized = _equalize_histogram(
        scaled, histogram_method, adaptive_clip_limit, adaptive_kernel_size
    )

    if not return_stats:
        return equalized

    original = np.asarray(volume, dtype=np.float32)
    outside = np.count_nonzero((original < window_min) | (original > window_max))
    clip_fraction = float(outside) / float(original.size)

    stats = NormalizationStats(
        original_min=float(original.min()),
        original_max=float(original.max()),
        window_min=float(window_min),
        window_max=float(window_max),
        target_min=float(target_min),
        target_max=float(target_max),
        clip_fraction=clip_fraction,
    )

    return equalized, stats


__all__ = [
    "HistogramMethod",
    "NormalizationStats",
    "clip_hounsfield",
    "normalize_intensity",
]
