"""Ground truth generation for nodule center heatmaps."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "HeatmapConfig",
    "generate_center_heatmap",
]


@dataclass(frozen=True, slots=True)
class HeatmapConfig:
    """Configuration for Gaussian heatmap generation."""

    sigma_mm: float = 3.0
    truncate: float = 3.0
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mode: Literal["max", "sum"] = "max"
    normalize: bool = True
    dtype: np.dtype = np.dtype(np.float32)


def generate_center_heatmap(
    volume_shape: Sequence[int],
    centers: Iterable[Sequence[float]],
    config: HeatmapConfig | None = None,
) -> np.ndarray:
    """Create a 3D heatmap with Gaussian peaks at the provided centers.

    Args:
        volume_shape: Target heatmap shape in ``(z, y, x)`` voxel order.
        centers: Iterable of ``(z, y, x)`` coordinates. Coordinates can be
            floating-point and are interpreted in voxel space.
        config: Optional ``HeatmapConfig`` overriding defaults.

    Returns:
        ``numpy.ndarray`` containing a normalised heatmap with values in
        ``[0, 1]`` when normalisation is enabled.

    Raises:
        ValueError: If ``volume_shape`` is not three-dimensional or if
            ``sigma_mm``/``truncate`` are non-positive.
    """

    cfg = config or HeatmapConfig()
    if len(volume_shape) != 3:
        raise ValueError("volume_shape must be three-dimensional")
    if cfg.sigma_mm <= 0:
        raise ValueError("sigma_mm must be positive")
    if cfg.truncate <= 0:
        raise ValueError("truncate must be positive")

    spacing = np.asarray(cfg.spacing, dtype=np.float32)
    if spacing.shape != (3,):
        raise ValueError("spacing must define three values (z, y, x)")
    if np.any(spacing <= 0):
        raise ValueError("spacing values must be positive")

    sigma_vox = np.asarray(cfg.sigma_mm / spacing, dtype=np.float32)
    radius = np.ceil(cfg.truncate * sigma_vox).astype(int)
    dtype = np.dtype(cfg.dtype)
    heatmap = np.zeros(tuple(int(dim) for dim in volume_shape), dtype=dtype)

    for raw_center in centers:
        center = np.asarray(raw_center, dtype=np.float32)
        if center.shape != (3,):
            raise ValueError("centers must contain (z, y, x) triplets")

        z_min = max(0, int(np.floor(center[0] - radius[0])))
        y_min = max(0, int(np.floor(center[1] - radius[1])))
        x_min = max(0, int(np.floor(center[2] - radius[2])))

        z_max = min(volume_shape[0], int(np.ceil(center[0] + radius[0]) + 1))
        y_max = min(volume_shape[1], int(np.ceil(center[1] + radius[1]) + 1))
        x_max = min(volume_shape[2], int(np.ceil(center[2] + radius[2]) + 1))

        if z_min >= z_max or y_min >= y_max or x_min >= x_max:
            continue

        z_coords = np.arange(z_min, z_max, dtype=np.float32) - center[0]
        y_coords = np.arange(y_min, y_max, dtype=np.float32) - center[1]
        x_coords = np.arange(x_min, x_max, dtype=np.float32) - center[2]

        z_term = (z_coords[:, None, None] ** 2) / (2.0 * sigma_vox[0] ** 2)
        y_term = (y_coords[None, :, None] ** 2) / (2.0 * sigma_vox[1] ** 2)
        x_term = (x_coords[None, None, :] ** 2) / (2.0 * sigma_vox[2] ** 2)
        patch = np.exp(-(z_term + y_term + x_term)).astype(dtype, copy=False)

        window = np.s_[z_min:z_max, y_min:y_max, x_min:x_max]
        if cfg.mode == "sum":
            heatmap[window] += patch
        else:
            np.maximum(heatmap[window], patch, out=heatmap[window])

    if cfg.normalize:
        max_value = float(heatmap.max())
        if max_value > 0:
            heatmap /= max_value

    return heatmap
