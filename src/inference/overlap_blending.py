"""Overlap blending strategies for merging predictions from overlapping patches."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


class BlendMode(str, Enum):
    """Blending strategies for overlapping predictions.

    - AVERAGE: Simple average of overlapping predictions
    - GAUSSIAN: Gaussian-weighted average (higher weight at patch center)
    - LINEAR: Linear distance-weighted average
    """

    AVERAGE = "average"
    GAUSSIAN = "gaussian"
    LINEAR = "linear"


class OverlapBlending:
    """Handle merging of overlapping patch predictions.

    This class provides methods to blend predictions from overlapping patches
    using different weighting strategies to avoid artifacts at patch boundaries.

    Attributes:
        window_size: Size of patches (depth, height, width).
        mode: Blending strategy to use.
    """

    def __init__(
        self,
        window_size: tuple[int, int, int],
        mode: BlendMode | str = BlendMode.AVERAGE,
    ) -> None:
        """Initialize overlap blending.

        Args:
        ----
            window_size: Patch size as (depth, height, width).
            mode: Blending mode (average, gaussian, or linear).

        Raises:
        ------
            ValueError: If window_size has invalid dimensions or mode is unknown.

        """
        if len(window_size) != 3:
            raise ValueError("window_size must be a 3-tuple")
        if any(s <= 0 for s in window_size):
            raise ValueError("window_size values must be positive")

        self.window_size = window_size

        if isinstance(mode, str):
            try:
                self.mode = BlendMode(mode)
            except ValueError as e:
                raise ValueError(
                    f"Unknown blending mode: {mode}. "
                    f"Must be one of {[m.value for m in BlendMode]}"
                ) from e
        else:
            self.mode = mode

        # Pre-compute blending weights
        self._weight_map = self._create_weight_map()

    def _create_weight_map(self) -> np.ndarray:
        """Create weight map for blending based on mode.

        Returns:
        -------
            3D weight array with shape window_size, values in [0, 1].

        """
        dw, hw, ww = self.window_size

        if self.mode == BlendMode.AVERAGE:
            return np.ones((dw, hw, ww), dtype=np.float32)

        if self.mode == BlendMode.GAUSSIAN:
            return self._gaussian_weights(dw, hw, ww)

        if self.mode == BlendMode.LINEAR:
            return self._linear_weights(dw, hw, ww)

        raise ValueError(f"Unknown blending mode: {self.mode}")

    @staticmethod
    def _gaussian_weights(d: int, h: int, w: int) -> NDArray[np.float32]:
        """Create Gaussian-weighted map.

        Args:
        ----
            d, h, w: Dimensions of weight map.

        Returns:
        -------
            3D Gaussian weight map.

        """
        # Create 1D Gaussian for each dimension
        sigma_ratio = 0.25  # Gaussian sigma as fraction of size

        gz = np.exp(-0.5 * ((np.arange(d) - d / 2) / (sigma_ratio * d)) ** 2)
        gy = np.exp(-0.5 * ((np.arange(h) - h / 2) / (sigma_ratio * h)) ** 2)
        gx = np.exp(-0.5 * ((np.arange(w) - w / 2) / (sigma_ratio * w)) ** 2)

        # Combine dimensions (outer product)
        weights = np.outer(gz, gy).reshape(d, h, 1) * gx.reshape(1, 1, w)

        # Normalize to [0, 1]
        weights = weights / weights.max()

        return cast(NDArray[np.float32], weights.astype(np.float32))

    @staticmethod
    def _linear_weights(d: int, h: int, w: int) -> NDArray[np.float32]:
        """Create linear distance-based weights.

        Args:
        ----
            d, h, w: Dimensions of weight map.

        Returns:
        -------
            3D linear weight map.

        """
        # Distance from patch center
        dz = np.abs(np.arange(d) - d / 2) / (d / 2)
        dy = np.abs(np.arange(h) - h / 2) / (h / 2)
        dx = np.abs(np.arange(w) - w / 2) / (w / 2)

        # Linear falloff from center: 1 at center, 0 at edges
        wz = np.maximum(1 - dz, 0)
        wy = np.maximum(1 - dy, 0)
        wx = np.maximum(1 - dx, 0)

        # Combine dimensions
        weights = np.outer(wz, wy).reshape(d, h, 1) * wx.reshape(1, 1, w)

        return cast(NDArray[np.float32], weights.astype(np.float32))

    def blend(
        self,
        predictions: list[np.ndarray],
        positions: list[tuple[int, int, int]],
        output_shape: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend overlapping patch predictions.

        Args:
        ----
            predictions: List of prediction arrays from patches.
            positions: List of (z, y, x) positions for each patch.
            output_shape: Target output shape (depth, height, width).

        Returns:
        -------
            Tuple of:
            - Blended prediction array
            - Weight/count array

        Raises:
        ------
            ValueError: If inputs are inconsistent or empty.

        """
        if not predictions:
            raise ValueError("predictions list cannot be empty")

        if len(predictions) != len(positions):
            raise ValueError("predictions and positions must have same length")

        # Get number of channels from first prediction
        num_channels = predictions[0].shape[0]
        dw, hw, ww = self.window_size

        # Initialize output
        output_shape_with_channels = (num_channels,) + output_shape
        blended = np.zeros(output_shape_with_channels, dtype=np.float32)
        weights = np.zeros(output_shape, dtype=np.float32)

        # Accumulate weighted predictions
        for pred, (z, y, x) in zip(predictions, positions, strict=True):
            if pred.shape[0] != num_channels:
                raise ValueError(
                    f"Inconsistent number of channels: "
                    f"expected {num_channels}, got {pred.shape[0]}"
                )

            # Check bounds
            if (z + dw > output_shape[0] or y + hw > output_shape[1] or
                    x + ww > output_shape[2]):
                raise ValueError(
                    f"Patch at position {(z, y, x)} exceeds output shape {output_shape}"
                )

            # Apply weights to prediction
            weighted_pred = pred * self._weight_map

            # Accumulate
            blended[:, z : z + dw, y : y + hw, x : x + ww] += weighted_pred
            weights[z : z + dw, y : y + hw, x : x + ww] += self._weight_map

        # Normalize by weights
        for c in range(num_channels):
            blended[c] = np.divide(
                blended[c],
                weights,
                where=weights > 0,
                out=np.zeros_like(blended[c]),
            )

        return blended, weights

    def blend_with_counts(
        self,
        assembled: np.ndarray,
        counts: np.ndarray,
    ) -> NDArray[np.float32]:
        """Apply blending to pre-assembled predictions.

        This is useful when predictions have already been assembled
        but without proper blending weights.

        Args:
        ----
            assembled: Already assembled prediction array.
            counts: Count/weight array from assembly.

        Returns:
        -------
            Blended prediction array.

        Raises:
        ------
            ValueError: If inputs have incompatible shapes.

        """
        if assembled.ndim != 4:
            raise ValueError("assembled must be 4D (channels, d, h, w)")

        if counts.ndim != 3:
            raise ValueError("counts must be 3D (d, h, w)")

        if assembled.shape[1:] != counts.shape:
            raise ValueError("assembled and counts spatial dimensions must match")

        num_channels = assembled.shape[0]
        blended = np.zeros_like(assembled)

        # Apply weight-based blending
        for c in range(num_channels):
            blended[c] = np.divide(
                assembled[c],
                counts,
                where=counts > 0,
                out=np.zeros_like(assembled[c]),
            )

        return blended
