"""Sliding window inference for patch-based 3D predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import nn


class SlidingWindowInference:
    """Perform sliding window inference on 3D volumes.

    This class extracts patches from a 3D volume using a sliding window,
    passes them through a model, and assembles the predictions back into
    the original volume space.

    Attributes:
        window_size: Size of each patch (depth, height, width).
        stride: Distance between patch centers. If None, defaults to window_size
            (non-overlapping patches).
    """

    def __init__(
        self,
        window_size: tuple[int, int, int],
        stride: tuple[int, int, int] | None = None,
    ) -> None:
        """Initialize sliding window inference.

        Args:
        ----
            window_size: Patch size as (depth, height, width).
            stride: Step size between patches. Defaults to window_size
                (non-overlapping).

        Raises:
        ------
            ValueError: If window_size or stride has invalid dimensions.

        """
        if len(window_size) != 3:
            raise ValueError("window_size must be a 3-tuple")
        if any(s <= 0 for s in window_size):
            raise ValueError("window_size values must be positive")

        self.window_size = window_size

        if stride is None:
            self.stride = window_size
        else:
            if len(stride) != 3:
                raise ValueError("stride must be a 3-tuple")
            if any(s <= 0 for s in stride):
                raise ValueError("stride values must be positive")
            self.stride = stride

    def extract_patches(
        self,
        volume: np.ndarray,
        pad_value: float = 0.0,
    ) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
        """Extract patches from a 3D volume using sliding window.

        Args:
        ----
            volume: 3D array with shape (depth, height, width).
            pad_value: Value to pad volume with if needed.

        Returns:
        -------
            Tuple of:
            - List of extracted patches
            - List of (z, y, x) positions for each patch

        Raises:
        ------
            ValueError: If volume has incorrect shape.

        """
        if volume.ndim != 3:
            raise ValueError("volume must be 3D")

        # Pad volume if necessary
        padded_volume = self._pad_volume(volume, pad_value)
        patches = []
        positions = []

        d, h, w = self.stride
        dw, hw, ww = self.window_size

        # Extract patches with sliding window
        for z in range(0, padded_volume.shape[0] - dw + 1, d):
            for y in range(0, padded_volume.shape[1] - hw + 1, h):
                for x in range(0, padded_volume.shape[2] - ww + 1, w):
                    patch = padded_volume[z : z + dw, y : y + hw, x : x + ww]
                    patches.append(patch.copy())
                    positions.append((z, y, x))

        return patches, positions

    def _pad_volume(
        self,
        volume: np.ndarray,
        pad_value: float,
    ) -> np.ndarray:
        """Pad volume so it's divisible by stride.

        Args:
        ----
            volume: Input 3D array.
            pad_value: Value to pad with.

        Returns:
        -------
            Padded volume.

        """
        current_shape = np.array(volume.shape)
        window = np.array(self.window_size)
        stride = np.array(self.stride)

        # Calculate how much padding is needed
        pad_shape = current_shape.copy()
        for i in range(3):
            if pad_shape[i] < window[i]:
                pad_shape[i] = window[i]
            else:
                remainder = (pad_shape[i] - window[i]) % stride[i]
                if remainder != 0:
                    pad_shape[i] += stride[i] - remainder

        # Create padding tuples
        pad_widths = []
        for i in range(3):
            deficit = pad_shape[i] - current_shape[i]
            if deficit == 0:
                pad_widths.append((0, 0))
            else:
                # Center the original volume in padding
                pad_before = deficit // 2
                pad_after = deficit - pad_before
                pad_widths.append((pad_before, pad_after))

        if any(p != (0, 0) for p in pad_widths):
            padded = np.pad(
                volume,
                tuple(pad_widths),
                mode="constant",
                constant_values=pad_value,
            )
        else:
            padded = volume.copy()

        return padded

    def __call__(
        self,
        volume: np.ndarray,
        model: nn.Module,
        device: torch.device | str = "cpu",
        pad_value: float = 0.0,
        return_logits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on a 3D volume using sliding window.

        Args:
        ----
            volume: Input 3D array with shape (depth, height, width).
            model: PyTorch model to use for inference.
            device: Device to run model on.
            pad_value: Value to pad volume with.
            return_logits: If True, return raw logits instead of probabilities.

        Returns:
        -------
            Tuple of:
            - Prediction array with same shape as input volume
            - Count array for overlap handling

        """
        if volume.ndim != 3:
            raise ValueError("volume must be 3D")

        # Get patches
        patches, positions = self.extract_patches(volume, pad_value)

        if not patches:
            raise ValueError("No patches extracted from volume")

        # Run inference on patches
        predictions = []
        with torch.no_grad():
            for patch in patches:
                # Normalize to [0, 1] if needed
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
                patch_tensor = patch_tensor.to(device)

                # Model inference
                output = model(patch_tensor)

                # Apply softmax/sigmoid if not returning logits
                if not return_logits:
                    if output.shape[1] > 1:  # Multi-class: apply softmax
                        output = F.softmax(output, dim=1)
                    else:  # Binary: apply sigmoid
                        output = torch.sigmoid(output)

                predictions.append(output.cpu().numpy())

        # Assemble predictions
        assembled = self._assemble_predictions(
            volume.shape,
            predictions,
            positions,
            pad_value,
        )

        return assembled

    def _assemble_predictions(
        self,
        original_shape: tuple[int, int, int],
        predictions: list[np.ndarray],
        positions: list[tuple[int, int, int]],
        pad_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble patch predictions back into volume.

        Args:
        ----
            original_shape: Shape of original (unpadded) volume.
            predictions: List of prediction arrays from model.
            positions: List of (z, y, x) positions for each patch.
            pad_value: Padding value used.

        Returns:
        -------
            Tuple of assembled prediction and count arrays.

        """
        # Get padded shape
        padded_volume = np.zeros(original_shape)
        padded_volume[:] = pad_value
        padded_volume = self._pad_volume(padded_volume, pad_value)
        padded_shape = padded_volume.shape

        # Get number of output channels from first prediction (batch, channels, d, h, w)
        first_pred = predictions[0]
        num_channels = 1 if first_pred.ndim == 4 else first_pred.shape[1]

        dw, hw, ww = self.window_size        # Initialize output arrays
        assembled_shape = (num_channels,) + padded_shape
        assembled = np.zeros(assembled_shape, dtype=np.float32)
        count = np.zeros(padded_shape, dtype=np.float32)

        # Accumulate predictions
        for pred, (z, y, x) in zip(predictions, positions, strict=True):
            pred = pred[0]  # Remove batch dimension
            if pred.ndim == 3:
                # Single channel - add channel dimension
                pred = pred[np.newaxis, ...]
            assembled[:, z : z + dw, y : y + hw, x : x + ww] += pred
            count[z : z + dw, y : y + hw, x : x + ww] += 1.0

        # Average overlapping regions
        for c in range(num_channels):
            assembled[c] = np.divide(
                assembled[c],
                count,
                where=count > 0,
                out=np.zeros_like(assembled[c]),
            )

        # Crop back to original shape
        pad_widths = self._get_pad_widths(original_shape, padded_shape)
        assembled = self._crop_array(assembled, pad_widths)
        count = self._crop_array(count, pad_widths)

        return assembled, count

    @staticmethod
    def _get_pad_widths(
        original_shape: tuple[int, int, int],
        padded_shape: tuple[int, int, int],
    ) -> list[tuple[int, int]]:
        """Calculate padding amounts for each dimension.

        Args:
        ----
            original_shape: Shape before padding.
            padded_shape: Shape after padding.

        Returns:
        -------
            List of (pad_before, pad_after) tuples.

        """
        pad_widths = []
        for orig, padded in zip(original_shape, padded_shape, strict=True):
            deficit = padded - orig
            if deficit == 0:
                pad_widths.append((0, 0))
            else:
                pad_before = deficit // 2
                pad_after = deficit - pad_before
                pad_widths.append((pad_before, pad_after))
        return pad_widths

    @staticmethod
    def _crop_array(
        array: np.ndarray,
        pad_widths: list[tuple[int, int]],
    ) -> np.ndarray:
        """Crop array to remove padding.

        Args:
        ----
            array: Array to crop (may include channel dimension).
            pad_widths: Padding amounts for each spatial dimension.

        Returns:
        -------
            Cropped array.

        """
        has_channel = array.ndim == 4
        if has_channel:
            # Handle channel dimension
            cropped = array
            for i, (pad_before, pad_after) in enumerate(pad_widths, 1):
                if pad_after == 0:
                    cropped = np.take(
                        cropped,
                        range(pad_before, cropped.shape[i]),
                        axis=i,
                    )
                else:
                    cropped = np.take(
                        cropped,
                        range(pad_before, cropped.shape[i] - pad_after),
                        axis=i,
                    )
            return cropped

        # 3D array
        cropped = array
        for i, (pad_before, pad_after) in enumerate(pad_widths):
            if pad_after == 0:
                cropped = np.take(
                    cropped,
                    range(pad_before, cropped.shape[i]),
                    axis=i,
                )
            else:
                cropped = np.take(
                    cropped,
                    range(pad_before, cropped.shape[i] - pad_after),
                    axis=i,
                )
        return cropped
