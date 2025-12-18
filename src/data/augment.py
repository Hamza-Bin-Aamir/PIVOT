"""3D data augmentation utilities for medical imaging volumes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

TensorLike = torch.Tensor


class Compose:
    """Compose multiple augmentation transforms into a single callable."""

    def __init__(self, transforms: Iterable) -> None:
        self.transforms = list(transforms)

    def __call__(self, tensor: TensorLike) -> TensorLike:
        result = tensor
        for transform in self.transforms:
            result = transform(result)
        return result


class RandomFlip3D:
    """Randomly flip a 3D volume across spatial dimensions."""

    def __init__(self, prob: float = 0.5, dims: Sequence[int] = (1, 2, 3)) -> None:
        if not 0.0 <= prob <= 1.0:
            msg = f"Probability must lie in [0, 1], got {prob}"
            raise ValueError(msg)
        self.prob = prob
        self.dims = tuple(dims)

    def __call__(self, tensor: TensorLike) -> TensorLike:
        if tensor.ndim != 4:
            msg = f"Expected tensor with shape (C, D, H, W); got {tensor.shape}"
            raise ValueError(msg)

        result = tensor.clone()
        rand = torch.rand(len(self.dims), device=result.device)
        for idx, dim in enumerate(self.dims):
            if rand[idx] < self.prob:
                result = torch.flip(result, dims=(dim,))
        return result


class RandomRotate90:
    """Randomly rotate a 3D volume in 90° increments."""

    def __init__(self, prob: float = 0.5) -> None:
        if not 0.0 <= prob <= 1.0:
            msg = f"Probability must lie in [0, 1], got {prob}"
            raise ValueError(msg)
        self.prob = prob
        self.axes = ((1, 2), (1, 3), (2, 3))

    def __call__(self, tensor: TensorLike) -> TensorLike:
        if tensor.ndim != 4:
            msg = f"Expected tensor with shape (C, D, H, W); got {tensor.shape}"
            raise ValueError(msg)

        result = tensor.clone()
        if torch.rand(1, device=result.device) >= self.prob:
            return result

        axis_idx = torch.randint(len(self.axes), (1,), device=result.device).item()
        k = int(torch.randint(1, 4, (1,), device=result.device).item())
        axes = self.axes[int(axis_idx)]
        rotated = torch.rot90(result, k=k, dims=axes)
        # Restore canonical axis order (C, D, H, W) after rotation swaps dims.
        dims_order = list(range(rotated.ndim))
        dims_order[axes[0]], dims_order[axes[1]] = dims_order[axes[1]], dims_order[axes[0]]
        return rotated.permute(dims_order).contiguous()


class RandomGaussianNoise:
    """Inject Gaussian noise into a 3D tensor."""

    def __init__(self, prob: float = 0.15, mean: float = 0.0, std: float = 0.01) -> None:
        if not 0.0 <= prob <= 1.0:
            msg = f"Probability must lie in [0, 1], got {prob}"
            raise ValueError(msg)
        if std < 0:
            msg = f"Standard deviation must be non-negative, got {std}"
            raise ValueError(msg)
        self.prob = prob
        self.mean = mean
        self.std = std

    def __call__(self, tensor: TensorLike) -> TensorLike:
        if torch.rand(1, device=tensor.device) >= self.prob or self.std == 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


class RandomIntensityScale:
    """Apply random intensity scaling and shifting."""

    def __init__(
        self,
        prob: float = 0.2,
        scale_range: tuple[float, float] = (0.9, 1.1),
        shift_range: tuple[float, float] = (-0.05, 0.05),
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            msg = f"Probability must lie in [0, 1], got {prob}"
            raise ValueError(msg)
        if scale_range[0] <= 0 or scale_range[1] <= 0:
            msg = f"Scale range must be positive, got {scale_range}"
            raise ValueError(msg)
        if scale_range[0] > scale_range[1]:
            msg = f"Scale range min must be <= max, got {scale_range}"
            raise ValueError(msg)
        if shift_range[0] > shift_range[1]:
            msg = f"Shift range min must be <= max, got {shift_range}"
            raise ValueError(msg)

        self.prob = prob
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(self, tensor: TensorLike) -> TensorLike:
        if torch.rand(1, device=tensor.device) >= self.prob:
            return tensor

        scale = torch.empty(1, device=tensor.device).uniform_(*self.scale_range).item()
        shift = torch.empty(1, device=tensor.device).uniform_(*self.shift_range).item()
        return tensor * scale + shift


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for the default 3D augmentation pipeline."""

    flip_prob: float = 0.5
    rotate_prob: float = 0.3
    noise_prob: float = 0.1
    noise_std: float = 0.01
    intensity_prob: float = 0.2
    intensity_scale_range: tuple[float, float] = (0.9, 1.1)
    intensity_shift_range: tuple[float, float] = (-0.05, 0.05)


def build_default_augmentation_pipeline(config: AugmentationConfig | None = None) -> Compose:
    """Construct a standard augmentation pipeline for training."""

    cfg = config or AugmentationConfig()

    transforms = [
        RandomFlip3D(prob=cfg.flip_prob),
        RandomRotate90(prob=cfg.rotate_prob),
        RandomGaussianNoise(prob=cfg.noise_prob, std=cfg.noise_std),
        RandomIntensityScale(
            prob=cfg.intensity_prob,
            scale_range=cfg.intensity_scale_range,
            shift_range=cfg.intensity_shift_range,
        ),
    ]

    return Compose(transforms)


__all__ = [
    "AugmentationConfig",
    "Compose",
    "RandomFlip3D",
    "RandomRotate90",
    "RandomGaussianNoise",
    "RandomIntensityScale",
    "build_default_augmentation_pipeline",
]
