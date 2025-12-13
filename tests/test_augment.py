"""Tests for 3D data augmentation utilities."""

from __future__ import annotations

import torch

from src.data.augment import (
    AugmentationConfig,
    RandomFlip3D,
    RandomGaussianNoise,
    RandomIntensityScale,
    RandomRotate90,
    build_default_augmentation_pipeline,
)


def _sample_volume() -> torch.Tensor:
    return torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)


def test_random_flip_applies_with_probability_one() -> None:
    tensor = _sample_volume()
    transform = RandomFlip3D(prob=1.0, dims=(1,))

    flipped = transform(tensor)

    expected = torch.flip(tensor, dims=(1,))
    assert torch.allclose(flipped, expected)


def test_random_rotate90_preserves_shape() -> None:
    tensor = _sample_volume()
    transform = RandomRotate90(prob=1.0)

    rotated = transform(tensor)

    assert rotated.shape == tensor.shape


def test_random_gaussian_noise_zero_std_is_no_op() -> None:
    tensor = _sample_volume()
    transform = RandomGaussianNoise(prob=1.0, std=0.0)

    noisy = transform(tensor)

    assert torch.allclose(noisy, tensor)


def test_random_intensity_scale_applies_shift_and_scale() -> None:
    tensor = _sample_volume()
    transform = RandomIntensityScale(prob=1.0, scale_range=(2.0, 2.0), shift_range=(1.0, 1.0))

    augmented = transform(tensor)

    expected = tensor * 2.0 + 1.0
    assert torch.allclose(augmented, expected)


def test_build_default_pipeline_executes_all_transforms() -> None:
    tensor = _sample_volume()
    config = AugmentationConfig(
        flip_prob=1.0,
        rotate_prob=1.0,
        noise_prob=0.0,
        intensity_prob=1.0,
        intensity_scale_range=(1.0, 1.0),
        intensity_shift_range=(0.0, 0.0),
    )

    pipeline = build_default_augmentation_pipeline(config)

    torch.manual_seed(0)
    output = pipeline(tensor)

    assert isinstance(output, torch.Tensor)
    assert output.shape == tensor.shape
    assert output.dtype == tensor.dtype
