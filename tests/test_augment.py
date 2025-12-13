"""Tests for 3D data augmentation utilities."""

from __future__ import annotations

import pytest
import torch

from src.data.augment import (
    AugmentationConfig,
    Compose,
    RandomFlip3D,
    RandomGaussianNoise,
    RandomIntensityScale,
    RandomRotate90,
    build_default_augmentation_pipeline,
)


def _sample_volume() -> torch.Tensor:
    return torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)


def _constant_rand(value: float):
    def _rand(*args, **kwargs):
        device = kwargs.get("device")
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            shape = tuple(args[0])
        elif len(args) == 1 and isinstance(args[0], int):
            shape = (args[0],)
        else:
            shape = tuple(args)
        return torch.full(shape, value, device=device)

    return _rand


def _randint_sequence(values: list[int]):
    iterator = iter(values)

    def _randint(*args, **kwargs):
        device = kwargs.get("device")
        size = args[-1]
        if isinstance(size, int):
            size = (size,)
        value = next(iterator)
        return torch.full(size, value, device=device, dtype=torch.int64)

    return _randint


def test_compose_runs_transforms_in_order() -> None:
    calls: list[str] = []

    def transform_a(tensor: torch.Tensor) -> torch.Tensor:
        calls.append("a")
        return tensor + 1

    def transform_b(tensor: torch.Tensor) -> torch.Tensor:
        calls.append("b")
        return tensor * 2

    compose = Compose([transform_a, transform_b])

    result = compose(torch.tensor([1.0]))

    assert calls == ["a", "b"]
    assert torch.equal(result, torch.tensor([4.0]))


def test_random_flip_applies_across_all_dims(monkeypatch) -> None:
    tensor = _sample_volume()
    transform = RandomFlip3D(prob=1.0)

    monkeypatch.setattr(torch, "rand", _constant_rand(0.0))

    flipped = transform(tensor)

    expected = tensor
    for dim in (1, 2, 3):
        expected = torch.flip(expected, dims=(dim,))
    assert torch.allclose(flipped, expected)


def test_random_flip_returns_original_when_threshold_not_met(monkeypatch) -> None:
    tensor = _sample_volume()
    transform = RandomFlip3D(prob=0.0)

    monkeypatch.setattr(torch, "rand", _constant_rand(0.0))

    flipped = transform(tensor)

    assert torch.allclose(flipped, tensor)


def test_random_flip_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        RandomFlip3D(prob=1.5)

    transform = RandomFlip3D()
    with pytest.raises(ValueError):
        transform(torch.ones(3, 3, 3))


def test_random_rotate90_applies_rotation(monkeypatch) -> None:
    tensor = _sample_volume()
    transform = RandomRotate90(prob=1.0)

    monkeypatch.setattr(torch, "rand", _constant_rand(0.0))
    monkeypatch.setattr(torch, "randint", _randint_sequence([1, 2]))

    rotated = transform(tensor)

    manual = torch.rot90(tensor.clone(), k=2, dims=(1, 3))
    dims_order = list(range(manual.ndim))
    dims_order[1], dims_order[3] = dims_order[3], dims_order[1]
    expected = manual.permute(dims_order).contiguous()

    assert torch.allclose(rotated, expected)


def test_random_rotate90_skips_when_probability_not_met(monkeypatch) -> None:
    tensor = _sample_volume()
    transform = RandomRotate90(prob=0.5)

    monkeypatch.setattr(torch, "rand", _constant_rand(1.0))

    rotated = transform(tensor)

    assert torch.allclose(rotated, tensor)


def test_random_rotate90_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        RandomRotate90(prob=-0.1)

    transform = RandomRotate90()
    with pytest.raises(ValueError):
        transform(torch.ones(3, 3, 3))


def test_random_gaussian_noise_zero_std_is_no_op(monkeypatch) -> None:
    tensor = _sample_volume()
    transform = RandomGaussianNoise(prob=1.0, std=0.0)

    monkeypatch.setattr(torch, "rand", _constant_rand(0.0))

    noisy = transform(tensor)

    assert torch.allclose(noisy, tensor)


def test_random_gaussian_noise_adds_expected_noise(monkeypatch) -> None:
    tensor = _sample_volume()
    transform = RandomGaussianNoise(prob=1.0, mean=0.5, std=0.2)

    monkeypatch.setattr(torch, "rand", _constant_rand(0.0))
    monkeypatch.setattr(torch, "randn_like", lambda t: torch.ones_like(t))

    noisy = transform(tensor)

    assert torch.allclose(noisy, tensor + 0.5 + 0.2)


def test_random_gaussian_noise_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        RandomGaussianNoise(prob=1.5)

    with pytest.raises(ValueError):
        RandomGaussianNoise(prob=0.5, std=-0.1)


def test_random_intensity_scale_applies_shift_and_scale(monkeypatch) -> None:
    tensor = _sample_volume()
    transform = RandomIntensityScale(
        prob=1.0,
        scale_range=(2.0, 2.0),
        shift_range=(1.0, 1.0),
    )

    monkeypatch.setattr(torch, "rand", _constant_rand(0.0))

    augmented = transform(tensor)

    expected = tensor * 2.0 + 1.0
    assert torch.allclose(augmented, expected)


def test_random_intensity_scale_returns_original_when_probability_not_met(monkeypatch) -> None:
    tensor = _sample_volume()
    transform = RandomIntensityScale(prob=0.5)

    monkeypatch.setattr(torch, "rand", _constant_rand(1.0))

    augmented = transform(tensor)

    assert torch.allclose(augmented, tensor)


def test_random_intensity_scale_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        RandomIntensityScale(prob=1.5)

    with pytest.raises(ValueError):
        RandomIntensityScale(prob=0.5, scale_range=(-1.0, 1.0))

    with pytest.raises(ValueError):
        RandomIntensityScale(prob=0.5, scale_range=(1.0, 0.5))

    with pytest.raises(ValueError):
        RandomIntensityScale(prob=0.5, shift_range=(0.1, -0.1))


def test_build_default_pipeline_executes_all_transforms(monkeypatch) -> None:
    tensor = _sample_volume()
    config = AugmentationConfig(
        flip_prob=1.0,
        rotate_prob=1.0,
        noise_prob=0.0,
        intensity_prob=1.0,
        intensity_scale_range=(1.0, 1.0),
        intensity_shift_range=(0.0, 0.0),
    )

    monkeypatch.setattr(torch, "rand", _constant_rand(0.0))
    monkeypatch.setattr(torch, "randint", _randint_sequence([0, 1]))

    pipeline = build_default_augmentation_pipeline(config)

    output = pipeline(tensor)

    assert isinstance(output, torch.Tensor)
    assert output.shape == tensor.shape
    assert output.dtype == tensor.dtype


def test_build_default_pipeline_uses_default_config(monkeypatch) -> None:
    tensor = _sample_volume()

    monkeypatch.setattr(torch, "rand", _constant_rand(0.0))
    monkeypatch.setattr(torch, "randn_like", lambda t: torch.zeros_like(t))
    monkeypatch.setattr(torch, "randint", _randint_sequence([0, 1]))

    pipeline = build_default_augmentation_pipeline()

    output = pipeline(tensor)

    assert isinstance(pipeline, Compose)
    assert output.shape == tensor.shape
