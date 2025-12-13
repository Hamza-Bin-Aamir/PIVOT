"""Tests for model architecture."""

from __future__ import annotations

import torch

from src.model.unet import UNet3D


def test_unet_attributes_exposed() -> None:
    model = UNet3D(in_channels=2, out_channels=3, init_features=16, depth=2)

    assert model.in_channels == 2
    assert model.out_channels == 3
    assert model.init_features == 16
    assert model.depth == 2


def test_unet_forward_identity() -> None:
    model = UNet3D()
    input_tensor = torch.randn(1, 1, 4, 4, 4)

    output = model(input_tensor)

    assert output.shape == input_tensor.shape
    assert torch.allclose(output, input_tensor)
