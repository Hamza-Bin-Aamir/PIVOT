"""Tests for dataset module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.dataset import LUNADataset


def _write_sample(directory: Path, name: str, array: np.ndarray) -> Path:
    path = directory / name
    np.save(path, array)
    return Path(f"{path}.npy")


def test_luna_dataset_sorts_samples(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    second = _write_sample(train_dir, "sample_b", np.zeros((2, 2, 2), dtype=np.float32))
    first = _write_sample(train_dir, "sample_a", np.ones((2, 2, 2), dtype=np.float32))

    dataset = LUNADataset(tmp_path, split="train")

    assert len(dataset) == 2
    assert dataset.samples == [first, second]


def test_luna_dataset_adds_channel_and_applies_transform(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _write_sample(train_dir, "sample", np.ones((2, 3, 4), dtype=np.float32))

    def transform(tensor: torch.Tensor) -> torch.Tensor:
        return tensor * 2

    dataset = LUNADataset(tmp_path, split="train", transform=transform)
    sample = dataset[0]

    assert sample["image"].shape[0] == 1
    assert torch.allclose(sample["image"], torch.ones((1, 2, 3, 4)) * 2)
    assert sample["path"].endswith("sample.npy")


def test_luna_dataset_preserves_existing_channel_dimension(tmp_path: Path) -> None:
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    array = np.arange(1 * 3 * 4 * 5, dtype=np.float32).reshape(1, 3, 4, 5)
    _write_sample(val_dir, "sample", array)

    dataset = LUNADataset(tmp_path, split="val", transform=None)
    sample = dataset[0]

    assert sample["image"].shape == (1, 3, 4, 5)
    assert torch.allclose(sample["image"], torch.from_numpy(array))
