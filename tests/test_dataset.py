"""Tests for dataset module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import LUNADataset


def _write_volume(
    directory: Path,
    name: str,
    *,
    image: np.ndarray,
    mask: np.ndarray | None = None,
    heatmap: np.ndarray | None = None,
    positive_centers: np.ndarray | None = None,
    negative_centers: np.ndarray | None = None,
    triage: float | None = None,
) -> Path:
    path = directory / f"{name}.npz"
    kwargs: dict[str, np.ndarray] = {"image": image.astype(np.float32)}
    if mask is not None:
        kwargs["mask"] = mask.astype(np.float32)
    if heatmap is not None:
        kwargs["heatmap"] = heatmap.astype(np.float32)
    if positive_centers is not None:
        kwargs["positive_centers"] = positive_centers.astype(np.float32)
    if negative_centers is not None:
        kwargs["negative_centers"] = negative_centers.astype(np.float32)
    if triage is not None:
        kwargs["triage"] = np.array([triage], dtype=np.float32)
    np.savez(path, **kwargs)
    return path


def _prepare_dataset(tmp_path: Path) -> Path:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.zeros((32, 32, 32), dtype=np.float32)
    mask = np.zeros_like(image)
    mask[14:18, 14:18, 14:18] = 1
    heatmap = np.zeros_like(image)
    heatmap[16, 16, 16] = 1

    positive_centers = np.array([[16, 16, 16], [20, 20, 20]], dtype=np.float32)
    negative_centers = np.array([[8, 8, 8], [24, 24, 24], [10, 24, 12]], dtype=np.float32)

    return _write_volume(
        train_dir,
        "sample",
        image=image,
        mask=mask,
        heatmap=heatmap,
        positive_centers=positive_centers,
        negative_centers=negative_centers,
        triage=7.5,
    )


def test_luna_dataset_balances_positive_and_negative_patches(tmp_path: Path) -> None:
    created = _prepare_dataset(tmp_path)

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=4,
        cache_size=2,
        seed=7,
    )

    assert dataset.samples == [created]
    assert len(dataset) == 4

    positives = 0
    negatives = 0
    for sample in dataset:
        assert sample["image"].shape == (1, 8, 8, 8)
        assert sample["series_uid"] == "sample"
        assert sample["path"].endswith("sample.npz")
        if sample["is_positive"]:
            positives += 1
            assert "mask" in sample
            assert torch.count_nonzero(sample["mask"]).item() > 0
        else:
            negatives += 1
            assert "mask" in sample
            assert torch.count_nonzero(sample["mask"]).item() == 0
        assert "heatmap" in sample
        assert torch.all(sample["heatmap"] >= 0)
        assert sample["triage"].dtype == torch.float32
        assert len(sample["center"]) == 3

    assert positives == negatives == 2


def test_luna_dataset_caches_volumes(tmp_path: Path) -> None:
    _prepare_dataset(tmp_path)

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=2,
        cache_size=1,
    )

    dataset._main_state["cache_hits"] = 0
    dataset._main_state["cache_misses"] = 0

    _ = dataset[0]
    _ = dataset[1]
    _ = dataset[0]

    assert dataset._main_state["cache_hits"] >= 1
    assert len(dataset._main_state["cache"]) <= 1


def test_luna_dataset_supports_tensor_only_transform(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    mask = np.zeros_like(image)
    mask[10:14, 10:14, 10:14] = 1
    positive_centers = np.array([[12, 12, 12]], dtype=np.float32)
    negative_centers = np.array([[5, 5, 5]], dtype=np.float32)
    _write_volume(
        train_dir,
        "sample",
        image=image,
        mask=mask,
        positive_centers=positive_centers,
        negative_centers=negative_centers,
    )

    def transform(image_tensor: torch.Tensor) -> torch.Tensor:
        return image_tensor * 3

    base_dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=2,
        transform=None,
    )

    transformed_dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=2,
        transform=transform,
        seed=base_dataset.seed,
    )

    base_sample = base_dataset[0]
    transformed_sample = transformed_dataset[0]

    assert torch.allclose(transformed_sample["image"], base_sample["image"] * 3)
    assert torch.allclose(transformed_sample["image"], transformed_dataset[0]["image"])


def test_luna_dataset_raises_with_invalid_fraction(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _write_volume(train_dir, "sample", image=np.zeros((8, 8, 8), dtype=np.float32))

    with pytest.raises(ValueError):
        LUNADataset(tmp_path, positive_fraction=0.0)

    with pytest.raises(ValueError):
        LUNADataset(tmp_path, positive_fraction=1.5)


def test_luna_dataset_raises_with_invalid_patch_size(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _write_volume(train_dir, "sample", image=np.zeros((8, 8, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="patch_size must contain three positive integers"):
        LUNADataset(tmp_path, patch_size=(8, 8))

    with pytest.raises(ValueError, match="patch_size must contain three positive integers"):
        LUNADataset(tmp_path, patch_size=(8, -1, 8))


def test_luna_dataset_raises_with_invalid_patches_per_volume(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _write_volume(train_dir, "sample", image=np.zeros((8, 8, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="patches_per_volume must be positive"):
        LUNADataset(tmp_path, patches_per_volume=0)

    with pytest.raises(ValueError, match="patches_per_volume must be positive"):
        LUNADataset(tmp_path, patches_per_volume=-5)


def test_luna_dataset_raises_with_invalid_cache_size(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _write_volume(train_dir, "sample", image=np.zeros((8, 8, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="cache_size must be >= 0"):
        LUNADataset(tmp_path, cache_size=-1)


def test_luna_dataset_raises_when_no_files_found(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No preprocessed volumes found"):
        LUNADataset(tmp_path, split="train")


def test_luna_dataset_loads_npy_files(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((16, 16, 16), dtype=np.float32)
    path = train_dir / "sample.npy"
    np.save(path, image)

    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=1)

    assert len(dataset.samples) == 1
    sample = dataset[0]
    assert sample["image"].shape == (1, 8, 8, 8)


def test_luna_dataset_raises_with_wrong_dimensions(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    bad_image = np.ones((16, 16), dtype=np.float32)
    path = train_dir / "bad.npz"
    np.savez(path, image=bad_image)

    with pytest.raises(ValueError, match="Expected image volume with 3 dimensions"):
        LUNADataset(tmp_path, split="train")


def test_luna_dataset_raises_with_wrong_dimensions_npy(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    bad_image = np.ones((16, 16), dtype=np.float32)
    path = train_dir / "bad.npy"
    np.save(path, bad_image)

    with pytest.raises(ValueError, match="Expected image volume with 3 dimensions in .npy file"):
        LUNADataset(tmp_path, split="train")


def test_luna_dataset_without_mask_or_heatmap(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    path = train_dir / "sample.npz"
    np.savez(path, image=image)

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=2,
        include_mask=True,
        include_heatmap=True,
    )

    assert len(dataset) > 0
    sample = dataset[0]
    assert "mask" not in sample
    assert "heatmap" not in sample


def test_luna_dataset_includes_spacing(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    spacing = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    path = train_dir / "sample.npz"
    np.savez(path, image=image, spacing=spacing)

    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=1)

    sample = dataset[0]
    assert "spacing" in sample
    assert len(sample["spacing"]) == 3
    assert sample["spacing"] == (0.5, 0.5, 1.0)


def test_luna_dataset_transform_returns_dict(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    center = np.array([[12, 12, 12]], dtype=np.float32)
    _write_volume(train_dir, "sample", image=image, positive_centers=center)

    def dict_transform(sample: dict) -> dict:
        sample["image"] = sample["image"] * 2
        sample["custom_field"] = torch.tensor([42.0])
        return sample

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=1,
        transform=dict_transform,
    )

    sample = dataset[0]
    # Image should be doubled by transform, verify nonzero values are 2
    assert sample["image"].shape == (1, 8, 8, 8)
    assert torch.max(sample["image"]) == 2.0
    assert "custom_field" in sample
    assert sample["custom_field"] == 42.0


def test_luna_dataset_transform_returns_invalid_type(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((16, 16, 16), dtype=np.float32)
    _write_volume(train_dir, "sample", image=image)

    def bad_transform(sample: dict) -> list:
        return [1, 2, 3]

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=1,
        transform=bad_transform,
    )

    with pytest.raises(TypeError, match="Transform must return a Tensor or dict"):
        _ = dataset[0]


def test_luna_dataset_with_cache_disabled(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((16, 16, 16), dtype=np.float32)
    _write_volume(train_dir, "sample", image=image)

    dataset = LUNADataset(
        tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=2, cache_size=0
    )

    state = dataset._main_state
    state["cache_hits"] = 0
    state["cache_misses"] = 0

    _ = dataset[0]
    _ = dataset[1]

    assert state["cache_misses"] == 2
    assert state["cache_hits"] == 0
    assert len(state["cache"]) == 0


def test_luna_dataset_fallback_center_when_no_samples(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    mask = np.zeros_like(image)
    positive_centers = np.empty((0, 3), dtype=np.float32)
    negative_centers = np.empty((0, 3), dtype=np.float32)

    _write_volume(
        train_dir,
        "sample",
        image=image,
        mask=mask,
        positive_centers=positive_centers,
        negative_centers=negative_centers,
    )

    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=2)

    assert len(dataset) >= 1
    sample = dataset[0]
    assert sample["image"].shape == (1, 8, 8, 8)


def test_luna_dataset_patch_extraction_with_padding(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((10, 10, 10), dtype=np.float32)
    center = np.array([[2, 2, 2]], dtype=np.float32)
    _write_volume(train_dir, "sample", image=image, positive_centers=center)

    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=1)

    sample = dataset[0]
    assert sample["image"].shape == (1, 8, 8, 8)


def test_luna_dataset_derives_positive_from_heatmap_when_no_mask(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    heatmap = np.zeros_like(image)
    heatmap[12, 12, 12] = 1.0
    heatmap[18, 18, 18] = 0.9

    _write_volume(train_dir, "sample", image=image, heatmap=heatmap)

    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=4)

    assert len(dataset) > 0
    has_positive = False
    for sample in dataset:
        if sample["is_positive"]:
            has_positive = True
            break
    assert has_positive


def test_luna_dataset_without_include_mask(tmp_path: Path) -> None:
    _prepare_dataset(tmp_path)

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=2,
        include_mask=False,
    )

    sample = dataset[0]
    assert "mask" not in sample


def test_luna_dataset_without_include_heatmap(tmp_path: Path) -> None:
    _prepare_dataset(tmp_path)

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=2,
        include_heatmap=False,
    )

    sample = dataset[0]
    assert "heatmap" not in sample


def test_luna_dataset_large_patch_extraction(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((8, 8, 8), dtype=np.float32)
    center = np.array([[4, 4, 4]], dtype=np.float32)
    _write_volume(train_dir, "sample", image=image, positive_centers=center)

    dataset = LUNADataset(tmp_path, split="train", patch_size=(16, 16, 16), patches_per_volume=1)

    sample = dataset[0]
    assert sample["image"].shape == (1, 16, 16, 16)


def test_luna_dataset_mask_with_no_positives(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    mask = np.zeros_like(image)  # All zeros

    _write_volume(train_dir, "sample", image=image, mask=mask)

    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=2)

    # Should still produce samples (negatives)
    assert len(dataset) > 0


def test_luna_dataset_mask_all_ones(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    mask = np.ones_like(image)  # All ones - no negatives

    _write_volume(train_dir, "sample", image=image, mask=mask)

    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=4)

    # Should still produce samples (positives + fallback)
    assert len(dataset) > 0


def test_luna_dataset_many_positive_centers_subsampled(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((32, 32, 32), dtype=np.float32)
    # Create many more centers than patches_per_volume
    mask = np.zeros_like(image)
    for i in range(5, 25, 2):
        for j in range(5, 25, 2):
            mask[i, j, 10] = 1

    _write_volume(train_dir, "sample", image=image, mask=mask)

    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=4)

    # Should subsample from many available centers
    assert len(dataset) == 4


def test_luna_dataset_transform_returns_tensor(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    center = np.array([[12, 12, 12]], dtype=np.float32)
    _write_volume(train_dir, "sample", image=image, positive_centers=center)

    def tensor_transform(sample: dict) -> torch.Tensor:
        return sample["image"] * 3

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=1,
        transform=tensor_transform,
    )

    sample = dataset[0]
    # Transform returns tensor; should replace image
    assert torch.max(sample["image"]) == 3.0


def test_luna_dataset_uniform_sampling_with_zero_shape(tmp_path: Path) -> None:
    """Edge case: _sample_uniform_centers with invalid shape"""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    # Force a scenario that might exercise uniform sampling
    mask = np.zeros_like(image)
    positive_centers = np.empty((0, 3), dtype=np.float32)
    # Provide very few negative centers to trigger uniform sampling
    negative_centers = np.array([[2, 2, 2]], dtype=np.float32)

    _write_volume(
        train_dir,
        "sample",
        image=image,
        mask=mask,
        positive_centers=positive_centers,
        negative_centers=negative_centers,
    )

    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=10)

    # Should handle uniform sampling when negatives are insufficient
    assert len(dataset) > 0


def test_luna_dataset_worker_state_initialization(tmp_path: Path) -> None:
    """Test worker-specific RNG state initialization in multiprocessing"""
    from unittest.mock import MagicMock, patch

    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    center = np.array([[12, 12, 12]], dtype=np.float32)
    _write_volume(train_dir, "sample", image=image, positive_centers=center)

    dataset = LUNADataset(
        tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=2, seed=42
    )

    # Simulate worker context
    mock_worker = MagicMock()
    mock_worker.id = 3

    with patch("src.data.dataset.get_worker_info", return_value=mock_worker):
        state = dataset._get_state()
        # Worker state should be created with offset seed
        assert 3 in dataset._worker_states
        assert state is dataset._worker_states[3]
        # Calling again should return cached state
        state2 = dataset._get_state()
        assert state is state2


def test_luna_dataset_worker_state_with_none_seed(tmp_path: Path) -> None:
    """Test worker state initialization when seed is None"""
    from unittest.mock import MagicMock, patch

    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    center = np.array([[12, 12, 12]], dtype=np.float32)
    _write_volume(train_dir, "sample", image=image, positive_centers=center)

    dataset = LUNADataset(
        tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=2, seed=None
    )

    mock_worker = MagicMock()
    mock_worker.id = 1

    with patch("src.data.dataset.get_worker_info", return_value=mock_worker):
        dataset._get_state()
        # Should use 0 as base when seed is None
        assert 1 in dataset._worker_states


def test_luna_dataset_sample_centers_when_count_exceeds_available(tmp_path: Path) -> None:
    """Test _sample_centers when requesting more centers than available"""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    # Only 2 positive centers
    positive_centers = np.array([[10, 10, 10], [14, 14, 14]], dtype=np.float32)
    _write_volume(train_dir, "sample", image=image, positive_centers=positive_centers)

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=10,  # Request more than available
        positive_fraction=1.0,  # All positives
    )

    # Should get all available positives without error
    positive_count = sum(1 for i in range(len(dataset)) if dataset[i]["is_positive"])
    # Will get the 2 positives we have
    assert positive_count == 2


def test_luna_dataset_empty_negative_candidates_triggers_uniform_sampling(tmp_path: Path) -> None:
    """Test uniform sampling when no negative candidates exist within bounds"""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    # Small volume where all voxels might be positive or out of bounds for patch
    image = np.ones((12, 12, 12), dtype=np.float32)
    mask = np.ones_like(image)  # All positive, no negatives
    positive_centers = np.array([[6, 6, 6]], dtype=np.float32)
    negative_centers = np.empty((0, 3), dtype=np.float32)  # No pre-computed negatives

    _write_volume(
        train_dir,
        "sample",
        image=image,
        mask=mask,
        positive_centers=positive_centers,
        negative_centers=negative_centers,
    )

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=4,
        positive_fraction=0.5,
    )

    # Should generate samples via uniform sampling
    assert len(dataset) > 0


def test_luna_dataset_patch_shape_correction_crop_then_pad(tmp_path: Path) -> None:
    """Test _ensure_patch_shape with a patch that needs both cropping and padding"""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    # Create a scenario that might produce an odd-shaped patch
    image = np.ones((15, 15, 15), dtype=np.float32)
    # Edge center that might cause extraction issues
    center = np.array([[1, 1, 14]], dtype=np.float32)
    _write_volume(train_dir, "sample", image=image, positive_centers=center)

    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=1)

    sample = dataset[0]
    # Should still produce correct shape despite edge extraction
    assert sample["image"].shape == (1, 8, 8, 8)


def test_luna_dataset_negative_centers_sampling_with_bounds_filtering(tmp_path: Path) -> None:
    """Test that negative centers outside patch bounds are filtered correctly"""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((32, 32, 32), dtype=np.float32)
    mask = np.zeros_like(image)
    # Centers too close to edges (will be filtered by bounds check for 16x16x16 patch)
    negative_centers = np.array(
        [
            [1, 1, 1],
            [30, 30, 30],
            [2, 2, 2],
            [16, 16, 16],
        ],  # Out of bounds  # Out of bounds  # Out of bounds  # Valid
        dtype=np.float32,
    )

    _write_volume(
        train_dir,
        "sample",
        image=image,
        mask=mask,
        negative_centers=negative_centers,
    )

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(16, 16, 16),  # Large patch requiring bounds filtering
        patches_per_volume=4,
        positive_fraction=0.01,  # Minimal positives, mostly negatives
    )

    # Should handle filtering and fallback to uniform sampling
    assert len(dataset) >= 1


def test_luna_dataset_cache_eviction_beyond_limit(tmp_path: Path) -> None:
    """Test LRU cache eviction when cache size is exceeded"""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    # Create multiple volumes
    for i in range(5):
        image = np.ones((16, 16, 16), dtype=np.float32) * i
        center = np.array([[8, 8, 8]], dtype=np.float32)
        _write_volume(train_dir, f"sample_{i}", image=image, positive_centers=center)

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=1,
        cache_size=2,  # Small cache
    )

    # Access multiple volumes to trigger eviction
    _ = dataset[0]  # sample_0
    _ = dataset[1]  # sample_1
    _ = dataset[2]  # sample_2 - should evict sample_0

    # Cache should only hold 2 volumes
    assert len(dataset._main_state["cache"]) <= 2


def test_luna_dataset_zero_shape_uniform_sampling(tmp_path: Path) -> None:
    """Test _sample_uniform_centers edge case with degenerate shape"""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    _write_volume(train_dir, "sample", image=image)

    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=2)

    # Test the edge case directly
    rng = np.random.default_rng(42)

    # Edge case: count <= 0
    result = dataset._sample_uniform_centers(0, rng, np.array([10, 10, 10]))
    assert result.shape == (0, 3)

    result = dataset._sample_uniform_centers(-5, rng, np.array([10, 10, 10]))
    assert result.shape == (0, 3)

    # Edge case: shape with zero or negative values
    result = dataset._sample_uniform_centers(3, rng, np.array([0, 10, 10]))
    assert result.shape == (3, 3)
    assert np.all(result[:, 0] == 0)  # Zero dimension should produce zeros

    result = dataset._sample_uniform_centers(2, rng, np.array([-5, 10, 10]))
    assert result.shape == (2, 3)


def test_luna_dataset_select_negative_with_zero_count(tmp_path: Path) -> None:
    """Test _select_negative_centers when count is zero or negative"""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((24, 24, 24), dtype=np.float32)
    _write_volume(train_dir, "sample", image=image)

    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=1)

    rng = np.random.default_rng(42)
    candidates = np.array([[10, 10, 10], [12, 12, 12]], dtype=np.float32)

    # Edge case: count <= 0
    result = dataset._select_negative_centers(candidates, 0, rng, (24, 24, 24))
    assert result == []

    result = dataset._select_negative_centers(candidates, -3, rng, (24, 24, 24))
    assert result == []


def test_luna_dataset_vstack_with_empty_chosen(tmp_path: Path) -> None:
    """Test the branch where chosen.size is 0 when combining with sampled centers"""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((16, 16, 16), dtype=np.float32)
    mask = np.zeros_like(image)
    # Provide negative centers that will all be out of bounds for a large patch
    negative_centers = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)

    _write_volume(
        train_dir,
        "sample",
        image=image,
        mask=mask,
        negative_centers=negative_centers,
    )

    # Large patch size to make provided centers out of bounds
    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(14, 14, 14),
        patches_per_volume=5,
        positive_fraction=0.01,  # Minimal positives
    )

    # Should trigger uniform sampling fallback when all candidates filtered
    assert len(dataset) > 0


def test_luna_dataset_patch_shape_ensure_exact_crop(tmp_path: Path) -> None:
    """Test _ensure_patch_shape cropping logic when patch is too large"""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((20, 20, 20), dtype=np.float32)
    center = np.array([[10, 10, 10]], dtype=np.float32)
    _write_volume(train_dir, "sample", image=image, positive_centers=center)

    # Test internal method directly to hit cropping branches
    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=1)

    # Create a patch that's too large and needs cropping
    oversized_patch = np.ones((12, 12, 12), dtype=np.float32)
    result = dataset._ensure_patch_shape(oversized_patch)
    assert result.shape == (8, 8, 8)

    # Create a patch that's too small and needs padding
    undersized_patch = np.ones((5, 5, 5), dtype=np.float32)
    result = dataset._ensure_patch_shape(undersized_patch)
    assert result.shape == (8, 8, 8)

    # Create a patch with mixed dimensions (some too large, some too small)
    mixed_patch = np.ones((10, 5, 8), dtype=np.float32)
    result = dataset._ensure_patch_shape(mixed_patch)
    assert result.shape == (8, 8, 8)


def test_luna_dataset_fallback_center_generation(tmp_path: Path) -> None:
    """Test fallback to center point when no positive or negative samples found"""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    # Small volume with tiny patch requirement to trigger fallback
    image = np.ones((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(image)  # No positives
    # Provide empty arrays
    positive_centers = np.empty((0, 3), dtype=np.float32)
    negative_centers = np.empty((0, 3), dtype=np.float32)

    _write_volume(
        train_dir,
        "sample",
        image=image,
        mask=mask,
        positive_centers=positive_centers,
        negative_centers=negative_centers,
    )

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(6, 6, 6),
        patches_per_volume=2,
    )

    # Should create fallback center at volume center
    assert len(dataset) >= 1
    sample = dataset[0]
    # Fallback center should be roughly at (3.5, 3.5, 3.5) for 8^3 volume
    assert sample["center"][0] >= 0 and sample["center"][0] < 8


def test_luna_dataset_sample_centers_empty_input(tmp_path: Path) -> None:
    """Test _sample_centers with empty input"""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    image = np.ones((16, 16, 16), dtype=np.float32)
    _write_volume(train_dir, "sample", image=image)

    dataset = LUNADataset(tmp_path, split="train", patch_size=(8, 8, 8), patches_per_volume=1)

    rng = np.random.default_rng(42)

    # Empty centers list
    result = dataset._sample_centers([], 5, rng)
    assert result == []

    # Empty numpy array
    empty_array = np.empty((0, 3), dtype=np.float32)
    result = dataset._sample_centers(empty_array, 5, rng)
    assert result == []


def test_luna_dataset_both_selections_empty_triggers_fallback(tmp_path: Path) -> None:
    """Test the specific case where both positive and negative selections are empty, triggering fallback center"""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    # Create a very constrained scenario:
    # - Tiny volume (10x10x10)
    # - No mask or metadata (no positives from mask/heatmap)
    # - Empty pre-computed centers
    # - Large patch size (8x8x8) that makes most locations out of bounds
    image = np.ones((10, 10, 10), dtype=np.float32)
    positive_centers = np.empty((0, 3), dtype=np.float32)
    negative_centers = np.empty((0, 3), dtype=np.float32)

    _write_volume(
        train_dir,
        "sample",
        image=image,
        positive_centers=positive_centers,
        negative_centers=negative_centers,
    )

    dataset = LUNADataset(
        tmp_path,
        split="train",
        patch_size=(8, 8, 8),
        patches_per_volume=1,
    )

    # Should have created at least one sample via fallback or uniform sampling
    assert len(dataset) >= 1
    sample = dataset[0]
    # Just verify we got a valid sample
    assert sample["image"].shape == (1, 8, 8, 8)
    assert "center" in sample
    assert len(sample["center"]) == 3
