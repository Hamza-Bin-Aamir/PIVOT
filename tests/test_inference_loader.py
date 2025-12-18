"""Tests for inference data loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.inference.inference_loader import InferenceDataLoader


class TestInferenceDataLoaderInit:
    """Test inference loader initialization."""

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_init_valid_directory(self, mock_metadata, mock_discover):
        """Test initialization with valid directory."""
        mock_discover.return_value = [Path("test.npy")]
        mock_metadata.return_value = {}

        loader = InferenceDataLoader(data_dir="/tmp")

        assert loader.data_dir == Path("/tmp")
        assert loader.normalize is True
        assert loader.normalize_range == (0.0, 1.0)

    def test_init_invalid_directory(self):
        """Test initialization with non-existent directory."""
        with pytest.raises(ValueError, match="data_dir does not exist"):
            InferenceDataLoader(data_dir="/nonexistent/path")

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_init_empty_directory(self, mock_metadata, mock_discover):
        """Test initialization with empty directory."""
        mock_discover.return_value = []
        mock_metadata.return_value = {}

        with pytest.raises(ValueError, match="No volume files found"):
            InferenceDataLoader(data_dir="/tmp")

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_init_custom_normalize_range(self, mock_metadata, mock_discover):
        """Test initialization with custom normalization range."""
        mock_discover.return_value = [Path("test.npy")]
        mock_metadata.return_value = {}

        loader = InferenceDataLoader(
            data_dir="/tmp",
            normalize_range=(-1.0, 1.0),
        )

        assert loader.normalize_range == (-1.0, 1.0)


class TestDiscoverVolumes:
    """Test volume discovery."""

    def test_discover_volumes_numpy(self, tmp_path):
        """Test discovering numpy files."""
        # Create test files
        (tmp_path / "volume1.npy").touch()
        (tmp_path / "volume2.npy").touch()

        loader = InferenceDataLoader.__new__(InferenceDataLoader)
        loader.data_dir = tmp_path

        volumes = loader._discover_volumes()

        assert len(volumes) == 2
        assert all(v.suffix == ".npy" for v in volumes)

    def test_discover_volumes_nifti(self, tmp_path):
        """Test discovering NIfTI files."""
        (tmp_path / "volume1.nii").touch()
        (tmp_path / "volume2.nii.gz").touch()

        loader = InferenceDataLoader.__new__(InferenceDataLoader)
        loader.data_dir = tmp_path

        volumes = loader._discover_volumes()

        assert len(volumes) == 2

    def test_discover_volumes_mixed(self, tmp_path):
        """Test discovering mixed file formats."""
        (tmp_path / "vol1.npy").touch()
        (tmp_path / "vol2.nii").touch()
        (tmp_path / "vol3.nii.gz").touch()

        loader = InferenceDataLoader.__new__(InferenceDataLoader)
        loader.data_dir = tmp_path

        volumes = loader._discover_volumes()

        assert len(volumes) == 3

    def test_discover_volumes_empty(self, tmp_path):
        """Test discovering volumes in empty directory."""
        loader = InferenceDataLoader.__new__(InferenceDataLoader)
        loader.data_dir = tmp_path

        volumes = loader._discover_volumes()

        assert len(volumes) == 0

    def test_discover_volumes_sorted(self, tmp_path):
        """Test that discovered volumes are sorted."""
        (tmp_path / "volume_z.npy").touch()
        (tmp_path / "volume_a.npy").touch()
        (tmp_path / "volume_m.npy").touch()

        loader = InferenceDataLoader.__new__(InferenceDataLoader)
        loader.data_dir = tmp_path

        volumes = loader._discover_volumes()

        names = [v.name for v in volumes]
        assert names == sorted(names)


class TestLoadVolumeFile:
    """Test volume file loading."""

    def test_load_numpy_file(self, tmp_path):
        """Test loading numpy file."""
        volume = np.random.rand(64, 64, 64).astype(np.float32)
        path = tmp_path / "test.npy"
        np.save(path, volume)

        loaded = InferenceDataLoader._load_volume_file(path)

        assert isinstance(loaded, np.ndarray)
        assert loaded.dtype == np.float32
        assert np.allclose(loaded, volume)

    def test_load_numpy_converts_dtype(self, tmp_path):
        """Test that numpy loading converts to float32."""
        volume = np.random.randint(0, 100, (64, 64, 64), dtype=np.int32)
        path = tmp_path / "test.npy"
        np.save(path, volume)

        loaded = InferenceDataLoader._load_volume_file(path)

        assert loaded.dtype == np.float32

    def test_load_unsupported_format(self, tmp_path):
        """Test loading unsupported file format."""
        path = tmp_path / "test.txt"
        path.write_text("dummy")

        with pytest.raises(ValueError, match="Unsupported file format"):
            InferenceDataLoader._load_volume_file(path)


class TestLen:
    """Test __len__ method."""

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_len(self, mock_metadata, mock_discover):
        """Test length of dataset."""
        mock_discover.return_value = [Path(f"vol{i}.npy") for i in range(5)]
        mock_metadata.return_value = {}

        loader = InferenceDataLoader(data_dir="/tmp")

        assert len(loader) == 5


class TestGetItem:
    """Test __getitem__ method."""

    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_volume_file")
    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    def test_getitem_valid_index(self, mock_discover, mock_load, mock_metadata):
        """Test getting item by valid index."""
        volume = np.ones((64, 64, 64), dtype=np.float32)
        mock_discover.return_value = [Path("test.npy")]
        mock_load.return_value = volume
        mock_metadata.return_value = {"test": {"shape": (64, 64, 64)}}

        loader = InferenceDataLoader(data_dir="/tmp")
        item = loader[0]

        assert "volume" in item
        assert "metadata" in item
        assert "path" in item
        assert "name" in item
        assert "shape" in item
        assert isinstance(item["volume"], torch.Tensor)
        assert item["volume"].shape == (1, 64, 64, 64)

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_getitem_negative_index(self, mock_metadata, mock_discover):
        """Test getting item with negative index."""
        mock_discover.return_value = [Path("vol1.npy"), Path("vol2.npy")]
        mock_metadata.return_value = {}

        loader = InferenceDataLoader(data_dir="/tmp")

        with pytest.raises(IndexError, match="Index .* out of range"):
            loader[-1]

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_getitem_out_of_bounds(self, mock_metadata, mock_discover):
        """Test getting item with out-of-bounds index."""
        mock_discover.return_value = [Path("test.npy")]
        mock_metadata.return_value = {}

        loader = InferenceDataLoader(data_dir="/tmp")

        with pytest.raises(IndexError, match="Index .* out of range"):
            loader[5]


class TestNormalizeVolume:
    """Test volume normalization."""

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_normalize_to_01_range(self, mock_metadata, mock_discover):
        """Test normalization to [0, 1]."""
        mock_discover.return_value = [Path("test.npy")]
        mock_metadata.return_value = {}

        loader = InferenceDataLoader(data_dir="/tmp", normalize=True)

        volume = np.array([[[0.0, 100.0]]], dtype=np.float32)
        normalized = loader._normalize_volume(volume)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_normalize_to_custom_range(self, mock_metadata, mock_discover):
        """Test normalization to custom range."""
        mock_discover.return_value = [Path("test.npy")]
        mock_metadata.return_value = {}

        loader = InferenceDataLoader(
            data_dir="/tmp",
            normalize=True,
            normalize_range=(-1.0, 1.0),
        )

        volume = np.array([[[0.0, 100.0]]], dtype=np.float32)
        normalized = loader._normalize_volume(volume)

        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_normalize_constant_volume(self, mock_metadata, mock_discover):
        """Test normalization of constant volume."""
        mock_discover.return_value = [Path("test.npy")]
        mock_metadata.return_value = {}

        loader = InferenceDataLoader(data_dir="/tmp", normalize=True)

        volume = np.ones((10, 10, 10), dtype=np.float32) * 5.0
        normalized = loader._normalize_volume(volume)

        # Should return unchanged for constant volume
        assert np.allclose(normalized, 5.0)


class TestGetMetadata:
    """Test getting metadata."""

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_get_metadata_valid_index(self, mock_metadata_loader, mock_discover):
        """Test getting metadata by valid index."""
        test_metadata = {"test": {"shape": (64, 64, 64), "mean": 0.5}}
        mock_discover.return_value = [Path("test.npy")]
        mock_metadata_loader.return_value = test_metadata

        loader = InferenceDataLoader(data_dir="/tmp")
        metadata = loader.get_metadata(0)

        assert metadata["shape"] == (64, 64, 64)

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_get_metadata_invalid_index(self, mock_metadata, mock_discover):
        """Test getting metadata with invalid index."""
        mock_discover.return_value = [Path("test.npy")]
        mock_metadata.return_value = {}

        loader = InferenceDataLoader(data_dir="/tmp")

        with pytest.raises(IndexError, match="Index .* out of range"):
            loader.get_metadata(5)


class TestGetByName:
    """Test getting volume by name."""

    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_volume_file")
    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    def test_get_by_name_valid(self, mock_discover, mock_load, mock_metadata):
        """Test getting volume by valid name."""
        volume = np.ones((64, 64, 64), dtype=np.float32)
        mock_discover.return_value = [Path("test.npy")]
        mock_load.return_value = volume
        mock_metadata.return_value = {}

        loader = InferenceDataLoader(data_dir="/tmp")
        item = loader.get_by_name("test")

        assert "volume" in item
        assert item["name"] == "test"

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_get_by_name_not_found(self, mock_metadata, mock_discover):
        """Test getting volume with non-existent name."""
        mock_discover.return_value = [Path("test.npy")]
        mock_metadata.return_value = {}

        loader = InferenceDataLoader(data_dir="/tmp")

        with pytest.raises(ValueError, match="Volume .* not found"):
            loader.get_by_name("nonexistent")

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_get_by_name_multiple_matches(self, mock_metadata, mock_discover):
        """Test getting volume when multiple matches exist."""
        mock_discover.return_value = [Path("test.npy"), Path("test.nii")]
        mock_metadata.return_value = {}

        loader = InferenceDataLoader(data_dir="/tmp")

        with pytest.raises(ValueError, match="Multiple volumes"):
            loader.get_by_name("test")


class TestListVolumes:
    """Test listing volumes."""

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_list_volumes(self, mock_metadata, mock_discover):
        """Test listing all volume names."""
        paths = [Path(f"vol{i}.npy") for i in range(3)]
        mock_discover.return_value = paths
        mock_metadata.return_value = {}

        loader = InferenceDataLoader(data_dir="/tmp")
        names = loader.list_volumes()

        assert names == ["vol0", "vol1", "vol2"]

    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    def test_list_volumes_empty(self, mock_metadata, mock_discover):
        """Test listing volumes from empty directory."""
        mock_discover.return_value = []
        mock_metadata.return_value = {}

        # This will raise ValueError in __init__ for empty directory
        with pytest.raises(ValueError, match="No volume files found"):
            InferenceDataLoader(data_dir="/tmp")


class TestGetStatistics:
    """Test getting dataset statistics."""

    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    def test_get_statistics(self, mock_discover, mock_metadata):
        """Test getting statistics."""
        metadata = {
            "vol1": {"shape": (64, 64, 64), "min": 0.0, "max": 1.0, "mean": 0.5},
            "vol2": {"shape": (128, 128, 128), "min": -1.0, "max": 2.0, "mean": 0.6},
        }
        mock_discover.return_value = [Path("vol1.npy"), Path("vol2.npy")]
        mock_metadata.return_value = metadata

        loader = InferenceDataLoader(data_dir="/tmp")
        stats = loader.get_statistics()

        assert stats["num_volumes"] == 2
        assert len(stats["shapes"]) == 2
        assert stats["intensity_range"]["min"] == pytest.approx(-1.0)
        assert stats["intensity_range"]["max"] == pytest.approx(2.0)

    @patch("src.inference.inference_loader.InferenceDataLoader._load_metadata")
    @patch("src.inference.inference_loader.InferenceDataLoader._discover_volumes")
    def test_get_statistics_with_errors(self, mock_discover, mock_metadata):
        """Test statistics with some failed loads."""
        metadata = {
            "vol1": {"shape": (64, 64, 64), "min": 0.0, "max": 1.0, "mean": 0.5},
            "vol2": {"error": "failed to load"},
        }
        mock_discover.return_value = [Path("vol1.npy"), Path("vol2.npy")]
        mock_metadata.return_value = metadata

        loader = InferenceDataLoader(data_dir="/tmp")
        stats = loader.get_statistics()

        # Should only count valid volumes
        assert stats["num_volumes"] == 1
