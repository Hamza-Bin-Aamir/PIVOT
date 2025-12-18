"""Data loader for inference with full volume support."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

if TYPE_CHECKING:
    pass


class InferenceDataLoader(Dataset):
    """Data loader for inference on full CT volumes.

    This loader loads complete CT volumes (without patching) and provides
    metadata needed for inference, assembly, and post-processing.

    Attributes:
        data_dir: Directory containing preprocessed CT volumes.
        volumes: List of volume file paths.
        metadata: Dictionary of metadata for each volume.
    """

    def __init__(
        self,
        data_dir: str | Path,
        normalize: bool = True,
        normalize_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Initialize inference data loader.

        Args:
        ----
            data_dir: Path to directory containing volumes.
            normalize: Whether to normalize volumes to range.
            normalize_range: Target normalization range (min, max).

        Raises:
        ------
            ValueError: If data_dir doesn't exist or is empty.

        """
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise ValueError(f"data_dir does not exist: {data_dir}")

        # Discover volume files
        self.volumes = self._discover_volumes()

        if not self.volumes:
            raise ValueError(f"No volume files found in {data_dir}")

        self.normalize = normalize
        self.normalize_range = normalize_range

        # Load metadata for all volumes
        self.metadata = self._load_metadata()

    def _discover_volumes(self) -> list[Path]:
        """Discover volume files in data directory.

        Supports .npy, .nii, and .nii.gz formats.

        Returns:
        -------
            Sorted list of volume file paths.

        """
        volumes = []

        # Look for numpy arrays
        volumes.extend(sorted(self.data_dir.glob("*.npy")))

        # Look for NIfTI files
        volumes.extend(sorted(self.data_dir.glob("*.nii")))
        volumes.extend(sorted(self.data_dir.glob("*.nii.gz")))

        return volumes

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        """Load metadata for all volumes.

        Returns:
        -------
            Dictionary mapping volume names to metadata.

        """
        metadata = {}

        for vol_path in self.volumes:
            try:
                volume = self._load_volume_file(vol_path)
                metadata[vol_path.stem] = {
                    "shape": volume.shape,
                    "dtype": str(volume.dtype),
                    "min": float(np.min(volume)),
                    "max": float(np.max(volume)),
                    "mean": float(np.mean(volume)),
                    "std": float(np.std(volume)),
                    "path": str(vol_path),
                }
            except Exception as e:
                # Log but don't fail on metadata loading
                print(f"Warning: Failed to load metadata for {vol_path}: {e}")
                metadata[vol_path.stem] = {
                    "path": str(vol_path),
                    "error": str(e),
                }

        return metadata

    @staticmethod
    def _load_volume_file(path: Path) -> NDArray[np.float32]:
        """Load volume from file.

        Args:
        ----
            path: Path to volume file.

        Returns:
        -------
            3D numpy array.

        Raises:
        ------
            ValueError: If file format is unsupported.

        """
        if path.suffix == ".npy":
            return cast(NDArray[np.float32], np.load(path).astype(np.float32))

        if path.suffix == ".nii" or path.suffixes[-2:] == [".nii", ".gz"]:
            try:
                import nibabel as nib

                img = nib.load(path)
                return cast(NDArray[np.float32], img.get_fdata().astype(np.float32))  # type: ignore[attr-defined]
            except ImportError:
                raise ValueError("nibabel required for NIfTI file support")

        raise ValueError(f"Unsupported file format: {path.suffix}")

    def __len__(self) -> int:
        """Return number of volumes.

        Returns:
        -------
            Number of volumes in dataset.

        """
        return len(self.volumes)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a volume by index.

        Args:
        ----
            idx: Volume index.

        Returns:
        -------
            Dictionary containing:
            - "volume": 3D tensor (C, D, H, W) where C=1
            - "metadata": Volume metadata
            - "path": Path to volume file
            - "name": Volume name

        Raises:
        ------
            IndexError: If index is out of range.

        """
        if idx < 0 or idx >= len(self.volumes):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        vol_path = self.volumes[idx]
        vol_name = vol_path.stem

        # Load volume
        volume = self._load_volume_file(vol_path)

        # Normalize if requested
        if self.normalize:
            volume = self._normalize_volume(volume)

        # Convert to tensor and add channel dimension
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)

        return {
            "volume": volume_tensor,
            "metadata": self.metadata.get(vol_name, {}),
            "path": str(vol_path),
            "name": vol_name,
            "shape": volume.shape,
        }

    def _normalize_volume(self, volume: np.ndarray) -> NDArray[np.float32]:
        """Normalize volume to target range.

        Args:
        ----
            volume: Input volume.

        Returns:
        -------
            Normalized volume.

        """
        vmin = np.min(volume)
        vmax = np.max(volume)

        if vmax == vmin:
            # Constant volume, return as is
            return volume

        # Normalize to [0, 1]
        normalized = (volume - vmin) / (vmax - vmin)

        # Scale to target range
        tmin, tmax = self.normalize_range
        normalized = normalized * (tmax - tmin) + tmin

        return cast(NDArray[np.float32], normalized.astype(np.float32))

    def get_metadata(self, idx: int) -> dict[str, Any]:
        """Get metadata for a volume by index.

        Args:
        ----
            idx: Volume index.

        Returns:
        -------
            Metadata dictionary.

        """
        if idx < 0 or idx >= len(self.volumes):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        vol_name = self.volumes[idx].stem
        return self.metadata.get(vol_name, {})

    def get_by_name(self, name: str) -> dict[str, Any]:
        """Get a volume by name.

        Args:
        ----
            name: Volume name (without extension).

        Returns:
        -------
            Dictionary with volume and metadata.

        Raises:
        ------
            ValueError: If volume not found.

        """
        # Find matching volume
        matching = [i for i, v in enumerate(self.volumes) if v.stem == name]

        if not matching:
            raise ValueError(f"Volume '{name}' not found in dataset")

        if len(matching) > 1:
            raise ValueError(f"Multiple volumes with name '{name}' found")

        return self[matching[0]]

    def list_volumes(self) -> list[str]:
        """List all volume names.

        Returns:
        -------
            List of volume names.

        """
        return [v.stem for v in self.volumes]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics across all volumes.

        Returns:
        -------
            Dictionary with aggregate statistics.

        """
        if not self.metadata:
            return {}

        valid_metadata = [
            m for m in self.metadata.values()
            if "shape" in m and "error" not in m
        ]

        if not valid_metadata:
            return {}

        shapes = [m["shape"] for m in valid_metadata]
        mins = [m["min"] for m in valid_metadata]
        maxs = [m["max"] for m in valid_metadata]
        means = [m["mean"] for m in valid_metadata]

        return {
            "num_volumes": len(valid_metadata),
            "shapes": shapes,
            "shape_range": {
                "min": tuple(np.min(shapes, axis=0)),
                "max": tuple(np.max(shapes, axis=0)),
            },
            "intensity_range": {
                "min": float(np.min(mins)),
                "max": float(np.max(maxs)),
            },
            "intensity_mean": float(np.mean(means)),
        }
