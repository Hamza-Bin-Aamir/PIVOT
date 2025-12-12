"""Dataset classes for loading and batching CT scan data."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class LUNADataset(Dataset):
    """Dataset class for LUNA16 data.

    Loads preprocessed numpy arrays from disk.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        patch_size: tuple = (96, 96, 96),
        transform: Any | None = None,
    ):
        """Initialize LUNA dataset.

        Args:
            data_dir: Path to processed data directory
            split: Dataset split ('train' or 'val')
            patch_size: Size of 3D patches
            transform: Optional transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size = patch_size
        self.transform = transform

        # Load file list
        self.samples = self._load_samples()

    def _load_samples(self) -> list[Path]:
        """Load list of sample files."""
        split_dir = self.data_dir / self.split
        samples = list(split_dir.glob("*.npy"))
        return sorted(samples)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'image' and optional 'label' tensors
        """
        # Load numpy array
        sample_path = self.samples[idx]
        data = np.load(sample_path)

        # Convert to tensor
        image = torch.from_numpy(data).float()

        # Add channel dimension if needed
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return {"image": image, "path": str(sample_path)}
