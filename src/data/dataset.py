"""Dataset classes for loading and batching CT scan data."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

PatchSize = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _PatchTarget:
    sample_index: int
    center: tuple[float, float, float]
    is_positive: bool


class LUNADataset(Dataset):
    """Patch-based dataset for preprocessed LUNA/LIDC volumes.

    The dataset expects preprocessed volumes stored as ``.npz`` files within
    ``{data_dir}/{split}``. Each archive should contain at least an ``image``
    array of shape ``(Z, Y, X)``. Optional keys include ``mask`` (binary nodule
    mask), ``heatmap`` (centre heatmap), ``triage`` (scalar score), and pre-
    computed ``positive_centers``/``negative_centers`` arrays with ``(N, 3)``
    voxel coordinates. When explicit centres are not provided, the dataset will
    derive them from the mask/heatmap.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        split: str = "train",
        patch_size: Sequence[int] = (128, 128, 128),
        patches_per_volume: int = 16,
        positive_fraction: float = 0.5,
        cache_size: int = 4,
        seed: int | None = 1337,
        include_mask: bool = True,
        include_heatmap: bool = True,
        transform: Any | None = None,
    ) -> None:
        if positive_fraction <= 0 or positive_fraction > 1:
            msg = f"positive_fraction must be in (0, 1], got {positive_fraction}"
            raise ValueError(msg)
        if patches_per_volume <= 0:
            msg = f"patches_per_volume must be positive, got {patches_per_volume}"
            raise ValueError(msg)
        patch_tuple = tuple(int(v) for v in patch_size)
        if len(patch_tuple) != 3 or any(v <= 0 for v in patch_tuple):
            raise ValueError("patch_size must contain three positive integers")
        if cache_size < 0:
            raise ValueError("cache_size must be >= 0")

        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size: PatchSize = patch_tuple
        self.patches_per_volume = int(patches_per_volume)
        self.positive_fraction = float(positive_fraction)
        self.cache_size = int(cache_size)
        self.seed = seed
        self.include_mask = include_mask
        self.include_heatmap = include_heatmap
        self.transform = transform

        self.samples: list[Path] = self._discover_samples()
        if not self.samples:
            raise FileNotFoundError(
                f"No preprocessed volumes found in {(self.data_dir / self.split)!s}"
            )

        self._series_ids: list[str] = [path.stem for path in self.samples]
        self._main_state = self._create_state(seed)
        self._worker_states: dict[int, dict[str, Any]] = {}
        self._targets: list[_PatchTarget] = []

        self._build_targets()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        target = self._targets[index]
        state = self._get_state()
        volume = self._load_volume(self.samples[target.sample_index], state)

        image_patch = self._extract_patch(volume["image"], target.center)
        image_tensor = torch.from_numpy(image_patch).unsqueeze(0)

        sample: dict[str, Any] = {
            "image": image_tensor,
            "is_positive": target.is_positive,
            "center": tuple(float(c) for c in target.center),
            "series_uid": self._series_ids[target.sample_index],
            "path": str(self.samples[target.sample_index]),
        }

        if self.include_mask and volume.get("mask") is not None:
            mask_patch = self._extract_patch(volume["mask"], target.center)
            sample["mask"] = torch.from_numpy(mask_patch).unsqueeze(0)

        if self.include_heatmap and volume.get("heatmap") is not None:
            heatmap_patch = self._extract_patch(volume["heatmap"], target.center)
            sample["heatmap"] = torch.from_numpy(heatmap_patch).unsqueeze(0)

        if volume.get("triage") is not None:
            sample["triage"] = torch.tensor(float(volume["triage"]), dtype=torch.float32)

        if volume.get("spacing") is not None:
            sample["spacing"] = tuple(float(x) for x in volume["spacing"])

        if self.transform is not None:
            try:
                transformed = self.transform(sample)
            except TypeError:
                sample["image"] = self.transform(sample["image"])
            else:
                if isinstance(transformed, dict):
                    sample = transformed
                elif isinstance(transformed, torch.Tensor):
                    sample["image"] = transformed
                else:
                    msg = "Transform must return a Tensor or dict when invoked with sample"
                    raise TypeError(msg)

        return sample

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _build_targets(self) -> None:
        rng = np.random.default_rng(self.seed)
        for sample_index, path in enumerate(self.samples):
            volume = self._load_volume(path, self._main_state)
            shape = volume["image"].shape

            positives = self._collect_positive_centers(volume, rng)
            negatives = self._collect_negative_candidates(volume, rng)

            desired_positive = min(
                len(positives),
                max(1, int(round(self.patches_per_volume * self.positive_fraction)))
                if positives
                else 0,
            )

            if desired_positive:
                positive_selection = self._sample_centers(positives, desired_positive, rng)
            else:
                positive_selection = []

            if positive_selection:
                desired_negative = len(positive_selection)
            else:
                desired_negative = min(len(negatives), self.patches_per_volume)
                if desired_negative == 0:
                    desired_negative = max(1, self.patches_per_volume)

            negative_selection = self._select_negative_centers(
                negatives, desired_negative, rng, shape
            )

            for center in positive_selection:
                self._targets.append(_PatchTarget(sample_index, center, True))
            for center in negative_selection:
                self._targets.append(_PatchTarget(sample_index, center, False))

        rng.shuffle(self._targets)

    # ------------------------------------------------------------------
    # Volume discovery & loading
    # ------------------------------------------------------------------
    def _discover_samples(self) -> list[Path]:
        split_dir = self.data_dir / self.split
        candidates = list(split_dir.glob("*.npz")) + list(split_dir.glob("*.npy"))
        candidates.sort()
        return candidates

    def _create_state(self, seed: int | None) -> dict[str, Any]:
        return {
            "cache": OrderedDict(),
            "cache_hits": 0,
            "cache_misses": 0,
            "rng": np.random.default_rng(seed),
        }

    def _get_state(self) -> dict[str, Any]:
        worker = get_worker_info()
        if worker is None:
            return self._main_state
        state = self._worker_states.get(worker.id)
        if state is None:
            base_seed = 0 if self.seed is None else self.seed
            state = self._create_state(base_seed + worker.id)
            self._worker_states[worker.id] = state
        return state

    def _load_volume(self, path: Path, state: dict[str, Any]) -> dict[str, Any]:
        cache = state["cache"]
        key = str(path)
        if self.cache_size > 0 and key in cache:
            state["cache_hits"] += 1
            cache.move_to_end(key)
            return cache[key]

        volume = self._read_volume(path)
        state["cache_misses"] += 1
        if self.cache_size > 0:
            cache[key] = volume
            if len(cache) > self.cache_size:
                cache.popitem(last=False)
        return volume

    def _read_volume(self, path: Path) -> dict[str, Any]:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                image = np.asarray(data["image"], dtype=np.float32)
                if image.ndim != 3:
                    raise ValueError("Expected image volume with 3 dimensions")

                payload: dict[str, Any] = {"image": np.ascontiguousarray(image)}

                if "mask" in data:
                    payload["mask"] = np.ascontiguousarray(
                        np.asarray(data["mask"], dtype=np.float32)
                    )
                if "heatmap" in data:
                    payload["heatmap"] = np.ascontiguousarray(
                        np.asarray(data["heatmap"], dtype=np.float32)
                    )
                if "triage" in data:
                    triage = np.asarray(data["triage"], dtype=np.float32)
                    payload["triage"] = float(np.squeeze(triage))
                if "spacing" in data:
                    spacing = np.asarray(data["spacing"], dtype=np.float32)
                    payload["spacing"] = tuple(float(x) for x in spacing.reshape(-1)[:3])
                if "positive_centers" in data:
                    payload["positive_centers"] = np.asarray(
                        data["positive_centers"], dtype=np.float32
                    )
                if "negative_centers" in data:
                    payload["negative_centers"] = np.asarray(
                        data["negative_centers"], dtype=np.float32
                    )
        else:
            array = np.load(path)
            image = np.asarray(array, dtype=np.float32)
            if image.ndim != 3:
                raise ValueError("Expected image volume with 3 dimensions in .npy file")
            payload = {"image": np.ascontiguousarray(image)}

        return payload

    # ------------------------------------------------------------------
    # Centre selection helpers
    # ------------------------------------------------------------------
    def _collect_positive_centers(
        self, volume: dict[str, Any], rng: np.random.Generator
    ) -> list[tuple[float, float, float]]:
        if "positive_centers" in volume:
            centers = np.asarray(volume["positive_centers"], dtype=np.float32)
            centers = centers.reshape(-1, 3)
            return [tuple(map(float, row)) for row in centers]

        mask = volume.get("mask")
        if mask is None:
            heatmap = volume.get("heatmap")
            if heatmap is None:
                return []
            coords = np.argwhere(heatmap > 0.5)
        else:
            coords = np.argwhere(mask > 0.5)

        if coords.size == 0:
            return []

        coords = coords.astype(np.float32)
        if coords.shape[0] > self.patches_per_volume:
            indices = rng.choice(coords.shape[0], size=self.patches_per_volume, replace=False)
            coords = coords[indices]
        return [tuple(map(float, row)) for row in coords]

    def _collect_negative_candidates(
        self,
        volume: dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        if "negative_centers" in volume:
            candidates = np.asarray(volume["negative_centers"], dtype=np.float32)
            return candidates.reshape(-1, 3)

        mask = volume.get("mask")
        if mask is None:
            return np.empty((0, 3), dtype=np.float32)

        inverse = np.argwhere(mask <= 0.5).astype(np.float32)
        if inverse.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        if inverse.shape[0] > self.patches_per_volume * 4:
            indices = rng.choice(inverse.shape[0], size=self.patches_per_volume * 4, replace=False)
            inverse = inverse[indices]

        return inverse

    def _sample_centers(
        self,
        centers: Iterable[tuple[float, float, float]] | np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> list[tuple[float, float, float]]:
        array = np.asarray(list(centers), dtype=np.float32)
        if array.size == 0:
            return []
        array = array.reshape(-1, 3)
        if count >= array.shape[0]:
            return [tuple(map(float, row)) for row in array]
        indices = rng.choice(array.shape[0], size=count, replace=False)
        return [tuple(map(float, array[i])) for i in indices]

    def _select_negative_centers(
        self,
        candidates: np.ndarray,
        count: int,
        rng: np.random.Generator,
        shape: Iterable[int],
    ) -> list[tuple[float, float, float]]:
        if count <= 0:
            return []

        shape_arr = np.array(tuple(int(s) for s in shape), dtype=np.int32)
        half = np.array(self.patch_size, dtype=np.int32) // 2

        def within_bounds(points: np.ndarray) -> np.ndarray:
            if points.size == 0:
                return points
            mins = points >= half
            maxs = points <= (shape_arr - 1 - half)
            mask = np.all(mins & maxs, axis=1)
            return points[mask]

        filtered = within_bounds(candidates.astype(np.float32))

        if filtered.shape[0] >= count:
            indices = rng.choice(filtered.shape[0], size=count, replace=False)
            chosen = filtered[indices]
        else:
            chosen = filtered
            remaining = count - chosen.shape[0]
            if remaining > 0:
                sampled = self._sample_uniform_centers(remaining, rng, shape_arr)
                chosen = np.vstack([chosen, sampled]) if chosen.size else sampled

        return [tuple(map(float, row)) for row in chosen.astype(np.float32)]

    def _sample_uniform_centers(
        self, count: int, rng: np.random.Generator, shape: np.ndarray
    ) -> np.ndarray:
        if count <= 0:
            return np.empty((0, 3), dtype=np.float32)
        if np.any(shape <= 0):
            return np.zeros((count, 3), dtype=np.float32)

        coords = rng.uniform(low=0.0, high=np.maximum(shape - 1, 1), size=(count, 3))
        return coords.astype(np.float32)

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------
    def _extract_patch(self, volume: np.ndarray, center: tuple[float, float, float]) -> np.ndarray:
        center_arr = np.asarray(center, dtype=np.float32)
        size = np.asarray(self.patch_size, dtype=np.int32)
        half = size // 2

        center_int = np.round(center_arr).astype(np.int32)
        mins = center_int - half
        maxs = mins + size

        shape = np.array(volume.shape, dtype=np.int32)
        pad_before = np.maximum(0, -mins)
        pad_after = np.maximum(0, maxs - shape)

        valid_min = np.maximum(mins, 0)
        valid_max = np.minimum(maxs, shape)

        slices = tuple(slice(int(a), int(b)) for a, b in zip(valid_min, valid_max, strict=True))
        patch = volume[slices]

        if pad_before.any() or pad_after.any():
            padding = tuple((int(pad_before[i]), int(pad_after[i])) for i in range(3))
            patch = np.pad(patch, padding, mode="constant", constant_values=0)

        patch = self._ensure_patch_shape(patch)
        return np.ascontiguousarray(patch, dtype=np.float32)

    def _ensure_patch_shape(self, patch: np.ndarray) -> np.ndarray:
        target = np.array(self.patch_size, dtype=np.int32)
        current = np.array(patch.shape, dtype=np.int32)

        if np.array_equal(target, current):
            return patch

        # Crop if too large
        if np.any(current > target):
            start = np.maximum(0, (current - target) // 2)
            end = start + target
            slices = tuple(slice(int(s), int(e)) for s, e in zip(start, end, strict=True))
            patch = patch[slices]
            current = np.array(patch.shape, dtype=np.int32)

        # Pad if still smaller
        if np.any(current < target):
            pad_width = []
            for dim, tgt in zip(current, target, strict=True):
                deficit = int(tgt - dim)
                pad_width.append((0, deficit))
            patch = np.pad(patch, tuple(pad_width), mode="constant", constant_values=0)
            slices = tuple(slice(0, int(t)) for t in target)
            patch = patch[slices]

        return patch
