"""Tests for preprocessing helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.data import preprocess
from src.data.preprocess import apply_hu_windowing, normalize_to_range


def test_apply_hu_windowing_clips_to_window() -> None:
    volume = np.array([-2000, -600, 0, 500], dtype=np.float32)
    windowed = apply_hu_windowing(volume, window_center=-600, window_width=1500)
    assert np.isclose(windowed.min(), -1350.0)
    assert np.isclose(windowed.max(), 150.0)


def test_normalize_to_range_outputs_target_interval() -> None:
    volume = np.array([0.0, 5.0, 10.0], dtype=np.float32)
    normalized = normalize_to_range(volume, target_min=-1.0, target_max=1.0)
    assert np.isclose(normalized.min(), -1.0)
    assert np.isclose(normalized.max(), 1.0)


def test_normalize_to_range_handles_constant_volume() -> None:
    volume = np.full((3,), 7.0, dtype=np.float32)
    normalized = normalize_to_range(volume)
    assert np.allclose(normalized, 0.0)


def test_normalize_to_range_rejects_invalid_target_range() -> None:
    volume = np.array([0.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError):
        normalize_to_range(volume, target_min=1.0, target_max=0.0)


def test_resample_to_isotropic_invokes_simpleitk(monkeypatch) -> None:
    class DummyImage:
        def __init__(self) -> None:
            self._spacing = (2.0, 1.0, 0.5)
            self._size = (64, 32, 16)

        def GetSpacing(self):
            return self._spacing

        def GetSize(self):
            return self._size

        def GetDirection(self):
            return (1.0,) * 9

        def GetOrigin(self):
            return (0.0, 0.0, 0.0)

        def GetPixelIDValue(self):
            return 1

    class FakeFilter:
        def __init__(self) -> None:
            self.size = None
            self.spacing = None
            self.interpolator = None
            self.executed = False

        def SetOutputSpacing(self, spacing):
            self.spacing = spacing

        def SetSize(self, size):
            self.size = size

        def SetOutputDirection(self, direction):
            self.direction = direction

        def SetOutputOrigin(self, origin):
            self.origin = origin

        def SetTransform(self, transform):
            self.transform = transform

        def SetDefaultPixelValue(self, value):
            self.default_value = value

        def SetInterpolator(self, interpolator):
            self.interpolator = interpolator

        def Execute(self, image):
            self.executed = True
            return {"image": image, "size": self.size}

    fake_filter = FakeFilter()
    monkeypatch.setattr(preprocess.sitk, "ResampleImageFilter", lambda: fake_filter)

    image = DummyImage()
    result = preprocess.resample_to_isotropic(
        image, target_spacing=(1.0, 1.0, 1.0), interpolator=42
    )

    assert fake_filter.size == [128, 32, 8]
    assert fake_filter.spacing == (1.0, 1.0, 1.0)
    assert fake_filter.interpolator == 42
    assert result["image"] is image


def test_preprocess_ct_scan_runs_pipeline(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "volume.mhd"
    image_path.write_text("dummy", encoding="utf-8")

    class DummyImage:
        pass

    read_calls: list[str] = []
    monkeypatch.setattr(
        preprocess.sitk, "ReadImage", lambda path: read_calls.append(path) or DummyImage()
    )

    resample_calls: list[tuple[object, tuple[float, float, float]]] = []

    def fake_resample(image, target_spacing):
        resample_calls.append((image, target_spacing))
        return "resampled"

    monkeypatch.setattr(preprocess, "resample_to_isotropic", fake_resample)

    monkeypatch.setattr(
        preprocess.sitk, "GetArrayFromImage", lambda image: np.ones((2, 2, 2), dtype=np.float32)
    )

    normalize_calls: list[dict[str, object]] = []

    def fake_normalize(array, **kwargs):
        normalize_calls.append(kwargs)
        if kwargs.get("return_stats"):
            return array * 2, {"stats": True}
        return array * 2

    monkeypatch.setattr(preprocess, "normalize_intensity", fake_normalize)

    result = preprocess.preprocess_ct_scan(image_path)

    assert isinstance(result, np.ndarray)
    assert read_calls == [str(image_path)]
    assert resample_calls[0][1] == (1.0, 1.0, 1.0)
    assert normalize_calls[-1]["return_stats"] is False

    result_with_stats = preprocess.preprocess_ct_scan(image_path, return_stats=True)

    assert isinstance(result_with_stats, tuple)
    assert normalize_calls[-1]["return_stats"] is True


def test_preprocess_main_outputs_message(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["preprocess", "--input_dir", "in", "--output_dir", "out"])

    preprocess.main()

    captured = capsys.readouterr()
    assert "Preprocessing CT scans" in captured.out
    assert "in" in captured.out
    assert "out" in captured.out
