"""Tests for multi-slice viewer."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.viz.viewer import (
    MultiSliceViewer,
    SliceOrientation,
    ViewerConfig,
    view_volume,
)


class TestViewerConfig:
    """Test ViewerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ViewerConfig()
        assert config.orientation == SliceOrientation.AXIAL
        assert config.figsize == (10.0, 8.0)
        assert config.dpi == 100
        assert config.cmap == "gray"
        assert config.vmin is None
        assert config.vmax is None
        assert config.show_slider is True
        assert config.show_info is True
        assert config.interpolation == "nearest"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ViewerConfig(
            orientation=SliceOrientation.SAGITTAL,
            figsize=(12.0, 10.0),
            dpi=150,
            cmap="hot",
            vmin=-1000.0,
            vmax=400.0,
            show_slider=False,
            show_info=False,
            interpolation="bilinear",
        )
        assert config.orientation == SliceOrientation.SAGITTAL
        assert config.figsize == (12.0, 10.0)
        assert config.dpi == 150
        assert config.cmap == "hot"
        assert config.vmin == -1000.0
        assert config.vmax == 400.0
        assert config.show_slider is False
        assert config.show_info is False
        assert config.interpolation == "bilinear"

    def test_invalid_dpi(self):
        """Test that invalid dpi raises ValueError."""
        with pytest.raises(ValueError, match="dpi must be"):
            ViewerConfig(dpi=0)
        with pytest.raises(ValueError, match="dpi must be"):
            ViewerConfig(dpi=-1)

    def test_invalid_vmin_vmax(self):
        """Test that vmin >= vmax raises ValueError."""
        with pytest.raises(ValueError, match="vmin .* must be < vmax"):
            ViewerConfig(vmin=100.0, vmax=50.0)
        with pytest.raises(ValueError, match="vmin .* must be < vmax"):
            ViewerConfig(vmin=100.0, vmax=100.0)


class TestMultiSliceViewer:
    """Test MultiSliceViewer class."""

    def test_initialization_3d_volume(self):
        """Test viewer initialization with 3D volume."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        viewer = MultiSliceViewer(volume)
        assert viewer.volume.shape == (50, 256, 256)
        assert viewer.mask is None
        assert viewer.num_slices == 50
        assert viewer.current_slice == 25

    def test_initialization_with_mask(self):
        """Test viewer initialization with mask."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        mask = np.random.rand(50, 256, 256) > 0.5
        viewer = MultiSliceViewer(volume, mask=mask)
        assert viewer.mask is not None
        assert viewer.mask.shape == volume.shape

    def test_initialization_with_custom_config(self):
        """Test viewer initialization with custom config."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        config = ViewerConfig(orientation=SliceOrientation.CORONAL, dpi=150)
        viewer = MultiSliceViewer(volume, config=config)
        assert viewer.config.orientation == SliceOrientation.CORONAL
        assert viewer.config.dpi == 150

    def test_invalid_volume_shape(self):
        """Test that non-3D volume raises ValueError."""
        volume_2d = np.random.rand(256, 256).astype(np.float32)
        with pytest.raises(ValueError, match="volume must be 3D"):
            MultiSliceViewer(volume_2d)

    def test_mismatched_mask_shape(self):
        """Test that mismatched mask shape raises ValueError."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        mask = np.random.rand(30, 256, 256) > 0.5
        with pytest.raises(ValueError, match="mask shape .* != volume shape"):
            MultiSliceViewer(volume, mask=mask)

    def test_axial_orientation(self):
        """Test axial slice orientation setup."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        config = ViewerConfig(orientation=SliceOrientation.AXIAL)
        viewer = MultiSliceViewer(volume, config=config)
        assert viewer.axis == 0
        assert viewer.num_slices == 50
        assert viewer.slice_label == "Axial Slice"

    def test_sagittal_orientation(self):
        """Test sagittal slice orientation setup."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        config = ViewerConfig(orientation=SliceOrientation.SAGITTAL)
        viewer = MultiSliceViewer(volume, config=config)
        assert viewer.axis == 2
        assert viewer.num_slices == 256
        assert viewer.slice_label == "Sagittal Slice"

    def test_coronal_orientation(self):
        """Test coronal slice orientation setup."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        config = ViewerConfig(orientation=SliceOrientation.CORONAL)
        viewer = MultiSliceViewer(volume, config=config)
        assert viewer.axis == 1
        assert viewer.num_slices == 256
        assert viewer.slice_label == "Coronal Slice"

    def test_get_slice_axial(self):
        """Test getting axial slice."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        viewer = MultiSliceViewer(volume)
        slice_data = viewer._get_slice(25)
        assert slice_data.shape == (256, 256)
        np.testing.assert_array_equal(slice_data, volume[25, :, :])

    def test_get_slice_sagittal(self):
        """Test getting sagittal slice."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        config = ViewerConfig(orientation=SliceOrientation.SAGITTAL)
        viewer = MultiSliceViewer(volume, config=config)
        slice_data = viewer._get_slice(128)
        assert slice_data.shape == (50, 256)
        np.testing.assert_array_equal(slice_data, volume[:, :, 128])

    def test_get_slice_coronal(self):
        """Test getting coronal slice."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        config = ViewerConfig(orientation=SliceOrientation.CORONAL)
        viewer = MultiSliceViewer(volume, config=config)
        slice_data = viewer._get_slice(128)
        assert slice_data.shape == (50, 256)
        np.testing.assert_array_equal(slice_data, volume[:, 128, :])

    def test_get_mask_slice_with_mask(self):
        """Test getting mask slice when mask is present."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        mask = np.random.rand(50, 256, 256) > 0.5
        viewer = MultiSliceViewer(volume, mask=mask)
        mask_slice = viewer._get_mask_slice(25)
        assert mask_slice is not None
        assert mask_slice.shape == (256, 256)
        np.testing.assert_array_equal(mask_slice, mask[25, :, :])

    def test_get_mask_slice_without_mask(self):
        """Test getting mask slice when no mask is present."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        viewer = MultiSliceViewer(volume)
        mask_slice = viewer._get_mask_slice(25)
        assert mask_slice is None

    def test_show_with_slider(self):
        """Test showing viewer with slider."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        viewer = MultiSliceViewer(volume)
        fig = viewer.show()
        assert fig is not None
        assert viewer.slider is not None
        plt.close(fig)

    def test_show_without_slider(self):
        """Test showing viewer without slider."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        config = ViewerConfig(show_slider=False)
        viewer = MultiSliceViewer(volume, config=config)
        fig = viewer.show()
        assert fig is not None
        assert viewer.slider is None
        plt.close(fig)

    def test_show_with_start_slice(self):
        """Test showing viewer with specified start slice."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        viewer = MultiSliceViewer(volume)
        fig = viewer.show(start_slice=10)
        assert viewer.current_slice == 10
        plt.close(fig)

    def test_show_with_out_of_bounds_start_slice(self):
        """Test showing viewer with out-of-bounds start slice gets clipped."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        viewer = MultiSliceViewer(volume)
        fig = viewer.show(start_slice=100)
        assert viewer.current_slice == 49  # Clipped to max
        plt.close(fig)

    def test_set_orientation(self):
        """Test changing viewer orientation."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        viewer = MultiSliceViewer(volume)
        fig = viewer.show()

        # Change to sagittal
        viewer.set_orientation(SliceOrientation.SAGITTAL)
        assert viewer.config.orientation == SliceOrientation.SAGITTAL
        assert viewer.axis == 2
        assert viewer.num_slices == 256

        plt.close(fig)

    def test_get_current_slice(self):
        """Test getting the current slice."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        viewer = MultiSliceViewer(volume)
        viewer.current_slice = 20
        current = viewer.get_current_slice()
        assert current.shape == (256, 256)
        np.testing.assert_array_equal(current, volume[20, :, :])

    def test_update_display_bounds_checking(self):
        """Test that update_display clips to valid slice indices."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        config = ViewerConfig(show_slider=False)
        viewer = MultiSliceViewer(volume, config=config)
        fig = viewer.show()

        # Try to update to out-of-bounds indices
        viewer._update_display(-5)
        assert viewer.current_slice == 0

        viewer._update_display(100)
        assert viewer.current_slice == 49

        plt.close(fig)


class TestViewVolume:
    """Test view_volume convenience function."""

    def test_basic_view(self):
        """Test basic volume viewing."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        fig = view_volume(volume)
        assert fig is not None
        plt.close(fig)

    def test_view_with_mask(self):
        """Test viewing with mask overlay."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        mask = np.random.rand(50, 256, 256) > 0.5
        fig = view_volume(volume, mask=mask)
        assert fig is not None
        plt.close(fig)

    def test_view_with_orientation(self):
        """Test viewing with specific orientation."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        fig = view_volume(volume, orientation=SliceOrientation.CORONAL)
        assert fig is not None
        plt.close(fig)

    def test_view_with_start_slice(self):
        """Test viewing with specific start slice."""
        volume = np.random.rand(50, 256, 256).astype(np.float32)
        fig = view_volume(volume, start_slice=20)
        assert fig is not None
        plt.close(fig)


class TestSliceOrientation:
    """Test SliceOrientation enum."""

    def test_orientation_values(self):
        """Test that SliceOrientation has expected values."""
        assert SliceOrientation.AXIAL == "axial"
        assert SliceOrientation.SAGITTAL == "sagittal"
        assert SliceOrientation.CORONAL == "coronal"
