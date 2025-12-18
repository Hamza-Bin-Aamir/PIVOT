"""Tests for 3D lung rendering."""

import numpy as np
import plotly.graph_objects as go
import pytest

from src.viz.rendering_3d import (
    RenderingConfig,
    render_comparison,
    render_lung_surface,
    render_volume,
    render_with_nodules,
)


class TestRenderingConfig:
    """Test RenderingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RenderingConfig()
        assert config.colorscale == "gray"
        assert config.opacity == 0.3
        assert config.show_axes is True
        assert config.camera_eye == (1.5, 1.5, 1.5)
        assert config.lighting is not None
        assert config.surface_count == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_lighting = {"ambient": 0.6, "diffuse": 0.9}
        config = RenderingConfig(
            colorscale="viridis",
            opacity=0.5,
            show_axes=False,
            camera_eye=(2.0, 2.0, 2.0),
            lighting=custom_lighting,
            surface_count=15,
        )
        assert config.colorscale == "viridis"
        assert config.opacity == 0.5
        assert config.show_axes is False
        assert config.camera_eye == (2.0, 2.0, 2.0)
        assert config.lighting == custom_lighting
        assert config.surface_count == 15

    def test_default_lighting(self):
        """Test that default lighting is set correctly."""
        config = RenderingConfig()
        assert "ambient" in config.lighting
        assert "diffuse" in config.lighting
        assert "specular" in config.lighting
        assert "roughness" in config.lighting

    def test_invalid_opacity(self):
        """Test that invalid opacity raises ValueError."""
        with pytest.raises(ValueError, match="opacity must be in"):
            RenderingConfig(opacity=-0.1)
        with pytest.raises(ValueError, match="opacity must be in"):
            RenderingConfig(opacity=1.5)

    def test_invalid_surface_count(self):
        """Test that invalid surface_count raises ValueError."""
        with pytest.raises(ValueError, match="surface_count must be"):
            RenderingConfig(surface_count=0)
        with pytest.raises(ValueError, match="surface_count must be"):
            RenderingConfig(surface_count=-1)


class TestRenderLungSurface:
    """Test render_lung_surface function."""

    def test_basic_rendering(self):
        """Test basic lung surface rendering."""
        mask = np.zeros((50, 64, 64), dtype=bool)
        mask[20:35, 25:40, 25:40] = True

        fig = render_lung_surface(mask)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Mesh3d)

    def test_with_custom_spacing(self):
        """Test rendering with custom voxel spacing."""
        mask = np.zeros((50, 64, 64), dtype=bool)
        mask[20:35, 25:40, 25:40] = True
        spacing = (2.5, 1.0, 1.0)

        fig = render_lung_surface(mask, spacing=spacing)
        assert isinstance(fig, go.Figure)

    def test_with_custom_config(self):
        """Test rendering with custom configuration."""
        mask = np.zeros((50, 64, 64), dtype=bool)
        mask[20:35, 25:40, 25:40] = True
        config = RenderingConfig(opacity=0.5, colorscale="viridis")

        fig = render_lung_surface(mask, config=config)
        assert isinstance(fig, go.Figure)
        assert fig.data[0].opacity == 0.5

    def test_with_custom_title(self):
        """Test rendering with custom title."""
        mask = np.zeros((50, 64, 64), dtype=bool)
        mask[20:35, 25:40, 25:40] = True

        fig = render_lung_surface(mask, title="Custom Title")
        assert fig.layout.title.text == "Custom Title"

    def test_invalid_mask_shape(self):
        """Test that non-3D mask raises ValueError."""
        mask_2d = np.zeros((64, 64), dtype=bool)
        with pytest.raises(ValueError, match="mask must be 3D"):
            render_lung_surface(mask_2d)

    def test_invalid_spacing_length(self):
        """Test that invalid spacing length raises ValueError."""
        mask = np.zeros((50, 64, 64), dtype=bool)
        mask[20:35, 25:40, 25:40] = True
        with pytest.raises(ValueError, match="spacing must have 3 values"):
            render_lung_surface(mask, spacing=(1.0, 1.0))

    def test_invalid_spacing_values(self):
        """Test that non-positive spacing raises ValueError."""
        mask = np.zeros((50, 64, 64), dtype=bool)
        mask[20:35, 25:40, 25:40] = True
        with pytest.raises(ValueError, match="spacing values must be > 0"):
            render_lung_surface(mask, spacing=(1.0, 0.0, 1.0))

    def test_empty_mask(self):
        """Test that empty mask raises ValueError."""
        mask = np.zeros((50, 64, 64), dtype=bool)
        with pytest.raises(ValueError, match="mask is empty"):
            render_lung_surface(mask)


class TestRenderVolume:
    """Test render_volume function."""

    def test_basic_volume_rendering(self):
        """Test basic volume rendering."""
        volume = np.random.rand(50, 64, 64).astype(np.float32)

        fig = render_volume(volume)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Volume)

    def test_with_custom_spacing(self):
        """Test volume rendering with custom spacing."""
        volume = np.random.rand(50, 64, 64).astype(np.float32)
        spacing = (2.5, 1.0, 1.0)

        fig = render_volume(volume, spacing=spacing)
        assert isinstance(fig, go.Figure)

    def test_with_custom_config(self):
        """Test volume rendering with custom configuration."""
        volume = np.random.rand(50, 64, 64).astype(np.float32)
        config = RenderingConfig(opacity=0.2, colorscale="Viridis")

        fig = render_volume(volume, config=config)
        assert isinstance(fig, go.Figure)
        assert fig.data[0].opacity == 0.2
        assert fig.data[0].colorscale is not None  # Config was applied

    def test_with_custom_title(self):
        """Test volume rendering with custom title."""
        volume = np.random.rand(50, 64, 64).astype(np.float32)

        fig = render_volume(volume, title="Custom Volume")
        assert fig.layout.title.text == "Custom Volume"

    def test_constant_volume(self):
        """Test rendering of constant-valued volume."""
        volume = np.ones((50, 64, 64), dtype=np.float32) * 100.0

        fig = render_volume(volume)
        assert isinstance(fig, go.Figure)

    def test_invalid_volume_shape(self):
        """Test that non-3D volume raises ValueError."""
        volume_2d = np.random.rand(64, 64).astype(np.float32)
        with pytest.raises(ValueError, match="volume must be 3D"):
            render_volume(volume_2d)

    def test_invalid_spacing_length(self):
        """Test that invalid spacing length raises ValueError."""
        volume = np.random.rand(50, 64, 64).astype(np.float32)
        with pytest.raises(ValueError, match="spacing must have 3 values"):
            render_volume(volume, spacing=(1.0, 1.0))

    def test_invalid_spacing_values(self):
        """Test that non-positive spacing raises ValueError."""
        volume = np.random.rand(50, 64, 64).astype(np.float32)
        with pytest.raises(ValueError, match="spacing values must be > 0"):
            render_volume(volume, spacing=(1.0, -1.0, 1.0))


class TestRenderWithNodules:
    """Test render_with_nodules function."""

    def test_basic_rendering_with_nodules(self):
        """Test rendering lung with nodule markers."""
        lung_mask = np.zeros((50, 64, 64), dtype=bool)
        lung_mask[20:35, 25:40, 25:40] = True
        centers = np.array([[25, 30, 30], [30, 35, 35]], dtype=np.float64)

        fig = render_with_nodules(lung_mask, centers)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Lung surface + nodules

    def test_with_nodule_radii(self):
        """Test rendering with specified nodule radii."""
        lung_mask = np.zeros((50, 64, 64), dtype=bool)
        lung_mask[20:35, 25:40, 25:40] = True
        centers = np.array([[25, 30, 30]], dtype=np.float64)
        radii = np.array([5.0], dtype=np.float64)

        fig = render_with_nodules(lung_mask, centers, nodule_radii=radii)
        assert isinstance(fig, go.Figure)

    def test_with_custom_spacing(self):
        """Test rendering with custom spacing."""
        lung_mask = np.zeros((50, 64, 64), dtype=bool)
        lung_mask[20:35, 25:40, 25:40] = True
        centers = np.array([[25, 30, 30]], dtype=np.float64)
        spacing = (2.5, 1.0, 1.0)

        fig = render_with_nodules(lung_mask, centers, spacing=spacing)
        assert isinstance(fig, go.Figure)

    def test_with_no_nodules(self):
        """Test rendering with empty nodule array."""
        lung_mask = np.zeros((50, 64, 64), dtype=bool)
        lung_mask[20:35, 25:40, 25:40] = True
        centers = np.zeros((0, 3), dtype=np.float64)

        fig = render_with_nodules(lung_mask, centers)
        assert isinstance(fig, go.Figure)

    def test_invalid_lung_mask_shape(self):
        """Test that non-3D lung mask raises ValueError."""
        mask_2d = np.zeros((64, 64), dtype=bool)
        centers = np.array([[25, 30, 30]], dtype=np.float64)
        with pytest.raises(ValueError, match="lung_mask must be 3D"):
            render_with_nodules(mask_2d, centers)

    def test_invalid_centers_shape(self):
        """Test that invalid centers shape raises ValueError."""
        lung_mask = np.zeros((50, 64, 64), dtype=bool)
        lung_mask[20:35, 25:40, 25:40] = True
        centers = np.array([25, 30, 30], dtype=np.float64)  # 1D instead of 2D
        with pytest.raises(ValueError, match="nodule_centers must have shape"):
            render_with_nodules(lung_mask, centers)

    def test_invalid_radii_shape(self):
        """Test that invalid radii shape raises ValueError."""
        lung_mask = np.zeros((50, 64, 64), dtype=bool)
        lung_mask[20:35, 25:40, 25:40] = True
        centers = np.array([[25, 30, 30]], dtype=np.float64)
        radii = np.array([[5.0]], dtype=np.float64)  # 2D instead of 1D
        with pytest.raises(ValueError, match="nodule_radii must be 1D"):
            render_with_nodules(lung_mask, centers, nodule_radii=radii)

    def test_mismatched_radii_length(self):
        """Test that mismatched radii length raises ValueError."""
        lung_mask = np.zeros((50, 64, 64), dtype=bool)
        lung_mask[20:35, 25:40, 25:40] = True
        centers = np.array([[25, 30, 30], [30, 35, 35]], dtype=np.float64)
        radii = np.array([5.0], dtype=np.float64)  # Wrong length
        with pytest.raises(ValueError, match="nodule_radii length"):
            render_with_nodules(lung_mask, centers, nodule_radii=radii)


class TestRenderComparison:
    """Test render_comparison function."""

    def test_basic_comparison(self):
        """Test basic prediction vs ground truth comparison."""
        pred = np.zeros((50, 64, 64), dtype=bool)
        pred[20:32, 25:37, 25:37] = True
        gt = np.zeros((50, 64, 64), dtype=bool)
        gt[22:34, 27:39, 27:39] = True

        fig = render_comparison(pred, gt)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Both masks

    def test_with_custom_spacing(self):
        """Test comparison with custom spacing."""
        pred = np.zeros((50, 64, 64), dtype=bool)
        pred[20:32, 25:37, 25:37] = True
        gt = np.zeros((50, 64, 64), dtype=bool)
        gt[22:34, 27:39, 27:39] = True
        spacing = (2.5, 1.0, 1.0)

        fig = render_comparison(pred, gt, spacing=spacing)
        assert isinstance(fig, go.Figure)

    def test_with_custom_config(self):
        """Test comparison with custom configuration."""
        pred = np.zeros((50, 64, 64), dtype=bool)
        pred[20:32, 25:37, 25:37] = True
        gt = np.zeros((50, 64, 64), dtype=bool)
        gt[22:34, 27:39, 27:39] = True
        config = RenderingConfig(opacity=0.4)

        fig = render_comparison(pred, gt, config=config)
        assert isinstance(fig, go.Figure)

    def test_only_prediction(self):
        """Test comparison with only prediction mask."""
        pred = np.zeros((50, 64, 64), dtype=bool)
        pred[20:32, 25:37, 25:37] = True
        gt = np.zeros((50, 64, 64), dtype=bool)

        fig = render_comparison(pred, gt)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_only_ground_truth(self):
        """Test comparison with only ground truth mask."""
        pred = np.zeros((50, 64, 64), dtype=bool)
        gt = np.zeros((50, 64, 64), dtype=bool)
        gt[22:34, 27:39, 27:39] = True

        fig = render_comparison(pred, gt)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_mismatched_shapes(self):
        """Test that mismatched mask shapes raise ValueError."""
        pred = np.zeros((50, 64, 64), dtype=bool)
        gt = np.zeros((60, 64, 64), dtype=bool)
        with pytest.raises(ValueError, match="prediction_mask shape .* != ground_truth_mask shape"):
            render_comparison(pred, gt)

    def test_invalid_mask_dimensions(self):
        """Test that non-3D masks raise ValueError."""
        pred = np.zeros((64, 64), dtype=bool)
        gt = np.zeros((64, 64), dtype=bool)
        with pytest.raises(ValueError, match="masks must be 3D"):
            render_comparison(pred, gt)

    def test_both_masks_empty(self):
        """Test that both empty masks raise ValueError."""
        pred = np.zeros((50, 64, 64), dtype=bool)
        gt = np.zeros((50, 64, 64), dtype=bool)
        with pytest.raises(ValueError, match="both masks are empty"):
            render_comparison(pred, gt)
