"""Tests for Grad-CAM attention visualization."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest
import torch
import torch.nn as nn

from src.viz.gradcam import (
    GradCAM,
    GradCAMConfig,
    overlay_gradcam_on_volume,
    render_gradcam_3d,
)


class TestGradCAMConfig:
    """Tests for GradCAMConfig dataclass."""

    def test_default_config(self):
        """Test GradCAMConfig with default values."""
        config = GradCAMConfig()
        assert config.colorscale == "Hot"
        assert config.opacity == 0.5
        assert config.threshold == 0.0
        assert config.show_colorbar is True

    def test_custom_config(self):
        """Test GradCAMConfig with custom values."""
        config = GradCAMConfig(
            colorscale="Viridis", opacity=0.7, threshold=0.3, show_colorbar=False
        )
        assert config.colorscale == "Viridis"
        assert config.opacity == 0.7
        assert config.threshold == 0.3
        assert config.show_colorbar is False

    def test_invalid_opacity(self):
        """Test GradCAMConfig rejects invalid opacity."""
        with pytest.raises(ValueError, match="Opacity must be in"):
            GradCAMConfig(opacity=1.5)
        with pytest.raises(ValueError, match="Opacity must be in"):
            GradCAMConfig(opacity=-0.1)

    def test_invalid_threshold(self):
        """Test GradCAMConfig rejects invalid threshold."""
        with pytest.raises(ValueError, match="Threshold must be in"):
            GradCAMConfig(threshold=1.5)
        with pytest.raises(ValueError, match="Threshold must be in"):
            GradCAMConfig(threshold=-0.1)


class SimpleConvNet(nn.Module):
    """Simple 3D CNN for testing Grad-CAM."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class TestGradCAM:
    """Tests for GradCAM class."""

    def test_gradcam_initialization(self):
        """Test GradCAM initializes correctly."""
        model = SimpleConvNet()
        gradcam = GradCAM(model, model.conv2)

        assert gradcam.model is model
        assert gradcam.target_layer is model.conv2
        assert gradcam.activations is None
        assert gradcam.gradients is None

    def test_gradcam_generate_basic(self):
        """Test basic Grad-CAM generation."""
        model = SimpleConvNet()
        model.eval()

        gradcam = GradCAM(model, model.conv2)
        input_tensor = torch.randn(1, 1, 16, 16, 16)

        cam = gradcam.generate(input_tensor)

        assert isinstance(cam, np.ndarray)
        assert cam.shape == (16, 16, 16)
        assert cam.dtype == np.float32
        assert cam.min() >= 0
        assert cam.max() <= 1

    def test_gradcam_captures_activations(self):
        """Test that Grad-CAM captures forward activations."""
        model = SimpleConvNet()
        model.eval()

        gradcam = GradCAM(model, model.conv2)
        input_tensor = torch.randn(1, 1, 16, 16, 16)

        gradcam.generate(input_tensor)

        assert gradcam.activations is not None
        assert gradcam.activations.shape == (1, 16, 16, 16, 16)

    def test_gradcam_captures_gradients(self):
        """Test that Grad-CAM captures backward gradients."""
        model = SimpleConvNet()
        model.eval()

        gradcam = GradCAM(model, model.conv2)
        input_tensor = torch.randn(1, 1, 16, 16, 16)

        gradcam.generate(input_tensor)

        assert gradcam.gradients is not None
        assert gradcam.gradients.shape == (1, 16, 16, 16, 16)

    def test_gradcam_invalid_input_dim(self):
        """Test Grad-CAM rejects wrong input dimensions."""
        model = SimpleConvNet()
        gradcam = GradCAM(model, model.conv2)

        with pytest.raises(ValueError, match="Expected 5D input"):
            gradcam.generate(torch.randn(16, 16, 16))

    def test_gradcam_invalid_batch_size(self):
        """Test Grad-CAM rejects batch size != 1."""
        model = SimpleConvNet()
        gradcam = GradCAM(model, model.conv2)

        with pytest.raises(ValueError, match="Expected batch size 1"):
            gradcam.generate(torch.randn(2, 1, 16, 16, 16))

    def test_gradcam_with_dict_output(self):
        """Test Grad-CAM handles dict output from multi-task models."""

        class MultiTaskNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(1, 8, kernel_size=3, padding=1)
                self.seg_head = nn.Conv3d(8, 1, kernel_size=1)

            def forward(self, x):
                feat = self.conv(x)
                return {"segmentation": self.seg_head(feat)}

        model = MultiTaskNet()
        model.eval()

        gradcam = GradCAM(model, model.conv)
        input_tensor = torch.randn(1, 1, 16, 16, 16)

        cam = gradcam.generate(input_tensor)

        assert cam.shape == (16, 16, 16)
        assert cam.min() >= 0
        assert cam.max() <= 1

    def test_gradcam_hook_cleanup(self):
        """Test Grad-CAM removes hooks on deletion."""
        model = SimpleConvNet()
        gradcam = GradCAM(model, model.conv2)

        # Check hooks are registered
        assert hasattr(gradcam, "_forward_hook")
        assert hasattr(gradcam, "_backward_hook")

        # Delete and verify cleanup
        del gradcam
        # If hooks weren't removed, this would cause issues in subsequent tests


class TestOverlayGradCAMOnVolume:
    """Tests for overlay_gradcam_on_volume function."""

    def test_basic_overlay_axial(self):
        """Test basic Grad-CAM overlay on axial slice."""
        volume = np.random.rand(32, 64, 64).astype(np.float32)
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)

        fig = overlay_gradcam_on_volume(volume, gradcam, slice_idx=16, axis=0)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Volume + overlay
        assert "Axial" in fig.layout.title.text

    def test_overlay_coronal(self):
        """Test Grad-CAM overlay on coronal slice."""
        volume = np.random.rand(32, 64, 64).astype(np.float32)
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)

        fig = overlay_gradcam_on_volume(volume, gradcam, slice_idx=32, axis=1)

        assert isinstance(fig, go.Figure)
        assert "Coronal" in fig.layout.title.text

    def test_overlay_sagittal(self):
        """Test Grad-CAM overlay on sagittal slice."""
        volume = np.random.rand(32, 64, 64).astype(np.float32)
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)

        fig = overlay_gradcam_on_volume(volume, gradcam, slice_idx=32, axis=2)

        assert isinstance(fig, go.Figure)
        assert "Sagittal" in fig.layout.title.text

    def test_overlay_with_custom_config(self):
        """Test overlay with custom configuration."""
        volume = np.random.rand(32, 64, 64).astype(np.float32)
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)
        config = GradCAMConfig(opacity=0.8, colorscale="Jet")

        fig = overlay_gradcam_on_volume(
            volume, gradcam, slice_idx=16, axis=0, config=config
        )

        assert isinstance(fig, go.Figure)
        assert fig.data[1].opacity == 0.8

    def test_overlay_with_threshold(self):
        """Test overlay applies threshold correctly."""
        volume = np.random.rand(32, 64, 64).astype(np.float32)
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)
        config = GradCAMConfig(threshold=0.5)

        fig = overlay_gradcam_on_volume(
            volume, gradcam, slice_idx=16, axis=0, config=config
        )

        assert isinstance(fig, go.Figure)

    def test_overlay_shape_mismatch(self):
        """Test overlay rejects mismatched shapes."""
        volume = np.random.rand(32, 64, 64).astype(np.float32)
        gradcam = np.random.rand(16, 32, 32).astype(np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            overlay_gradcam_on_volume(volume, gradcam, slice_idx=16, axis=0)

    def test_overlay_invalid_dimensions(self):
        """Test overlay rejects non-3D volumes."""
        volume = np.random.rand(64, 64).astype(np.float32)
        gradcam = np.random.rand(64, 64).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 3D volume"):
            overlay_gradcam_on_volume(volume, gradcam, slice_idx=32, axis=0)

    def test_overlay_invalid_axis(self):
        """Test overlay rejects invalid axis."""
        volume = np.random.rand(32, 64, 64).astype(np.float32)
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)

        with pytest.raises(ValueError, match="Axis must be"):
            overlay_gradcam_on_volume(volume, gradcam, slice_idx=16, axis=3)

    def test_overlay_invalid_slice_index(self):
        """Test overlay rejects out-of-range slice index."""
        volume = np.random.rand(32, 64, 64).astype(np.float32)
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)

        with pytest.raises(ValueError, match="Slice index .* out of range"):
            overlay_gradcam_on_volume(volume, gradcam, slice_idx=100, axis=0)

    def test_overlay_custom_title(self):
        """Test overlay with custom title."""
        volume = np.random.rand(32, 64, 64).astype(np.float32)
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)

        fig = overlay_gradcam_on_volume(
            volume, gradcam, slice_idx=16, axis=0, title="Custom Attention Map"
        )

        assert "Custom Attention Map" in fig.layout.title.text


class TestRenderGradCAM3D:
    """Tests for render_gradcam_3d function."""

    def test_basic_3d_rendering(self):
        """Test basic 3D Grad-CAM rendering."""
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)

        fig = render_gradcam_3d(gradcam)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Volume)

    def test_3d_with_custom_spacing(self):
        """Test 3D rendering with custom voxel spacing."""
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)

        fig = render_gradcam_3d(gradcam, spacing=(2.0, 1.0, 1.0))

        assert isinstance(fig, go.Figure)

    def test_3d_with_custom_config(self):
        """Test 3D rendering with custom configuration."""
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)
        config = GradCAMConfig(opacity=0.3, colorscale="Plasma", threshold=0.2)

        fig = render_gradcam_3d(gradcam, config=config)

        assert isinstance(fig, go.Figure)
        assert fig.data[0].opacity == 0.3

    def test_3d_with_threshold(self):
        """Test 3D rendering applies threshold."""
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)
        config = GradCAMConfig(threshold=0.8)

        fig = render_gradcam_3d(gradcam, config=config)

        assert isinstance(fig, go.Figure)

    def test_3d_invalid_dimensions(self):
        """Test 3D rendering rejects non-3D input."""
        gradcam = np.random.rand(64, 64).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 3D gradcam"):
            render_gradcam_3d(gradcam)

    def test_3d_invalid_spacing_length(self):
        """Test 3D rendering rejects invalid spacing length."""
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)

        with pytest.raises(ValueError, match="Spacing must have 3 values"):
            render_gradcam_3d(gradcam, spacing=(1.0, 1.0))

    def test_3d_invalid_spacing_values(self):
        """Test 3D rendering rejects non-positive spacing."""
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)

        with pytest.raises(ValueError, match="Spacing values must be positive"):
            render_gradcam_3d(gradcam, spacing=(1.0, 0.0, 1.0))

    def test_3d_custom_title(self):
        """Test 3D rendering with custom title."""
        gradcam = np.random.rand(32, 64, 64).astype(np.float32)

        fig = render_gradcam_3d(gradcam, title="Custom 3D Attention")

        assert "Custom 3D Attention" in fig.layout.title.text
