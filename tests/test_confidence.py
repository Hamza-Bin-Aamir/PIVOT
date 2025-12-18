"""Tests for confidence and uncertainty visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest

from src.viz.confidence import (
    ConfidenceConfig,
    plot_calibration_curve,
    plot_confidence_histogram,
    visualize_confidence_map,
    visualize_uncertainty_regions,
)


class TestConfidenceConfig:
    """Tests for ConfidenceConfig dataclass."""

    def test_default_config(self):
        """Test ConfidenceConfig with default values."""
        config = ConfidenceConfig()
        assert config.low_conf_threshold == 0.3
        assert config.high_conf_threshold == 0.7
        assert config.colorscale == "RdYlGn"
        assert config.show_colorbar is True

    def test_custom_config(self):
        """Test ConfidenceConfig with custom values."""
        config = ConfidenceConfig(
            low_conf_threshold=0.2,
            high_conf_threshold=0.8,
            colorscale="Viridis",
            show_colorbar=False,
        )
        assert config.low_conf_threshold == 0.2
        assert config.high_conf_threshold == 0.8
        assert config.colorscale == "Viridis"
        assert config.show_colorbar is False

    def test_invalid_low_threshold(self):
        """Test ConfidenceConfig rejects invalid low threshold."""
        with pytest.raises(ValueError, match="low_conf_threshold must be in"):
            ConfidenceConfig(low_conf_threshold=1.5)
        with pytest.raises(ValueError, match="low_conf_threshold must be in"):
            ConfidenceConfig(low_conf_threshold=-0.1)

    def test_invalid_high_threshold(self):
        """Test ConfidenceConfig rejects invalid high threshold."""
        with pytest.raises(ValueError, match="high_conf_threshold must be in"):
            ConfidenceConfig(high_conf_threshold=1.5)
        with pytest.raises(ValueError, match="high_conf_threshold must be in"):
            ConfidenceConfig(high_conf_threshold=-0.1)

    def test_threshold_ordering(self):
        """Test ConfidenceConfig enforces low < high threshold."""
        with pytest.raises(ValueError, match="low_conf_threshold must be <"):
            ConfidenceConfig(low_conf_threshold=0.8, high_conf_threshold=0.3)


class TestVisualizeConfidenceMap:
    """Tests for visualize_confidence_map function."""

    def test_basic_confidence_map_axial(self):
        """Test basic confidence map visualization on axial slice."""
        confidence = np.random.rand(32, 64, 64).astype(np.float32)

        fig = visualize_confidence_map(confidence, slice_idx=16, axis=0)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)
        assert "Axial" in fig.layout.title.text

    def test_confidence_map_coronal(self):
        """Test confidence map on coronal slice."""
        confidence = np.random.rand(32, 64, 64).astype(np.float32)

        fig = visualize_confidence_map(confidence, slice_idx=32, axis=1)

        assert isinstance(fig, go.Figure)
        assert "Coronal" in fig.layout.title.text

    def test_confidence_map_sagittal(self):
        """Test confidence map on sagittal slice."""
        confidence = np.random.rand(32, 64, 64).astype(np.float32)

        fig = visualize_confidence_map(confidence, slice_idx=32, axis=2)

        assert isinstance(fig, go.Figure)
        assert "Sagittal" in fig.layout.title.text

    def test_confidence_map_with_custom_config(self):
        """Test confidence map with custom configuration."""
        confidence = np.random.rand(32, 64, 64).astype(np.float32)
        config = ConfidenceConfig(colorscale="Plasma")

        fig = visualize_confidence_map(
            confidence, slice_idx=16, axis=0, config=config
        )

        assert isinstance(fig, go.Figure)

    def test_confidence_map_invalid_dimensions(self):
        """Test confidence map rejects non-3D input."""
        confidence = np.random.rand(64, 64).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 3D confidence map"):
            visualize_confidence_map(confidence, slice_idx=32, axis=0)

    def test_confidence_map_invalid_axis(self):
        """Test confidence map rejects invalid axis."""
        confidence = np.random.rand(32, 64, 64).astype(np.float32)

        with pytest.raises(ValueError, match="Axis must be"):
            visualize_confidence_map(confidence, slice_idx=16, axis=3)

    def test_confidence_map_invalid_slice_index(self):
        """Test confidence map rejects out-of-range slice index."""
        confidence = np.random.rand(32, 64, 64).astype(np.float32)

        with pytest.raises(ValueError, match="Slice index .* out of range"):
            visualize_confidence_map(confidence, slice_idx=100, axis=0)

    def test_confidence_map_custom_title(self):
        """Test confidence map with custom title."""
        confidence = np.random.rand(32, 64, 64).astype(np.float32)

        fig = visualize_confidence_map(
            confidence, slice_idx=16, axis=0, title="Custom Confidence"
        )

        assert "Custom Confidence" in fig.layout.title.text


class TestVisualizeUncertaintyRegions:
    """Tests for visualize_uncertainty_regions function."""

    def test_basic_uncertainty_regions(self):
        """Test basic uncertainty region visualization."""
        confidence = np.random.rand(32, 64, 64).astype(np.float32)

        fig = visualize_uncertainty_regions(confidence, slice_idx=16, axis=0)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_uncertainty_regions_classification(self):
        """Test uncertainty regions classify confidence correctly."""
        # Create confidence map with known values
        confidence = np.ones((10, 10, 10), dtype=np.float32) * 0.5
        confidence[:3, :, :] = 0.2  # Low confidence
        confidence[7:, :, :] = 0.9  # High confidence

        fig = visualize_uncertainty_regions(
            confidence, slice_idx=5, axis=0
        )

        assert isinstance(fig, go.Figure)

    def test_uncertainty_regions_with_custom_config(self):
        """Test uncertainty regions with custom thresholds."""
        confidence = np.random.rand(32, 64, 64).astype(np.float32)
        config = ConfidenceConfig(low_conf_threshold=0.4, high_conf_threshold=0.6)

        fig = visualize_uncertainty_regions(
            confidence, slice_idx=16, axis=0, config=config
        )

        assert isinstance(fig, go.Figure)

    def test_uncertainty_regions_coronal(self):
        """Test uncertainty regions on coronal slice."""
        confidence = np.random.rand(32, 64, 64).astype(np.float32)

        fig = visualize_uncertainty_regions(confidence, slice_idx=32, axis=1)

        assert isinstance(fig, go.Figure)
        assert "Coronal" in fig.layout.title.text

    def test_uncertainty_regions_invalid_dimensions(self):
        """Test uncertainty regions rejects non-3D input."""
        confidence = np.random.rand(64, 64).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 3D confidence map"):
            visualize_uncertainty_regions(confidence, slice_idx=32, axis=0)

    def test_uncertainty_regions_invalid_axis(self):
        """Test uncertainty regions rejects invalid axis."""
        confidence = np.random.rand(32, 64, 64).astype(np.float32)

        with pytest.raises(ValueError, match="Axis must be"):
            visualize_uncertainty_regions(confidence, slice_idx=16, axis=3)

    def test_uncertainty_regions_custom_title(self):
        """Test uncertainty regions with custom title."""
        confidence = np.random.rand(32, 64, 64).astype(np.float32)

        fig = visualize_uncertainty_regions(
            confidence, slice_idx=16, axis=0, title="Custom Uncertainty"
        )

        assert "Custom Uncertainty" in fig.layout.title.text


class TestPlotCalibrationCurve:
    """Tests for plot_calibration_curve function."""

    def test_basic_calibration_curve(self):
        """Test basic calibration curve plotting."""
        predictions = np.random.rand(1000).astype(np.float32)
        targets = np.random.randint(0, 2, 1000).astype(np.float32)

        fig = plot_calibration_curve(predictions, targets)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_calibration_curve_perfect_calibration(self):
        """Test calibration curve with perfectly calibrated predictions."""
        # Create perfectly calibrated predictions
        targets = np.random.randint(0, 2, 1000).astype(np.float32)
        predictions = targets.copy()

        fig = plot_calibration_curve(predictions, targets, n_bins=5)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_calibration_curve_custom_bins(self):
        """Test calibration curve with custom number of bins."""
        predictions = np.random.rand(1000).astype(np.float32)
        targets = np.random.randint(0, 2, 1000).astype(np.float32)

        fig = plot_calibration_curve(predictions, targets, n_bins=20)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_calibration_curve_custom_title(self):
        """Test calibration curve with custom title."""
        predictions = np.random.rand(1000).astype(np.float32)
        targets = np.random.randint(0, 2, 1000).astype(np.float32)

        fig = plot_calibration_curve(
            predictions, targets, title="Custom Calibration"
        )

        assert isinstance(fig, plt.Figure)
        assert "Custom Calibration" in fig.axes[0].get_title()
        plt.close(fig)

    def test_calibration_curve_invalid_predictions_dim(self):
        """Test calibration curve rejects non-1D predictions."""
        predictions = np.random.rand(100, 10).astype(np.float32)
        targets = np.random.randint(0, 2, 100).astype(np.float32)

        with pytest.raises(ValueError, match="Predictions must be 1D"):
            plot_calibration_curve(predictions, targets)

    def test_calibration_curve_invalid_targets_dim(self):
        """Test calibration curve rejects non-1D targets."""
        predictions = np.random.rand(100).astype(np.float32)
        targets = np.random.randint(0, 2, (100, 10)).astype(np.float32)

        with pytest.raises(ValueError, match="Targets must be 1D"):
            plot_calibration_curve(predictions, targets)

    def test_calibration_curve_length_mismatch(self):
        """Test calibration curve rejects mismatched lengths."""
        predictions = np.random.rand(100).astype(np.float32)
        targets = np.random.randint(0, 2, 50).astype(np.float32)

        with pytest.raises(ValueError, match="Length mismatch"):
            plot_calibration_curve(predictions, targets)

    def test_calibration_curve_invalid_bins(self):
        """Test calibration curve rejects invalid bin count."""
        predictions = np.random.rand(100).astype(np.float32)
        targets = np.random.randint(0, 2, 100).astype(np.float32)

        with pytest.raises(ValueError, match="n_bins must be >= 2"):
            plot_calibration_curve(predictions, targets, n_bins=1)


class TestPlotConfidenceHistogram:
    """Tests for plot_confidence_histogram function."""

    def test_basic_histogram_3d(self):
        """Test basic confidence histogram with 3D input."""
        confidence = np.random.rand(32, 64, 64).astype(np.float32)

        fig = plot_confidence_histogram(confidence)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_histogram_1d_input(self):
        """Test confidence histogram with 1D input."""
        confidence = np.random.rand(1000).astype(np.float32)

        fig = plot_confidence_histogram(confidence)

        assert isinstance(fig, go.Figure)

    def test_histogram_with_custom_config(self):
        """Test histogram with custom threshold configuration."""
        confidence = np.random.rand(1000).astype(np.float32)
        config = ConfidenceConfig(low_conf_threshold=0.2, high_conf_threshold=0.8)

        fig = plot_confidence_histogram(confidence, config=config)

        assert isinstance(fig, go.Figure)

    def test_histogram_custom_title(self):
        """Test histogram with custom title."""
        confidence = np.random.rand(1000).astype(np.float32)

        fig = plot_confidence_histogram(
            confidence, title="Custom Confidence Distribution"
        )

        assert "Custom Confidence Distribution" in fig.layout.title.text

    def test_histogram_invalid_dimensions(self):
        """Test histogram rejects invalid dimensions."""
        confidence = np.random.rand(10, 10).astype(np.float32)

        with pytest.raises(ValueError, match="Confidence must be 1D or 3D"):
            plot_confidence_histogram(confidence)
