"""Tests for 2D overlay visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.viz.overlay import (
    ColorMap,
    OverlayConfig,
    create_2d_overlay,
    overlay_detections,
    overlay_masks,
)


class TestOverlayConfig:
    """Test OverlayConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OverlayConfig()
        assert config.alpha == 0.3
        assert config.linewidth == 2.0
        assert config.markersize == 8.0
        assert config.show_confidence is True
        assert config.confidence_decimals == 2
        assert config.contour_only is False
        assert config.figsize == (10.0, 10.0)
        assert config.dpi == 100

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OverlayConfig(
            alpha=0.5,
            linewidth=3.0,
            markersize=10.0,
            show_confidence=False,
            confidence_decimals=3,
            contour_only=True,
            figsize=(12.0, 12.0),
            dpi=150,
        )
        assert config.alpha == 0.5
        assert config.linewidth == 3.0
        assert config.markersize == 10.0
        assert config.show_confidence is False
        assert config.confidence_decimals == 3
        assert config.contour_only is True
        assert config.figsize == (12.0, 12.0)
        assert config.dpi == 150

    def test_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            OverlayConfig(alpha=-0.1)
        with pytest.raises(ValueError, match="alpha must be in"):
            OverlayConfig(alpha=1.5)

    def test_invalid_linewidth(self):
        """Test that invalid linewidth raises ValueError."""
        with pytest.raises(ValueError, match="linewidth must be"):
            OverlayConfig(linewidth=0)
        with pytest.raises(ValueError, match="linewidth must be"):
            OverlayConfig(linewidth=-1)

    def test_invalid_markersize(self):
        """Test that invalid markersize raises ValueError."""
        with pytest.raises(ValueError, match="markersize must be"):
            OverlayConfig(markersize=0)

    def test_invalid_confidence_decimals(self):
        """Test that invalid confidence_decimals raises ValueError."""
        with pytest.raises(ValueError, match="confidence_decimals must be"):
            OverlayConfig(confidence_decimals=-1)

    def test_invalid_dpi(self):
        """Test that invalid dpi raises ValueError."""
        with pytest.raises(ValueError, match="dpi must be"):
            OverlayConfig(dpi=0)


class TestCreate2DOverlay:
    """Test create_2d_overlay function."""

    def test_basic_image_only(self):
        """Test overlay with only base image."""
        image = np.random.rand(256, 256).astype(np.float32)
        fig = create_2d_overlay(image)
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_with_prediction_mask(self):
        """Test overlay with prediction mask."""
        image = np.random.rand(256, 256).astype(np.float32)
        pred_mask = np.zeros((256, 256), dtype=bool)
        pred_mask[100:150, 100:150] = True

        fig = create_2d_overlay(image, prediction_mask=pred_mask)
        assert fig is not None
        plt.close(fig)

    def test_with_ground_truth_mask(self):
        """Test overlay with ground truth mask."""
        image = np.random.rand(256, 256).astype(np.float32)
        gt_mask = np.zeros((256, 256), dtype=bool)
        gt_mask[80:130, 80:130] = True

        fig = create_2d_overlay(image, ground_truth_mask=gt_mask)
        assert fig is not None
        plt.close(fig)

    def test_with_both_masks(self):
        """Test overlay with both prediction and ground truth masks."""
        image = np.random.rand(256, 256).astype(np.float32)
        pred_mask = np.zeros((256, 256), dtype=bool)
        pred_mask[100:150, 100:150] = True
        gt_mask = np.zeros((256, 256), dtype=bool)
        gt_mask[80:130, 80:130] = True

        fig = create_2d_overlay(image, prediction_mask=pred_mask, ground_truth_mask=gt_mask)
        assert fig is not None
        plt.close(fig)

    def test_contour_only_mode(self):
        """Test contour-only visualization mode."""
        image = np.random.rand(256, 256).astype(np.float32)
        mask = np.zeros((256, 256), dtype=bool)
        mask[100:150, 100:150] = True
        config = OverlayConfig(contour_only=True)

        fig = create_2d_overlay(image, prediction_mask=mask, config=config)
        assert fig is not None
        plt.close(fig)

    def test_custom_config(self):
        """Test overlay with custom configuration."""
        image = np.random.rand(256, 256).astype(np.float32)
        mask = np.zeros((256, 256), dtype=bool)
        mask[100:150, 100:150] = True
        config = OverlayConfig(alpha=0.5, linewidth=3.0, figsize=(8.0, 8.0))

        fig = create_2d_overlay(image, prediction_mask=mask, config=config)
        assert fig is not None
        plt.close(fig)

    def test_invalid_image_shape(self):
        """Test that non-2D image raises ValueError."""
        image_3d = np.random.rand(10, 256, 256).astype(np.float32)
        with pytest.raises(ValueError, match="image must be 2D"):
            create_2d_overlay(image_3d)

    def test_mismatched_prediction_mask_shape(self):
        """Test that mismatched prediction mask shape raises ValueError."""
        image = np.random.rand(256, 256).astype(np.float32)
        mask = np.zeros((128, 128), dtype=bool)
        with pytest.raises(ValueError, match="prediction_mask shape"):
            create_2d_overlay(image, prediction_mask=mask)

    def test_mismatched_ground_truth_mask_shape(self):
        """Test that mismatched ground truth mask shape raises ValueError."""
        image = np.random.rand(256, 256).astype(np.float32)
        mask = np.zeros((128, 128), dtype=bool)
        with pytest.raises(ValueError, match="ground_truth_mask shape"):
            create_2d_overlay(image, ground_truth_mask=mask)


class TestOverlayMasks:
    """Test overlay_masks function."""

    def test_single_mask(self):
        """Test overlay with single named mask."""
        image = np.random.rand(256, 256).astype(np.float32)
        masks = {"prediction": np.zeros((256, 256), dtype=bool)}
        masks["prediction"][100:150, 100:150] = True

        fig = overlay_masks(image, masks)
        assert fig is not None
        plt.close(fig)

    def test_multiple_masks(self):
        """Test overlay with multiple named masks."""
        image = np.random.rand(256, 256).astype(np.float32)
        masks = {
            "prediction": np.zeros((256, 256), dtype=bool),
            "ground_truth": np.zeros((256, 256), dtype=bool),
            "alternative": np.zeros((256, 256), dtype=bool),
        }
        masks["prediction"][50:100, 50:100] = True
        masks["ground_truth"][80:130, 80:130] = True
        masks["alternative"][120:170, 120:170] = True

        fig = overlay_masks(image, masks)
        assert fig is not None
        plt.close(fig)

    def test_empty_masks_dict(self):
        """Test overlay with empty masks dictionary."""
        image = np.random.rand(256, 256).astype(np.float32)
        masks = {}

        fig = overlay_masks(image, masks)
        assert fig is not None
        plt.close(fig)

    def test_invalid_image_shape(self):
        """Test that non-2D image raises ValueError."""
        image_3d = np.random.rand(10, 256, 256).astype(np.float32)
        masks = {"test": np.zeros((256, 256), dtype=bool)}
        with pytest.raises(ValueError, match="image must be 2D"):
            overlay_masks(image_3d, masks)

    def test_mismatched_mask_shape(self):
        """Test that mismatched mask shape raises ValueError."""
        image = np.random.rand(256, 256).astype(np.float32)
        masks = {"test": np.zeros((128, 128), dtype=bool)}
        with pytest.raises(ValueError, match="mask 'test' shape"):
            overlay_masks(image, masks)


class TestOverlayDetections:
    """Test overlay_detections function."""

    def test_basic_detections(self):
        """Test overlay with basic detection centers."""
        image = np.random.rand(256, 256).astype(np.float32)
        centers = np.array([[100, 100], [150, 150]], dtype=np.float64)

        fig = overlay_detections(image, centers)
        assert fig is not None
        plt.close(fig)

    def test_detections_with_confidence(self):
        """Test overlay with detection centers and confidence scores."""
        image = np.random.rand(256, 256).astype(np.float32)
        centers = np.array([[100, 100], [150, 150]], dtype=np.float64)
        confidences = np.array([0.95, 0.87], dtype=np.float64)

        fig = overlay_detections(image, centers, confidences)
        assert fig is not None
        plt.close(fig)

    def test_detections_with_ground_truth(self):
        """Test overlay with both predictions and ground truth."""
        image = np.random.rand(256, 256).astype(np.float32)
        centers = np.array([[100, 100], [150, 150]], dtype=np.float64)
        gt_centers = np.array([[105, 105], [145, 145]], dtype=np.float64)

        fig = overlay_detections(image, centers, ground_truth_centers=gt_centers)
        assert fig is not None
        plt.close(fig)

    def test_detections_with_all_info(self):
        """Test overlay with detections, confidences, and ground truth."""
        image = np.random.rand(256, 256).astype(np.float32)
        centers = np.array([[100, 100], [150, 150]], dtype=np.float64)
        confidences = np.array([0.95, 0.87], dtype=np.float64)
        gt_centers = np.array([[105, 105], [145, 145]], dtype=np.float64)

        fig = overlay_detections(
            image, centers, confidences, ground_truth_centers=gt_centers
        )
        assert fig is not None
        plt.close(fig)

    def test_no_confidence_display(self):
        """Test overlay without showing confidence values."""
        image = np.random.rand(256, 256).astype(np.float32)
        centers = np.array([[100, 100]], dtype=np.float64)
        confidences = np.array([0.95], dtype=np.float64)
        config = OverlayConfig(show_confidence=False)

        fig = overlay_detections(image, centers, confidences, config=config)
        assert fig is not None
        plt.close(fig)

    def test_empty_detections(self):
        """Test overlay with no detection centers."""
        image = np.random.rand(256, 256).astype(np.float32)
        centers = np.zeros((0, 2), dtype=np.float64)

        fig = overlay_detections(image, centers)
        assert fig is not None
        plt.close(fig)

    def test_empty_ground_truth(self):
        """Test overlay with empty ground truth centers."""
        image = np.random.rand(256, 256).astype(np.float32)
        centers = np.array([[100, 100]], dtype=np.float64)
        gt_centers = np.zeros((0, 2), dtype=np.float64)

        fig = overlay_detections(image, centers, ground_truth_centers=gt_centers)
        assert fig is not None
        plt.close(fig)

    def test_invalid_image_shape(self):
        """Test that non-2D image raises ValueError."""
        image_3d = np.random.rand(10, 256, 256).astype(np.float32)
        centers = np.array([[100, 100]], dtype=np.float64)
        with pytest.raises(ValueError, match="image must be 2D"):
            overlay_detections(image_3d, centers)

    def test_invalid_centers_shape(self):
        """Test that invalid centers shape raises ValueError."""
        image = np.random.rand(256, 256).astype(np.float32)
        centers = np.array([100, 100], dtype=np.float64)  # 1D instead of 2D
        with pytest.raises(ValueError, match="centers must have shape"):
            overlay_detections(image, centers)

    def test_invalid_confidences_shape(self):
        """Test that invalid confidences shape raises ValueError."""
        image = np.random.rand(256, 256).astype(np.float32)
        centers = np.array([[100, 100]], dtype=np.float64)
        confidences = np.array([[0.95]], dtype=np.float64)  # 2D instead of 1D
        with pytest.raises(ValueError, match="confidences must be 1D"):
            overlay_detections(image, centers, confidences)

    def test_mismatched_confidences_length(self):
        """Test that mismatched confidences length raises ValueError."""
        image = np.random.rand(256, 256).astype(np.float32)
        centers = np.array([[100, 100], [150, 150]], dtype=np.float64)
        confidences = np.array([0.95], dtype=np.float64)  # Wrong length
        with pytest.raises(ValueError, match="confidences length"):
            overlay_detections(image, centers, confidences)

    def test_invalid_ground_truth_centers_shape(self):
        """Test that invalid ground truth centers shape raises ValueError."""
        image = np.random.rand(256, 256).astype(np.float32)
        centers = np.array([[100, 100]], dtype=np.float64)
        gt_centers = np.array([100, 100], dtype=np.float64)  # 1D instead of 2D
        with pytest.raises(ValueError, match="ground_truth_centers must have shape"):
            overlay_detections(image, centers, ground_truth_centers=gt_centers)


class TestColorMap:
    """Test ColorMap enum."""

    def test_color_map_values(self):
        """Test that ColorMap has expected values."""
        assert ColorMap.PREDICTION == "prediction"
        assert ColorMap.GROUND_TRUTH == "ground_truth"
        assert ColorMap.OVERLAY == "overlay"
        assert ColorMap.HEAT == "heat"
        assert ColorMap.COOL == "cool"
