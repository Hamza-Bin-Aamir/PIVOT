"""2D overlay visualization for medical images with predictions and ground truth.

This module provides functionality to create overlays of:
- Detection centers (points)
- Segmentation masks (contours and filled regions)
- Bounding boxes
- Confidence values

Supports multiple color schemes and transparency levels for different visualization needs.
"""

from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from numpy.typing import NDArray


class ColorMap(str, Enum):
    """Predefined color maps for visualization."""

    PREDICTION = "prediction"  # Red for predictions
    GROUND_TRUTH = "ground_truth"  # Green for ground truth
    OVERLAY = "overlay"  # Red + Green = Yellow overlap
    HEAT = "heat"  # Hot colormap for confidence/probability
    COOL = "cool"  # Cool colormap for alternative view


@dataclass
class OverlayConfig:
    """Configuration for overlay visualization.

    Attributes:
        alpha: Transparency level for overlays (0.0 = transparent, 1.0 = opaque).
        linewidth: Width of contour lines in pixels.
        markersize: Size of detection center markers in points.
        show_confidence: Whether to display confidence scores as text.
        confidence_decimals: Number of decimal places for confidence values.
        contour_only: If True, only show mask contours; if False, show filled masks.
        figsize: Figure size in inches (width, height).
        dpi: Dots per inch for figure resolution.
    """

    alpha: float = 0.3
    linewidth: float = 2.0
    markersize: float = 8.0
    show_confidence: bool = True
    confidence_decimals: int = 2
    contour_only: bool = False
    figsize: tuple[float, float] = (10.0, 10.0)
    dpi: int = 100

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.linewidth <= 0:
            raise ValueError(f"linewidth must be > 0, got {self.linewidth}")
        if self.markersize <= 0:
            raise ValueError(f"markersize must be > 0, got {self.markersize}")
        if self.confidence_decimals < 0:
            raise ValueError(
                f"confidence_decimals must be >= 0, got {self.confidence_decimals}"
            )
        if self.dpi <= 0:
            raise ValueError(f"dpi must be > 0, got {self.dpi}")


def _get_color_values(colormap: ColorMap) -> tuple[float, float, float]:
    """Get RGB values for a specific color map.

    Args:
        colormap: Color map identifier.

    Returns:
        RGB tuple with values in [0, 1].
    """
    color_mapping = {
        ColorMap.PREDICTION: (1.0, 0.0, 0.0),  # Red
        ColorMap.GROUND_TRUTH: (0.0, 1.0, 0.0),  # Green
        ColorMap.OVERLAY: (1.0, 1.0, 0.0),  # Yellow
        ColorMap.HEAT: (1.0, 0.0, 0.0),  # Red (base for hot colormap)
        ColorMap.COOL: (0.0, 0.0, 1.0),  # Blue (base for cool colormap)
    }
    return color_mapping[colormap]


def create_2d_overlay(
    image: NDArray[np.float32],
    prediction_mask: NDArray[np.bool_] | None = None,
    ground_truth_mask: NDArray[np.bool_] | None = None,
    config: OverlayConfig | None = None,
) -> Figure:
    """Create a 2D overlay visualization with optional masks.

    Args:
        image: 2D grayscale image, shape (H, W).
        prediction_mask: Optional predicted segmentation mask, shape (H, W).
        ground_truth_mask: Optional ground truth segmentation mask, shape (H, W).
        config: Visualization configuration. Uses defaults if None.

    Returns:
        Matplotlib Figure object with the overlay.

    Raises:
        ValueError: If image is not 2D or masks have incompatible shapes.

    Example:
        >>> image = np.random.rand(256, 256).astype(np.float32)
        >>> pred = np.zeros((256, 256), dtype=bool)
        >>> pred[100:150, 100:150] = True
        >>> fig = create_2d_overlay(image, prediction_mask=pred)
        >>> plt.show()
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")

    if config is None:
        config = OverlayConfig()

    # Validate mask shapes
    if prediction_mask is not None and prediction_mask.shape != image.shape:
        raise ValueError(
            f"prediction_mask shape {prediction_mask.shape} != image shape {image.shape}"
        )
    if ground_truth_mask is not None and ground_truth_mask.shape != image.shape:
        raise ValueError(
            f"ground_truth_mask shape {ground_truth_mask.shape} != image shape {image.shape}"
        )

    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Display base image
    ax.imshow(image, cmap="gray", interpolation="nearest")

    # Overlay ground truth mask (green)
    if ground_truth_mask is not None:
        _overlay_single_mask(
            ax, ground_truth_mask, ColorMap.GROUND_TRUTH, config, label="Ground Truth"
        )

    # Overlay prediction mask (red)
    if prediction_mask is not None:
        _overlay_single_mask(
            ax, prediction_mask, ColorMap.PREDICTION, config, label="Prediction"
        )

    ax.set_title("2D Overlay Visualization")
    ax.axis("off")

    # Add legend if masks are present
    if prediction_mask is not None or ground_truth_mask is not None:
        ax.legend(loc="upper right")

    plt.tight_layout()
    return fig


def _overlay_single_mask(
    ax: Axes,
    mask: NDArray[np.bool_],
    colormap: ColorMap,
    config: OverlayConfig,
    label: str,
) -> None:
    """Overlay a single mask on an axis.

    Args:
        ax: Matplotlib axis to draw on.
        mask: Binary mask to overlay, shape (H, W).
        colormap: Color scheme for the mask.
        config: Visualization configuration.
        label: Label for the legend.
    """
    color = _get_color_values(colormap)

    if config.contour_only:
        # Draw contours only
        ax.contour(
            mask.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=config.linewidth,
            label=label,
        )
    else:
        # Draw filled mask with transparency
        masked_array = np.ma.masked_where(~mask, mask)
        cmap = ListedColormap([[0, 0, 0, 0], list(color) + [config.alpha]])  # type: ignore[arg-type]
        ax.imshow(masked_array, cmap=cmap, interpolation="nearest", label=label)

        # Also draw contours for clarity
        ax.contour(
            mask.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=config.linewidth,
        )


def overlay_masks(
    image: NDArray[np.float32],
    masks: dict[str, NDArray[np.bool_]],
    config: OverlayConfig | None = None,
) -> Figure:
    """Create overlay with multiple labeled masks.

    Args:
        image: 2D grayscale image, shape (H, W).
        masks: Dictionary mapping mask names to binary masks.
        config: Visualization configuration. Uses defaults if None.

    Returns:
        Matplotlib Figure with all masks overlaid.

    Raises:
        ValueError: If image is not 2D or any mask has incompatible shape.

    Example:
        >>> image = np.random.rand(256, 256).astype(np.float32)
        >>> masks = {
        ...     "prediction": np.zeros((256, 256), dtype=bool),
        ...     "ground_truth": np.zeros((256, 256), dtype=bool),
        ... }
        >>> masks["prediction"][50:100, 50:100] = True
        >>> masks["ground_truth"][80:130, 80:130] = True
        >>> fig = overlay_masks(image, masks)
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")

    if config is None:
        config = OverlayConfig()

    # Validate mask shapes
    for name, mask in masks.items():
        if mask.shape != image.shape:
            raise ValueError(f"mask '{name}' shape {mask.shape} != image shape {image.shape}")

    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    ax.imshow(image, cmap="gray", interpolation="nearest")

    # Overlay each mask with different colors
    colors = [
        ColorMap.PREDICTION,
        ColorMap.GROUND_TRUTH,
        ColorMap.OVERLAY,
        ColorMap.HEAT,
        ColorMap.COOL,
    ]

    for idx, (name, mask) in enumerate(masks.items()):
        color = colors[idx % len(colors)]
        _overlay_single_mask(ax, mask, color, config, label=name)

    ax.set_title("Multi-Mask Overlay")
    ax.axis("off")
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig


def overlay_detections(
    image: NDArray[np.float32],
    centers: NDArray[np.float64],
    confidences: NDArray[np.float64] | None = None,
    ground_truth_centers: NDArray[np.float64] | None = None,
    config: OverlayConfig | None = None,
) -> Figure:
    """Create overlay with detection centers and optional confidence scores.

    Args:
        image: 2D grayscale image, shape (H, W).
        centers: Detection center coordinates, shape (N, 2) as (row, col).
        confidences: Optional confidence scores, shape (N,).
        ground_truth_centers: Optional ground truth centers, shape (M, 2) as (row, col).
        config: Visualization configuration. Uses defaults if None.

    Returns:
        Matplotlib Figure with detection points overlaid.

    Raises:
        ValueError: If inputs have invalid shapes.

    Example:
        >>> image = np.random.rand(256, 256).astype(np.float32)
        >>> centers = np.array([[100, 100], [150, 150]], dtype=np.float64)
        >>> confidences = np.array([0.95, 0.87], dtype=np.float64)
        >>> fig = overlay_detections(image, centers, confidences)
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"centers must have shape (N, 2), got {centers.shape}")

    if confidences is not None and confidences.ndim != 1:
        raise ValueError(f"confidences must be 1D, got shape {confidences.shape}")

    if confidences is not None and len(confidences) != len(centers):
        raise ValueError(
            f"confidences length {len(confidences)} != centers length {len(centers)}"
        )

    if ground_truth_centers is not None and (
        ground_truth_centers.ndim != 2 or ground_truth_centers.shape[1] != 2
    ):
        raise ValueError(
            f"ground_truth_centers must have shape (M, 2), got {ground_truth_centers.shape}"
        )

    if config is None:
        config = OverlayConfig()

    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    ax.imshow(image, cmap="gray", interpolation="nearest")

    # Plot ground truth centers (green)
    if ground_truth_centers is not None and len(ground_truth_centers) > 0:
        gt_color = _get_color_values(ColorMap.GROUND_TRUTH)
        ax.scatter(
            ground_truth_centers[:, 1],  # col -> x
            ground_truth_centers[:, 0],  # row -> y
            c=[gt_color],
            s=config.markersize**2,
            marker="o",
            edgecolors="white",
            linewidths=1.5,
            label="Ground Truth",
        )

    # Plot prediction centers (red)
    if len(centers) > 0:
        pred_color = _get_color_values(ColorMap.PREDICTION)
        ax.scatter(
            centers[:, 1],  # col -> x
            centers[:, 0],  # row -> y
            c=[pred_color],
            s=config.markersize**2,
            marker="x",
            linewidths=config.linewidth,
            label="Prediction",
        )

        # Add confidence text labels
        if config.show_confidence and confidences is not None:
            for _idx, (center, conf) in enumerate(zip(centers, confidences, strict=True)):
                text = f"{conf:.{config.confidence_decimals}f}"
                ax.text(
                    center[1],
                    center[0] - 5,
                    text,
                    color="red",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
                )

    ax.set_title("Detection Overlay")
    ax.axis("off")

    # Add legend if detections present
    if len(centers) > 0 or (ground_truth_centers is not None and len(ground_truth_centers) > 0):
        ax.legend(loc="upper right")

    plt.tight_layout()
    return fig
