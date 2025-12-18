"""Confidence and uncertainty visualization for model predictions.

This module provides tools to visualize prediction confidence, uncertainty,
and calibration for nodule detection and classification tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray


@dataclass
class ConfidenceConfig:
    """Configuration for confidence visualization.

    Attributes:
        low_conf_threshold: Threshold below which confidence is low [0, 1]
        high_conf_threshold: Threshold above which confidence is high [0, 1]
        colorscale: Plotly colorscale for confidence heatmaps
        show_colorbar: Whether to display colorbar
    """

    low_conf_threshold: float = 0.3
    high_conf_threshold: float = 0.7
    colorscale: str = "RdYlGn"
    show_colorbar: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.low_conf_threshold <= 1:
            msg = f"low_conf_threshold must be in [0, 1], got {self.low_conf_threshold}"
            raise ValueError(msg)
        if not 0 <= self.high_conf_threshold <= 1:
            msg = f"high_conf_threshold must be in [0, 1], got {self.high_conf_threshold}"
            raise ValueError(msg)
        if self.low_conf_threshold >= self.high_conf_threshold:
            msg = "low_conf_threshold must be < high_conf_threshold"
            raise ValueError(msg)


def visualize_confidence_map(
    confidence: NDArray[np.float32],
    slice_idx: int,
    axis: int = 0,
    config: ConfidenceConfig | None = None,
    title: str = "Confidence Map",
) -> go.Figure:
    """Visualize confidence scores as a 2D heatmap on a slice.

    Args:
        confidence: Confidence scores [D, H, W] with values in [0, 1]
        slice_idx: Index of slice to visualize
        axis: Axis along which to slice (0=axial, 1=coronal, 2=sagittal)
        config: Visualization configuration
        title: Plot title

    Returns:
        Plotly figure with confidence heatmap

    Raises:
        ValueError: If parameters are invalid
    """
    if confidence.ndim != 3:
        msg = f"Expected 3D confidence map, got {confidence.ndim}D"
        raise ValueError(msg)
    if not 0 <= axis <= 2:
        msg = f"Axis must be 0, 1, or 2, got {axis}"
        raise ValueError(msg)
    if not 0 <= slice_idx < confidence.shape[axis]:
        msg = f"Slice index {slice_idx} out of range [0, {confidence.shape[axis]})"
        raise ValueError(msg)

    if config is None:
        config = ConfidenceConfig()

    # Extract slice
    if axis == 0:
        conf_slice = confidence[slice_idx, :, :]
        axis_name = "Axial"
    elif axis == 1:
        conf_slice = confidence[:, slice_idx, :]
        axis_name = "Coronal"
    else:
        conf_slice = confidence[:, :, slice_idx]
        axis_name = "Sagittal"

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=conf_slice,
            colorscale=config.colorscale,
            zmin=0,
            zmax=1,
            showscale=config.show_colorbar,
            colorbar={"title": "Confidence"} if config.show_colorbar else None,
            hovertemplate="Confidence: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{title} ({axis_name} Slice {slice_idx})",
        xaxis={"title": "X", "showgrid": False},
        yaxis={"title": "Y", "showgrid": False, "scaleanchor": "x"},
        width=600,
        height=600,
    )

    return fig


def visualize_uncertainty_regions(
    confidence: NDArray[np.float32],
    slice_idx: int,
    axis: int = 0,
    config: ConfidenceConfig | None = None,
    title: str = "Uncertainty Regions",
) -> go.Figure:
    """Visualize high/medium/low confidence regions on a slice.

    Args:
        confidence: Confidence scores [D, H, W] with values in [0, 1]
        slice_idx: Index of slice to visualize
        axis: Axis along which to slice (0=axial, 1=coronal, 2=sagittal)
        config: Visualization configuration
        title: Plot title

    Returns:
        Plotly figure with uncertainty regions

    Raises:
        ValueError: If parameters are invalid
    """
    if confidence.ndim != 3:
        msg = f"Expected 3D confidence map, got {confidence.ndim}D"
        raise ValueError(msg)
    if not 0 <= axis <= 2:
        msg = f"Axis must be 0, 1, or 2, got {axis}"
        raise ValueError(msg)
    if not 0 <= slice_idx < confidence.shape[axis]:
        msg = f"Slice index {slice_idx} out of range [0, {confidence.shape[axis]})"
        raise ValueError(msg)

    if config is None:
        config = ConfidenceConfig()

    # Extract slice
    if axis == 0:
        conf_slice = confidence[slice_idx, :, :]
        axis_name = "Axial"
    elif axis == 1:
        conf_slice = confidence[:, slice_idx, :]
        axis_name = "Coronal"
    else:
        conf_slice = confidence[:, :, slice_idx]
        axis_name = "Sagittal"

    # Classify regions
    regions = np.zeros_like(conf_slice)
    regions[conf_slice < config.low_conf_threshold] = 0  # Low confidence (red)
    regions[
        (conf_slice >= config.low_conf_threshold)
        & (conf_slice < config.high_conf_threshold)
    ] = 1  # Medium (yellow)
    regions[conf_slice >= config.high_conf_threshold] = 2  # High (green)

    # Create categorical heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=regions,
            colorscale=[[0, "red"], [0.5, "yellow"], [1, "green"]],
            zmin=0,
            zmax=2,
            showscale=True,
            colorbar={
                "title": "Confidence",
                "tickvals": [0, 1, 2],
                "ticktext": ["Low", "Medium", "High"],
            },
            hovertemplate="Region: %{z}<br>Confidence: %{customdata:.3f}<extra></extra>",
            customdata=conf_slice,
        )
    )

    fig.update_layout(
        title=f"{title} ({axis_name} Slice {slice_idx})",
        xaxis={"title": "X", "showgrid": False},
        yaxis={"title": "Y", "showgrid": False, "scaleanchor": "x"},
        width=600,
        height=600,
    )

    return fig


def plot_calibration_curve(
    predictions: NDArray[np.float32],
    targets: NDArray[np.float32],
    n_bins: int = 10,
    title: str = "Calibration Curve",
) -> plt.Figure:
    """Plot calibration curve showing predicted vs actual probabilities.

    Args:
        predictions: Predicted probabilities [N,] in [0, 1]
        targets: Binary ground truth labels [N,] in {0, 1}
        n_bins: Number of bins for grouping predictions
        title: Plot title

    Returns:
        Matplotlib figure with calibration curve

    Raises:
        ValueError: If inputs are invalid
    """
    if predictions.ndim != 1:
        msg = f"Predictions must be 1D, got {predictions.ndim}D"
        raise ValueError(msg)
    if targets.ndim != 1:
        msg = f"Targets must be 1D, got {targets.ndim}D"
        raise ValueError(msg)
    if len(predictions) != len(targets):
        msg = f"Length mismatch: {len(predictions)} vs {len(targets)}"
        raise ValueError(msg)
    if n_bins < 2:
        msg = f"n_bins must be >= 2, got {n_bins}"
        raise ValueError(msg)

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Compute fraction of positives in each bin
    bin_true_probs = np.zeros(n_bins)
    bin_pred_probs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])

        if mask.sum() > 0:
            bin_true_probs[i] = targets[mask].mean()
            bin_pred_probs[i] = predictions[mask].mean()
            bin_counts[i] = mask.sum()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)
    ax1.plot(
        bin_pred_probs,
        bin_true_probs,
        "o-",
        label="Model calibration",
        linewidth=2,
        markersize=8,
    )
    ax1.set_xlabel("Predicted Probability", fontsize=12)
    ax1.set_ylabel("Actual Probability", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Plot histogram of predictions
    ax2.hist(predictions, bins=n_bins, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Predicted Probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Prediction Distribution", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_confidence_histogram(
    confidence: NDArray[np.float32],
    config: ConfidenceConfig | None = None,
    title: str = "Confidence Distribution",
) -> go.Figure:
    """Plot histogram of confidence scores across volume.

    Args:
        confidence: Confidence scores [D, H, W] or [N,]
        config: Visualization configuration for threshold lines
        title: Plot title

    Returns:
        Plotly figure with confidence histogram

    Raises:
        ValueError: If confidence has invalid shape
    """
    if confidence.ndim not in (1, 3):
        msg = f"Confidence must be 1D or 3D, got {confidence.ndim}D"
        raise ValueError(msg)

    if config is None:
        config = ConfidenceConfig()

    # Flatten if 3D
    conf_flat = confidence.flatten()

    # Create histogram
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=conf_flat,
            nbinsx=50,
            marker={"color": "steelblue", "line": {"color": "black", "width": 1}},
            name="Confidence",
        )
    )

    # Add threshold lines
    fig.add_vline(
        x=config.low_conf_threshold,
        line={"color": "red", "width": 2, "dash": "dash"},
        annotation_text="Low Threshold",
    )
    fig.add_vline(
        x=config.high_conf_threshold,
        line={"color": "green", "width": 2, "dash": "dash"},
        annotation_text="High Threshold",
    )

    fig.update_layout(
        title=title,
        xaxis={"title": "Confidence Score", "range": [0, 1]},
        yaxis={"title": "Count"},
        showlegend=False,
        width=800,
        height=500,
    )

    return fig
