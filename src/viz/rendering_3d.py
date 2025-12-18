"""3D rendering for lung volumes and nodules.

This module provides functionality to create 3D visualizations of CT lung scans:
- Surface rendering of lung segmentation masks
- Volume rendering with opacity control
- Nodule highlighting and annotation
- Interactive camera controls
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from skimage import measure


@dataclass
class RenderingConfig:
    """Configuration for 3D rendering.

    Attributes:
        colorscale: Plotly colorscale name for surface/volume.
        opacity: Opacity level for surface rendering (0.0-1.0).
        show_axes: Whether to show axis labels and grid.
        camera_eye: Camera position as (x, y, z) tuple.
        lighting: Lighting configuration dict for surface plots.
        surface_count: Number of isosurface levels for volume rendering.
    """

    colorscale: str = "gray"
    opacity: float = 0.3
    show_axes: bool = True
    camera_eye: tuple[float, float, float] = (1.5, 1.5, 1.5)
    lighting: dict[str, Any] | None = None
    surface_count: int = 10

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"opacity must be in [0, 1], got {self.opacity}")
        if self.surface_count <= 0:
            raise ValueError(f"surface_count must be > 0, got {self.surface_count}")

        # Set default lighting if not provided
        if self.lighting is None:
            self.lighting = {
                "ambient": 0.5,
                "diffuse": 0.8,
                "specular": 0.3,
                "roughness": 0.5,
            }


def render_lung_surface(
    mask: NDArray[np.bool_],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    config: RenderingConfig | None = None,
    title: str = "Lung Surface Rendering",
) -> go.Figure:
    """Create a 3D surface rendering of a lung segmentation mask.

    Uses marching cubes algorithm to extract isosurface from binary mask.

    Args:
        mask: Binary segmentation mask, shape (D, H, W).
        spacing: Voxel spacing in mm (z, y, x).
        config: Rendering configuration. Uses defaults if None.
        title: Title for the plot.

    Returns:
        Plotly Figure object with interactive 3D surface.

    Raises:
        ValueError: If mask is not 3D or spacing is invalid.

    Example:
        >>> mask = np.zeros((100, 256, 256), dtype=bool)
        >>> mask[30:70, 80:180, 80:180] = True
        >>> fig = render_lung_surface(mask)
        >>> fig.show()
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {mask.shape}")

    if len(spacing) != 3:
        raise ValueError(f"spacing must have 3 values, got {len(spacing)}")

    if any(s <= 0 for s in spacing):
        raise ValueError(f"spacing values must be > 0, got {spacing}")

    if not np.any(mask):
        raise ValueError("mask is empty (all False)")

    if config is None:
        config = RenderingConfig()

    # Extract surface using marching cubes
    # Level 0.5 gives the boundary between True/False voxels
    verts, faces, normals, _ = measure.marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=spacing
    )

    # Create mesh3d trace
    trace = go.Mesh3d(
        x=verts[:, 2],  # x-axis (width)
        y=verts[:, 1],  # y-axis (height)
        z=verts[:, 0],  # z-axis (depth)
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=config.opacity,
        color="lightblue",
        flatshading=True,
        lighting=config.lighting,
        name="Lung Surface",
    )

    # Create figure
    fig = go.Figure(data=[trace])

    # Update layout
    fig.update_layout(
        title=title,
        scene={
            "xaxis": {"visible": config.show_axes, "title": "X (mm)"},
            "yaxis": {"visible": config.show_axes, "title": "Y (mm)"},
            "zaxis": {"visible": config.show_axes, "title": "Z (mm)"},
            "camera": {"eye": {"x": config.camera_eye[0], "y": config.camera_eye[1], "z": config.camera_eye[2]}},
            "aspectmode": "data",
        },
        showlegend=True,
    )

    return fig


def render_volume(
    volume: NDArray[np.float32],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    config: RenderingConfig | None = None,
    title: str = "3D Volume Rendering",
) -> go.Figure:
    """Create a 3D volume rendering with opacity-based visualization.

    Args:
        volume: 3D volume data, shape (D, H, W).
        spacing: Voxel spacing in mm (z, y, x).
        config: Rendering configuration. Uses defaults if None.
        title: Title for the plot.

    Returns:
        Plotly Figure object with interactive 3D volume.

    Raises:
        ValueError: If volume is not 3D or spacing is invalid.

    Example:
        >>> volume = np.random.rand(100, 128, 128).astype(np.float32)
        >>> fig = render_volume(volume)
        >>> fig.show()
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {volume.shape}")

    if len(spacing) != 3:
        raise ValueError(f"spacing must have 3 values, got {len(spacing)}")

    if any(s <= 0 for s in spacing):
        raise ValueError(f"spacing values must be > 0, got {spacing}")

    if config is None:
        config = RenderingConfig()

    # Normalize volume to [0, 1] for opacity mapping
    vol_min, vol_max = volume.min(), volume.max()
    volume_norm = (volume - vol_min) / (vol_max - vol_min) if vol_max > vol_min else volume.copy()

    # Create volume trace
    trace = go.Volume(
        x=np.arange(volume.shape[2]) * spacing[2],
        y=np.arange(volume.shape[1]) * spacing[1],
        z=np.arange(volume.shape[0]) * spacing[0],
        value=volume_norm.flatten(),
        opacity=config.opacity,
        surface_count=config.surface_count,
        colorscale=config.colorscale,
        name="Volume",
    )

    # Create figure
    fig = go.Figure(data=[trace])

    # Update layout
    fig.update_layout(
        title=title,
        scene={
            "xaxis": {"visible": config.show_axes, "title": "X (mm)"},
            "yaxis": {"visible": config.show_axes, "title": "Y (mm)"},
            "zaxis": {"visible": config.show_axes, "title": "Z (mm)"},
            "camera": {"eye": {"x": config.camera_eye[0], "y": config.camera_eye[1], "z": config.camera_eye[2]}},
            "aspectmode": "data",
        },
        showlegend=True,
    )

    return fig


def render_with_nodules(
    lung_mask: NDArray[np.bool_],
    nodule_centers: NDArray[np.float64],
    nodule_radii: NDArray[np.float64] | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    config: RenderingConfig | None = None,
    title: str = "Lung with Nodules",
) -> go.Figure:
    """Render lung surface with highlighted nodule locations.

    Args:
        lung_mask: Binary lung segmentation mask, shape (D, H, W).
        nodule_centers: Nodule center coordinates, shape (N, 3) as (z, y, x) in voxels.
        nodule_radii: Optional nodule radii in mm, shape (N,). If None, uses default marker size.
        spacing: Voxel spacing in mm (z, y, x).
        config: Rendering configuration. Uses defaults if None.
        title: Title for the plot.

    Returns:
        Plotly Figure with lung surface and nodule markers.

    Raises:
        ValueError: If inputs have invalid shapes or values.

    Example:
        >>> lung_mask = np.zeros((100, 256, 256), dtype=bool)
        >>> lung_mask[30:70, 80:180, 80:180] = True
        >>> centers = np.array([[50, 130, 130], [60, 140, 140]], dtype=np.float64)
        >>> fig = render_with_nodules(lung_mask, centers)
        >>> fig.show()
    """
    if lung_mask.ndim != 3:
        raise ValueError(f"lung_mask must be 3D, got shape {lung_mask.shape}")

    if nodule_centers.ndim != 2 or nodule_centers.shape[1] != 3:
        raise ValueError(f"nodule_centers must have shape (N, 3), got {nodule_centers.shape}")

    if nodule_radii is not None and nodule_radii.ndim != 1:
        raise ValueError(f"nodule_radii must be 1D, got shape {nodule_radii.shape}")

    if nodule_radii is not None and len(nodule_radii) != len(nodule_centers):
        raise ValueError(
            f"nodule_radii length {len(nodule_radii)} != nodule_centers length {len(nodule_centers)}"
        )

    if config is None:
        config = RenderingConfig()

    # Render lung surface
    fig = render_lung_surface(lung_mask, spacing=spacing, config=config, title=title)

    # Convert nodule centers from voxel to physical coordinates
    centers_mm = nodule_centers * np.array(spacing)

    # Add nodule markers
    if nodule_radii is not None:
        # Scale marker sizes based on radii
        marker_sizes = nodule_radii * 2  # Diameter for visual representation
    else:
        marker_sizes = np.full(len(nodule_centers), 10.0)

    nodule_trace = go.Scatter3d(
        x=centers_mm[:, 2],  # x-axis (width)
        y=centers_mm[:, 1],  # y-axis (height)
        z=centers_mm[:, 0],  # z-axis (depth)
        mode="markers",
        marker={
            "size": marker_sizes,
            "color": "red",
            "opacity": 0.8,
            "line": {"color": "darkred", "width": 2},
        },
        name="Nodules",
    )

    fig.add_trace(nodule_trace)

    return fig


def render_comparison(
    prediction_mask: NDArray[np.bool_],
    ground_truth_mask: NDArray[np.bool_],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    config: RenderingConfig | None = None,
    title: str = "Prediction vs Ground Truth",
) -> go.Figure:
    """Render prediction and ground truth masks side by side for comparison.

    Args:
        prediction_mask: Predicted segmentation mask, shape (D, H, W).
        ground_truth_mask: Ground truth segmentation mask, shape (D, H, W).
        spacing: Voxel spacing in mm (z, y, x).
        config: Rendering configuration. Uses defaults if None.
        title: Title for the plot.

    Returns:
        Plotly Figure with both masks rendered with different colors.

    Raises:
        ValueError: If masks have different shapes or invalid dimensions.

    Example:
        >>> pred = np.zeros((100, 128, 128), dtype=bool)
        >>> pred[40:60, 50:70, 50:70] = True
        >>> gt = np.zeros((100, 128, 128), dtype=bool)
        >>> gt[42:62, 52:72, 52:72] = True
        >>> fig = render_comparison(pred, gt)
        >>> fig.show()
    """
    if prediction_mask.shape != ground_truth_mask.shape:
        raise ValueError(
            f"prediction_mask shape {prediction_mask.shape} != "
            f"ground_truth_mask shape {ground_truth_mask.shape}"
        )

    if prediction_mask.ndim != 3:
        raise ValueError(f"masks must be 3D, got shape {prediction_mask.shape}")

    if not np.any(prediction_mask) and not np.any(ground_truth_mask):
        raise ValueError("both masks are empty")

    if config is None:
        config = RenderingConfig()

    traces = []

    # Render prediction mask (red)
    if np.any(prediction_mask):
        verts_pred, faces_pred, _, _ = measure.marching_cubes(
            prediction_mask.astype(np.float32), level=0.5, spacing=spacing
        )
        pred_trace = go.Mesh3d(
            x=verts_pred[:, 2],
            y=verts_pred[:, 1],
            z=verts_pred[:, 0],
            i=faces_pred[:, 0],
            j=faces_pred[:, 1],
            k=faces_pred[:, 2],
            opacity=config.opacity,
            color="red",
            flatshading=True,
            lighting=config.lighting,
            name="Prediction",
        )
        traces.append(pred_trace)

    # Render ground truth mask (green)
    if np.any(ground_truth_mask):
        verts_gt, faces_gt, _, _ = measure.marching_cubes(
            ground_truth_mask.astype(np.float32), level=0.5, spacing=spacing
        )
        gt_trace = go.Mesh3d(
            x=verts_gt[:, 2],
            y=verts_gt[:, 1],
            z=verts_gt[:, 0],
            i=faces_gt[:, 0],
            j=faces_gt[:, 1],
            k=faces_gt[:, 2],
            opacity=config.opacity,
            color="green",
            flatshading=True,
            lighting=config.lighting,
            name="Ground Truth",
        )
        traces.append(gt_trace)

    # Create figure
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title=title,
        scene={
            "xaxis": {"visible": config.show_axes, "title": "X (mm)"},
            "yaxis": {"visible": config.show_axes, "title": "Y (mm)"},
            "zaxis": {"visible": config.show_axes, "title": "Z (mm)"},
            "camera": {"eye": {"x": config.camera_eye[0], "y": config.camera_eye[1], "z": config.camera_eye[2]}},
            "aspectmode": "data",
        },
        showlegend=True,
    )

    return fig
