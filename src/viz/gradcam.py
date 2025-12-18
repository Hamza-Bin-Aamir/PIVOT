"""Grad-CAM attention visualization for 3D medical imaging models.

This module implements Gradient-weighted Class Activation Mapping (Grad-CAM)
for visualizing where 3D CNN models focus their attention when making predictions.
Useful for understanding and debugging model decisions in nodule detection.

Reference:
    Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization." In ICCV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray


@dataclass
class GradCAMConfig:
    """Configuration for Grad-CAM visualization.

    Attributes:
        colorscale: Plotly colorscale name for heatmap. Default: 'Hot'
        opacity: Overlay opacity [0, 1]. Default: 0.5
        threshold: Activation threshold for visualization [0, 1]. Default: 0.0
        show_colorbar: Whether to show colorbar. Default: True
    """

    colorscale: str = "Hot"
    opacity: float = 0.5
    threshold: float = 0.0
    show_colorbar: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.opacity <= 1:
            msg = f"Opacity must be in [0, 1], got {self.opacity}"
            raise ValueError(msg)
        if not 0 <= self.threshold <= 1:
            msg = f"Threshold must be in [0, 1], got {self.threshold}"
            raise ValueError(msg)


class GradCAM:
    """Grad-CAM attention map generator for 3D CNNs.

    Computes gradient-weighted class activation maps to visualize which regions
    of the input volume most influence the model's predictions.

    Attributes:
        model: The neural network model (in eval mode)
        target_layer: The convolutional layer to extract activations from
        activations: Stored forward activations
        gradients: Stored backward gradients
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Initialize Grad-CAM.

        Args:
            model: Neural network model to analyze
            target_layer: Target convolutional layer for activation extraction
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)  # type: ignore[arg-type]

    def _save_activation(
        self, _module: nn.Module, _input: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Hook to save forward activations."""
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Hook to save backward gradients."""
        self.gradients = grad_output[0].detach()

    def generate(
        self, input_tensor: torch.Tensor, target_class: int | None = None
    ) -> NDArray[np.float32]:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input volume [1, C, D, H, W]
            target_class: Target class index for multi-class models.
                         For binary segmentation, use None (default).

        Returns:
            Grad-CAM heatmap as numpy array [D, H, W] with values in [0, 1]

        Raises:
            ValueError: If input has wrong shape or no gradients computed
        """
        if input_tensor.dim() != 5:
            msg = f"Expected 5D input [1, C, D, H, W], got {input_tensor.dim()}D"
            raise ValueError(msg)
        if input_tensor.size(0) != 1:
            msg = f"Expected batch size 1, got {input_tensor.size(0)}"
            raise ValueError(msg)

        # Ensure model is in eval mode but gradients are enabled
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Handle different output types
        if isinstance(output, dict):
            # Multi-task model - use segmentation by default
            output = output.get("segmentation", list(output.values())[0])

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        if target_class is not None:
            # Multi-class: backprop specific class
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
        else:
            # Binary: backprop entire output
            output.backward(gradient=torch.ones_like(output), retain_graph=True)

        # Check if gradients were computed
        if self.gradients is None or self.activations is None:
            msg = "No gradients or activations captured. Check target_layer."
            raise ValueError(msg)

        # Compute channel-wise weights (global average pooling of gradients)
        # Shape: [1, C, D, H, W] -> [1, C, 1, 1, 1]
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)

        # Weighted combination of activations
        # Shape: [1, C, D, H, W] * [1, C, 1, 1, 1] -> [1, C, D, H, W]
        weighted_activations = weights * self.activations

        # Sum over channels and apply ReLU (only positive influences)
        # Shape: [1, C, D, H, W] -> [1, D, H, W]
        cam = weighted_activations.sum(dim=1, keepdim=False)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_np: NDArray[np.float32] = cam.squeeze(0).cpu().numpy()  # [D, H, W]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)

        return cam_np.astype(np.float32)

    def __del__(self) -> None:
        """Remove hooks on deletion."""
        if hasattr(self, "_forward_hook"):
            self._forward_hook.remove()
        if hasattr(self, "_backward_hook"):
            self._backward_hook.remove()


def overlay_gradcam_on_volume(
    volume: NDArray[np.float32],
    gradcam: NDArray[np.float32],
    slice_idx: int,
    axis: int = 0,
    config: GradCAMConfig | None = None,
    title: str = "Grad-CAM Attention Overlay",
) -> go.Figure:
    """Overlay Grad-CAM heatmap on a 2D slice of the volume.

    Args:
        volume: Input CT volume [D, H, W]
        gradcam: Grad-CAM heatmap [D, H, W] with values in [0, 1]
        slice_idx: Index of slice to visualize
        axis: Axis along which to slice (0=axial, 1=coronal, 2=sagittal)
        config: Visualization configuration
        title: Plot title

    Returns:
        Plotly figure with overlay visualization

    Raises:
        ValueError: If shapes don't match or parameters are invalid
    """
    if volume.shape != gradcam.shape:
        msg = f"Shape mismatch: volume {volume.shape} vs gradcam {gradcam.shape}"
        raise ValueError(msg)
    if volume.ndim != 3:
        msg = f"Expected 3D volume, got {volume.ndim}D"
        raise ValueError(msg)
    if not 0 <= axis <= 2:
        msg = f"Axis must be 0, 1, or 2, got {axis}"
        raise ValueError(msg)
    if not 0 <= slice_idx < volume.shape[axis]:
        msg = f"Slice index {slice_idx} out of range [0, {volume.shape[axis]})"
        raise ValueError(msg)

    if config is None:
        config = GradCAMConfig()

    # Extract slices
    if axis == 0:
        vol_slice = volume[slice_idx, :, :]
        cam_slice = gradcam[slice_idx, :, :]
        axis_name = "Axial"
    elif axis == 1:
        vol_slice = volume[:, slice_idx, :]
        cam_slice = gradcam[:, slice_idx, :]
        axis_name = "Coronal"
    else:
        vol_slice = volume[:, :, slice_idx]
        cam_slice = gradcam[:, :, slice_idx]
        axis_name = "Sagittal"

    # Normalize volume for display
    vol_min, vol_max = vol_slice.min(), vol_slice.max()
    if vol_max > vol_min:
        vol_norm = (vol_slice - vol_min) / (vol_max - vol_min)
    else:
        vol_norm = vol_slice.copy()

    # Apply threshold to Grad-CAM
    cam_masked = cam_slice.copy()
    cam_masked[cam_masked < config.threshold] = 0

    # Create figure with base volume and Grad-CAM overlay
    fig = go.Figure()

    # Base volume (grayscale)
    fig.add_trace(
        go.Heatmap(
            z=vol_norm,
            colorscale="Gray",
            showscale=False,
            hovertemplate="Intensity: %{z:.3f}<extra></extra>",
        )
    )

    # Grad-CAM overlay (hot colormap)
    fig.add_trace(
        go.Heatmap(
            z=cam_masked,
            colorscale=config.colorscale,
            opacity=config.opacity,
            showscale=config.show_colorbar,
            colorbar={"title": "Attention", "x": 1.1} if config.show_colorbar else None,
            hovertemplate="Attention: %{z:.3f}<extra></extra>",
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


def render_gradcam_3d(
    gradcam: NDArray[np.float32],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    config: GradCAMConfig | None = None,
    title: str = "3D Grad-CAM Attention",
) -> go.Figure:
    """Render 3D isosurface visualization of Grad-CAM attention.

    Args:
        gradcam: Grad-CAM heatmap [D, H, W] with values in [0, 1]
        spacing: Voxel spacing (z, y, x) in mm
        config: Visualization configuration
        title: Plot title

    Returns:
        Plotly figure with 3D isosurface visualization

    Raises:
        ValueError: If gradcam has wrong shape or spacing is invalid
    """
    if gradcam.ndim != 3:
        msg = f"Expected 3D gradcam, got {gradcam.ndim}D"
        raise ValueError(msg)
    if len(spacing) != 3:
        msg = f"Spacing must have 3 values, got {len(spacing)}"
        raise ValueError(msg)
    if any(s <= 0 for s in spacing):
        msg = f"Spacing values must be positive, got {spacing}"
        raise ValueError(msg)

    if config is None:
        config = GradCAMConfig()

    # Apply threshold
    gradcam_thresh = gradcam.copy()
    gradcam_thresh[gradcam_thresh < config.threshold] = 0

    # Create coordinate grids
    d, h, w = gradcam.shape
    z = np.arange(d) * spacing[0]
    y = np.arange(h) * spacing[1]
    x = np.arange(w) * spacing[2]

    # Create volume trace
    fig = go.Figure(
        data=go.Volume(
            x=x,
            y=y,
            z=z,
            value=gradcam_thresh.flatten(),
            opacity=config.opacity,
            surface_count=15,
            colorscale=config.colorscale,
            showscale=config.show_colorbar,
            colorbar={"title": "Attention"} if config.show_colorbar else None,
        )
    )

    fig.update_layout(
        title=title,
        scene={
            "xaxis": {"title": "X (mm)"},
            "yaxis": {"title": "Y (mm)"},
            "zaxis": {"title": "Z (mm)"},
            "aspectmode": "data",
        },
        showlegend=False,
    )

    return fig
