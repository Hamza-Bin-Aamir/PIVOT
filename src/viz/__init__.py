"""Visualization tools for medical imaging and model outputs."""

from .confidence import (
    ConfidenceConfig,
    plot_calibration_curve,
    plot_confidence_histogram,
    visualize_confidence_map,
    visualize_uncertainty_regions,
)
from .gradcam import (
    GradCAM,
    GradCAMConfig,
    overlay_gradcam_on_volume,
    render_gradcam_3d,
)
from .overlay import (
    ColorMap,
    OverlayConfig,
    create_2d_overlay,
    overlay_detections,
    overlay_masks,
)
from .rendering_3d import (
    RenderingConfig,
    render_comparison,
    render_lung_surface,
    render_volume,
    render_with_nodules,
)
from .viewer import MultiSliceViewer, SliceOrientation, ViewerConfig

__all__ = [
    # Overlay visualization
    "create_2d_overlay",
    "overlay_masks",
    "overlay_detections",
    "ColorMap",
    "OverlayConfig",
    # Multi-slice viewer
    "MultiSliceViewer",
    "SliceOrientation",
    "ViewerConfig",
    # 3D rendering
    "render_lung_surface",
    "render_volume",
    "render_with_nodules",
    "render_comparison",
    "RenderingConfig",
    # Grad-CAM attention
    "GradCAM",
    "GradCAMConfig",
    "overlay_gradcam_on_volume",
    "render_gradcam_3d",
    # Confidence visualization
    "ConfidenceConfig",
    "visualize_confidence_map",
    "visualize_uncertainty_regions",
    "plot_calibration_curve",
    "plot_confidence_histogram",
]
