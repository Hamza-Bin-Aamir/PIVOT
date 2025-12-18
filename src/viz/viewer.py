"""Multi-slice viewer for 3D medical imaging volumes.

This module provides an interactive viewer to navigate through 3D medical scans
along different anatomical planes (axial, sagittal, coronal).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from numpy.typing import NDArray


class SliceOrientation(str, Enum):
    """Anatomical slice orientations for 3D volumes."""

    AXIAL = "axial"  # Transverse plane (z-axis, typically patient's feet to head)
    SAGITTAL = "sagittal"  # Lateral plane (x-axis, left to right)
    CORONAL = "coronal"  # Frontal plane (y-axis, front to back)


@dataclass
class ViewerConfig:
    """Configuration for multi-slice viewer.

    Attributes:
        orientation: Slice orientation to display.
        figsize: Figure size in inches (width, height).
        dpi: Dots per inch for figure resolution.
        cmap: Matplotlib colormap name.
        vmin: Minimum value for color mapping. None for auto.
        vmax: Maximum value for color mapping. None for auto.
        show_slider: Whether to show the interactive slider.
        show_info: Whether to show slice information overlay.
        interpolation: Interpolation method for display.
    """

    orientation: SliceOrientation = SliceOrientation.AXIAL
    figsize: tuple[float, float] = (10.0, 8.0)
    dpi: int = 100
    cmap: str = "gray"
    vmin: float | None = None
    vmax: float | None = None
    show_slider: bool = True
    show_info: bool = True
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "nearest"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.dpi <= 0:
            raise ValueError(f"dpi must be > 0, got {self.dpi}")
        if self.vmin is not None and self.vmax is not None and self.vmin >= self.vmax:
            raise ValueError(f"vmin ({self.vmin}) must be < vmax ({self.vmax})")


class MultiSliceViewer:
    """Interactive viewer for navigating 3D medical imaging volumes.

    Provides functionality to:
    - View slices along different anatomical planes
    - Navigate through slices with keyboard or slider
    - Overlay masks or annotations
    - Display metadata and orientation information

    Example:
        >>> volume = np.random.rand(100, 256, 256).astype(np.float32)
        >>> viewer = MultiSliceViewer(volume)
        >>> fig = viewer.show()
        >>> plt.show()
    """

    def __init__(
        self,
        volume: NDArray[np.float32],
        mask: NDArray[np.bool_] | None = None,
        config: ViewerConfig | None = None,
    ) -> None:
        """Initialize the multi-slice viewer.

        Args:
            volume: 3D medical imaging volume, shape (D, H, W).
            mask: Optional 3D binary mask to overlay, shape (D, H, W).
            config: Viewer configuration. Uses defaults if None.

        Raises:
            ValueError: If volume is not 3D or mask shape doesn't match.
        """
        if volume.ndim != 3:
            raise ValueError(f"volume must be 3D, got shape {volume.shape}")

        if mask is not None and mask.shape != volume.shape:
            raise ValueError(f"mask shape {mask.shape} != volume shape {volume.shape}")

        self.volume = volume
        self.mask = mask
        self.config = config if config is not None else ViewerConfig()

        # Set up slicing based on orientation
        self._setup_orientation()

        # Current slice index
        self.current_slice = self.num_slices // 2

        # Figure and axes (created in show())
        self.fig: Figure | None = None
        self.ax: plt.Axes | None = None
        self.slider: Slider | None = None

    def _setup_orientation(self) -> None:
        """Set up orientation-specific parameters."""
        if self.config.orientation == SliceOrientation.AXIAL:
            self.axis = 0
            self.num_slices = self.volume.shape[0]
            self.slice_label = "Axial Slice"
        elif self.config.orientation == SliceOrientation.SAGITTAL:
            self.axis = 2
            self.num_slices = self.volume.shape[2]
            self.slice_label = "Sagittal Slice"
        elif self.config.orientation == SliceOrientation.CORONAL:
            self.axis = 1
            self.num_slices = self.volume.shape[1]
            self.slice_label = "Coronal Slice"
        else:
            raise ValueError(f"Unknown orientation: {self.config.orientation}")

    def _get_slice(self, index: int) -> NDArray[np.float32]:
        """Get a 2D slice at the specified index.

        Args:
            index: Slice index along the current orientation axis.

        Returns:
            2D slice array.
        """
        if self.axis == 0:
            return self.volume[index, :, :]
        elif self.axis == 1:
            return self.volume[:, index, :]
        else:  # axis == 2
            return self.volume[:, :, index]

    def _get_mask_slice(self, index: int) -> NDArray[np.bool_] | None:
        """Get a 2D mask slice at the specified index.

        Args:
            index: Slice index along the current orientation axis.

        Returns:
            2D mask slice or None if no mask.
        """
        if self.mask is None:
            return None

        if self.axis == 0:
            return self.mask[index, :, :]
        elif self.axis == 1:
            return self.mask[:, index, :]
        else:  # axis == 2
            return self.mask[:, :, index]

    def _update_display(self, index: int) -> None:
        """Update the display with a new slice.

        Args:
            index: Slice index to display.
        """
        if self.ax is None:
            return

        index = int(np.clip(index, 0, self.num_slices - 1))
        self.current_slice = index

        # Clear previous display
        self.ax.clear()

        # Get and display slice
        slice_data = self._get_slice(index)
        self.ax.imshow(
            slice_data,
            cmap=self.config.cmap,
            vmin=self.config.vmin,
            vmax=self.config.vmax,
            interpolation=self.config.interpolation,
        )

        # Overlay mask if present
        mask_slice = self._get_mask_slice(index)
        if mask_slice is not None and np.any(mask_slice):
            # Show mask contours in red
            self.ax.contour(
                mask_slice.astype(float),
                levels=[0.5],
                colors=["red"],
                linewidths=2.0,
            )

        # Add slice information
        if self.config.show_info:
            info_text = f"{self.slice_label}: {index + 1}/{self.num_slices}"
            self.ax.text(
                0.02,
                0.98,
                info_text,
                transform=self.ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        self.ax.axis("off")

        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def _on_slider_change(self, val: float) -> None:
        """Handle slider value changes.

        Args:
            val: New slider value (slice index).
        """
        self._update_display(int(val))

    def _on_key_press(self, event: Any) -> None:  # noqa: ANN401
        """Handle keyboard events for slice navigation.

        Args:
            event: Matplotlib key press event.
        """
        if event.key == "up" or event.key == "right":
            new_index = min(self.current_slice + 1, self.num_slices - 1)
            if self.slider is not None:
                self.slider.set_val(new_index)
            else:
                self._update_display(new_index)
        elif event.key == "down" or event.key == "left":
            new_index = max(self.current_slice - 1, 0)
            if self.slider is not None:
                self.slider.set_val(new_index)
            else:
                self._update_display(new_index)

    def show(self, start_slice: int | None = None) -> Figure:
        """Display the multi-slice viewer.

        Args:
            start_slice: Initial slice index to display. If None, shows middle slice.

        Returns:
            Matplotlib Figure object.

        Example:
            >>> viewer = MultiSliceViewer(volume)
            >>> fig = viewer.show(start_slice=50)
            >>> plt.show()
        """
        if start_slice is not None:
            self.current_slice = np.clip(start_slice, 0, self.num_slices - 1)

        # Create figure
        if self.config.show_slider:
            self.fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            # Main axes for image
            self.ax = self.fig.add_axes((0.1, 0.2, 0.8, 0.7))
            # Slider axes
            slider_ax = self.fig.add_axes((0.1, 0.05, 0.8, 0.03))
            self.slider = Slider(
                slider_ax,
                "Slice",
                0,
                self.num_slices - 1,
                valinit=self.current_slice,
                valstep=1,
            )
            self.slider.on_changed(self._on_slider_change)
        else:
            self.fig, self.ax = plt.subplots(
                figsize=self.config.figsize, dpi=self.config.dpi
            )

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # Initial display
        self._update_display(self.current_slice)

        return self.fig

    def set_orientation(self, orientation: SliceOrientation) -> None:
        """Change the viewing orientation.

        Args:
            orientation: New slice orientation.

        Example:
            >>> viewer.set_orientation(SliceOrientation.SAGITTAL)
            >>> viewer.show()
        """
        self.config.orientation = orientation
        self._setup_orientation()
        self.current_slice = self.num_slices // 2

        if self.slider is not None:
            self.slider.valmax = self.num_slices - 1
            self.slider.set_val(self.current_slice)

        self._update_display(self.current_slice)

    def get_current_slice(self) -> NDArray[np.float32]:
        """Get the currently displayed slice.

        Returns:
            2D array of the current slice.
        """
        return self._get_slice(self.current_slice)


def view_volume(
    volume: NDArray[np.float32],
    mask: NDArray[np.bool_] | None = None,
    orientation: SliceOrientation = SliceOrientation.AXIAL,
    start_slice: int | None = None,
) -> Figure:
    """Convenience function to quickly view a 3D volume.

    Args:
        volume: 3D medical imaging volume, shape (D, H, W).
        mask: Optional 3D binary mask to overlay, shape (D, H, W).
        orientation: Slice orientation to display.
        start_slice: Initial slice index to display.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> volume = np.random.rand(100, 256, 256).astype(np.float32)
        >>> fig = view_volume(volume, orientation=SliceOrientation.AXIAL)
        >>> plt.show()
    """
    config = ViewerConfig(orientation=orientation)
    viewer = MultiSliceViewer(volume, mask=mask, config=config)
    return viewer.show(start_slice=start_slice)
