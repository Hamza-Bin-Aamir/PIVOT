"""Structured output formatting for inference results.

This module provides utilities to format raw inference outputs into standardized,
structured data formats suitable for downstream consumption (e.g., JSON, database
storage, visualization tools).

The formatter takes raw model predictions (segmentation masks, center heatmaps,
bounding boxes, triage scores) and packages them into a clean, validated output
structure with proper metadata and confidence scores.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "NoduleDetection",
    "InferenceMetadata",
    "StructuredInferenceOutput",
    "StructuredOutputFormatter",
    "OutputFormat",
]

OutputFormat = Literal["dict", "json"]


@dataclass(slots=True)
class NoduleDetection:
    """Single nodule detection with properties and confidence scores.

    Attributes:
        center: 3D coordinates of nodule center in voxel space (z, y, x)
        confidence: Detection confidence score [0, 1]
        triage_score: Malignancy triage probability [0, 1]
        diameter_mm: Estimated nodule diameter in millimeters
        volume_mm3: Estimated nodule volume in cubic millimeters
        bbox_min: Bounding box minimum corner (z, y, x) in voxel space
        bbox_max: Bounding box maximum corner (z, y, x) in voxel space
        properties: Additional properties (e.g., spiculation, calcification)
    """

    center: tuple[float, float, float]
    confidence: float
    triage_score: float
    diameter_mm: float
    volume_mm3: float
    bbox_min: tuple[int, int, int]
    bbox_max: tuple[int, int, int]
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate detection fields."""
        if not 0 <= self.confidence <= 1:
            msg = f"Confidence must be in [0, 1], got {self.confidence}"
            raise ValueError(msg)
        if not 0 <= self.triage_score <= 1:
            msg = f"Triage score must be in [0, 1], got {self.triage_score}"
            raise ValueError(msg)
        if self.diameter_mm < 0:
            msg = f"Diameter must be non-negative, got {self.diameter_mm}"
            raise ValueError(msg)
        if self.volume_mm3 < 0:
            msg = f"Volume must be non-negative, got {self.volume_mm3}"
            raise ValueError(msg)


@dataclass(slots=True)
class InferenceMetadata:
    """Metadata for inference run.

    Attributes:
        scan_id: Unique identifier for the CT scan
        timestamp: ISO-8601 timestamp of inference
        model_version: Version identifier for the model
        spacing: Voxel spacing in mm (z, y, x)
        shape: Volume shape (D, H, W)
        processing_time_ms: Inference duration in milliseconds
        config: Additional configuration parameters
    """

    scan_id: str
    timestamp: str
    model_version: str
    spacing: tuple[float, float, float]
    shape: tuple[int, int, int]
    processing_time_ms: float
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructuredInferenceOutput:
    """Complete structured output from inference pipeline.

    Attributes:
        metadata: Inference metadata
        detections: List of nodule detections
        summary: High-level summary statistics
    """

    metadata: InferenceMetadata
    detections: list[NoduleDetection]
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute default summary statistics if not provided."""
        if not self.summary:
            self.summary = {
                "num_detections": len(self.detections),
                "max_triage_score": (
                    max(d.triage_score for d in self.detections)
                    if self.detections
                    else 0.0
                ),
                "high_risk_count": sum(
                    1 for d in self.detections if d.triage_score >= 0.7
                ),
                "medium_risk_count": sum(
                    1 for d in self.detections if 0.3 <= d.triage_score < 0.7
                ),
                "low_risk_count": sum(
                    1 for d in self.detections if d.triage_score < 0.3
                ),
            }


class StructuredOutputFormatter:
    """Format raw inference outputs into structured, validated format.

    This class takes raw model predictions and organizes them into a clean
    structured format with proper validation and metadata.

    Args:
        model_version: Version identifier for the model
        spacing: Default voxel spacing in mm (z, y, x) if not provided per-sample
        min_confidence: Minimum confidence threshold for including detections
        max_detections: Maximum number of detections to include (top-k by confidence)

    Examples:
        >>> formatter = StructuredOutputFormatter(model_version="v1.0.0")
        >>> output = formatter.format(
        ...     centers=[(10, 20, 30)],
        ...     confidences=[0.95],
        ...     triage_scores=[0.8],
        ...     diameters=[12.5],
        ...     scan_id="SCAN001",
        ...     spacing=(1.0, 1.0, 1.0),
        ...     shape=(128, 256, 256),
        ... )
        >>> output_dict = formatter.to_dict(output)
    """

    def __init__(
        self,
        model_version: str = "unknown",
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        min_confidence: float = 0.0,
        max_detections: int | None = None,
    ) -> None:
        """Initialize formatter with configuration.

        Args:
            model_version: Model version identifier
            spacing: Default voxel spacing in mm (z, y, x)
            min_confidence: Minimum confidence to include detection
            max_detections: Maximum detections to return (None = unlimited)

        Raises:
            ValueError: If min_confidence not in [0, 1]
            ValueError: If max_detections is negative
        """
        if not 0 <= min_confidence <= 1:
            msg = f"min_confidence must be in [0, 1], got {min_confidence}"
            raise ValueError(msg)
        if max_detections is not None and max_detections < 0:
            msg = f"max_detections must be non-negative, got {max_detections}"
            raise ValueError(msg)

        self.model_version = model_version
        self.default_spacing = spacing
        self.min_confidence = min_confidence
        self.max_detections = max_detections

    def format(
        self,
        centers: list[tuple[float, float, float]] | NDArray[np.float32],
        confidences: list[float] | NDArray[np.float32],
        triage_scores: list[float] | NDArray[np.float32],
        diameters: list[float] | NDArray[np.float32],
        scan_id: str,
        *,
        spacing: tuple[float, float, float] | None = None,
        shape: tuple[int, int, int] | None = None,
        processing_time_ms: float = 0.0,
        bounding_boxes: list[tuple[tuple[int, int, int], tuple[int, int, int]]] | None = None,
        properties: list[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> StructuredInferenceOutput:
        """Format raw inference outputs into structured format.

        Args:
            centers: Nodule center coordinates (N, 3) in voxel space (z, y, x)
            confidences: Detection confidence scores (N,) in [0, 1]
            triage_scores: Triage probability scores (N,) in [0, 1]
            diameters: Nodule diameters in mm (N,)
            scan_id: Unique scan identifier
            spacing: Voxel spacing in mm (z, y, x), uses default if None
            shape: Volume shape (D, H, W), required if bboxes not provided
            processing_time_ms: Processing duration in milliseconds
            bounding_boxes: Optional bounding boxes as (min, max) tuples
            properties: Optional per-detection properties
            config: Optional configuration dict for metadata

        Returns:
            Structured inference output with validated detections

        Raises:
            ValueError: If input arrays have mismatched lengths
            ValueError: If no valid detections remain after filtering
        """
        # Convert to lists if numpy arrays
        centers_list = self._to_list(centers)
        confidences_list = self._to_list(confidences)
        triage_list = self._to_list(triage_scores)
        diameters_list = self._to_list(diameters)

        # Validate input lengths
        n = len(centers_list)
        if not (len(confidences_list) == len(triage_list) == len(diameters_list) == n):
            msg = "All input arrays must have same length"
            raise ValueError(msg)

        # Use defaults if not provided
        spacing = spacing or self.default_spacing
        shape = shape or (0, 0, 0)  # Placeholder if not provided
        config = config or {}

        # Filter by confidence threshold
        valid_indices = [i for i, c in enumerate(confidences_list) if c >= self.min_confidence]

        # Sort by confidence (descending)
        valid_indices.sort(key=lambda i: confidences_list[i], reverse=True)

        # Limit to max detections
        if self.max_detections is not None:
            valid_indices = valid_indices[: self.max_detections]

        # Create detection objects
        detections: list[NoduleDetection] = []
        for i in valid_indices:
            center = centers_list[i]
            if isinstance(center, (list, np.ndarray)):
                center = tuple(float(x) for x in center)  # type: ignore[assignment]

            diameter = float(diameters_list[i])
            volume = self._estimate_volume(diameter)

            # Get bounding box if provided, otherwise estimate from center/diameter
            if bounding_boxes and i < len(bounding_boxes):
                bbox_min, bbox_max = bounding_boxes[i]
            else:
                bbox_min, bbox_max = self._estimate_bbox(center, diameter, spacing, shape)

            # Get properties if provided
            props = properties[i] if properties and i < len(properties) else {}

            detection = NoduleDetection(
                center=center,  # type: ignore[arg-type]
                confidence=float(confidences_list[i]),
                triage_score=float(triage_list[i]),
                diameter_mm=diameter,
                volume_mm3=volume,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                properties=props,
            )
            detections.append(detection)

        # Create metadata
        metadata = InferenceMetadata(
            scan_id=scan_id,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            model_version=self.model_version,
            spacing=spacing,
            shape=shape,
            processing_time_ms=processing_time_ms,
            config=config,
        )

        return StructuredInferenceOutput(metadata=metadata, detections=detections)

    def to_dict(self, output: StructuredInferenceOutput) -> dict[str, Any]:
        """Convert structured output to dictionary.

        Args:
            output: Structured inference output

        Returns:
            Dictionary representation
        """
        return asdict(output)

    def to_json(self, output: StructuredInferenceOutput, *, indent: int = 2) -> str:
        """Convert structured output to JSON string.

        Args:
            output: Structured inference output
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(output), indent=indent)

    def save(
        self,
        output: StructuredInferenceOutput,
        filepath: str | Path,
        *,
        format: OutputFormat = "json",
    ) -> None:
        """Save structured output to file.

        Args:
            output: Structured inference output
            filepath: Output file path
            format: Output format ("json" or "dict" for pickle)

        Raises:
            ValueError: If format is not supported
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(filepath, "w") as f:
                f.write(self.to_json(output))
        elif format == "dict":
            import pickle

            with open(filepath, "wb") as f:
                pickle.dump(self.to_dict(output), f)
        else:
            msg = f"Unsupported format: {format}"
            raise ValueError(msg)

    def _to_list(self, arr: list[Any] | NDArray[Any]) -> list[Any]:
        """Convert array-like to list.

        Args:
            arr: Input array or list

        Returns:
            List representation
        """
        if isinstance(arr, np.ndarray):
            result: list[Any] = arr.tolist()
            return result
        return list(arr)

    def _estimate_volume(self, diameter: float) -> float:
        """Estimate nodule volume assuming spherical shape.

        Args:
            diameter: Nodule diameter in mm

        Returns:
            Estimated volume in mm³
        """
        radius = diameter / 2.0
        return (4.0 / 3.0) * np.pi * (radius**3)

    def _estimate_bbox(
        self,
        center: tuple[float, float, float],
        diameter: float,
        spacing: tuple[float, float, float],
        shape: tuple[int, int, int],
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Estimate bounding box from center and diameter.

        Args:
            center: Center coordinates (z, y, x) in voxel space
            diameter: Diameter in mm
            spacing: Voxel spacing (z, y, x) in mm
            shape: Volume shape (D, H, W)

        Returns:
            Tuple of (bbox_min, bbox_max) in voxel coordinates
        """
        # Convert diameter to voxels in each dimension
        radius_voxels = [diameter / (2 * s) for s in spacing]

        # Calculate bbox bounds
        bbox_min = tuple(
            max(0, int(center[i] - radius_voxels[i])) for i in range(3)
        )
        bbox_max = tuple(
            min(shape[i] - 1, int(center[i] + radius_voxels[i])) for i in range(3)
        )

        return bbox_min, bbox_max  # type: ignore[return-value]
