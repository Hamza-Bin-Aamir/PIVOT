"""Post-processing filters for nodule detections.

This module provides filtering functions to refine detected nodules based on
physical properties (size, confidence) and spatial relationships (overlap).
"""

import logging

from .nodule_properties import NoduleProperties

logger = logging.getLogger(__name__)


def filter_by_size(
    detections: list[NoduleProperties],
    min_volume_mm3: float | None = None,
    max_volume_mm3: float | None = None,
    min_diameter_mm: float | None = None,
    max_diameter_mm: float | None = None,
) -> list[NoduleProperties]:
    """Filter nodule detections based on size constraints.

    Args:
        detections: List of nodule properties to filter.
        min_volume_mm3: Minimum volume in mm³. Defaults to None (no minimum).
        max_volume_mm3: Maximum volume in mm³. Defaults to None (no maximum).
        min_diameter_mm: Minimum diameter in mm. Defaults to None (no minimum).
        max_diameter_mm: Maximum diameter in mm. Defaults to None (no maximum).

    Returns:
        Filtered list of nodule properties meeting size criteria.

    Examples:
        >>> # Filter for medium-sized nodules (3-30mm diameter)
        >>> filtered = filter_by_size(
        ...     detections,
        ...     min_diameter_mm=3.0,
        ...     max_diameter_mm=30.0
        ... )
    """
    filtered: list[NoduleProperties] = []
    for detection in detections:
        # Check volume constraints
        if min_volume_mm3 is not None and detection.volume_mm3 < min_volume_mm3:
            continue
        if max_volume_mm3 is not None and detection.volume_mm3 > max_volume_mm3:
            continue

        # Check diameter constraints - compare against max of (z,y,x) dimensions
        diameter = max(detection.diameter_mm)
        if min_diameter_mm is not None and diameter < min_diameter_mm:
            continue
        if max_diameter_mm is not None and diameter > max_diameter_mm:
            continue

        filtered.append(detection)

    logger.debug(
        f"Size filter: {len(detections)} -> {len(filtered)} detections "
        f"(vol: {min_volume_mm3}-{max_volume_mm3} mm³, "
        f"diam: {min_diameter_mm}-{max_diameter_mm} mm)"
    )
    return filtered


def filter_by_confidence(
    detections: list[NoduleProperties], min_confidence: float = 0.5
) -> list[NoduleProperties]:
    """Filter nodule detections based on confidence score.

    Args:
        detections: List of nodule properties to filter.
        min_confidence: Minimum confidence score (0-1). Defaults to 0.5.

    Returns:
        Filtered list of nodule properties meeting confidence threshold.

    Examples:
        >>> # Filter for high-confidence detections
        >>> filtered = filter_by_confidence(detections, min_confidence=0.7)
    """
    filtered: list[NoduleProperties] = []
    for detection in detections:
        if detection.confidence is not None and detection.confidence >= min_confidence:
            filtered.append(detection)
        elif detection.confidence is None:
            # Include detections without confidence scores
            filtered.append(detection)

    logger.debug(
        f"Confidence filter: {len(detections)} -> {len(filtered)} detections "
        f"(threshold: {min_confidence})"
    )
    return filtered


def compute_iou_3d(
    bbox1: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    bbox2: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> float:
    """Compute 3D Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bounding box ((z_min, z_max), (y_min, y_max), (x_min, x_max)).
        bbox2: Second bounding box ((z_min, z_max), (y_min, y_max), (x_min, x_max)).

    Returns:
        IoU score between 0 and 1.

    Examples:
        >>> bbox1 = ((0, 10), (0, 10), (0, 10))  # 1000 voxel cube
        >>> bbox2 = ((5, 15), (5, 15), (5, 15))  # Overlapping cube
        >>> iou = compute_iou_3d(bbox1, bbox2)  # 125/1875 = 0.0667
    """
    # Compute intersection
    z_inter = max(0, min(bbox1[0][1], bbox2[0][1]) - max(bbox1[0][0], bbox2[0][0]))
    y_inter = max(0, min(bbox1[1][1], bbox2[1][1]) - max(bbox1[1][0], bbox2[1][0]))
    x_inter = max(0, min(bbox1[2][1], bbox2[2][1]) - max(bbox1[2][0], bbox2[2][0]))
    intersection = z_inter * y_inter * x_inter

    # Compute union
    vol1 = (
        (bbox1[0][1] - bbox1[0][0])
        * (bbox1[1][1] - bbox1[1][0])
        * (bbox1[2][1] - bbox1[2][0])
    )
    vol2 = (
        (bbox2[0][1] - bbox2[0][0])
        * (bbox2[1][1] - bbox2[1][0])
        * (bbox2[2][1] - bbox2[2][0])
    )
    union = vol1 + vol2 - intersection

    return intersection / union if union > 0 else 0.0


def filter_by_overlap(
    detections: list[NoduleProperties], iou_threshold: float = 0.3
) -> list[NoduleProperties]:
    """Filter overlapping nodule detections using Non-Maximum Suppression (NMS).

    Keeps the detection with highest confidence when two detections overlap
    beyond the IoU threshold. Processes detections in descending confidence order.

    Args:
        detections: List of nodule properties to filter.
        iou_threshold: IoU threshold above which detections are considered overlapping.
            Defaults to 0.3.

    Returns:
        Filtered list of non-overlapping nodule properties.

    Examples:
        >>> # Suppress overlapping detections
        >>> filtered = filter_by_overlap(detections, iou_threshold=0.3)
    """
    if not detections:
        return []

    # Sort by confidence (descending), handling None values
    sorted_detections = sorted(
        detections, key=lambda d: d.confidence if d.confidence is not None else 0.0, reverse=True
    )

    kept: list[NoduleProperties] = []
    for detection in sorted_detections:
        # Check if this detection overlaps with any kept detection
        overlaps = False
        for kept_detection in kept:
            iou = compute_iou_3d(detection.bbox, kept_detection.bbox)
            if iou > iou_threshold:
                overlaps = True
                break

        if not overlaps:
            kept.append(detection)

    logger.debug(
        f"Overlap filter: {len(detections)} -> {len(kept)} detections "
        f"(IoU threshold: {iou_threshold})"
    )
    return kept


def apply_all_filters(
    detections: list[NoduleProperties],
    min_volume_mm3: float | None = None,
    max_volume_mm3: float | None = None,
    min_diameter_mm: float | None = None,
    max_diameter_mm: float | None = None,
    min_confidence: float | None = None,
    iou_threshold: float | None = None,
) -> list[NoduleProperties]:
    """Apply all post-processing filters in sequence.

    Filter order: size -> confidence -> overlap (NMS).

    Args:
        detections: List of nodule properties to filter.
        min_volume_mm3: Minimum volume in mm³. Defaults to None (no minimum).
        max_volume_mm3: Maximum volume in mm³. Defaults to None (no maximum).
        min_diameter_mm: Minimum diameter in mm. Defaults to None (no minimum).
        max_diameter_mm: Maximum diameter in mm. Defaults to None (no maximum).
        min_confidence: Minimum confidence score (0-1). Defaults to None (no filtering).
        iou_threshold: IoU threshold for NMS. Defaults to None (no filtering).

    Returns:
        Filtered list of nodule properties.

    Examples:
        >>> # Full pipeline: size + confidence + overlap filtering
        >>> filtered = apply_all_filters(
        ...     detections,
        ...     min_diameter_mm=3.0,
        ...     max_diameter_mm=30.0,
        ...     min_confidence=0.5,
        ...     iou_threshold=0.3
        ... )
    """
    result = detections

    # Apply size filters
    if any(
        x is not None for x in [min_volume_mm3, max_volume_mm3, min_diameter_mm, max_diameter_mm]
    ):
        result = filter_by_size(
            result,
            min_volume_mm3=min_volume_mm3,
            max_volume_mm3=max_volume_mm3,
            min_diameter_mm=min_diameter_mm,
            max_diameter_mm=max_diameter_mm,
        )

    # Apply confidence filter
    if min_confidence is not None:
        result = filter_by_confidence(result, min_confidence=min_confidence)

    # Apply overlap filter (NMS)
    if iou_threshold is not None:
        result = filter_by_overlap(result, iou_threshold=iou_threshold)

    logger.info(
        f"Applied all filters: {len(detections)} -> {len(result)} final detections"
    )
    return result
