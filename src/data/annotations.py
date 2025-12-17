"""Utilities for parsing LUNA16 and LIDC-IDRI annotation metadata."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

__all__ = [
    "LUNA16Annotation",
    "LIDCAnnotation",
    "LIDCRoi",
    "group_annotations_by_series",
    "parse_luna16_annotations",
    "parse_lidc_annotations",
]


@dataclass(frozen=True, slots=True)
class LUNA16Annotation:
    """Structured representation of a single LUNA16 nodule annotation."""

    series_uid: str
    center_xyz: tuple[float, float, float]
    diameter_mm: float


@dataclass(frozen=True, slots=True)
class LIDCRoi:
    """Single 2D region of interest (ROI) for a LIDC-IDRI annotation slice."""

    image_uid: str | None
    z_position: float
    xy_coords: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class LIDCAnnotation:
    """Structured representation of a LIDC-IDRI nodule annotation."""

    series_uid: str
    nodule_id: str
    reading_session_id: str
    rois: tuple[LIDCRoi, ...]
    characteristics: tuple[tuple[str, int], ...]


def parse_luna16_annotations(
    csv_path: str | Path,
    *,
    min_diameter_mm: float | None = None,
    allowed_series_uids: set[str] | None = None,
) -> list[LUNA16Annotation]:
    """Parse the official LUNA16 annotations.csv file.

    Args:
        csv_path: Path to ``annotations.csv``.
        min_diameter_mm: Optional minimum diameter threshold. Annotations with
            smaller diameters are discarded.
        allowed_series_uids: Optional whitelist limiting results to the
            provided series identifiers.

    Returns:
        A list of ``LUNA16Annotation`` instances ordered as they appear in the
        source file.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If the CSV header is missing required columns or a row
            contains invalid numeric data.
    """

    if min_diameter_mm is not None and min_diameter_mm < 0:
        msg = f"min_diameter_mm must be non-negative, got {min_diameter_mm}"
        raise ValueError(msg)

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    if allowed_series_uids is not None:
        allowed_series = {uid.strip() for uid in allowed_series_uids}
    else:
        allowed_series = None

    annotations: list[LUNA16Annotation] = []
    required_columns = {"seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = required_columns - fieldnames
        if missing:
            raise ValueError(f"Missing columns in annotations file: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            series_uid = (row.get("seriesuid") or "").strip()
            if not series_uid:
                raise ValueError(f"Row {line_number} missing seriesuid")

            if allowed_series is not None and series_uid not in allowed_series:
                continue

            try:
                coord_x = float(row["coordX"])
                coord_y = float(row["coordY"])
                coord_z = float(row["coordZ"])
                diameter = float(row["diameter_mm"])
            except (TypeError, ValueError) as exc:
                msg = f"Invalid numeric value in row {line_number}: {row}"
                raise ValueError(msg) from exc

            if min_diameter_mm is not None and diameter < min_diameter_mm:
                continue

            annotations.append(
                LUNA16Annotation(
                    series_uid=series_uid,
                    center_xyz=(coord_x, coord_y, coord_z),
                    diameter_mm=diameter,
                )
            )

    return annotations


def group_annotations_by_series(
    annotations: Iterable[LUNA16Annotation],
) -> dict[str, list[LUNA16Annotation]]:
    """Group annotations by their ``series_uid``."""

    groups: dict[str, list[LUNA16Annotation]] = defaultdict(list)
    for annotation in annotations:
        groups[annotation.series_uid].append(annotation)
    return dict(groups)


def parse_lidc_annotations(
    xml_path: str | Path,
    *,
    min_roi_count: int | None = None,
    allowed_nodule_ids: set[str] | None = None,
) -> list[LIDCAnnotation]:
    """Parse a LIDC-IDRI XML annotation file.

    Args:
        xml_path: Path to a single LIDC-IDRI XML annotation file.
        min_roi_count: Optional minimum number of ROIs required for a nodule
            to be returned. ROIs marked as excluded or without coordinates are
            discarded prior to counting.
        allowed_nodule_ids: Optional whitelist limiting the returned nodules.

    Returns:
        A list of ``LIDCAnnotation`` instances, one per unblinded nodule found
        in the XML file. The order matches the order encountered in the
        document.

    Raises:
        FileNotFoundError: If ``xml_path`` does not exist.
        ValueError: If the XML content is malformed or missing required
            fields.
    """

    if min_roi_count is not None and min_roi_count < 1:
        msg = f"min_roi_count must be positive, got {min_roi_count}"
        raise ValueError(msg)

    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse XML file: {path}") from exc

    series_uid = (root.findtext(".//SeriesInstanceUid") or "").strip()
    if not series_uid:
        raise ValueError("SeriesInstanceUid missing from annotation file")

    annotations: list[LIDCAnnotation] = []
    reading_sessions = list(root.findall(".//readingSession"))

    for session_index, session in enumerate(reading_sessions):
        session_identifier = (session.findtext("servicingRadiologistID") or "").strip()
        if not session_identifier:
            session_identifier = f"session-{session_index + 1}"

        for nodule in session.findall("unblindedReadNodule"):
            nodule_id = (nodule.findtext("noduleID") or "").strip()
            if not nodule_id:
                raise ValueError("Encountered nodule without noduleID")
            if allowed_nodule_ids is not None and nodule_id not in allowed_nodule_ids:
                continue

            rois: list[LIDCRoi] = []
            for roi in nodule.findall("roi"):
                inclusion_value = (roi.findtext("inclusion") or "true").strip().lower()
                if inclusion_value in {"false", "0"}:
                    continue

                z_text = roi.findtext("imageZposition")
                if z_text is None or not z_text.strip():
                    raise ValueError(f"ROI missing imageZposition for nodule {nodule_id}")
                try:
                    z_position = float(z_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid imageZposition '{z_text}' for nodule {nodule_id}"
                    ) from exc

                image_uid = (roi.findtext("imageUID") or roi.findtext("imageSOP_UID") or "").strip()
                if not image_uid:
                    image_uid_value: str | None = None
                else:
                    image_uid_value = image_uid

                coords: list[tuple[float, float]] = []
                for edge_map in roi.findall("edgeMap"):
                    x_text = edge_map.findtext("xCoord")
                    y_text = edge_map.findtext("yCoord")
                    if x_text is None or y_text is None:
                        raise ValueError(f"edgeMap missing coordinates in nodule {nodule_id}")
                    try:
                        x_val = float(x_text)
                        y_val = float(y_text)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid edge coordinate in nodule {nodule_id}: ({x_text}, {y_text})"
                        ) from exc
                    coords.append((x_val, y_val))

                if not coords:
                    continue

                rois.append(
                    LIDCRoi(
                        image_uid=image_uid_value,
                        z_position=z_position,
                        xy_coords=tuple(coords),
                    )
                )

            if not rois:
                continue
            if min_roi_count is not None and len(rois) < min_roi_count:
                continue

            characteristics_node = nodule.find("characteristics")
            characteristics: tuple[tuple[str, int], ...] = ()
            if characteristics_node is not None:
                parsed_characteristics: list[tuple[str, int]] = []
                for child in list(characteristics_node):
                    value = child.text
                    if value is None or not value.strip():
                        continue
                    try:
                        parsed_characteristics.append((child.tag, int(float(value))))
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid characteristic value '{value}' for {child.tag} in nodule {nodule_id}"
                        ) from exc

                if parsed_characteristics:
                    parsed_characteristics.sort(key=lambda item: item[0])
                    characteristics = tuple(parsed_characteristics)

            annotations.append(
                LIDCAnnotation(
                    series_uid=series_uid,
                    nodule_id=nodule_id,
                    reading_session_id=session_identifier,
                    rois=tuple(rois),
                    characteristics=characteristics,
                )
            )

    return annotations
