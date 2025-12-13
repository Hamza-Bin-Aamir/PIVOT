"""Utilities for parsing LUNA16 annotation metadata."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "LUNA16Annotation",
    "group_annotations_by_series",
    "parse_luna16_annotations",
]


@dataclass(frozen=True, slots=True)
class LUNA16Annotation:
    """Structured representation of a single LUNA16 nodule annotation."""

    series_uid: str
    center_xyz: tuple[float, float, float]
    diameter_mm: float


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
