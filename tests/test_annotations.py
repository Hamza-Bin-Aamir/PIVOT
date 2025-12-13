"""Tests for LUNA16 annotation parsing utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.data.annotations import (
    LUNA16Annotation,
    group_annotations_by_series,
    parse_luna16_annotations,
)


def _write_annotations(tmp_path: Path, content: str) -> Path:
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def test_parse_luna16_annotations_reads_rows() -> None:
    content = (
        "seriesuid,coordX,coordY,coordZ,diameter_mm\n"
        "1.3.6.1,10.0,11.0,12.0,4.5\n"
        "1.3.6.2,-5.5,0.0,2.5,6.2\n"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = _write_annotations(Path(tmp_dir), content)

        annotations = parse_luna16_annotations(csv_path)

    assert annotations == [
        LUNA16Annotation("1.3.6.1", (10.0, 11.0, 12.0), 4.5),
        LUNA16Annotation("1.3.6.2", (-5.5, 0.0, 2.5), 6.2),
    ]


def test_parse_luna16_annotations_applies_filters() -> None:
    content = (
        "seriesuid,coordX,coordY,coordZ,diameter_mm\n"
        "1.3.6.1,10.0,11.0,12.0,4.5\n"
        "1.3.6.2,-5.5,0.0,2.5,6.2\n"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = _write_annotations(Path(tmp_dir), content)

        annotations = parse_luna16_annotations(
            csv_path,
            min_diameter_mm=5.0,
            allowed_series_uids={"1.3.6.1", "1.3.6.2"},
        )

    assert annotations == [LUNA16Annotation("1.3.6.2", (-5.5, 0.0, 2.5), 6.2)]


def test_parse_luna16_annotations_raises_for_missing_columns() -> None:
    content = "seriesuid,coordX,coordY\n1,2,3\n"

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = _write_annotations(Path(tmp_dir), content)

        with pytest.raises(ValueError, match="Missing columns"):
            parse_luna16_annotations(csv_path)


def test_parse_luna16_annotations_accepts_pathlike() -> None:
    content = "seriesuid,coordX,coordY,coordZ,diameter_mm\n1.3.6.1,10.0,11.0,12.0,4.5\n"

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = _write_annotations(Path(tmp_dir), content)

        annotations = parse_luna16_annotations(Path(csv_path))

    assert annotations[0].series_uid == "1.3.6.1"


def test_group_annotations_by_series_preserves_membership() -> None:
    annotations = [
        LUNA16Annotation("uid-a", (0.0, 0.0, 0.0), 5.0),
        LUNA16Annotation("uid-a", (1.0, 0.0, 0.0), 4.0),
        LUNA16Annotation("uid-b", (2.0, 0.0, 0.0), 6.0),
    ]

    grouped = group_annotations_by_series(annotations)

    assert set(grouped.keys()) == {"uid-a", "uid-b"}
    assert grouped["uid-a"] == annotations[:2]
    assert grouped["uid-b"] == [annotations[2]]


def test_parse_luna16_annotations_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        parse_luna16_annotations(Path("/tmp/missing.csv"), min_diameter_mm=-1)


def test_parse_luna16_annotations_raises_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Annotation file not found"):
        parse_luna16_annotations(missing)


def test_parse_luna16_annotations_raises_when_series_uid_missing() -> None:
    content = "seriesuid,coordX,coordY,coordZ,diameter_mm\n,1.0,2.0,3.0,4.0\n"

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = _write_annotations(Path(tmp_dir), content)

        with pytest.raises(ValueError, match="missing seriesuid"):
            parse_luna16_annotations(csv_path)


def test_parse_luna16_annotations_raises_for_invalid_numeric_value() -> None:
    content = "seriesuid,coordX,coordY,coordZ,diameter_mm\nuid-a,not-a-number,2.0,3.0,4.0\n"

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = _write_annotations(Path(tmp_dir), content)

        with pytest.raises(ValueError, match="Invalid numeric value"):
            parse_luna16_annotations(csv_path)


def test_parse_luna16_annotations_strips_allowed_series_entries() -> None:
    content = "seriesuid,coordX,coordY,coordZ,diameter_mm\nuid-a,1.0,2.0,3.0,4.0\n"

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = _write_annotations(Path(tmp_dir), content)

        annotations = parse_luna16_annotations(csv_path, allowed_series_uids={"  uid-a  "})

    assert len(annotations) == 1


def test_parse_luna16_annotations_excludes_disallowed_series() -> None:
    content = (
        "seriesuid,coordX,coordY,coordZ,diameter_mm\nuid-a,1.0,2.0,3.0,4.0\nuid-b,4.0,3.0,2.0,5.0\n"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = _write_annotations(Path(tmp_dir), content)

        annotations = parse_luna16_annotations(csv_path, allowed_series_uids={"uid-b"})

    assert [annotation.series_uid for annotation in annotations] == ["uid-b"]
