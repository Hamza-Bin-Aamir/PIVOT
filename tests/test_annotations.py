"""Tests for annotation parsing utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.data.annotations import (
    LIDCAnnotation,
    LIDCRoi,
    LUNA16Annotation,
    group_annotations_by_series,
    parse_lidc_annotations,
    parse_luna16_annotations,
)


def _write_annotations(tmp_path: Path, content: str) -> Path:
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def _write_xml(tmp_path: Path, filename: str, content: str) -> Path:
    xml_path = tmp_path / filename
    xml_path.write_text(content, encoding="utf-8")
    return xml_path


def _minimal_lidc_xml(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<LidcReadMessage>\n"
        "  <ResponseHeader><SeriesInstanceUid>1.2.3</SeriesInstanceUid></ResponseHeader>\n"
        f"  {body}\n"
        "</LidcReadMessage>\n"
    )


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


def test_parse_lidc_annotations_parses_unblinded_nodules(tmp_path: Path) -> None:
    xml_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<LidcReadMessage>\n"
        "  <ResponseHeader>\n"
        "    <SeriesInstanceUid>1.2.3.4</SeriesInstanceUid>\n"
        "  </ResponseHeader>\n"
        "  <readingSession>\n"
        "    <servicingRadiologistID>RAD-1</servicingRadiologistID>\n"
        "    <unblindedReadNodule>\n"
        "      <noduleID>Nodule-1</noduleID>\n"
        "      <roi>\n"
        "        <imageUID>IMG-1</imageUID>\n"
        "        <inclusion>TRUE</inclusion>\n"
        "        <imageZposition>-123.4</imageZposition>\n"
        "        <edgeMap><xCoord>10</xCoord><yCoord>20</yCoord></edgeMap>\n"
        "        <edgeMap><xCoord>11</xCoord><yCoord>21</yCoord></edgeMap>\n"
        "      </roi>\n"
        "      <roi>\n"
        "        <imageUID>IMG-1-excluded</imageUID>\n"
        "        <inclusion>FALSE</inclusion>\n"
        "        <imageZposition>-123.4</imageZposition>\n"
        "        <edgeMap><xCoord>12</xCoord><yCoord>22</yCoord></edgeMap>\n"
        "      </roi>\n"
        "      <characteristics>\n"
        "        <malignancy>4</malignancy>\n"
        "        <subtlety>2</subtlety>\n"
        "      </characteristics>\n"
        "    </unblindedReadNodule>\n"
        "    <unblindedReadNodule>\n"
        "      <noduleID>Nodule-2</noduleID>\n"
        "      <roi>\n"
        "        <imageUID>IMG-2A</imageUID>\n"
        "        <inclusion>TRUE</inclusion>\n"
        "        <imageZposition>-120.0</imageZposition>\n"
        "        <edgeMap><xCoord>5</xCoord><yCoord>6</yCoord></edgeMap>\n"
        "      </roi>\n"
        "      <roi>\n"
        "        <imageUID>IMG-2B</imageUID>\n"
        "        <inclusion>TRUE</inclusion>\n"
        "        <imageZposition>-119.5</imageZposition>\n"
        "        <edgeMap><xCoord>15</xCoord><yCoord>16</yCoord></edgeMap>\n"
        "        <edgeMap><xCoord>17</xCoord><yCoord>18</yCoord></edgeMap>\n"
        "      </roi>\n"
        "    </unblindedReadNodule>\n"
        "  </readingSession>\n"
        "</LidcReadMessage>\n"
    )

    xml_path = _write_xml(tmp_path, "lidc.xml", xml_content)

    annotations = parse_lidc_annotations(xml_path)

    assert len(annotations) == 2

    first = annotations[0]
    assert isinstance(first, LIDCAnnotation)
    assert first.series_uid == "1.2.3.4"
    assert first.nodule_id == "Nodule-1"
    assert first.reading_session_id == "RAD-1"
    assert len(first.rois) == 1

    roi = first.rois[0]
    assert isinstance(roi, LIDCRoi)
    assert roi.image_uid == "IMG-1"
    assert roi.z_position == pytest.approx(-123.4)
    assert roi.xy_coords == ((10.0, 20.0), (11.0, 21.0))
    assert dict(first.characteristics) == {"malignancy": 4, "subtlety": 2}

    second = annotations[1]
    assert second.nodule_id == "Nodule-2"
    assert len(second.rois) == 2


def test_parse_lidc_annotations_supports_filters(tmp_path: Path) -> None:
    xml_content = (
        '<?xml version="1.0"?><LidcReadMessage>\n'
        "  <ResponseHeader><SeriesInstanceUid>1</SeriesInstanceUid></ResponseHeader>\n"
        "  <readingSession>\n"
        "    <unblindedReadNodule>\n"
        "      <noduleID>A</noduleID>\n"
        "      <roi><inclusion>TRUE</inclusion><imageZposition>0</imageZposition><edgeMap><xCoord>1</xCoord><yCoord>2</yCoord></edgeMap></roi>\n"
        "    </unblindedReadNodule>\n"
        "    <unblindedReadNodule>\n"
        "      <noduleID>B</noduleID>\n"
        "      <roi><inclusion>TRUE</inclusion><imageZposition>0</imageZposition><edgeMap><xCoord>3</xCoord><yCoord>4</yCoord></edgeMap></roi>\n"
        "      <roi><inclusion>TRUE</inclusion><imageZposition>1</imageZposition><edgeMap><xCoord>5</xCoord><yCoord>6</yCoord></edgeMap></roi>\n"
        "    </unblindedReadNodule>\n"
        "  </readingSession>\n"
        "</LidcReadMessage>\n"
    )

    xml_path = _write_xml(tmp_path, "lidc_filter.xml", xml_content)

    annotations = parse_lidc_annotations(xml_path, min_roi_count=2)
    assert [annotation.nodule_id for annotation in annotations] == ["B"]

    annotations = parse_lidc_annotations(xml_path, allowed_nodule_ids={"A"})
    assert [annotation.nodule_id for annotation in annotations] == ["A"]


def test_parse_lidc_annotations_raises_when_series_uid_missing(tmp_path: Path) -> None:
    xml_path = _write_xml(
        tmp_path,
        "missing_uid.xml",
        "<LidcReadMessage><readingSession /></LidcReadMessage>",
    )

    with pytest.raises(ValueError, match="SeriesInstanceUid missing"):
        parse_lidc_annotations(xml_path)


def test_parse_lidc_annotations_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Annotation file not found"):
        parse_lidc_annotations(tmp_path / "missing.xml")


def test_parse_lidc_annotations_rejects_non_positive_min_roi_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_roi_count must be positive"):
        parse_lidc_annotations(tmp_path / "whatever.xml", min_roi_count=0)


def test_parse_lidc_annotations_raises_on_parse_error(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, "invalid.xml", "<not-xml>")

    with pytest.raises(ValueError, match="Failed to parse XML"):
        parse_lidc_annotations(xml_path)


def test_parse_lidc_annotations_generates_session_identifier(tmp_path: Path) -> None:
    xml_body = (
        "<readingSession>\n"
        "  <unblindedReadNodule>\n"
        "    <noduleID>N1</noduleID>\n"
        "    <roi><inclusion>true</inclusion><imageZposition>0</imageZposition>"
        "      <edgeMap><xCoord>1</xCoord><yCoord>1</yCoord></edgeMap></roi>\n"
        "  </unblindedReadNodule>\n"
        "</readingSession>"
    )
    xml_path = _write_xml(tmp_path, "no_radiologist.xml", _minimal_lidc_xml(xml_body))

    annotations = parse_lidc_annotations(xml_path)

    assert annotations[0].reading_session_id == "session-1"


def test_parse_lidc_annotations_raises_for_missing_nodule_id(tmp_path: Path) -> None:
    xml_body = (
        "<readingSession>\n"
        "  <unblindedReadNodule>\n"
        "    <noduleID></noduleID>\n"
        "    <roi><inclusion>true</inclusion><imageZposition>0</imageZposition>"
        "      <edgeMap><xCoord>1</xCoord><yCoord>1</yCoord></edgeMap></roi>\n"
        "  </unblindedReadNodule>\n"
        "</readingSession>"
    )
    xml_path = _write_xml(tmp_path, "missing_nodule_id.xml", _minimal_lidc_xml(xml_body))

    with pytest.raises(ValueError, match="nodule without noduleID"):
        parse_lidc_annotations(xml_path)


def test_parse_lidc_annotations_raises_for_missing_z_position(tmp_path: Path) -> None:
    xml_body = (
        "<readingSession>\n"
        "  <unblindedReadNodule>\n"
        "    <noduleID>N1</noduleID>\n"
        "    <roi><inclusion>true</inclusion></roi>\n"
        "  </unblindedReadNodule>\n"
        "</readingSession>"
    )
    xml_path = _write_xml(tmp_path, "missing_z.xml", _minimal_lidc_xml(xml_body))

    with pytest.raises(ValueError, match="ROI missing imageZposition"):
        parse_lidc_annotations(xml_path)


def test_parse_lidc_annotations_raises_for_invalid_z_position(tmp_path: Path) -> None:
    xml_body = (
        "<readingSession>\n"
        "  <unblindedReadNodule>\n"
        "    <noduleID>N1</noduleID>\n"
        "    <roi><inclusion>true</inclusion><imageZposition>not-a-number</imageZposition>\n"
        "      <edgeMap><xCoord>1</xCoord><yCoord>1</yCoord></edgeMap></roi>\n"
        "  </unblindedReadNodule>\n"
        "</readingSession>"
    )
    xml_path = _write_xml(tmp_path, "invalid_z.xml", _minimal_lidc_xml(xml_body))

    with pytest.raises(ValueError, match="Invalid imageZposition"):
        parse_lidc_annotations(xml_path)


def test_parse_lidc_annotations_skips_rois_without_edges(tmp_path: Path) -> None:
    xml_body = (
        "<readingSession>\n"
        "  <unblindedReadNodule>\n"
        "    <noduleID>N1</noduleID>\n"
        "    <roi><inclusion>true</inclusion><imageZposition>0</imageZposition></roi>\n"
        "    <roi><inclusion>true</inclusion><imageZposition>1</imageZposition>\n"
        "      <edgeMap><xCoord>1</xCoord><yCoord>2</yCoord></edgeMap></roi>\n"
        "  </unblindedReadNodule>\n"
        "</readingSession>"
    )
    xml_path = _write_xml(tmp_path, "skip_empty_roi.xml", _minimal_lidc_xml(xml_body))

    annotations = parse_lidc_annotations(xml_path)

    assert len(annotations[0].rois) == 1
    assert annotations[0].rois[0].z_position == pytest.approx(1.0)


def test_parse_lidc_annotations_raises_when_edge_map_missing_coordinate(tmp_path: Path) -> None:
    xml_body = (
        "<readingSession>\n"
        "  <unblindedReadNodule>\n"
        "    <noduleID>N1</noduleID>\n"
        "    <roi><inclusion>true</inclusion><imageZposition>0</imageZposition>\n"
        "      <edgeMap><xCoord>1</xCoord></edgeMap></roi>\n"
        "  </unblindedReadNodule>\n"
        "</readingSession>"
    )
    xml_path = _write_xml(tmp_path, "missing_coord.xml", _minimal_lidc_xml(xml_body))

    with pytest.raises(ValueError, match="edgeMap missing coordinates"):
        parse_lidc_annotations(xml_path)


def test_parse_lidc_annotations_raises_for_invalid_edge_coordinate(tmp_path: Path) -> None:
    xml_body = (
        "<readingSession>\n"
        "  <unblindedReadNodule>\n"
        "    <noduleID>N1</noduleID>\n"
        "    <roi><inclusion>true</inclusion><imageZposition>0</imageZposition>\n"
        "      <edgeMap><xCoord>oops</xCoord><yCoord>2</yCoord></edgeMap></roi>\n"
        "  </unblindedReadNodule>\n"
        "</readingSession>"
    )
    xml_path = _write_xml(tmp_path, "invalid_coord.xml", _minimal_lidc_xml(xml_body))

    with pytest.raises(ValueError, match="Invalid edge coordinate"):
        parse_lidc_annotations(xml_path)


def test_parse_lidc_annotations_skips_nodules_without_valid_rois(tmp_path: Path) -> None:
    xml_body = (
        "<readingSession>\n"
        "  <unblindedReadNodule>\n"
        "    <noduleID>SkipMe</noduleID>\n"
        "    <roi><inclusion>true</inclusion><imageZposition>0</imageZposition></roi>\n"
        "  </unblindedReadNodule>\n"
        "</readingSession>"
    )
    xml_path = _write_xml(tmp_path, "skip_nodule.xml", _minimal_lidc_xml(xml_body))

    annotations = parse_lidc_annotations(xml_path)

    assert annotations == []


def test_parse_lidc_annotations_supports_image_sop_uid(tmp_path: Path) -> None:
    xml_body = (
        "<readingSession>\n"
        "  <unblindedReadNodule>\n"
        "    <noduleID>N1</noduleID>\n"
        "    <roi><inclusion>true</inclusion><imageZposition>0</imageZposition>\n"
        "      <imageSOP_UID>SOP-1</imageSOP_UID>\n"
        "      <edgeMap><xCoord>1</xCoord><yCoord>1</yCoord></edgeMap></roi>\n"
        "  </unblindedReadNodule>\n"
        "</readingSession>"
    )
    xml_path = _write_xml(tmp_path, "sop_uid.xml", _minimal_lidc_xml(xml_body))

    annotations = parse_lidc_annotations(xml_path)

    assert annotations[0].rois[0].image_uid == "SOP-1"


def test_parse_lidc_annotations_sets_image_uid_none_when_missing(tmp_path: Path) -> None:
    xml_body = (
        "<readingSession>\n"
        "  <unblindedReadNodule>\n"
        "    <noduleID>N1</noduleID>\n"
        "    <roi><inclusion>true</inclusion><imageZposition>0</imageZposition>\n"
        "      <edgeMap><xCoord>1</xCoord><yCoord>1</yCoord></edgeMap></roi>\n"
        "  </unblindedReadNodule>\n"
        "</readingSession>"
    )
    xml_path = _write_xml(tmp_path, "missing_image_uid.xml", _minimal_lidc_xml(xml_body))

    annotations = parse_lidc_annotations(xml_path)

    assert annotations[0].rois[0].image_uid is None


def test_parse_lidc_annotations_skips_blank_characteristics(tmp_path: Path) -> None:
    xml_body = (
        "<readingSession>\n"
        "  <unblindedReadNodule>\n"
        "    <noduleID>N1</noduleID>\n"
        "    <roi><inclusion>true</inclusion><imageZposition>0</imageZposition>\n"
        "      <edgeMap><xCoord>1</xCoord><yCoord>1</yCoord></edgeMap></roi>\n"
        "    <characteristics><malignancy> </malignancy><subtlety>3</subtlety></characteristics>\n"
        "  </unblindedReadNodule>\n"
        "</readingSession>"
    )
    xml_path = _write_xml(tmp_path, "blank_characteristics.xml", _minimal_lidc_xml(xml_body))

    annotations = parse_lidc_annotations(xml_path)

    assert dict(annotations[0].characteristics) == {"subtlety": 3}


def test_parse_lidc_annotations_raises_for_invalid_characteristic_value(tmp_path: Path) -> None:
    xml_body = (
        "<readingSession>\n"
        "  <unblindedReadNodule>\n"
        "    <noduleID>N1</noduleID>\n"
        "    <roi><inclusion>true</inclusion><imageZposition>0</imageZposition>\n"
        "      <edgeMap><xCoord>1</xCoord><yCoord>1</yCoord></edgeMap></roi>\n"
        "    <characteristics><malignancy>not-int</malignancy></characteristics>\n"
        "  </unblindedReadNodule>\n"
        "</readingSession>"
    )
    xml_path = _write_xml(tmp_path, "invalid_characteristic.xml", _minimal_lidc_xml(xml_body))

    with pytest.raises(ValueError, match="Invalid characteristic value"):
        parse_lidc_annotations(xml_path)
