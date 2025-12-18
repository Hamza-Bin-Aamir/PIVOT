"""Unit tests for structured output formatter module."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.inference.output_formatter import (
    InferenceMetadata,
    NoduleDetection,
    StructuredInferenceOutput,
    StructuredOutputFormatter,
)


class TestNoduleDetection:
    """Tests for NoduleDetection dataclass."""

    def test_valid_detection(self) -> None:
        """Test creating a valid nodule detection."""
        detection = NoduleDetection(
            center=(10.5, 20.3, 30.1),
            confidence=0.95,
            triage_score=0.8,
            diameter_mm=12.5,
            volume_mm3=1021.0,
            bbox_min=(5, 15, 25),
            bbox_max=(15, 25, 35),
        )

        assert detection.center == (10.5, 20.3, 30.1)
        assert detection.confidence == 0.95
        assert detection.triage_score == 0.8
        assert detection.diameter_mm == 12.5
        assert detection.properties == {}

    def test_detection_with_properties(self) -> None:
        """Test detection with additional properties."""
        props = {"spiculation": True, "calcification": False}
        detection = NoduleDetection(
            center=(10.0, 20.0, 30.0),
            confidence=0.9,
            triage_score=0.7,
            diameter_mm=10.0,
            volume_mm3=523.0,
            bbox_min=(5, 15, 25),
            bbox_max=(15, 25, 35),
            properties=props,
        )

        assert detection.properties == props

    def test_invalid_confidence_raises_error(self) -> None:
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="Confidence must be"):
            NoduleDetection(
                center=(10.0, 20.0, 30.0),
                confidence=1.5,
                triage_score=0.7,
                diameter_mm=10.0,
                volume_mm3=523.0,
                bbox_min=(5, 15, 25),
                bbox_max=(15, 25, 35),
            )

    def test_invalid_triage_score_raises_error(self) -> None:
        """Test that invalid triage score raises ValueError."""
        with pytest.raises(ValueError, match="Triage score must be"):
            NoduleDetection(
                center=(10.0, 20.0, 30.0),
                confidence=0.9,
                triage_score=-0.1,
                diameter_mm=10.0,
                volume_mm3=523.0,
                bbox_min=(5, 15, 25),
                bbox_max=(15, 25, 35),
            )

    def test_negative_diameter_raises_error(self) -> None:
        """Test that negative diameter raises ValueError."""
        with pytest.raises(ValueError, match="Diameter must be"):
            NoduleDetection(
                center=(10.0, 20.0, 30.0),
                confidence=0.9,
                triage_score=0.7,
                diameter_mm=-5.0,
                volume_mm3=523.0,
                bbox_min=(5, 15, 25),
                bbox_max=(15, 25, 35),
            )

    def test_negative_volume_raises_error(self) -> None:
        """Test that negative volume raises ValueError."""
        with pytest.raises(ValueError, match="Volume must be"):
            NoduleDetection(
                center=(10.0, 20.0, 30.0),
                confidence=0.9,
                triage_score=0.7,
                diameter_mm=10.0,
                volume_mm3=-100.0,
                bbox_min=(5, 15, 25),
                bbox_max=(15, 25, 35),
            )


class TestStructuredInferenceOutput:
    """Tests for StructuredInferenceOutput dataclass."""

    def test_output_with_summary_calculation(self) -> None:
        """Test that summary statistics are calculated automatically."""
        metadata = InferenceMetadata(
            scan_id="SCAN001",
            timestamp="2024-01-01T00:00:00Z",
            model_version="v1.0.0",
            spacing=(1.0, 1.0, 1.0),
            shape=(128, 256, 256),
            processing_time_ms=100.0,
        )

        detections = [
            NoduleDetection(
                center=(10.0, 20.0, 30.0),
                confidence=0.9,
                triage_score=0.8,
                diameter_mm=10.0,
                volume_mm3=523.0,
                bbox_min=(5, 15, 25),
                bbox_max=(15, 25, 35),
            ),
            NoduleDetection(
                center=(50.0, 60.0, 70.0),
                confidence=0.7,
                triage_score=0.5,
                diameter_mm=8.0,
                volume_mm3=268.0,
                bbox_min=(45, 55, 65),
                bbox_max=(55, 65, 75),
            ),
        ]

        output = StructuredInferenceOutput(metadata=metadata, detections=detections)

        assert output.summary["num_detections"] == 2
        assert output.summary["max_triage_score"] == 0.8
        assert output.summary["high_risk_count"] == 1
        assert output.summary["medium_risk_count"] == 1
        assert output.summary["low_risk_count"] == 0

    def test_empty_detections_summary(self) -> None:
        """Test summary with no detections."""
        metadata = InferenceMetadata(
            scan_id="SCAN001",
            timestamp="2024-01-01T00:00:00Z",
            model_version="v1.0.0",
            spacing=(1.0, 1.0, 1.0),
            shape=(128, 256, 256),
            processing_time_ms=100.0,
        )

        output = StructuredInferenceOutput(metadata=metadata, detections=[])

        assert output.summary["num_detections"] == 0
        assert output.summary["max_triage_score"] == 0.0


class TestStructuredOutputFormatterInit:
    """Tests for StructuredOutputFormatter initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        formatter = StructuredOutputFormatter()

        assert formatter.model_version == "unknown"
        assert formatter.default_spacing == (1.0, 1.0, 1.0)
        assert formatter.min_confidence == 0.0
        assert formatter.max_detections is None

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        formatter = StructuredOutputFormatter(
            model_version="v2.1.0",
            spacing=(0.5, 0.5, 2.0),
            min_confidence=0.5,
            max_detections=10,
        )

        assert formatter.model_version == "v2.1.0"
        assert formatter.default_spacing == (0.5, 0.5, 2.0)
        assert formatter.min_confidence == 0.5
        assert formatter.max_detections == 10

    def test_invalid_min_confidence_raises_error(self) -> None:
        """Test that invalid min_confidence raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence must be"):
            StructuredOutputFormatter(min_confidence=1.5)

    def test_negative_max_detections_raises_error(self) -> None:
        """Test that negative max_detections raises ValueError."""
        with pytest.raises(ValueError, match="max_detections must be"):
            StructuredOutputFormatter(max_detections=-1)


class TestStructuredOutputFormatterFormat:
    """Tests for format method."""

    def test_format_basic(self) -> None:
        """Test basic formatting with minimal inputs."""
        formatter = StructuredOutputFormatter(model_version="v1.0.0")

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.9],
            triage_scores=[0.8],
            diameters=[10.0],
            scan_id="SCAN001",
            spacing=(1.0, 1.0, 1.0),
            shape=(128, 256, 256),
        )

        assert len(output.detections) == 1
        assert output.metadata.scan_id == "SCAN001"
        assert output.metadata.model_version == "v1.0.0"
        assert output.detections[0].center == (10.0, 20.0, 30.0)
        assert output.detections[0].confidence == 0.9

    def test_format_with_numpy_arrays(self) -> None:
        """Test formatting with numpy array inputs."""
        formatter = StructuredOutputFormatter()

        centers = np.array([[10.0, 20.0, 30.0], [50.0, 60.0, 70.0]])
        confidences = np.array([0.9, 0.7])
        triage_scores = np.array([0.8, 0.5])
        diameters = np.array([10.0, 8.0])

        output = formatter.format(
            centers=centers,
            confidences=confidences,
            triage_scores=triage_scores,
            diameters=diameters,
            scan_id="SCAN002",
            shape=(128, 256, 256),
        )

        assert len(output.detections) == 2

    def test_format_filters_by_confidence(self) -> None:
        """Test that low-confidence detections are filtered out."""
        formatter = StructuredOutputFormatter(min_confidence=0.7)

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0), (50.0, 60.0, 70.0)],
            confidences=[0.9, 0.5],
            triage_scores=[0.8, 0.6],
            diameters=[10.0, 8.0],
            scan_id="SCAN003",
        )

        assert len(output.detections) == 1
        assert output.detections[0].confidence == 0.9

    def test_format_limits_max_detections(self) -> None:
        """Test that max_detections limits output."""
        formatter = StructuredOutputFormatter(max_detections=2)

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0), (50.0, 60.0, 70.0), (90.0, 100.0, 110.0)],
            confidences=[0.9, 0.8, 0.7],
            triage_scores=[0.8, 0.7, 0.6],
            diameters=[10.0, 9.0, 8.0],
            scan_id="SCAN004",
        )

        assert len(output.detections) == 2
        # Should keep highest confidence detections
        assert output.detections[0].confidence == 0.9
        assert output.detections[1].confidence == 0.8

    def test_format_sorts_by_confidence(self) -> None:
        """Test that detections are sorted by confidence."""
        formatter = StructuredOutputFormatter()

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0), (50.0, 60.0, 70.0)],
            confidences=[0.7, 0.9],
            triage_scores=[0.6, 0.8],
            diameters=[8.0, 10.0],
            scan_id="SCAN005",
        )

        assert output.detections[0].confidence == 0.9
        assert output.detections[1].confidence == 0.7

    def test_format_with_mismatched_lengths_raises_error(self) -> None:
        """Test that mismatched input lengths raise ValueError."""
        formatter = StructuredOutputFormatter()

        with pytest.raises(ValueError, match="same length"):
            formatter.format(
                centers=[(10.0, 20.0, 30.0)],
                confidences=[0.9, 0.8],
                triage_scores=[0.8],
                diameters=[10.0],
                scan_id="SCAN006",
            )

    def test_format_with_bounding_boxes(self) -> None:
        """Test formatting with provided bounding boxes."""
        formatter = StructuredOutputFormatter()

        bboxes = [((5, 15, 25), (15, 25, 35))]

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.9],
            triage_scores=[0.8],
            diameters=[10.0],
            scan_id="SCAN007",
            bounding_boxes=bboxes,
        )

        assert output.detections[0].bbox_min == (5, 15, 25)
        assert output.detections[0].bbox_max == (15, 25, 35)

    def test_format_with_properties(self) -> None:
        """Test formatting with detection properties."""
        formatter = StructuredOutputFormatter()

        props = [{"spiculation": True}]

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.9],
            triage_scores=[0.8],
            diameters=[10.0],
            scan_id="SCAN008",
            properties=props,
        )

        assert output.detections[0].properties == {"spiculation": True}

    def test_volume_estimation(self) -> None:
        """Test that volume is estimated from diameter."""
        formatter = StructuredOutputFormatter()

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.9],
            triage_scores=[0.8],
            diameters=[10.0],
            scan_id="SCAN009",
        )

        # Volume of sphere with diameter 10mm
        expected_volume = (4.0 / 3.0) * np.pi * (5.0**3)
        assert np.isclose(output.detections[0].volume_mm3, expected_volume)


class TestStructuredOutputFormatterConversion:
    """Tests for output conversion methods."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        formatter = StructuredOutputFormatter(model_version="v1.0.0")

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.9],
            triage_scores=[0.8],
            diameters=[10.0],
            scan_id="SCAN010",
            spacing=(1.0, 1.0, 1.0),
            shape=(128, 256, 256),
        )

        result = formatter.to_dict(output)

        assert isinstance(result, dict)
        assert "metadata" in result
        assert "detections" in result
        assert result["metadata"]["scan_id"] == "SCAN010"

    def test_to_json(self) -> None:
        """Test conversion to JSON string."""
        formatter = StructuredOutputFormatter(model_version="v1.0.0")

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.9],
            triage_scores=[0.8],
            diameters=[10.0],
            scan_id="SCAN011",
        )

        json_str = formatter.to_json(output)

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["metadata"]["scan_id"] == "SCAN011"

    def test_to_json_with_indent(self) -> None:
        """Test JSON formatting with custom indentation."""
        formatter = StructuredOutputFormatter()

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.9],
            triage_scores=[0.8],
            diameters=[10.0],
            scan_id="SCAN012",
        )

        json_str = formatter.to_json(output, indent=4)

        assert "    " in json_str  # Check for 4-space indentation


class TestStructuredOutputFormatterSave:
    """Tests for save method."""

    def test_save_as_json(self, tmp_path: Path) -> None:
        """Test saving output as JSON file."""
        formatter = StructuredOutputFormatter()

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.9],
            triage_scores=[0.8],
            diameters=[10.0],
            scan_id="SCAN013",
        )

        filepath = tmp_path / "output.json"
        formatter.save(output, filepath, format="json")

        assert filepath.exists()
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded["metadata"]["scan_id"] == "SCAN013"

    def test_save_as_pickle(self, tmp_path: Path) -> None:
        """Test saving output as pickle file."""
        formatter = StructuredOutputFormatter()

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.9],
            triage_scores=[0.8],
            diameters=[10.0],
            scan_id="SCAN014",
        )

        filepath = tmp_path / "output.pkl"
        formatter.save(output, filepath, format="dict")

        assert filepath.exists()
        with open(filepath, "rb") as f:
            loaded = pickle.load(f)
        assert loaded["metadata"]["scan_id"] == "SCAN014"

    def test_save_creates_directories(self, tmp_path: Path) -> None:
        """Test that save creates parent directories."""
        formatter = StructuredOutputFormatter()

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.9],
            triage_scores=[0.8],
            diameters=[10.0],
            scan_id="SCAN015",
        )

        filepath = tmp_path / "nested" / "dir" / "output.json"
        formatter.save(output, filepath, format="json")

        assert filepath.exists()

    def test_save_with_invalid_format_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid format raises ValueError."""
        formatter = StructuredOutputFormatter()

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.9],
            triage_scores=[0.8],
            diameters=[10.0],
            scan_id="SCAN016",
        )

        filepath = tmp_path / "output.txt"

        with pytest.raises(ValueError, match="Unsupported format"):
            formatter.save(output, filepath, format="invalid")  # type: ignore[arg-type]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_detections_after_filtering(self) -> None:
        """Test handling when all detections are filtered out."""
        formatter = StructuredOutputFormatter(min_confidence=0.99)

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.5],
            triage_scores=[0.8],
            diameters=[10.0],
            scan_id="SCAN017",
        )

        assert len(output.detections) == 0
        assert output.summary["num_detections"] == 0

    def test_single_detection(self) -> None:
        """Test with single detection."""
        formatter = StructuredOutputFormatter()

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.9],
            triage_scores=[0.8],
            diameters=[10.0],
            scan_id="SCAN018",
        )

        assert len(output.detections) == 1

    def test_many_detections(self) -> None:
        """Test with many detections."""
        formatter = StructuredOutputFormatter()

        n = 100
        output = formatter.format(
            centers=[(float(i), float(i), float(i)) for i in range(n)],
            confidences=[0.9] * n,
            triage_scores=[0.8] * n,
            diameters=[10.0] * n,
            scan_id="SCAN019",
        )

        assert len(output.detections) == n

    def test_zero_diameter(self) -> None:
        """Test with zero diameter."""
        formatter = StructuredOutputFormatter()

        output = formatter.format(
            centers=[(10.0, 20.0, 30.0)],
            confidences=[0.9],
            triage_scores=[0.8],
            diameters=[0.0],
            scan_id="SCAN020",
        )

        assert output.detections[0].diameter_mm == 0.0
        assert output.detections[0].volume_mm3 == 0.0
