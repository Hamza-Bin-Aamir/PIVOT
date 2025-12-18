"""Tests for clinician report generation."""

from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.viz.report import (
    ClinicalReport,
    NoduleFinding,
    generate_report_from_predictions,
    plot_findings_summary,
)


class TestNoduleFinding:
    """Tests for NoduleFinding dataclass."""

    def test_valid_finding(self):
        """Test creating valid nodule finding."""
        finding = NoduleFinding(
            nodule_id=1,
            center=(100.0, 150.0, 200.0),
            size_mm=(8.5, 9.2, 7.8),
            confidence=0.95,
            malignancy_score=0.65,
            location="Right Lung, Upper Lobe",
        )

        assert finding.nodule_id == 1
        assert finding.center == (100.0, 150.0, 200.0)
        assert finding.size_mm == (8.5, 9.2, 7.8)
        assert finding.confidence == 0.95
        assert finding.malignancy_score == 0.65
        assert finding.location == "Right Lung, Upper Lobe"

    def test_invalid_confidence(self):
        """Test finding rejects invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            NoduleFinding(
                nodule_id=1,
                center=(100.0, 150.0, 200.0),
                size_mm=(8.0, 8.0, 8.0),
                confidence=1.5,
                malignancy_score=0.5,
            )

    def test_invalid_malignancy_score(self):
        """Test finding rejects invalid malignancy score."""
        with pytest.raises(ValueError, match="Malignancy score must be in"):
            NoduleFinding(
                nodule_id=1,
                center=(100.0, 150.0, 200.0),
                size_mm=(8.0, 8.0, 8.0),
                confidence=0.9,
                malignancy_score=-0.1,
            )

    def test_invalid_size(self):
        """Test finding rejects non-positive size."""
        with pytest.raises(ValueError, match="Size must be positive"):
            NoduleFinding(
                nodule_id=1,
                center=(100.0, 150.0, 200.0),
                size_mm=(0.0, 8.0, 8.0),
                confidence=0.9,
                malignancy_score=0.5,
            )


class TestClinicalReport:
    """Tests for ClinicalReport class."""

    def test_empty_report(self):
        """Test report with no findings."""
        report = ClinicalReport(
            patient_id="P12345",
            scan_date=datetime(2024, 1, 15),
            analysis_date=datetime(2024, 1, 16),
            findings=[],
        )

        assert report.patient_id == "P12345"
        assert len(report.findings) == 0
        assert "No nodules detected" in report.recommendation

    def test_report_with_findings(self):
        """Test report with nodule findings."""
        findings = [
            NoduleFinding(
                nodule_id=1,
                center=(100.0, 150.0, 200.0),
                size_mm=(8.5, 9.2, 7.8),
                confidence=0.95,
                malignancy_score=0.65,
            ),
            NoduleFinding(
                nodule_id=2,
                center=(120.0, 160.0, 180.0),
                size_mm=(5.2, 6.1, 5.5),
                confidence=0.88,
                malignancy_score=0.32,
            ),
        ]

        report = ClinicalReport(
            patient_id="P12345",
            scan_date=datetime(2024, 1, 15),
            analysis_date=datetime(2024, 1, 16),
            findings=findings,
        )

        assert len(report.findings) == 2
        assert "nodule(s) detected" in report.recommendation.lower()

    def test_high_priority_recommendation(self):
        """Test high priority recommendation generation."""
        findings = [
            NoduleFinding(
                nodule_id=1,
                center=(100.0, 150.0, 200.0),
                size_mm=(18.0, 18.0, 18.0),
                confidence=0.95,
                malignancy_score=0.85,
            )
        ]

        report = ClinicalReport(
            patient_id="P12345",
            scan_date=datetime(2024, 1, 15),
            analysis_date=datetime(2024, 1, 16),
            findings=findings,
        )

        assert "HIGH PRIORITY" in report.recommendation
        assert "Immediate" in report.recommendation

    def test_moderate_priority_recommendation(self):
        """Test moderate priority recommendation generation."""
        findings = [
            NoduleFinding(
                nodule_id=1,
                center=(100.0, 150.0, 200.0),
                size_mm=(10.0, 10.0, 10.0),
                confidence=0.90,
                malignancy_score=0.60,
            )
        ]

        report = ClinicalReport(
            patient_id="P12345",
            scan_date=datetime(2024, 1, 15),
            analysis_date=datetime(2024, 1, 16),
            findings=findings,
        )

        assert "MODERATE PRIORITY" in report.recommendation
        assert "3-6 months" in report.recommendation

    def test_low_priority_recommendation(self):
        """Test low priority recommendation generation."""
        findings = [
            NoduleFinding(
                nodule_id=1,
                center=(100.0, 150.0, 200.0),
                size_mm=(4.0, 4.0, 4.0),
                confidence=0.85,
                malignancy_score=0.25,
            )
        ]

        report = ClinicalReport(
            patient_id="P12345",
            scan_date=datetime(2024, 1, 15),
            analysis_date=datetime(2024, 1, 16),
            findings=findings,
        )

        assert "low concern" in report.recommendation
        assert "annual" in report.recommendation.lower()

    def test_custom_recommendation(self):
        """Test report with custom recommendation."""
        findings = [
            NoduleFinding(
                nodule_id=1,
                center=(100.0, 150.0, 200.0),
                size_mm=(8.0, 8.0, 8.0),
                confidence=0.90,
                malignancy_score=0.50,
            )
        ]

        custom_rec = "Custom recommendation text"
        report = ClinicalReport(
            patient_id="P12345",
            scan_date=datetime(2024, 1, 15),
            analysis_date=datetime(2024, 1, 16),
            findings=findings,
            recommendation=custom_rec,
        )

        assert report.recommendation == custom_rec

    def test_text_report_generation(self):
        """Test text report generation."""
        findings = [
            NoduleFinding(
                nodule_id=1,
                center=(100.0, 150.0, 200.0),
                size_mm=(8.0, 8.0, 8.0),
                confidence=0.95,
                malignancy_score=0.65,
                location="Right Lung, Upper Lobe",
            )
        ]

        report = ClinicalReport(
            patient_id="P12345",
            scan_date=datetime(2024, 1, 15),
            analysis_date=datetime(2024, 1, 16),
            findings=findings,
        )

        text = report.to_text()

        assert "P12345" in text
        assert "PULMONARY NODULE DETECTION REPORT" in text
        assert "Nodule #1" in text
        assert "Right Lung, Upper Lobe" in text
        assert "RECOMMENDATION" in text

    def test_dict_conversion(self):
        """Test report to dictionary conversion."""
        findings = [
            NoduleFinding(
                nodule_id=1,
                center=(100.0, 150.0, 200.0),
                size_mm=(8.0, 8.0, 8.0),
                confidence=0.95,
                malignancy_score=0.65,
            )
        ]

        report = ClinicalReport(
            patient_id="P12345",
            scan_date=datetime(2024, 1, 15),
            analysis_date=datetime(2024, 1, 16),
            findings=findings,
        )

        report_dict = report.to_dict()

        assert report_dict["patient_id"] == "P12345"
        assert report_dict["nodule_count"] == 1
        assert len(report_dict["findings"]) == 1
        assert report_dict["findings"][0]["nodule_id"] == 1


class TestPlotFindingsSummary:
    """Tests for plot_findings_summary function."""

    def test_basic_summary_plot(self):
        """Test basic findings summary plot."""
        findings = [
            NoduleFinding(
                nodule_id=i,
                center=(100.0 + i, 150.0, 200.0),
                size_mm=(5.0 + i, 6.0 + i, 5.5 + i),
                confidence=min(0.7 + i * 0.03, 0.99),  # Cap at 0.99
                malignancy_score=min(0.3 + i * 0.06, 0.99),  # Cap at 0.99
            )
            for i in range(10)
        ]

        fig = plot_findings_summary(findings)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 5  # 4 plots + 1 colorbar
        plt.close(fig)

    def test_summary_plot_custom_title(self):
        """Test summary plot with custom title."""
        findings = [
            NoduleFinding(
                nodule_id=1,
                center=(100.0, 150.0, 200.0),
                size_mm=(8.0, 8.0, 8.0),
                confidence=0.95,
                malignancy_score=0.65,
            )
        ]

        fig = plot_findings_summary(findings, title="Custom Summary")

        assert "Custom Summary" in fig._suptitle.get_text()
        plt.close(fig)

    def test_summary_plot_empty_findings(self):
        """Test summary plot rejects empty findings."""
        with pytest.raises(ValueError, match="Cannot plot empty findings"):
            plot_findings_summary([])


class TestGenerateReportFromPredictions:
    """Tests for generate_report_from_predictions function."""

    def test_basic_report_generation(self):
        """Test basic report generation from predictions."""
        centers = np.array([[100.0, 150.0, 200.0], [120.0, 160.0, 180.0]])
        sizes = np.array([[10.0, 10.0, 10.0], [8.0, 8.0, 8.0]])
        confidences = np.array([0.95, 0.88])
        malignancy_scores = np.array([0.65, 0.42])

        report = generate_report_from_predictions(
            patient_id="P12345",
            centers=centers,
            sizes=sizes,
            confidences=confidences,
            malignancy_scores=malignancy_scores,
        )

        assert report.patient_id == "P12345"
        assert len(report.findings) == 2

    def test_report_with_custom_spacing(self):
        """Test report generation with custom voxel spacing."""
        centers = np.array([[100.0, 150.0, 200.0]])
        sizes = np.array([[10.0, 10.0, 10.0]])
        confidences = np.array([0.95])
        malignancy_scores = np.array([0.65])
        spacing = (2.0, 1.0, 1.0)

        report = generate_report_from_predictions(
            patient_id="P12345",
            centers=centers,
            sizes=sizes,
            confidences=confidences,
            malignancy_scores=malignancy_scores,
            spacing=spacing,
        )

        # Size should be scaled by spacing
        assert report.findings[0].size_mm[0] == 20.0  # 10 * 2.0
        assert report.findings[0].size_mm[1] == 10.0  # 10 * 1.0

    def test_report_sorting_by_malignancy(self):
        """Test report sorts findings by malignancy score."""
        centers = np.array([[100.0, 150.0, 200.0], [120.0, 160.0, 180.0]])
        sizes = np.array([[10.0, 10.0, 10.0], [8.0, 8.0, 8.0]])
        confidences = np.array([0.95, 0.88])
        malignancy_scores = np.array([0.42, 0.85])  # Second is higher

        report = generate_report_from_predictions(
            patient_id="P12345",
            centers=centers,
            sizes=sizes,
            confidences=confidences,
            malignancy_scores=malignancy_scores,
        )

        # Highest malignancy should be first
        assert report.findings[0].malignancy_score == 0.85
        assert report.findings[1].malignancy_score == 0.42

    def test_invalid_centers_shape(self):
        """Test report generation rejects invalid centers shape."""
        centers = np.array([100.0, 150.0, 200.0])  # Should be [N, 3]
        sizes = np.array([[10.0, 10.0, 10.0]])
        confidences = np.array([0.95])
        malignancy_scores = np.array([0.65])

        with pytest.raises(ValueError, match="Centers must be"):
            generate_report_from_predictions(
                patient_id="P12345",
                centers=centers,
                sizes=sizes,
                confidences=confidences,
                malignancy_scores=malignancy_scores,
            )

    def test_invalid_sizes_shape(self):
        """Test report generation rejects invalid sizes shape."""
        centers = np.array([[100.0, 150.0, 200.0]])
        sizes = np.array([10.0, 10.0, 10.0])  # Should be [N, 3]
        confidences = np.array([0.95])
        malignancy_scores = np.array([0.65])

        with pytest.raises(ValueError, match="Sizes must be"):
            generate_report_from_predictions(
                patient_id="P12345",
                centers=centers,
                sizes=sizes,
                confidences=confidences,
                malignancy_scores=malignancy_scores,
            )

    def test_array_length_mismatch(self):
        """Test report generation rejects mismatched array lengths."""
        centers = np.array([[100.0, 150.0, 200.0], [120.0, 160.0, 180.0]])
        sizes = np.array([[10.0, 10.0, 10.0]])  # Length 1, should be 2
        confidences = np.array([0.95, 0.88])
        malignancy_scores = np.array([0.65, 0.42])

        with pytest.raises(ValueError, match="Array length mismatch"):
            generate_report_from_predictions(
                patient_id="P12345",
                centers=centers,
                sizes=sizes,
                confidences=confidences,
                malignancy_scores=malignancy_scores,
            )
