"""Tests for performance report generation."""

import json

import numpy as np
import pytest

from src.eval.pipeline import EvaluationResults
from src.eval.report import generate_performance_report


class TestGeneratePerformanceReport:
    """Test generate_performance_report function."""

    def test_text_format(self):
        """Test text report generation."""
        results = EvaluationResults(
            froc_metrics={"average_sensitivity": 0.85, "sensitivities": [0.7, 0.8, 0.9], "fppi_values": [1, 2, 4]},
            detection_metrics={"tp": 10, "fp": 2, "fn": 1, "precision": 0.83, "recall": 0.91, "f1_score": 0.87},
            segmentation_metrics={"mean_dice": 0.78, "std_dice": 0.12, "mean_iou": 0.65, "std_iou": 0.10},
            size_metrics={"volume_mae": 15.3, "volume_rmse": 20.1, "diameter_mae": 2.1, "diameter_rmse": 2.8},
            triage_metrics={
                "pearson_correlation": 0.72,
                "spearman_correlation": 0.68,
                "auc_roc": 0.88,
                "expected_calibration_error": 0.05,
            },
            num_images=5,
            num_predictions=12,
            num_ground_truths=11,
        )

        report = generate_performance_report(results, format="text")

        # Check content
        assert "EVALUATION PERFORMANCE REPORT" in report
        assert "DATASET SUMMARY" in report
        assert "Total Images:           5" in report
        assert "Total Detections:       12" in report
        assert "FROC CURVE METRICS" in report
        assert "Average Sensitivity:    0.8500" in report
        assert "CENTER DETECTION METRICS" in report
        assert "True Positives:         10" in report
        assert "Precision:              0.8300" in report
        assert "SEGMENTATION METRICS" in report
        assert "Mean Dice Score:        0.7800" in report
        assert "SIZE ESTIMATION METRICS" in report
        assert "Volume MAE (mm³):       15.3000" in report
        assert "TRIAGE PREDICTION METRICS" in report
        assert "Pearson Correlation:    0.7200" in report

    def test_json_format(self):
        """Test JSON report generation."""
        results = EvaluationResults(
            froc_metrics={"average_sensitivity": 0.85},
            detection_metrics={"tp": 10, "fp": 2, "fn": 1, "precision": 0.83, "recall": 0.91, "f1_score": 0.87},
            segmentation_metrics=None,
            size_metrics=None,
            triage_metrics=None,
            num_images=5,
            num_predictions=12,
            num_ground_truths=11,
        )

        report = generate_performance_report(results, format="json")
        data = json.loads(report)

        # Check structure
        assert "dataset_summary" in data
        assert data["dataset_summary"]["num_images"] == 5
        assert data["dataset_summary"]["num_detections"] == 12
        assert "froc_metrics" in data
        assert data["froc_metrics"]["average_sensitivity"] == 0.85
        assert "detection_metrics" in data
        assert data["detection_metrics"]["tp"] == 10
        assert data["segmentation_metrics"] is None
        assert data["size_metrics"] is None
        assert data["triage_metrics"] is None

    def test_html_format(self):
        """Test HTML report generation."""
        results = EvaluationResults(
            froc_metrics={"average_sensitivity": 0.85},
            detection_metrics={"tp": 10, "fp": 2, "fn": 1, "precision": 0.83, "recall": 0.91, "f1_score": 0.87},
            segmentation_metrics={"mean_dice": 0.78},
            size_metrics=None,
            triage_metrics=None,
            num_images=5,
            num_predictions=12,
            num_ground_truths=11,
        )

        report = generate_performance_report(results, format="html")

        # Check HTML structure
        assert "<!DOCTYPE html>" in report
        assert "<html>" in report
        assert "<title>Evaluation Performance Report</title>" in report
        assert "<h1>Evaluation Performance Report</h1>" in report
        assert "<h2>Dataset Summary</h2>" in report
        assert "<table>" in report
        assert "Total Images" in report
        assert "5" in report
        assert "<h2>FROC Curve Metrics</h2>" in report
        assert "Average Sensitivity" in report
        assert "0.8500" in report
        assert "<h2>Center Detection Metrics</h2>" in report
        assert "True Positives" in report
        assert "10" in report
        assert "<h2>Segmentation Metrics</h2>" in report
        assert "Mean Dice Score" in report
        assert "0.7800" in report

    def test_invalid_format(self):
        """Test error handling for invalid format."""
        results = EvaluationResults(
            froc_metrics=None,
            detection_metrics=None,
            segmentation_metrics=None,
            size_metrics=None,
            triage_metrics=None,
            num_images=0,
            num_predictions=0,
            num_ground_truths=0,
        )

        with pytest.raises(ValueError, match="Unsupported format"):
            generate_performance_report(results, format="xml")

    def test_empty_results(self):
        """Test report generation with no metrics."""
        results = EvaluationResults(
            froc_metrics=None,
            detection_metrics=None,
            segmentation_metrics=None,
            size_metrics=None,
            triage_metrics=None,
            num_images=0,
            num_predictions=0,
            num_ground_truths=0,
        )

        # Text format
        text_report = generate_performance_report(results, format="text")
        assert "DATASET SUMMARY" in text_report
        assert "Total Images:           0" in text_report
        assert "FROC CURVE METRICS" not in text_report

        # JSON format
        json_report = generate_performance_report(results, format="json")
        data = json.loads(json_report)
        assert data["froc_metrics"] is None
        assert data["detection_metrics"] is None

        # HTML format
        html_report = generate_performance_report(results, format="html")
        assert "<h2>Dataset Summary</h2>" in html_report
        assert "<h2>FROC Curve Metrics</h2>" not in html_report

    def test_partial_metrics(self):
        """Test report with some metrics missing."""
        results = EvaluationResults(
            froc_metrics={"average_sensitivity": 0.85},
            detection_metrics=None,
            segmentation_metrics={"mean_dice": 0.78},
            size_metrics=None,
            triage_metrics={"auc_roc": 0.88},
            num_images=3,
            num_predictions=5,
            num_ground_truths=4,
        )

        # Text format
        text_report = generate_performance_report(results, format="text")
        assert "FROC CURVE METRICS" in text_report
        assert "CENTER DETECTION METRICS" not in text_report
        assert "SEGMENTATION METRICS" in text_report
        assert "SIZE ESTIMATION METRICS" not in text_report
        assert "TRIAGE PREDICTION METRICS" in text_report

    def test_numpy_array_conversion(self):
        """Test conversion of numpy arrays to JSON."""
        results = EvaluationResults(
            froc_metrics={
                "sensitivities": np.array([0.7, 0.8, 0.9]),
                "fppi_values": np.array([1.0, 2.0, 4.0]),
                "average_sensitivity": np.float64(0.85),
            },
            detection_metrics={"tp": 10, "fp": 2, "fn": 1},
            segmentation_metrics=None,
            size_metrics=None,
            triage_metrics=None,
            num_images=3,
            num_predictions=12,
            num_ground_truths=11,
        )

        report = generate_performance_report(results, format="json")
        data = json.loads(report)

        # Check numpy arrays converted to lists
        assert isinstance(data["froc_metrics"]["sensitivities"], list)
        assert data["froc_metrics"]["sensitivities"] == [0.7, 0.8, 0.9]
        assert isinstance(data["froc_metrics"]["average_sensitivity"], float)

    def test_froc_sensitivity_range(self):
        """Test FROC sensitivity range calculation."""
        results = EvaluationResults(
            froc_metrics={"sensitivities": [0.6, 0.7, 0.95], "fppi_values": [1, 2, 4]},
            detection_metrics=None,
            segmentation_metrics=None,
            size_metrics=None,
            triage_metrics=None,
            num_images=2,
            num_predictions=5,
            num_ground_truths=5,
        )

        text_report = generate_performance_report(results, format="text")
        assert "Sensitivity Range:      0.6000 - 0.9500" in text_report
        assert "FPPI Range:             1.0000 - 4.0000" in text_report

    def test_all_detection_metrics(self):
        """Test detection metrics with distance stats."""
        results = EvaluationResults(
            froc_metrics=None,
            detection_metrics={
                "tp": 15,
                "fp": 3,
                "fn": 2,
                "precision": 0.83,
                "recall": 0.88,
                "f1_score": 0.85,
                "mean_distance": 2.5,
                "median_distance": 2.1,
            },
            segmentation_metrics=None,
            size_metrics=None,
            triage_metrics=None,
            num_images=4,
            num_predictions=18,
            num_ground_truths=17,
        )

        text_report = generate_performance_report(results, format="text")
        assert "Mean Distance (mm):     2.5000" in text_report
        assert "Median Distance (mm):   2.1000" in text_report

    def test_html_includes_styles(self):
        """Test HTML report includes CSS styles."""
        results = EvaluationResults(
            froc_metrics=None,
            detection_metrics=None,
            segmentation_metrics=None,
            size_metrics=None,
            triage_metrics=None,
            num_images=1,
            num_predictions=2,
            num_ground_truths=2,
        )

        html_report = generate_performance_report(results, format="html")

        # Check for CSS elements
        assert "<style>" in html_report
        assert "font-family:" in html_report
        assert ".container" in html_report
        assert ".metric-value" in html_report
        assert "background-color:" in html_report

    def test_html_table_structure(self):
        """Test HTML tables have proper structure."""
        results = EvaluationResults(
            froc_metrics={"average_sensitivity": 0.85},
            detection_metrics=None,
            segmentation_metrics=None,
            size_metrics=None,
            triage_metrics=None,
            num_images=1,
            num_predictions=2,
            num_ground_truths=2,
        )

        html_report = generate_performance_report(results, format="html")

        # Check table elements
        assert "<table>" in html_report
        assert "<tr>" in html_report
        assert "<th>Metric</th>" in html_report
        assert "<th>Value</th>" in html_report
        assert "<td>" in html_report
        assert "class='metric-value'" in html_report

    def test_comprehensive_report(self):
        """Test comprehensive report with all metrics."""
        results = EvaluationResults(
            froc_metrics={
                "sensitivities": [0.6, 0.75, 0.85, 0.92],
                "fppi_values": [0.5, 1.0, 2.0, 4.0],
                "average_sensitivity": 0.78,
            },
            detection_metrics={
                "tp": 25,
                "fp": 5,
                "fn": 3,
                "precision": 0.83,
                "recall": 0.89,
                "f1_score": 0.86,
                "mean_distance": 1.8,
                "median_distance": 1.5,
            },
            segmentation_metrics={"mean_dice": 0.82, "std_dice": 0.09, "mean_iou": 0.71, "std_iou": 0.08},
            size_metrics={"volume_mae": 12.5, "volume_rmse": 18.3, "diameter_mae": 1.8, "diameter_rmse": 2.3},
            triage_metrics={
                "pearson_correlation": 0.75,
                "spearman_correlation": 0.73,
                "auc_roc": 0.91,
                "expected_calibration_error": 0.04,
            },
            num_images=10,
            num_predictions=30,
            num_ground_truths=28,
        )

        # Test all three formats
        text_report = generate_performance_report(results, format="text")
        json_report = generate_performance_report(results, format="json")
        html_report = generate_performance_report(results, format="html")

        # Text assertions
        assert "Total Images:           10" in text_report
        assert "Average Sensitivity:    0.7800" in text_report
        assert "True Positives:         25" in text_report
        assert "Mean Dice Score:        0.8200" in text_report
        assert "Volume MAE (mm³):       12.5000" in text_report
        assert "AUC-ROC:                0.9100" in text_report

        # JSON assertions
        data = json.loads(json_report)
        assert data["dataset_summary"]["num_images"] == 10
        assert data["froc_metrics"]["average_sensitivity"] == 0.78
        assert data["detection_metrics"]["tp"] == 25
        assert data["segmentation_metrics"]["mean_dice"] == 0.82
        assert data["size_metrics"]["volume_mae"] == 12.5
        assert data["triage_metrics"]["auc_roc"] == 0.91

        # HTML assertions
        assert "10" in html_report
        assert "0.7800" in html_report
        assert "25" in html_report
        assert "0.8200" in html_report
        assert "12.5000" in html_report
        assert "0.9100" in html_report
