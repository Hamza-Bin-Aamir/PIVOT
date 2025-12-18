"""Performance report generation from evaluation results.

This module provides functionality to generate comprehensive, human-readable
performance reports from evaluation results. Supports multiple output formats
including text, JSON, and HTML.

Key features:
- Formatted text reports with metric summaries
- JSON export for programmatic access
- HTML reports with tables and visualizations
- Customizable reporting sections
- Support for partial results (missing metrics handled gracefully)

Example:
    >>> from eval.pipeline import EvaluationPipeline
    >>> from eval.report import generate_performance_report
    >>>
    >>> # Run evaluation
    >>> pipeline = EvaluationPipeline()
    >>> # ... add cases ...
    >>> results = pipeline.evaluate()
    >>>
    >>> # Generate text report
    >>> report = generate_performance_report(results, format='text')
    >>> print(report)
    >>>
    >>> # Generate HTML report
    >>> html_report = generate_performance_report(results, format='html')
    >>> with open('report.html', 'w') as f:
    >>>     f.write(html_report)
"""

import json
from typing import Any, Literal

import numpy as np

from .pipeline import EvaluationResults

ReportFormat = Literal["text", "json", "html"]


def generate_performance_report(
    results: EvaluationResults,
    format: ReportFormat = "text",  # noqa: A002
) -> str:
    """Generate a performance report from evaluation results.

    Args:
        results: Evaluation results from pipeline
        format: Output format ('text', 'json', or 'html')

    Returns:
        Formatted report string

    Raises:
        ValueError: If format is not supported
    """
    if format == "text":
        return _generate_text_report(results)
    elif format == "json":
        return _generate_json_report(results)
    elif format == "html":
        return _generate_html_report(results)
    else:
        raise ValueError(f"Unsupported format: {format}. Must be 'text', 'json', or 'html'")


def _generate_text_report(results: EvaluationResults) -> str:
    """Generate a text-formatted report.

    Args:
        results: EvaluationResults from pipeline evaluation

    Returns:
        Text report with sections for each metric
    """
    sections = []

    # Header
    sections.append("=" * 80)
    sections.append("EVALUATION PERFORMANCE REPORT")
    sections.append("=" * 80)
    sections.append("")

    # Dataset summary
    sections.append(_format_dataset_summary(results))
    sections.append("")

    # FROC metrics
    if results.froc_metrics:
        sections.append(_format_froc_metrics(results.froc_metrics))
        sections.append("")

    # Detection metrics
    if results.detection_metrics:
        sections.append(_format_detection_metrics(results.detection_metrics))
        sections.append("")

    # Segmentation metrics
    if results.segmentation_metrics:
        sections.append(_format_segmentation_metrics(results.segmentation_metrics))
        sections.append("")

    # Size estimation metrics
    if results.size_metrics:
        sections.append(_format_size_metrics(results.size_metrics))
        sections.append("")

    # Triage metrics
    if results.triage_metrics:
        sections.append(_format_triage_metrics(results.triage_metrics))
        sections.append("")

    sections.append("=" * 80)

    return "\n".join(sections)


def _format_dataset_summary(results: EvaluationResults) -> str:
    """Format dataset summary section."""
    lines = []
    lines.append("DATASET SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total Images:           {results.num_images}")
    lines.append(f"Total Detections:       {results.num_predictions}")
    lines.append(f"Total Ground Truths:    {results.num_ground_truths}")
    return "\n".join(lines)


def _format_froc_metrics(froc_metrics: dict) -> str:
    """Format FROC curve metrics section."""
    lines = []
    lines.append("FROC CURVE METRICS")
    lines.append("-" * 80)

    if "sensitivities" in froc_metrics and froc_metrics["sensitivities"] is not None:
        sens = np.array(froc_metrics["sensitivities"])
        lines.append(f"Sensitivity Range:      {sens.min():.4f} - {sens.max():.4f}")

    if "fppi_values" in froc_metrics and froc_metrics["fppi_values"] is not None:
        fppi = np.array(froc_metrics["fppi_values"])
        lines.append(f"FPPI Range:             {fppi.min():.4f} - {fppi.max():.4f}")

    if "average_sensitivity" in froc_metrics and froc_metrics["average_sensitivity"] is not None:
        lines.append(f"Average Sensitivity:    {froc_metrics['average_sensitivity']:.4f}")

    return "\n".join(lines)


def _format_detection_metrics(detection_metrics: dict) -> str:
    """Format center detection metrics section."""
    lines = []
    lines.append("CENTER DETECTION METRICS")
    lines.append("-" * 80)

    lines.append(f"True Positives:         {detection_metrics['tp']}")
    lines.append(f"False Positives:        {detection_metrics['fp']}")
    lines.append(f"False Negatives:        {detection_metrics['fn']}")

    if detection_metrics["precision"] is not None:
        lines.append(f"Precision:              {detection_metrics['precision']:.4f}")

    if detection_metrics["recall"] is not None:
        lines.append(f"Recall:                 {detection_metrics['recall']:.4f}")

    if detection_metrics["f1_score"] is not None:
        lines.append(f"F1 Score:               {detection_metrics['f1_score']:.4f}")

    if "mean_distance" in detection_metrics and detection_metrics["mean_distance"] is not None:
        lines.append(f"Mean Distance (mm):     {detection_metrics['mean_distance']:.4f}")

    if "median_distance" in detection_metrics and detection_metrics["median_distance"] is not None:
        lines.append(f"Median Distance (mm):   {detection_metrics['median_distance']:.4f}")

    return "\n".join(lines)


def _format_segmentation_metrics(segmentation_metrics: dict) -> str:
    """Format segmentation metrics section."""
    lines = []
    lines.append("SEGMENTATION METRICS")
    lines.append("-" * 80)

    if "mean_dice" in segmentation_metrics and segmentation_metrics["mean_dice"] is not None:
        lines.append(f"Mean Dice Score:        {segmentation_metrics['mean_dice']:.4f}")

    if "std_dice" in segmentation_metrics and segmentation_metrics["std_dice"] is not None:
        lines.append(f"Std Dice Score:         {segmentation_metrics['std_dice']:.4f}")

    if "mean_iou" in segmentation_metrics and segmentation_metrics["mean_iou"] is not None:
        lines.append(f"Mean IoU:               {segmentation_metrics['mean_iou']:.4f}")

    if "std_iou" in segmentation_metrics and segmentation_metrics["std_iou"] is not None:
        lines.append(f"Std IoU:                {segmentation_metrics['std_iou']:.4f}")

    return "\n".join(lines)


def _format_size_metrics(size_metrics: dict) -> str:
    """Format size estimation metrics section."""
    lines = []
    lines.append("SIZE ESTIMATION METRICS")
    lines.append("-" * 80)

    if "volume_mae" in size_metrics and size_metrics["volume_mae"] is not None:
        lines.append(f"Volume MAE (mm³):       {size_metrics['volume_mae']:.4f}")

    if "volume_rmse" in size_metrics and size_metrics["volume_rmse"] is not None:
        lines.append(f"Volume RMSE (mm³):      {size_metrics['volume_rmse']:.4f}")

    if "diameter_mae" in size_metrics and size_metrics["diameter_mae"] is not None:
        lines.append(f"Diameter MAE (mm):      {size_metrics['diameter_mae']:.4f}")

    if "diameter_rmse" in size_metrics and size_metrics["diameter_rmse"] is not None:
        lines.append(f"Diameter RMSE (mm):     {size_metrics['diameter_rmse']:.4f}")

    return "\n".join(lines)


def _format_triage_metrics(triage_metrics: dict) -> str:
    """Format triage prediction metrics section."""
    lines = []
    lines.append("TRIAGE PREDICTION METRICS")
    lines.append("-" * 80)

    if "pearson_correlation" in triage_metrics and triage_metrics["pearson_correlation"] is not None:
        lines.append(f"Pearson Correlation:    {triage_metrics['pearson_correlation']:.4f}")

    if "spearman_correlation" in triage_metrics and triage_metrics["spearman_correlation"] is not None:
        lines.append(f"Spearman Correlation:   {triage_metrics['spearman_correlation']:.4f}")

    if "auc_roc" in triage_metrics and triage_metrics["auc_roc"] is not None:
        lines.append(f"AUC-ROC:                {triage_metrics['auc_roc']:.4f}")

    if "expected_calibration_error" in triage_metrics and triage_metrics["expected_calibration_error"] is not None:
        lines.append(f"Expected Cal. Error:    {triage_metrics['expected_calibration_error']:.4f}")

    return "\n".join(lines)


def _generate_json_report(results: EvaluationResults) -> str:
    """Generate a JSON-formatted report.

    Args:
        results: EvaluationResults from pipeline evaluation

    Returns:
        JSON string with all metrics
    """
    # Convert to dict, handling numpy arrays
    report_dict = {
        "dataset_summary": {
            "num_images": results.num_images,
            "num_detections": results.num_predictions,
            "num_ground_truths": results.num_ground_truths,
        },
        "froc_metrics": _convert_to_json_serializable(results.froc_metrics) if results.froc_metrics else None,
        "detection_metrics": _convert_to_json_serializable(results.detection_metrics)
        if results.detection_metrics
        else None,
        "segmentation_metrics": _convert_to_json_serializable(results.segmentation_metrics)
        if results.segmentation_metrics
        else None,
        "size_metrics": _convert_to_json_serializable(results.size_metrics) if results.size_metrics else None,
        "triage_metrics": _convert_to_json_serializable(results.triage_metrics) if results.triage_metrics else None,
    }

    return json.dumps(report_dict, indent=2)


def _convert_to_json_serializable(obj: dict | list | np.ndarray | float | int | None) -> dict | list | float | int | None:  # noqa: E501
    """Convert numpy arrays and other non-serializable types to JSON-compatible types."""
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}  # type: ignore[no-any-return]
    elif isinstance(obj, list):
        result: list[Any] = [_convert_to_json_serializable(v) for v in obj]
        return result
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # type: ignore[return-value,no-any-return]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj  # type: ignore[no-any-return]


def _generate_html_report(results: EvaluationResults) -> str:
    """Generate an HTML-formatted report.

    Args:
        results: EvaluationResults from pipeline evaluation
        include_plots: Whether to include plot data (not implemented yet)

    Returns:
        HTML string with formatted tables
    """
    html_parts = []

    # HTML header
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html>")
    html_parts.append("<head>")
    html_parts.append("<meta charset='UTF-8'>")
    html_parts.append("<title>Evaluation Performance Report</title>")
    html_parts.append(_get_html_styles())
    html_parts.append("</head>")
    html_parts.append("<body>")

    # Main container
    html_parts.append("<div class='container'>")
    html_parts.append("<h1>Evaluation Performance Report</h1>")

    # Dataset summary
    html_parts.append(_format_html_dataset_summary(results))

    # FROC metrics
    if results.froc_metrics:
        html_parts.append(_format_html_froc_metrics(results.froc_metrics))

    # Detection metrics
    if results.detection_metrics:
        html_parts.append(_format_html_detection_metrics(results.detection_metrics))

    # Segmentation metrics
    if results.segmentation_metrics:
        html_parts.append(_format_html_segmentation_metrics(results.segmentation_metrics))

    # Size metrics
    if results.size_metrics:
        html_parts.append(_format_html_size_metrics(results.size_metrics))

    # Triage metrics
    if results.triage_metrics:
        html_parts.append(_format_html_triage_metrics(results.triage_metrics))

    html_parts.append("</div>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    return "\n".join(html_parts)


def _get_html_styles() -> str:
    """Get CSS styles for HTML report."""
    return """
<style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f5f5f5;
        margin: 0;
        padding: 20px;
    }
    .container {
        max-width: 1000px;
        margin: 0 auto;
        background-color: white;
        padding: 30px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #333;
        border-bottom: 3px solid #4CAF50;
        padding-bottom: 10px;
    }
    h2 {
        color: #555;
        margin-top: 30px;
        border-bottom: 2px solid #ddd;
        padding-bottom: 8px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    th, td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    th {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    tr:hover {
        background-color: #f5f5f5;
    }
    .metric-value {
        font-weight: bold;
        color: #2196F3;
    }
</style>
"""


def _format_html_dataset_summary(results: EvaluationResults) -> str:
    """Format dataset summary as HTML table."""
    html = []
    html.append("<h2>Dataset Summary</h2>")
    html.append("<table>")
    html.append("<tr><th>Metric</th><th>Value</th></tr>")
    html.append(f"<tr><td>Total Images</td><td class='metric-value'>{results.num_images}</td></tr>")
    html.append(
        f"<tr><td>Total Detections</td><td class='metric-value'>{results.num_predictions}</td></tr>"
    )
    html.append(
        f"<tr><td>Total Ground Truths</td><td class='metric-value'>{results.num_ground_truths}</td></tr>"
    )
    html.append("</table>")
    return "\n".join(html)


def _format_html_froc_metrics(froc_metrics: dict) -> str:
    """Format FROC metrics as HTML table."""
    html = []
    html.append("<h2>FROC Curve Metrics</h2>")
    html.append("<table>")
    html.append("<tr><th>Metric</th><th>Value</th></tr>")

    if "average_sensitivity" in froc_metrics and froc_metrics["average_sensitivity"] is not None:
        html.append(
            f"<tr><td>Average Sensitivity</td><td class='metric-value'>{froc_metrics['average_sensitivity']:.4f}</td></tr>"
        )

    if "sensitivities" in froc_metrics and froc_metrics["sensitivities"] is not None:
        sens = np.array(froc_metrics["sensitivities"])
        html.append(
            f"<tr><td>Sensitivity Range</td><td class='metric-value'>{sens.min():.4f} - {sens.max():.4f}</td></tr>"
        )

    html.append("</table>")
    return "\n".join(html)


def _format_html_detection_metrics(detection_metrics: dict) -> str:
    """Format detection metrics as HTML table."""
    html = []
    html.append("<h2>Center Detection Metrics</h2>")
    html.append("<table>")
    html.append("<tr><th>Metric</th><th>Value</th></tr>")

    html.append(f"<tr><td>True Positives</td><td class='metric-value'>{detection_metrics['tp']}</td></tr>")
    html.append(f"<tr><td>False Positives</td><td class='metric-value'>{detection_metrics['fp']}</td></tr>")
    html.append(f"<tr><td>False Negatives</td><td class='metric-value'>{detection_metrics['fn']}</td></tr>")

    if detection_metrics["precision"] is not None:
        html.append(
            f"<tr><td>Precision</td><td class='metric-value'>{detection_metrics['precision']:.4f}</td></tr>"
        )

    if detection_metrics["recall"] is not None:
        html.append(f"<tr><td>Recall</td><td class='metric-value'>{detection_metrics['recall']:.4f}</td></tr>")

    if detection_metrics["f1_score"] is not None:
        html.append(
            f"<tr><td>F1 Score</td><td class='metric-value'>{detection_metrics['f1_score']:.4f}</td></tr>"
        )

    html.append("</table>")
    return "\n".join(html)


def _format_html_segmentation_metrics(segmentation_metrics: dict) -> str:
    """Format segmentation metrics as HTML table."""
    html = []
    html.append("<h2>Segmentation Metrics</h2>")
    html.append("<table>")
    html.append("<tr><th>Metric</th><th>Value</th></tr>")

    if "mean_dice" in segmentation_metrics and segmentation_metrics["mean_dice"] is not None:
        html.append(
            f"<tr><td>Mean Dice Score</td><td class='metric-value'>{segmentation_metrics['mean_dice']:.4f}</td></tr>"
        )

    if "std_dice" in segmentation_metrics and segmentation_metrics["std_dice"] is not None:
        html.append(
            f"<tr><td>Std Dice Score</td><td class='metric-value'>{segmentation_metrics['std_dice']:.4f}</td></tr>"
        )

    if "mean_iou" in segmentation_metrics and segmentation_metrics["mean_iou"] is not None:
        html.append(
            f"<tr><td>Mean IoU</td><td class='metric-value'>{segmentation_metrics['mean_iou']:.4f}</td></tr>"
        )

    if "std_iou" in segmentation_metrics and segmentation_metrics["std_iou"] is not None:
        html.append(
            f"<tr><td>Std IoU</td><td class='metric-value'>{segmentation_metrics['std_iou']:.4f}</td></tr>"
        )

    html.append("</table>")
    return "\n".join(html)


def _format_html_size_metrics(size_metrics: dict) -> str:
    """Format size metrics as HTML table."""
    html = []
    html.append("<h2>Size Estimation Metrics</h2>")
    html.append("<table>")
    html.append("<tr><th>Metric</th><th>Value</th></tr>")

    if "volume_mae" in size_metrics and size_metrics["volume_mae"] is not None:
        html.append(
            f"<tr><td>Volume MAE (mm³)</td><td class='metric-value'>{size_metrics['volume_mae']:.4f}</td></tr>"
        )

    if "volume_rmse" in size_metrics and size_metrics["volume_rmse"] is not None:
        html.append(
            f"<tr><td>Volume RMSE (mm³)</td><td class='metric-value'>{size_metrics['volume_rmse']:.4f}</td></tr>"
        )

    if "diameter_mae" in size_metrics and size_metrics["diameter_mae"] is not None:
        html.append(
            f"<tr><td>Diameter MAE (mm)</td><td class='metric-value'>{size_metrics['diameter_mae']:.4f}</td></tr>"
        )

    if "diameter_rmse" in size_metrics and size_metrics["diameter_rmse"] is not None:
        html.append(
            f"<tr><td>Diameter RMSE (mm)</td><td class='metric-value'>{size_metrics['diameter_rmse']:.4f}</td></tr>"
        )

    html.append("</table>")
    return "\n".join(html)


def _format_html_triage_metrics(triage_metrics: dict) -> str:
    """Format triage metrics as HTML table."""
    html = []
    html.append("<h2>Triage Prediction Metrics</h2>")
    html.append("<table>")
    html.append("<tr><th>Metric</th><th>Value</th></tr>")

    if "pearson_correlation" in triage_metrics and triage_metrics["pearson_correlation"] is not None:
        html.append(
            f"<tr><td>Pearson Correlation</td><td class='metric-value'>{triage_metrics['pearson_correlation']:.4f}</td></tr>"
        )

    if "spearman_correlation" in triage_metrics and triage_metrics["spearman_correlation"] is not None:
        html.append(
            f"<tr><td>Spearman Correlation</td><td class='metric-value'>{triage_metrics['spearman_correlation']:.4f}</td></tr>"
        )

    if "auc_roc" in triage_metrics and triage_metrics["auc_roc"] is not None:
        html.append(
            f"<tr><td>AUC-ROC</td><td class='metric-value'>{triage_metrics['auc_roc']:.4f}</td></tr>"
        )

    if "expected_calibration_error" in triage_metrics and triage_metrics["expected_calibration_error"] is not None:
        html.append(
            f"<tr><td>Expected Cal. Error</td><td class='metric-value'>{triage_metrics['expected_calibration_error']:.4f}</td></tr>"
        )

    html.append("</table>")
    return "\n".join(html)
