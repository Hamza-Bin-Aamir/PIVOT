"""Clinician report generation for nodule detection results.

This module provides tools to generate comprehensive clinical reports
summarizing nodule detection, classification, and recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


@dataclass
class NoduleFinding:
    """Single nodule finding for clinical report.

    Attributes:
        nodule_id: Unique identifier for the nodule
        center: Center coordinates [z, y, x] in voxels
        size_mm: Estimated diameter in millimeters [z, y, x]
        confidence: Detection confidence score [0, 1]
        malignancy_score: Triage score [0, 1] indicating malignancy risk
        location: Anatomical location description
    """

    nodule_id: int
    center: tuple[float, float, float]
    size_mm: tuple[float, float, float]
    confidence: float
    malignancy_score: float
    location: str = "Unknown"

    def __post_init__(self) -> None:
        """Validate finding attributes."""
        if not 0 <= self.confidence <= 1:
            msg = f"Confidence must be in [0, 1], got {self.confidence}"
            raise ValueError(msg)
        if not 0 <= self.malignancy_score <= 1:
            msg = f"Malignancy score must be in [0, 1], got {self.malignancy_score}"
            raise ValueError(msg)
        if any(s <= 0 for s in self.size_mm):
            msg = f"Size must be positive, got {self.size_mm}"
            raise ValueError(msg)


@dataclass
class ClinicalReport:
    """Clinical report for nodule detection analysis.

    Attributes:
        patient_id: Patient identifier
        scan_date: Date of CT scan
        analysis_date: Date of AI analysis
        findings: List of nodule findings
        recommendation: Clinical recommendation text
        model_version: AI model version string
    """

    patient_id: str
    scan_date: datetime
    analysis_date: datetime
    findings: list[NoduleFinding]
    recommendation: str = ""
    model_version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Generate recommendation if not provided."""
        if not self.recommendation:
            self.recommendation = self._generate_recommendation()

    def _generate_recommendation(self) -> str:
        """Generate clinical recommendation based on findings."""
        if not self.findings:
            return "No nodules detected. Routine follow-up recommended."

        # Get highest malignancy score
        max_malignancy = max(f.malignancy_score for f in self.findings)
        nodule_count = len(self.findings)

        # Get largest nodule size
        max_size = max(max(f.size_mm) for f in self.findings)

        # Generate recommendation based on Lung-RADS guidelines
        if max_malignancy >= 0.8 or max_size >= 15:
            return (
                f"HIGH PRIORITY: {nodule_count} nodule(s) detected with high "
                f"malignancy risk (max score: {max_malignancy:.2f}). "
                "Immediate clinical evaluation and tissue diagnosis recommended."
            )
        elif max_malignancy >= 0.5 or max_size >= 8:
            return (
                f"MODERATE PRIORITY: {nodule_count} nodule(s) detected with "
                f"moderate concern (max score: {max_malignancy:.2f}). "
                "Follow-up CT in 3-6 months recommended."
            )
        else:
            return (
                f"{nodule_count} small nodule(s) detected with low concern "
                f"(max score: {max_malignancy:.2f}). "
                "Routine annual follow-up recommended."
            )

    def to_text(self) -> str:
        """Generate text report.

        Returns:
            Formatted text report string
        """
        lines = [
            "=" * 70,
            "PULMONARY NODULE DETECTION REPORT",
            "=" * 70,
            "",
            f"Patient ID: {self.patient_id}",
            f"Scan Date: {self.scan_date.strftime('%Y-%m-%d')}",
            f"Analysis Date: {self.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}",
            f"AI Model Version: {self.model_version}",
            "",
            "-" * 70,
            "FINDINGS",
            "-" * 70,
            "",
        ]

        if not self.findings:
            lines.append("No nodules detected.")
        else:
            lines.append(f"Total Nodules Detected: {len(self.findings)}")
            lines.append("")

            for finding in self.findings:
                avg_size = np.mean(finding.size_mm)
                lines.extend(
                    [
                        f"Nodule #{finding.nodule_id}:",
                        f"  Location: {finding.location}",
                        f"  Center (voxels): ({finding.center[0]:.1f}, "
                        f"{finding.center[1]:.1f}, {finding.center[2]:.1f})",
                        f"  Size (mm): {finding.size_mm[0]:.1f} x "
                        f"{finding.size_mm[1]:.1f} x {finding.size_mm[2]:.1f} "
                        f"(avg: {avg_size:.1f})",
                        f"  Detection Confidence: {finding.confidence:.1%}",
                        f"  Malignancy Risk: {finding.malignancy_score:.1%}",
                        "",
                    ]
                )

        lines.extend(
            [
                "-" * 70,
                "RECOMMENDATION",
                "-" * 70,
                "",
                self.recommendation,
                "",
                "=" * 70,
                "END OF REPORT",
                "=" * 70,
            ]
        )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary.

        Returns:
            Dictionary representation of report
        """
        return {
            "patient_id": self.patient_id,
            "scan_date": self.scan_date.isoformat(),
            "analysis_date": self.analysis_date.isoformat(),
            "model_version": self.model_version,
            "nodule_count": len(self.findings),
            "findings": [
                {
                    "nodule_id": f.nodule_id,
                    "center": list(f.center),
                    "size_mm": list(f.size_mm),
                    "confidence": f.confidence,
                    "malignancy_score": f.malignancy_score,
                    "location": f.location,
                }
                for f in self.findings
            ],
            "recommendation": self.recommendation,
        }


def plot_findings_summary(
    findings: list[NoduleFinding], title: str = "Nodule Findings Summary"
) -> plt.Figure:
    """Create summary visualization of nodule findings.

    Args:
        findings: List of nodule findings
        title: Plot title

    Returns:
        Matplotlib figure with summary plots

    Raises:
        ValueError: If findings list is empty
    """
    if not findings:
        msg = "Cannot plot empty findings list"
        raise ValueError(msg)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Size distribution
    sizes = [np.mean(f.size_mm) for f in findings]
    axes[0, 0].hist(sizes, bins=10, alpha=0.7, edgecolor="black", color="steelblue")
    axes[0, 0].set_xlabel("Average Size (mm)", fontsize=11)
    axes[0, 0].set_ylabel("Count", fontsize=11)
    axes[0, 0].set_title("Nodule Size Distribution", fontsize=12, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    # Confidence distribution
    confidences = [f.confidence for f in findings]
    axes[0, 1].hist(
        confidences, bins=10, alpha=0.7, edgecolor="black", color="forestgreen"
    )
    axes[0, 1].set_xlabel("Confidence Score", fontsize=11)
    axes[0, 1].set_ylabel("Count", fontsize=11)
    axes[0, 1].set_title("Detection Confidence", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)

    # Malignancy distribution
    malignancies = [f.malignancy_score for f in findings]
    axes[1, 0].hist(malignancies, bins=10, alpha=0.7, edgecolor="black", color="coral")
    axes[1, 0].set_xlabel("Malignancy Score", fontsize=11)
    axes[1, 0].set_ylabel("Count", fontsize=11)
    axes[1, 0].set_title("Malignancy Risk Distribution", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].grid(True, alpha=0.3)

    # Scatter: Size vs Malignancy
    axes[1, 1].scatter(
        sizes,
        malignancies,
        s=100,
        alpha=0.6,
        c=confidences,
        cmap="viridis",
        edgecolors="black",
    )
    axes[1, 1].set_xlabel("Average Size (mm)", fontsize=11)
    axes[1, 1].set_ylabel("Malignancy Score", fontsize=11)
    axes[1, 1].set_title(
        "Size vs Malignancy (colored by confidence)", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)

    # Add colorbar for scatter plot
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label("Confidence", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    return fig


def generate_report_from_predictions(
    patient_id: str,
    centers: NDArray[np.float32],
    sizes: NDArray[np.float32],
    confidences: NDArray[np.float32],
    malignancy_scores: NDArray[np.float32],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    scan_date: datetime | None = None,
) -> ClinicalReport:
    """Generate clinical report from model predictions.

    Args:
        patient_id: Patient identifier
        centers: Nodule centers [N, 3] in voxels
        sizes: Nodule sizes [N, 3] in voxels
        confidences: Detection confidences [N,]
        malignancy_scores: Malignancy scores [N,]
        spacing: Voxel spacing (z, y, x) in mm
        scan_date: Scan date (defaults to today)

    Returns:
        Generated clinical report

    Raises:
        ValueError: If input arrays have incompatible shapes
    """
    if centers.ndim != 2 or centers.shape[1] != 3:
        msg = f"Centers must be [N, 3], got {centers.shape}"
        raise ValueError(msg)
    if sizes.ndim != 2 or sizes.shape[1] != 3:
        msg = f"Sizes must be [N, 3], got {sizes.shape}"
        raise ValueError(msg)
    if confidences.ndim != 1:
        msg = f"Confidences must be 1D, got {confidences.ndim}D"
        raise ValueError(msg)
    if malignancy_scores.ndim != 1:
        msg = f"Malignancy scores must be 1D, got {malignancy_scores.ndim}D"
        raise ValueError(msg)

    n = len(centers)
    if len(sizes) != n or len(confidences) != n or len(malignancy_scores) != n:
        msg = f"Array length mismatch: {len(centers)}, {len(sizes)}, {len(confidences)}, {len(malignancy_scores)}"
        raise ValueError(msg)

    if scan_date is None:
        scan_date = datetime.now()

    # Convert to findings
    findings = []
    for i in range(n):
        # Convert voxel sizes to mm
        size_mm: tuple[float, float, float] = (
            float(sizes[i, 0] * spacing[0]),
            float(sizes[i, 1] * spacing[1]),
            float(sizes[i, 2] * spacing[2]),
        )
        center: tuple[float, float, float] = (
            float(centers[i, 0]),
            float(centers[i, 1]),
            float(centers[i, 2]),
        )

        finding = NoduleFinding(
            nodule_id=i + 1,
            center=center,
            size_mm=size_mm,
            confidence=float(confidences[i]),
            malignancy_score=float(malignancy_scores[i]),
            location=_infer_location(centers[i]),
        )
        findings.append(finding)

    # Sort by malignancy score (highest first)
    findings.sort(key=lambda f: f.malignancy_score, reverse=True)

    return ClinicalReport(
        patient_id=patient_id,
        scan_date=scan_date,
        analysis_date=datetime.now(),
        findings=findings,
    )


def _infer_location(center: NDArray[np.float32]) -> str:
    """Infer anatomical location from nodule center.

    Simple heuristic based on relative position.
    In production, this would use lung lobe segmentation.

    Args:
        center: Nodule center [z, y, x]

    Returns:
        Location description
    """
    # Simplified location inference (left/right, upper/middle/lower)
    z, _y, x = center

    # Assume image center is around x=256, z varies by slice
    side = "Right Lung" if x > 128 else "Left Lung"

    # Crude superior/inferior classification
    if z < 100:
        region = "Upper Lobe"
    elif z < 200:
        region = "Middle Region"
    else:
        region = "Lower Lobe"

    return f"{side}, {region}"
