"""Ground truth generation for triage scores.

The helpers in this module combine radiologist-provided malignancy ratings
with simple imaging cues to produce a 1-10 triage score. The intent is to
mirror the clinical heuristics documented for Issue #12 in the planning docs:

- Base score: map the averaged malignancy vote (1-5) onto a 1-10 triage scale.
- Size adjustment: larger nodules push the score upward; tiny nodules are
  slightly down-weighted.
- Morphology bonuses: spiculation (+2) and ground-glass appearance (+1).
- Growth bonus: documented or rated growth (+3) when temporal evidence exists.

All adjustments are intentionally coarse so the output remains interpretable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import mean

from .annotations import LIDCAnnotation

__all__ = [
    "TriageScoreBreakdown",
    "aggregate_lidc_characteristics",
    "compute_triage_score",
]


MIN_TRIAGE = 1
MAX_TRIAGE = 10


@dataclass(frozen=True, slots=True)
class TriageScoreBreakdown:
    """Detailed view of the triage score computation."""

    base_score: int
    size_adjustment: int
    spiculation_bonus: int
    ground_glass_bonus: int
    growth_bonus: int
    score: int


def aggregate_lidc_characteristics(
    annotations: Iterable[LIDCAnnotation],
) -> dict[str, float]:
    """Average per-characteristic ratings across LIDC sessions.

    Args:
        annotations: Collection of annotations for the same nodule.

    Returns:
        Dictionary mapping lower-case characteristic names to their mean score.
        Missing characteristics are omitted.
    """

    aggregated: defaultdict[str, list[int]] = defaultdict(list)
    for annotation in annotations:
        for name, value in annotation.characteristics:
            aggregated[name.lower()].append(value)

    return {name: float(mean(values)) for name, values in aggregated.items()}


def compute_triage_score(
    characteristics: Mapping[str, float],
    *,
    diameter_mm: float | None = None,
    growth_category: str | None = None,
    spiculation_threshold: float = 3.5,
    ground_glass_threshold: float = 2.5,
    growth_threshold: float = 4.0,
) -> TriageScoreBreakdown:
    """Generate a 1-10 triage score from annotation-derived cues.

    Args:
        characteristics: Mapping of averaged LIDC characteristics. Keys are
            compared in a case-insensitive way.
        diameter_mm: Optional nodule diameter in millimetres.
        growth_category: Optional free-text descriptor such as "growing" or
            "stable" extracted from longitudinal reports.
        spiculation_threshold: Minimum averaged spiculation rating that triggers
            the +2 bonus.
        ground_glass_threshold: Maximum averaged texture/internal structure
            rating considered ground-glass for the +1 bonus.
        growth_threshold: Minimum averaged temporal growth rating triggering
            the +3 bonus.

    Returns:
        ``TriageScoreBreakdown`` summarising all components.

    Raises:
        ValueError: If a malignancy score is not present in ``characteristics``.
    """

    lowered = {key.lower(): value for key, value in characteristics.items()}
    if "malignancy" not in lowered:
        raise ValueError("Malignancy score required to compute triage score")

    base = _base_score_from_malignancy(lowered["malignancy"])
    size_adj = _size_adjustment(diameter_mm)
    spiculation_bonus = 2 if _has_spiculation(lowered, spiculation_threshold) else 0
    ground_glass_bonus = 1 if _has_ground_glass(lowered, ground_glass_threshold) else 0
    growth_bonus = 3 if _is_growing(lowered, growth_category, growth_threshold) else 0

    raw_score = base + size_adj + spiculation_bonus + ground_glass_bonus + growth_bonus
    final_score = max(MIN_TRIAGE, min(MAX_TRIAGE, int(round(raw_score))))

    return TriageScoreBreakdown(
        base_score=int(round(base)),
        size_adjustment=size_adj,
        spiculation_bonus=spiculation_bonus,
        ground_glass_bonus=ground_glass_bonus,
        growth_bonus=growth_bonus,
        score=final_score,
    )


def _base_score_from_malignancy(malignancy: float) -> float:
    """Linearly map malignancy votes (1-5) onto the 1-10 triage scale."""

    clamped = max(1.0, min(5.0, float(malignancy)))
    return 1.0 + (clamped - 1.0) * (9.0 / 4.0)


def _size_adjustment(diameter_mm: float | None) -> int:
    """Coarse size-based adjustment."""

    if diameter_mm is None:
        return 0

    diameter = max(0.0, float(diameter_mm))
    if diameter < 4.0:
        return -1
    if diameter < 8.0:
        return 0
    if diameter < 12.0:
        return 1
    if diameter < 20.0:
        return 2
    return 3


def _has_spiculation(values: Mapping[str, float], threshold: float) -> bool:
    score = values.get("spiculation")
    return score is not None and score >= threshold


def _has_ground_glass(values: Mapping[str, float], threshold: float) -> bool:
    texture = values.get("texture")
    internal = values.get("internalstructure")
    return (texture is not None and texture <= threshold) or (
        internal is not None and internal <= threshold
    )


def _is_growing(
    values: Mapping[str, float],
    growth_category: str | None,
    threshold: float,
) -> bool:
    if growth_category is not None:
        label = growth_category.strip().lower()
        if label in {"growing", "increase", "increasing", "progressive"}:
            return True
    temporal = values.get("temporalgrowth")
    return temporal is not None and temporal >= threshold
