"""Tests for triage score ground truth helpers."""

from __future__ import annotations

import pytest

from src.data.annotations import LIDCAnnotation, LIDCRoi
from src.data.triage import (
    TriageScoreBreakdown,
    aggregate_lidc_characteristics,
    compute_triage_score,
)


def _roi() -> LIDCRoi:
    return LIDCRoi(image_uid=None, z_position=0.0, xy_coords=((0.0, 0.0), (1.0, 1.0)))


def test_aggregate_lidc_characteristics_produces_means() -> None:
    annotations = (
        LIDCAnnotation(
            series_uid="series",
            nodule_id="nodule",
            reading_session_id="session-1",
            rois=(_roi(),),
            characteristics=(
                ("malignancy", 5),
                ("spiculation", 4),
            ),
        ),
        LIDCAnnotation(
            series_uid="series",
            nodule_id="nodule",
            reading_session_id="session-2",
            rois=(_roi(),),
            characteristics=(
                ("malignancy", 4),
                ("spiculation", 2),
            ),
        ),
    )

    aggregated = aggregate_lidc_characteristics(annotations)

    assert aggregated["malignancy"] == pytest.approx(4.5)
    assert aggregated["spiculation"] == pytest.approx(3.0)


def test_compute_triage_score_requires_malignancy() -> None:
    with pytest.raises(ValueError):
        compute_triage_score({"texture": 2.0})


def test_compute_triage_score_combines_all_components() -> None:
    breakdown = compute_triage_score(
        {
            "malignancy": 4.0,
            "spiculation": 4.0,
            "texture": 2.0,
            "temporalgrowth": 4.5,
        },
        diameter_mm=18.0,
    )

    assert isinstance(breakdown, TriageScoreBreakdown)
    assert breakdown.base_score == 8
    assert breakdown.size_adjustment == 2
    assert breakdown.spiculation_bonus == 2
    assert breakdown.ground_glass_bonus == 1
    assert breakdown.growth_bonus == 3
    assert breakdown.score == 10


def test_compute_triage_score_handles_small_stable_nodule() -> None:
    breakdown = compute_triage_score({"malignancy": 2.0}, diameter_mm=3.5, growth_category="stable")

    assert breakdown.base_score == 3
    assert breakdown.size_adjustment == -1
    assert breakdown.growth_bonus == 0
    assert breakdown.score == 2


def test_compute_triage_score_growth_category_triggers_bonus() -> None:
    breakdown = compute_triage_score(
        {"malignancy": 3.0},
        diameter_mm=10.0,
        growth_category="Progressive",
    )

    assert breakdown.growth_bonus == 3
    assert breakdown.score >= breakdown.base_score


def test_compute_triage_score_defaults_to_zero_size_adjustment() -> None:
    breakdown = compute_triage_score({"malignancy": 3.0})

    assert breakdown.size_adjustment == 0


def test_compute_triage_score_mid_sized_nodule_has_neutral_adjustment() -> None:
    breakdown = compute_triage_score({"malignancy": 3.0}, diameter_mm=6.0)

    assert breakdown.size_adjustment == 0


def test_compute_triage_score_very_large_nodule_hits_max_adjustment() -> None:
    breakdown = compute_triage_score({"malignancy": 5.0}, diameter_mm=24.0)

    assert breakdown.size_adjustment == 3
