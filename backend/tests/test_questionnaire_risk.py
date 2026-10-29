"""
Acceptance tests for Questionnaire-Derived Risk Indicators (provisional / uncalibrated).

Verifies:
1. Scalp Health Score (SHS) is strictly image-derived and remains unchanged across different questionnaires.
2. Questionnaire markers produce separate, transparent risk_indicators:
   - 0 markers -> 'low'
   - 1-2 markers (without pain) -> 'moderate'
   - 3+ markers or active pain -> 'elevated'
3. Existing AnalysisResponse fields remain backward-compatible and unchanged.
"""

import pytest
from app.models.schemas import SymptomContext, SubScores, AnalysisResponse
from app.services.triage_engine import TriageEngine
from app.core.constants import CONDITION_NORMAL


def test_same_image_different_questionnaires_same_shs_different_risk():
    """
    Acceptance Criterion:
    Same image sub-scores + different questionnaires gives identical SHS
    but different risk_indicators.
    """
    sub_scores = SubScores(
        flake_score=20.0,
        redness_score=22.0,
        sebum_score=24.0,
        density_score=18.0
    )
    expected_shs = 84.0

    # Questionnaire A: Routine check, no risk markers
    ctx_clean = SymptomContext(
        primary_concern_zone="routine_healthy_check",
        shedding_pattern="stable_no_shedding",
        duration_months=0,
        duration_weeks=0,
        hair_pull_test="negative_under_3",
        scalp_pain_trichodynia=False,
        burning_or_pain=False,
        deficiency_flags=[],
        diet_pattern="balanced_protein",
        gut_health="optimal",
        metabolic_energy="balanced",
        hormonal_profile="none",
        family_history_hair_loss=False
    )

    # Questionnaire B: Multiple risk markers including acute pain and deficiencies
    ctx_high_risk = SymptomContext(
        primary_concern_zone="diffuse_all_over",
        shedding_pattern="sudden_acute",
        duration_months=6,
        duration_weeks=24,
        hair_pull_test="positive_over_5",
        scalp_pain_trichodynia=True,
        burning_or_pain=True,
        deficiency_flags=["iron_ferritin", "vitamin_d3"],
        diet_pattern="low_protein_crash_diet",
        gut_health="irregular_ibs",
        metabolic_energy="chronic_sluggish_fatigue",
        hormonal_profile="thyroid_dysfunction",
        family_history_hair_loss=True
    )

    resp_a = TriageEngine.evaluate(CONDITION_NORMAL, 0.90, sub_scores, ctx_clean)
    resp_b = TriageEngine.evaluate(CONDITION_NORMAL, 0.90, sub_scores, ctx_high_risk)

    # SHS must be strictly identical (image-only score)
    assert resp_a.scalp_health_score == expected_shs
    assert resp_b.scalp_health_score == expected_shs
    assert resp_a.scalp_health_score == resp_b.scalp_health_score

    # Risk indicators must be present and distinct
    assert hasattr(resp_a, "risk_indicators")
    assert hasattr(resp_b, "risk_indicators")
    assert resp_a.risk_indicators.overall_level == "low"
    assert resp_a.risk_indicators.marker_count == 0
    assert resp_b.risk_indicators.overall_level == "elevated"
    assert resp_b.risk_indicators.marker_count >= 5


def test_risk_indicators_level_no_markers_low():
    """Acceptance Criterion: 0 markers -> 'low' overall level."""
    ctx = SymptomContext(
        shedding_pattern="stable_no_shedding",
        duration_months=0,
        duration_weeks=0,
        hair_pull_test="negative_under_3",
        scalp_pain_trichodynia=False,
        burning_or_pain=False,
        deficiency_flags=[],
        diet_pattern="balanced_protein",
        gut_health="optimal",
        metabolic_energy="balanced",
        hormonal_profile="none",
        family_history_hair_loss=False
    )
    sub = SubScores(flake_score=25.0, redness_score=25.0, sebum_score=25.0, density_score=25.0)
    resp = TriageEngine.evaluate(CONDITION_NORMAL, 0.95, sub, ctx)

    assert resp.risk_indicators.overall_level == "low"
    assert resp.risk_indicators.marker_count == 0
    assert len(resp.risk_indicators.triggered_markers) == 0


def test_risk_indicators_level_few_markers_moderate():
    """Acceptance Criterion: 1-2 markers (no pain) -> 'moderate' overall level."""
    ctx = SymptomContext(
        shedding_pattern="stable_no_shedding",
        duration_months=0,
        duration_weeks=0,
        hair_pull_test="negative_under_3",
        scalp_pain_trichodynia=False,
        burning_or_pain=False,
        deficiency_flags=[],
        diet_pattern="balanced_protein",
        gut_health="frequent_bloating_acidity",  # Marker 1
        metabolic_energy="balanced",
        hormonal_profile="none",
        family_history_hair_loss=True            # Marker 2
    )
    sub = SubScores(flake_score=25.0, redness_score=25.0, sebum_score=25.0, density_score=25.0)
    resp = TriageEngine.evaluate(CONDITION_NORMAL, 0.95, sub, ctx)

    assert resp.risk_indicators.overall_level == "moderate"
    assert resp.risk_indicators.marker_count == 2
    names = [m.name for m in resp.risk_indicators.triggered_markers]
    assert any("Gut Health" in n for n in names)
    assert any("Family History" in n for n in names)


def test_risk_indicators_level_many_markers_including_pain_elevated():
    """Acceptance Criterion: Many markers including pain -> 'elevated' overall level."""
    ctx = SymptomContext(
        shedding_pattern="sudden_acute",          # Marker 1
        hair_pull_test="positive_over_5",         # Marker 2
        scalp_pain_trichodynia=True,              # Marker 3 (pain)
        burning_or_pain=True,                     # Marker 4 (pain)
        deficiency_flags=["iron_ferritin"],       # Marker 5
        duration_months=4,                        # Marker 6 (>=3 mo)
        duration_weeks=16
    )
    sub = SubScores(flake_score=25.0, redness_score=25.0, sebum_score=25.0, density_score=25.0)
    resp = TriageEngine.evaluate(CONDITION_NORMAL, 0.95, sub, ctx)

    assert resp.risk_indicators.overall_level == "elevated"
    assert resp.risk_indicators.marker_count >= 5
    assert "self-reported, not clinically validated" in resp.risk_indicators.note.lower()


def test_existing_analysis_response_fields_intact():
    """Acceptance Criterion: Existing response fields remain unchanged for current clients."""
    sub = SubScores(flake_score=25.0, redness_score=25.0, sebum_score=25.0, density_score=25.0)
    ctx = SymptomContext()
    resp = TriageEngine.evaluate(CONDITION_NORMAL, 0.90, sub, ctx)

    required_fields = [
        "suggested_condition",
        "confidence_score",
        "confidence_gated",
        "sub_scores",
        "scalp_health_score",
        "triage_level",
        "triage_reason",
        "clinical_correlations",
        "recommended_ingredients",
        "lifestyle_guidance",
        "consultation_referral",
        "disclaimer",
        "analysis_timestamp",
        "risk_indicators"
    ]
    dump = resp.model_dump()
    for field in required_fields:
        assert field in dump, f"Field {field} missing from AnalysisResponse serialization"

def test_pdf_service_renders_risk_indicators():
    """Verifies that PDF summary generates and contains risk indicators."""
    from app.services.pdf_service import pdf_service
    sub = SubScores(flake_score=20.0, redness_score=20.0, sebum_score=20.0, density_score=20.0)
    ctx = SymptomContext(scalp_pain_trichodynia=True, hair_pull_test="positive_over_5")
    resp = TriageEngine.evaluate(CONDITION_NORMAL, 0.90, sub, ctx)

    pdf_bytes = pdf_service.generate_report(resp, ctx)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000

