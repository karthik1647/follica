"""
Acceptance tests for condition classification bug fixes:
1. Heuristic mapping: family history + crown/vertex/diffuse thinning -> Androgenetic Alopecia (not Alopecia Areata).
2. Patchy spots -> Alopecia Areata.
3. Plaque / crusty scales -> Scalp Psoriasis.
4. Triage handling of CONDITION_ANDROGENETIC_ALOPECIA (Level 2 + dermatologist referral).
5. Triage handling of CONDITION_SCALP_PSORIASIS (Level 3 with pain).
6. CNN class list constraint: CLASSES remains exactly 5 classes.
"""

import pytest
from app.core.constants import (
    CONDITION_NORMAL,
    CONDITION_DRY_SCALP,
    CONDITION_SEBORRHEIC_DERMATITIS,
    CONDITION_ALOPECIA_AREATA,
    CONDITION_FOLLICULITIS,
    CONDITION_SCALP_PSORIASIS,
    TRIAGE_LEVEL_1_MINOR,
    TRIAGE_LEVEL_2_MONITORED,
    TRIAGE_LEVEL_3_ESCALATION,
    INGREDIENT_GUIDES
)
from app.models.schemas import SymptomContext, SubScores
from app.services.classifier_service import classifier_service, ScalpClassifierService
from app.services.triage_engine import TriageEngine


def test_constants_and_guides_contain_new_conditions():
    """Verify CONDITION_ANDROGENETIC_ALOPECIA and CONDITION_SCALP_PSORIASIS exist in constants and INGREDIENT_GUIDES."""
    from app.core import constants
    assert hasattr(constants, "CONDITION_ANDROGENETIC_ALOPECIA"), "CONDITION_ANDROGENETIC_ALOPECIA missing from constants.py"
    assert hasattr(constants, "CONDITION_SCALP_PSORIASIS"), "CONDITION_SCALP_PSORIASIS missing from constants.py"

    aga = constants.CONDITION_ANDROGENETIC_ALOPECIA
    psor = constants.CONDITION_SCALP_PSORIASIS

    assert aga in INGREDIENT_GUIDES, "CONDITION_ANDROGENETIC_ALOPECIA missing from INGREDIENT_GUIDES"
    assert psor in INGREDIENT_GUIDES, "CONDITION_SCALP_PSORIASIS missing from INGREDIENT_GUIDES"

    # Confirm CNN class list constraint: exactly 5 classes matching training
    assert len(ScalpClassifierService.CLASSES) == 5, "CNN CLASSES must remain exactly 5 classes matching training"


def test_heuristic_family_history_crown_returns_androgenetic_alopecia():
    """BUG 1 FIX: family_history + crown_vertex thinning -> Androgenetic Alopecia (not Alopecia Areata)."""
    from app.core.constants import CONDITION_ANDROGENETIC_ALOPECIA
    ctx = SymptomContext(
        primary_concern_zone="crown_vertex",
        family_history_hair_loss=True,
        duration_months=3,
        duration_weeks=12
    )
    condition, confidence = classifier_service._heuristic_classify(b"", ctx)
    assert condition == CONDITION_ANDROGENETIC_ALOPECIA, f"Expected {CONDITION_ANDROGENETIC_ALOPECIA}, got {condition}"


def test_heuristic_patchy_spots_returns_alopecia_areata():
    """Patchy spots -> Alopecia Areata."""
    ctx = SymptomContext(
        primary_concern_zone="patchy_spots",
        family_history_hair_loss=False,
        duration_months=1,
        duration_weeks=4
    )
    condition, confidence = classifier_service._heuristic_classify(b"", ctx)
    assert condition == CONDITION_ALOPECIA_AREATA, f"Expected {CONDITION_ALOPECIA_AREATA}, got {condition}"


def test_heuristic_plaque_crusty_returns_scalp_psoriasis():
    """BUG 2 FIX: plaque_crusty flakes -> Scalp Psoriasis."""
    ctx = SymptomContext(
        primary_concern_zone="crown_vertex",
        flake_type="plaque_crusty",
        itching=True,
        duration_months=2,
        duration_weeks=8
    )
    condition, confidence = classifier_service._heuristic_classify(b"", ctx)
    assert condition == CONDITION_SCALP_PSORIASIS, f"Expected {CONDITION_SCALP_PSORIASIS}, got {condition}"


def test_triage_handling_androgenetic_alopecia():
    """Androgenetic Alopecia gets Level 2 triage with dermatologist referral."""
    from app.core.constants import CONDITION_ANDROGENETIC_ALOPECIA
    sub = SubScores(flake_score=25.0, redness_score=25.0, sebum_score=25.0, density_score=15.0)
    ctx = SymptomContext(
        primary_concern_zone="crown_vertex",
        family_history_hair_loss=True,
        duration_months=6,
        duration_weeks=24
    )
    resp = TriageEngine.evaluate(CONDITION_ANDROGENETIC_ALOPECIA, 0.85, sub, ctx)

    assert resp.triage_level == TRIAGE_LEVEL_2_MONITORED
    assert resp.consultation_referral.required is True
    assert "dermatologist" in resp.triage_reason.lower() or "monitoring" in resp.triage_reason.lower()
    assert len(resp.recommended_ingredients) > 0


def test_triage_handling_scalp_psoriasis_with_pain():
    """Scalp Psoriasis with severe pain escalates to Level 3."""
    sub = SubScores(flake_score=10.0, redness_score=12.0, sebum_score=20.0, density_score=20.0)
    ctx = SymptomContext(
        primary_concern_zone="crown_vertex",
        flake_type="plaque_crusty",
        burning_or_pain=True,
        scalp_pain_trichodynia=True,
        duration_months=3,
        duration_weeks=12
    )
    resp = TriageEngine.evaluate(CONDITION_SCALP_PSORIASIS, 0.82, sub, ctx)

    assert resp.triage_level == TRIAGE_LEVEL_3_ESCALATION
    assert resp.consultation_referral.required is True
    assert len(resp.recommended_ingredients) > 0
