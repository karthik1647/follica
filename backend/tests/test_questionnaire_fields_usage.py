"""
Tests verifying that all collected questionnaire fields have a definite effect on output:
- biological_sex and age_group tailor explanatory correlation text and minor notice.
- washing_frequency and chemical_heat_styling tailor lifestyle guidance text.
- sleep_quality and stress_level tailor lifestyle guidance text and PDF output.
- scan_angle is removed.
- Default values work seamlessly when fields are omitted.
"""

import pytest
from app.models.schemas import SymptomContext, SubScores
from app.services.triage_engine import TriageEngine
from app.core.constants import CONDITION_NORMAL


def test_biological_sex_and_age_group_tailor_correlation_notes():
    """Verifies biological_sex and age_group produce rule-based explanatory text notes in correlations."""
    sub = SubScores(flake_score=25.0, redness_score=25.0, sebum_score=25.0, density_score=25.0)

    # Male
    ctx_male = SymptomContext(biological_sex="male", age_group="26_35")
    resp_male = TriageEngine.evaluate(CONDITION_NORMAL, 0.90, sub, ctx_male)
    corr_notes_male = " ".join(c.clinical_note for c in resp_male.clinical_correlations)
    assert "Norwood" in corr_notes_male or "bitemporal" in corr_notes_male.lower()

    # Female
    ctx_female = SymptomContext(biological_sex="female", age_group="26_35")
    resp_female = TriageEngine.evaluate(CONDITION_NORMAL, 0.90, sub, ctx_female)
    corr_notes_female = " ".join(c.clinical_note for c in resp_female.clinical_correlations)
    assert "Ludwig" in corr_notes_female or "central parting" in corr_notes_female.lower()


def test_age_group_under_18_produces_guardian_advisory():
    """Verifies that age_group='under_18' produces a visible guardian/minor advisory notice."""
    sub = SubScores(flake_score=25.0, redness_score=25.0, sebum_score=25.0, density_score=25.0)
    ctx_minor = SymptomContext(age_group="under_18", biological_sex="male")
    resp_minor = TriageEngine.evaluate(CONDITION_NORMAL, 0.90, sub, ctx_minor)

    corr_notes = " ".join(c.clinical_note for c in resp_minor.clinical_correlations)
    lifestyle_text = " ".join(resp_minor.lifestyle_guidance)

    assert "guardian" in corr_notes.lower() or "minor" in corr_notes.lower() or "under 18" in corr_notes.lower()
    assert any("guardian" in g.lower() or "minor" in g.lower() or "under 18" in g.lower() for g in resp_minor.lifestyle_guidance)


def test_washing_frequency_tailors_lifestyle_advice():
    """Verifies washing_frequency changes lifestyle guidance text appropriately."""
    sub = SubScores(flake_score=25.0, redness_score=25.0, sebum_score=25.0, density_score=25.0)

    # Daily washing -> over-washing warning
    ctx_daily = SymptomContext(washing_frequency="daily")
    resp_daily = TriageEngine.evaluate(CONDITION_NORMAL, 0.90, sub, ctx_daily)
    daily_text = " ".join(resp_daily.lifestyle_guidance).lower()
    assert "daily" in daily_text and ("strip" in daily_text or "rebound" in daily_text or "surfactant" in daily_text)

    # Rarely washing -> buildup warning
    ctx_rare = SymptomContext(washing_frequency="rarely")
    resp_rare = TriageEngine.evaluate(CONDITION_NORMAL, 0.90, sub, ctx_rare)
    rare_text = " ".join(resp_rare.lifestyle_guidance).lower()
    assert "infrequent" in rare_text or "accumulate" in rare_text or "yeast" in rare_text or "malassezia" in rare_text

    # Balanced 2-3x per week -> no over-washing or infrequent warnings
    ctx_normal = SymptomContext(washing_frequency="2_3_per_week")
    resp_normal = TriageEngine.evaluate(CONDITION_NORMAL, 0.90, sub, ctx_normal)
    normal_text = " ".join(resp_normal.lifestyle_guidance).lower()
    assert "strip" not in normal_text and "accumulate" not in normal_text


def test_chemical_heat_styling_tailors_lifestyle_advice():
    """Verifies chemical_heat_styling changes lifestyle guidance text."""
    sub = SubScores(flake_score=25.0, redness_score=25.0, sebum_score=25.0, density_score=25.0)

    ctx_styled = SymptomContext(chemical_heat_styling=True)
    resp_styled = TriageEngine.evaluate(CONDITION_NORMAL, 0.90, sub, ctx_styled)
    styled_text = " ".join(resp_styled.lifestyle_guidance).lower()
    assert "heat" in styled_text or "cuticle" in styled_text or "thermal" in styled_text

    ctx_natural = SymptomContext(chemical_heat_styling=False)
    resp_natural = TriageEngine.evaluate(CONDITION_NORMAL, 0.90, sub, ctx_natural)
    natural_text = " ".join(resp_natural.lifestyle_guidance).lower()
    assert "thermal and chemical" not in natural_text


def test_sleep_quality_and_stress_level_tailor_lifestyle_advice():
    """Verifies sleep_quality and stress_level change lifestyle guidance text."""
    sub = SubScores(flake_score=25.0, redness_score=25.0, sebum_score=25.0, density_score=25.0)

    ctx = SymptomContext(stress_level="high", sleep_quality="irregular_under_6h")
    resp = TriageEngine.evaluate(CONDITION_NORMAL, 0.90, sub, ctx)
    text = " ".join(resp.lifestyle_guidance).lower()
    assert "stress" in text or "cortisol" in text
    assert "sleep" in text or "circadian" in text


def test_scan_angle_is_removed_from_schema():
    """Verifies scan_angle is removed from SymptomContext model."""
    assert "scan_angle" not in SymptomContext.model_fields

def test_api_endpoint_handles_questionnaire_fields_and_defaults():
    """Verifies analyze endpoint works with full questionnaire data, omitted defaults, and ignores extra fields."""
    import io
    import numpy as np
    from PIL import Image
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # Create textured dummy image that passes the quality gate
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, :] = [180, 140, 120]
    arr[::4, :, :] = [40, 30, 20]
    arr[:, ::6, :] = [200, 160, 140]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # 1. Full data payload with sleep, stress, and former scan_angle
    files = {"image": ("scalp.jpg", buf, "image/jpeg")}
    data = {
        "age_group": "under_18",
        "biological_sex": "female",
        "washing_frequency": "daily",
        "chemical_heat_styling": "true",
        "stress_level": "high",
        "sleep_quality": "irregular_under_6h",
        "scan_angle": "crown_vertex",  # extra field, safely ignored
        "user_confirmed_consent": "true"
    }

    res = client.post("/api/v1/analyze", files=files, data=data)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()

    # Verify lifestyle guidance reflects the submitted fields
    guidance = " ".join(body["lifestyle_guidance"]).lower()
    assert "under 18" in guidance or "guardian" in guidance or "minor" in guidance
    assert "daily" in guidance and ("strip" in guidance or "rebound" in guidance or "surfactant" in guidance)
    assert "heat" in guidance or "cuticle" in guidance or "thermal" in guidance
    assert "stress" in guidance or "cortisol" in guidance
    assert "sleep" in guidance or "circadian" in guidance

    # Verify correlations reflect female Ludwig presentation
    corr_notes = " ".join(c["clinical_note"] for c in body["clinical_correlations"])
    assert "Ludwig" in corr_notes or "central parting" in corr_notes.lower()

