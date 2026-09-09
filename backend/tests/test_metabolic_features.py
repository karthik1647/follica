"""
Follica Metabolic, Gut Health & Endocrine Biomarker Test Suite.

Architectural Design Intent & Clinical Verification:
1. Verifies that the multi-modal clinical correlation engine incorporates the
   Traya 3-Science Model (Dermatology, Nutrition, Systemic/Ayurveda):
   - Gut Health (Bloating, Acidity, Constipation, IBS)
   - Daily Energy & Metabolic Rate (Chronic Fatigue, Rapid Weight Shifts)
   - Endocrine Profile (Thyroid Dysfunction, PCOS, Insulin Resistance)
2. Validates that these systemic markers correctly generate:
   - ClinicalCorrelationFactor objects with physiological explanations.
   - Enriched lifestyle guidance (prebiotic/probiotic nutrients and blood panel advice).
   - Structured physician hand-off notes for dermatologists.
   - Valid PDF reports without layout or text overflow issues.
"""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import SymptomContext, SubScores
from app.services.triage_engine import triage_engine
from app.services.pdf_service import pdf_service

client = TestClient(app)


def make_clean_test_image():
    """Generates a valid in-focus synthetic scalp image for API test client calls."""
    arr = np.full((300, 300, 3), [160, 120, 100], dtype=np.uint8)
    # Inject sharp hair lines for Laplacian focus gate
    arr[::6, :, 0] = np.clip(arr[::6, :, 0].astype(int) - 50, 0, 255).astype(np.uint8)
    arr[:, ::8, 1] = np.clip(arr[:, ::8, 1].astype(int) + 20, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_symptom_context_metabolic_defaults():
    """Verifies that SymptomContext defaults to optimal systemic parameters."""
    ctx = SymptomContext()
    assert ctx.gut_health == "optimal"
    assert ctx.metabolic_energy == "balanced"
    assert ctx.hormonal_profile == "none"


def test_symptom_context_metabolic_custom_values():
    """Verifies that custom metabolic, gut, and hormonal values parse correctly."""
    ctx = SymptomContext(
        gut_health="frequent_bloating_acidity",
        metabolic_energy="chronic_sluggish_fatigue",
        hormonal_profile="thyroid_dysfunction"
    )
    assert ctx.gut_health == "frequent_bloating_acidity"
    assert ctx.metabolic_energy == "chronic_sluggish_fatigue"
    assert ctx.hormonal_profile == "thyroid_dysfunction"


def test_gut_health_generates_malabsorption_correlation():
    """
    Verifies that self-reported gut dysbiosis / bloating generates
    a Gastrointestinal Absorption Barrier clinical correlation factor.
    """
    ctx = SymptomContext(
        gut_health="frequent_bloating_acidity",
        metabolic_energy="balanced",
        hormonal_profile="none"
    )
    sub = SubScores(flake_score=24.0, redness_score=24.0, sebum_score=24.0, density_score=24.0)

    res = triage_engine.evaluate(
        raw_condition="Normal",
        confidence=0.85,
        sub_scores=sub,
        context=ctx
    )

    factors = [c.factor for c in res.clinical_correlations]
    assert any("Gastrointestinal Absorption Barrier" in f for f in factors)
    # Check that lifestyle advice includes prebiotic/probiotic guidance
    assert any("gut-skin-hair axis" in l.lower() for l in res.lifestyle_guidance)


def test_endocrine_thyroid_generates_blood_panel_guidance():
    """
    Verifies that self-reported thyroid dysfunction flags an endocrine imbalance
    and recommends a clinical blood work panel.
    """
    ctx = SymptomContext(
        gut_health="optimal",
        metabolic_energy="chronic_sluggish_fatigue",
        hormonal_profile="thyroid_dysfunction"
    )
    sub = SubScores(flake_score=22.0, redness_score=22.0, sebum_score=22.0, density_score=15.0)

    res = triage_engine.evaluate(
        raw_condition="Androgenetic Alopecia",
        confidence=0.80,
        sub_scores=sub,
        context=ctx
    )

    factors = [c.factor for c in res.clinical_correlations]
    assert any("Endocrine Axis Imbalance" in f for f in factors)
    assert any("Systemic Metabolic Load" in f for f in factors)
    # Check that blood work advice is added
    assert any("blood work" in l.lower() or "tsh" in l.lower() for l in res.lifestyle_guidance)
    # Check clinical hand-off note contains systemic profile
    assert "Systemic Profile:" in res.consultation_referral.clinical_hand_off_notes


def test_api_analyze_with_metabolic_payload():
    """
    Verifies end-to-end multipart form submission to /api/v1/analyze
    with gut health, metabolic energy, and hormonal profile fields.
    """
    img_bytes = make_clean_test_image()
    files = {"image": ("scalp.jpg", img_bytes, "image/jpeg")}
    data = {
        "sensation": "painless",
        "gut_health": "chronic_constipation",
        "metabolic_energy": "chronic_sluggish_fatigue",
        "hormonal_profile": "pcos_irregular_cycles",
        "duration_months": 3
    }

    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    # Verify correlations contain the systemic drivers
    correlation_factors = [c["factor"] for c in res["clinical_correlations"]]
    assert any("Gastrointestinal Absorption Barrier" in f for f in correlation_factors)
    assert any("Endocrine Axis Imbalance" in f for f in correlation_factors)


def test_pdf_report_includes_metabolic_profile():
    """
    Verifies that the clinical PDF generator successfully embeds
    the metabolic and gut profile without compilation error.
    """
    img_bytes = make_clean_test_image()
    files = {"image": ("scalp.jpg", img_bytes, "image/jpeg")}
    analysis_res = client.post("/api/v1/analyze", files=files, data={
        "gut_health": "irregular_ibs",
        "metabolic_energy": "sudden_weight_shift",
        "hormonal_profile": "insulin_resistance"
    }).json()

    pdf_payload = {
        "analysis_data": analysis_res,
        "context": {
            "duration_months": 6,
            "duration_weeks": 26,
            "sensation": "painless",
            "gut_health": "irregular_ibs",
            "metabolic_energy": "sudden_weight_shift",
            "hormonal_profile": "insulin_resistance",
            "symptom_notes": "Patient reports IBS flareups concurrent with increased hair shedding."
        },
        "patient_ref_id": "STUDENT-METABOLIC-001"
    }

    response = client.post("/api/v1/report/pdf", json=pdf_payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert len(response.content) > 1000
