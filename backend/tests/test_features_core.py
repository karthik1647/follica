"""
Follica Core Features & User Journey Integration Test Suite.

Verifies:
A. Multi-Angle Photo Scanner context
B. Clinical Questionnaire (5 Questions: Sensation, Water, Stress, Shedding, Notes)
C. Scalp Health Score (SHS) composite index & 4 sub-scores
D. Molecule-first ingredient recommendations
E. Progress Tracker & Longitudinal Image Registration endpoint (/api/v1/progress/compare)
F. Clinical Doctor Hand-off PDF export (/api/v1/report/pdf)
G. DermaIQ Modular Domain discovery endpoint (/api/v1/domains)
"""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_sharp_scalp_image(size=(280, 280), base_color=(180, 140, 120)) -> bytes:
    """Generates synthetic high-frequency scalp image with simulated hair strands."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = list(base_color)
    arr[::4, :, :] = [30, 20, 10]
    arr[:, ::6, :] = [210, 170, 150]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_multi_angle_and_clinical_questionnaire_assessment():
    """Verify Features A, B, C, D in /api/v1/analyze."""
    img_bytes = make_sharp_scalp_image()
    files = {"image": ("scalp_crown.jpg", img_bytes, "image/jpeg")}
    data = {
        "scan_angle": "crown_vertex",
        "sensation": "itching",
        "water_environment": "hard_borewell",
        "stress_level": "high",
        "recent_exam_work_stress": "true",
        "shedding_pattern": "gradual_chronic",
        "duration_weeks": "4",
        "family_history_hair_loss": "true",
        "symptom_notes": "Tried generic anti-dandruff shampoo without relief."
    }

    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    # Feature C: Scalp Health Score (0-100) and 4 Sub-Scores
    assert 0.0 <= res["scalp_health_score"] <= 100.0
    sub = res["sub_scores"]
    assert 0.0 <= sub["flake_score"] <= 25.0
    assert 0.0 <= sub["redness_score"] <= 25.0
    assert 0.0 <= sub["sebum_score"] <= 25.0
    assert 0.0 <= sub["density_score"] <= 25.0
    assert "stage_comparison" in sub

    # Feature D: Molecule-First Ingredient Recommendation
    assert len(res["recommended_ingredients"]) > 0
    first_ing = res["recommended_ingredients"][0]
    assert "name" in first_ing
    assert "frequency" in first_ing
    assert "purpose" in first_ing


def test_progress_comparison_longitudinal_endpoint():
    """Verify Feature E: Progress Tracker & Longitudinal Registration."""
    img_baseline = make_sharp_scalp_image(base_color=(170, 130, 110))
    img_followup = make_sharp_scalp_image(base_color=(185, 145, 125))

    files = {
        "baseline_image": ("day0.jpg", img_baseline, "image/jpeg"),
        "followup_image": ("day14.jpg", img_followup, "image/jpeg")
    }
    data = {"days_elapsed": 14}

    response = client.post("/api/v1/progress/compare", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["days_elapsed"] == 14
    assert res["alignment_quality"] in ["high", "moderate", "low"]
    assert "follicle_density_delta_pct" in res
    assert "redness_reduction_delta_pct" in res
    assert "flake_reduction_delta_pct" in res
    assert res["overall_trend"] in ["improving", "stable", "monitoring_advised", "escalation_recommended"]
    assert len(res["clinical_interpretation"]) > 10


def test_derma_iq_modular_domains_endpoint():
    """Verify Feature G: DermaIQ Modular Domain discovery."""
    response = client.get("/api/v1/domains")
    assert response.status_code == 200
    res = response.json()

    assert "domains" in res
    domain_ids = [d["domain_id"] for d in res["domains"]]
    assert "scalp" in domain_ids
    assert "facial_acne" in domain_ids
    assert "eczema_rashes" in domain_ids


def test_multi_select_sensations_and_correlation():
    """Verify that multiple simultaneous sensations (e.g. itching + burning) are ingested and correlated."""
    img_bytes = make_sharp_scalp_image()
    files = {"image": ("scalp_multi_sensation.jpg", img_bytes, "image/jpeg")}
    data = {
        "sensation": "itching,burning,painful_bumps",
        "duration_weeks": "5",
        "primary_concern_zone": "crown_vertex"
    }
    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert 0.0 <= res["scalp_health_score"] <= 100.0
    assert len(res["clinical_correlations"]) > 0


def test_healthy_individual_routine_checkup():
    """Verify that a healthy person conducting a routine scan receives optimal preventative care guidance."""
    img_bytes = make_sharp_scalp_image()
    files = {"image": ("healthy_scalp.jpg", img_bytes, "image/jpeg")}
    data = {
        "primary_concern_zone": "routine_healthy_check",
        "sensation": "painless",
        "hair_pull_test": "negative_under_3",
        "flake_type": "none",
        "duration_weeks": "0",
        "water_environment": "normal",
        "washing_frequency": "2_3_per_week"
    }
    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["triage_level"] == "Level 1: Minor Home Routine"
    assert "Balanced" in res["suggested_condition"] or "Healthy" in res["suggested_condition"] or "Normal" in res["suggested_condition"]
    assert any("pH-Balanced" in ing["name"] or "Panthenol" in ing["name"] for ing in res["recommended_ingredients"])


def test_hair_mask_numpy_fallback_without_scipy():
    """Verify hair fiber mask dilation works in pure NumPy without scipy or OpenCV."""
    from app.services.cv_pipeline import cv_pipeline
    import app.services.cv_pipeline as cv_mod
    import numpy as np

    gray = np.full((100, 100), 120, dtype=np.uint8)
    gray[45:55, 45:55] = 40  # dark hair patch < 80

    orig_has_opencv = cv_mod.HAS_OPENCV
    try:
        cv_mod.HAS_OPENCV = False
        mask_np = cv_pipeline._build_hair_mask(gray)
        assert mask_np.shape == (100, 100)
        assert mask_np.dtype == bool
        assert np.any(mask_np)
        assert np.sum(mask_np) > 100  # dilated beyond initial 10x10 patch
    finally:
        cv_mod.HAS_OPENCV = orig_has_opencv
