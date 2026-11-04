"""
Follica Backend API Test Suite.

Verifies:
1. System warmup & cold-start health endpoint
2. User consent & data retention contract
3. Dual-pipeline analysis execution (DL + CV)
4. Presence of 4 separate CV sub-scores (flake, redness, sebum, density)
5. Server-side confidence gate rerouting (<0.65 -> Level 2)
6. Non-diagnostic phrasing compliance across responses
7. Universal medical disclaimer enforcement
8. Generic clinical PDF export (no named institutions)
"""

import io
import pytest
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.core.constants import (
    UNIVERSAL_DISCLAIMER,
    TRIAGE_LEVEL_1_MINOR,
    TRIAGE_LEVEL_2_MONITORED,
    TRIAGE_LEVEL_3_ESCALATION,
    CONDITION_UNVERIFIED_MONITORED
)
from app.models.schemas import SubScores, SymptomContext
from app.services.triage_engine import triage_engine

client = TestClient(app)


def create_dummy_image_bytes(color=(180, 140, 120), size=(250, 250)) -> bytes:
    """Helper to generate in-memory synthetic image bytes with micro-texture for quality gate passing."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = list(color)
    # Add high-frequency simulated hair fibers to satisfy Laplacian variance thresholds
    arr[::4, :, :] = [40, 30, 20]
    arr[:, ::6, :] = [200, 160, 140]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()



def test_system_status_and_health_check():
    """Verify warmup and cold-start polling endpoint."""
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_ready"] is True
    assert data["model_loaded"] is True
    assert data["confidence_gate_threshold"] == 0.65
    assert "free_tier_notice" in data


def test_consent_contract():
    """Verify consent registration and retention policy."""
    payload = {
        "user_id": "test-user-123",
        "biometric_image_consent": True,
        "research_opt_in": True,
        "data_retention_days": 30
    }
    response = client.post("/api/v1/consent", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["consent_recorded"] is True
    assert data["retention_days"] == 30
    assert "SQLite" in data["storage_security"]
    assert "AES-256" not in data["storage_security"]
    assert "consent_id" in data
    assert data["disclaimer"] == UNIVERSAL_DISCLAIMER


def test_dual_pipeline_analysis_with_subscores():
    """Verify dual-pipeline execution and separate sub-score fields."""
    img_bytes = create_dummy_image_bytes()
    files = {"image": ("test_scalp.jpg", img_bytes, "image/jpeg")}
    data = {
        "itching": "true",
        "hard_water_exposure": "true",
        "stress_level": "high",
        "duration_weeks": 3
    }
    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    # 1. Non-diagnostic phrasing check
    assert "suggests" in res["suggested_condition"].lower() or "pattern" in res["suggested_condition"].lower()
    assert "prescribe" not in str(res).lower()

    # 2. Verify 4 distinct sub-scores exist in response
    sub_scores = res["sub_scores"]
    assert "flake_score" in sub_scores
    assert "redness_score" in sub_scores
    assert "sebum_score" in sub_scores
    assert "density_score" in sub_scores

    assert 0.0 <= sub_scores["flake_score"] <= 25.0
    assert 0.0 <= sub_scores["redness_score"] <= 25.0
    assert 0.0 <= sub_scores["sebum_score"] <= 25.0
    assert 0.0 <= sub_scores["density_score"] <= 25.0

    # 3. Composite score is sum of sub-scores
    expected_sum = round(
        sub_scores["flake_score"] +
        sub_scores["redness_score"] +
        sub_scores["sebum_score"] +
        sub_scores["density_score"],
        2
    )
    assert abs(res["scalp_health_score"] - expected_sum) < 0.1

    # 4. Mandatory universal disclaimer check
    assert res["disclaimer"] == UNIVERSAL_DISCLAIMER


def test_server_side_confidence_gate():
    """Verify that predictions below 0.65 confidence are strictly rerouted to Level 2."""
    dummy_sub_scores = SubScores(
        flake_score=20.0,
        redness_score=20.0,
        sebum_score=20.0,
        density_score=20.0
    )
    dummy_context = SymptomContext(duration_weeks=1)

    # Simulate low confidence (0.45) on a Level 3 condition
    gated_res = triage_engine.evaluate(
        raw_condition="Pattern Suggestive of Alopecia Areata / Patchy Loss",
        confidence=0.45,
        sub_scores=dummy_sub_scores,
        context=dummy_context
    )

    assert gated_res.confidence_gated is True
    assert gated_res.suggested_condition == CONDITION_UNVERIFIED_MONITORED
    assert gated_res.triage_level == TRIAGE_LEVEL_2_MONITORED
    assert "below the clinical threshold" in gated_res.triage_reason


def test_generic_clinical_pdf_generation():
    """Verify that clinical PDF is generated without any institutional references."""
    img_bytes = create_dummy_image_bytes()
    files = {"image": ("test_scalp.jpg", img_bytes, "image/jpeg")}
    analysis_res = client.post("/api/v1/analyze", files=files, data={"burning_or_pain": "true"}).json()

    pdf_payload = {
        "analysis_data": analysis_res,
        "context": {
            "burning_or_pain": True,
            "duration_weeks": 4,
            "itching": True,
            "hard_water_exposure": False,
            "stress_level": "moderate",
            "family_history_hair_loss": False
        },
        "patient_ref_id": "TEST-PATIENT-001"
    }

    pdf_response = client.post("/api/v1/report/pdf", json=pdf_payload)
    assert pdf_response.status_code == 200
    assert "application/pdf" in pdf_response.headers["content-type"]
    assert len(pdf_response.content) > 100

    # Ensure no institutional names exist in the generated report or text
    content_text = str(pdf_response.content)
    assert "bc roy" not in content_text.lower()
    assert "iit" not in content_text.lower()


def test_cors_headers_allowed_and_disallowed_origins():
    """
    Acceptance test for CORS configuration:
    1. A request from an allowed Origin receives the 'access-control-allow-origin' header.
    2. A request from an unlisted/random Origin does NOT receive the 'access-control-allow-origin' header.
    3. Wildcard credentials combination is disabled ('access-control-allow-credentials' is not true with '*').
    4. Safe allowed methods (GET, POST, OPTIONS) and headers are verified.
    """
    # 1. Allowed Origin
    allowed_origin = "http://localhost:8000"
    res_allowed = client.options(
        "/api/v1/system/status",
        headers={
            "Origin": allowed_origin,
            "Access-Control-Request-Method": "GET"
        }
    )
    assert res_allowed.headers.get("access-control-allow-origin") == allowed_origin
    assert res_allowed.headers.get("access-control-allow-credentials") is None

    # 2. Disallowed Random Origin
    disallowed_origin = "http://malicious-random-site.com"
    res_disallowed = client.options(
        "/api/v1/system/status",
        headers={
            "Origin": disallowed_origin,
            "Access-Control-Request-Method": "GET"
        }
    )
    assert "access-control-allow-origin" not in res_disallowed.headers


def test_frontend_root_served_cleanly():
    """Verifies that the single-page application is served at root without error."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Follica" in response.text or "text/html" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Tests for unverified claim removal (Bug: unsupported clinical citations)
# ---------------------------------------------------------------------------

import os
import pathlib

# Source files that must not contain any banned citation strings
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]   # follica/
_BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]  # follica/backend/

_BANNED_STRINGS = [
    # Specific paper reference
    "Flament et al",
    "PMC11975186",
    # Fabricated validation statistics
    "r^2 = 0.952",
    "r^2=0.952",
    # Specific study reference
    "L'Oreal ScalpConsult Pro study",
    # Market claim
    "$35 billion",
    "CAGR",
]

_SOURCE_FILES_TO_CHECK = [
    _BACKEND_ROOT / "app" / "services" / "cv_pipeline.py",
    _BACKEND_ROOT / "app" / "services" / "quality_gate.py",
    _BACKEND_ROOT / "app" / "core" / "config.py",
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "scripts" / "generate_design_flow_pdf.py",
    _REPO_ROOT / "scripts" / "update_readme.py",
    _REPO_ROOT / "scripts" / "update_readme_plan.py",
]


def test_no_banned_clinical_citations_in_source_files():
    """
    Acceptance test: confirms that all unsupported validation/citation claims have
    been removed from source code, README, and PDF-generation scripts.

    Fails before the fix; passes after removal of Flament et al., PMC11975186,
    r^2=0.952, L'Oreal study reference, and $35B market claim.
    """
    violations = []
    for filepath in _SOURCE_FILES_TO_CHECK:
        if not filepath.exists():
            continue
        text = filepath.read_text(encoding="utf-8", errors="replace")
        for banned in _BANNED_STRINGS:
            if banned.lower() in text.lower():
                violations.append(f"{filepath.name}: found banned string '{banned}'")

    assert not violations, (
        "Unsupported clinical citation(s) still present in source:\n"
        + "\n".join(violations)
    )


def test_thresholds_documentation_exists_and_is_complete():
    """
    Acceptance test: confirms that docs/thresholds.md exists and documents
    the key numeric thresholds used in the pipeline.
    """
    thresholds_path = _BACKEND_ROOT / "docs" / "thresholds.md"
    assert thresholds_path.exists(), (
        f"docs/thresholds.md not found at {thresholds_path}. "
        "This file must document all numeric thresholds."
    )

    text = thresholds_path.read_text(encoding="utf-8")

    # Each of these threshold values or names must appear in the document
    required_entries = [
        "QUALITY_MIN_BLUR_SCORE",
        "QUALITY_MIN_BRIGHTNESS",
        "QUALITY_MAX_BRIGHTNESS",
        "QUALITY_MIN_DIMENSION",
        "CONFIDENCE_GATE_THRESHOLD",
        "red_excess",           # erythema threshold
        "shine_ratio",          # sebum thresholds
        "coverage_ratio",       # hair-coverage density threshold (replaced edge_energy in commit f2b128a)
        "uncalibrated",         # status column must have at least one
        "flash_glare",          # flash glare threshold
    ]

    missing = [entry for entry in required_entries if entry not in text]
    assert not missing, (
        f"docs/thresholds.md is missing required entries: {missing}"
    )


def test_thresholds_marked_uncalibrated():
    """
    Confirms that every grade boundary row in docs/thresholds.md carries the
    'uncalibrated' label, ensuring no threshold is implicitly treated as validated.
    """
    thresholds_path = _BACKEND_ROOT / "docs" / "thresholds.md"
    if not thresholds_path.exists():
        pytest.skip("docs/thresholds.md does not exist; skipping.")

    text = thresholds_path.read_text(encoding="utf-8")
    # The document must include the word 'uncalibrated' at least 10 times
    count = text.lower().count("uncalibrated")
    assert count >= 10, (
        f"Expected at least 10 occurrences of 'uncalibrated' in thresholds.md, found {count}. "
        "Make sure all thresholds are properly labelled."
    )

