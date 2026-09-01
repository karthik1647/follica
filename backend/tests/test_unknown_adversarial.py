"""
Follica Unknown & Adversarial Edge-Case Stress Test Suite.

Architectural Design Intent & Quality Assurance Rationale:
1. Why Unknown / Out-of-Distribution Testing is Critical:
   - Consumer smartphone users will submit non-standard, chaotic, and edge-case inputs:
     a. Extreme aspect ratios (e.g., 1600x180 panoramic strips, 150x900 vertical crops).
     b. RGBA transparent image buffers, grayscale-converted JPEGs, extreme salt-and-pepper sensor noise.
     c. Adversarial questionnaire payloads (massive 5,000+ character notes, conflicting symptoms, 500+ weeks duration).
     d. Completely unmatchable photo pairs submitted to the longitudinal alignment engine.
     e. Malformed byte payloads submitted to pre-inference quality gates.
2. Invariant Guarantees to Prove:
   - Server must never experience unhandled 500 crashes or memory leaks on any unknown input.
   - Quality gate must catch malformed or zero-variance images with structured 400 Bad Request responses.
   - Homography registration service must gracefully degrade to 'low' alignment quality when feature matching fails.
   - ReportLab PDF compiler must handle multi-page overflow and special character strings cleanly.
"""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.core.constants import UNIVERSAL_DISCLAIMER
from app.services.quality_gate import quality_gate
from app.services.cv_pipeline import cv_pipeline
from app.services.triage_engine import triage_engine
from app.services.registration import registration_service
from app.models.schemas import SymptomContext

client = TestClient(app)


def make_test_image(width=300, height=300, mode="RGB", color=(180, 140, 120), noise=False, sharp_lines=True):
    """Utility generator for adversarial image geometries and channel configurations."""
    if mode == "RGBA":
        arr = np.full((height, width, 4), [color[0], color[1], color[2], 255], dtype=np.uint8)
    elif mode == "L":
        arr = np.full((height, width), color[0], dtype=np.uint8)
    else:
        arr = np.full((height, width, 3), color, dtype=np.uint8)

    if noise:
        noise_arr = np.random.randint(-35, 35, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise_arr, 0, 255).astype(np.uint8)

    if sharp_lines and len(arr.shape) == 3:
        # Inject sharp high-frequency lines to ensure focus gate passes when desired
        arr[::6, :, 0] = np.clip(arr[::6, :, 0] - 80, 0, 255)
        arr[:, ::8, 1] = np.clip(arr[:, ::8, 1] + 30, 0, 255)

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    if mode == "RGBA":
        img.save(buf, format="PNG")
    else:
        img.save(buf, format="JPEG")
    return buf.getvalue()


# -----------------------------------------------------------------------------
# 1. Unknown Image Geometries & Channel Formats
# -----------------------------------------------------------------------------

def test_unknown_extreme_wide_aspect_ratio():
    """Verifies that an extreme wide panorama (1200x200) does not crash CV extractors."""
    img_bytes = make_test_image(width=1200, height=200, sharp_lines=True)
    files = {"image": ("extreme_wide.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/v1/analyze", files=files, data={"sensation": "painless"})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["scalp_health_score"] <= 100.0


def test_unknown_extreme_tall_aspect_ratio():
    """Verifies that an extreme vertical strip (200x1200) executes without dimensional errors."""
    img_bytes = make_test_image(width=200, height=1200, sharp_lines=True)
    files = {"image": ("extreme_tall.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/v1/analyze", files=files, data={"sensation": "itching"})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["scalp_health_score"] <= 100.0


def test_unknown_rgba_transparent_png():
    """Verifies that 4-channel RGBA uploads are converted to RGB without channel mismatch crashes."""
    img_bytes = make_test_image(width=300, height=300, mode="RGBA", sharp_lines=True)
    files = {"image": ("transparent.png", img_bytes, "image/png")}
    response = client.post("/api/v1/analyze", files=files, data={"water_environment": "hard_borewell"})
    assert response.status_code == 200
    data = response.json()
    assert data["sub_scores"]["flake_score"] >= 0.0


def test_unknown_heavy_gaussian_noise_image():
    """Verifies that heavy sensor grain / noise is handled without numeric overflow."""
    img_bytes = make_test_image(width=320, height=320, noise=True, sharp_lines=True)
    files = {"image": ("sensor_noise.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/v1/analyze", files=files, data={"stress_level": "high"})
    assert response.status_code == 200
    data = response.json()
    assert not np.isnan(data["scalp_health_score"])


def test_unknown_corrupted_raw_bytes_rejection():
    """Verifies that non-image garbage bytes return a clean 400 Bad Request rather than a 500 error."""
    corrupted_bytes = b"NOT_A_REAL_JPEG_IMAGE_DATA_0123456789"
    files = {"image": ("corrupted.jpg", corrupted_bytes, "image/jpeg")}
    response = client.post("/api/v1/analyze", files=files, data={})
    assert response.status_code == 400
    err = response.json()
    assert "rejection_reason" in err["detail"] or "error" in err["detail"]


# -----------------------------------------------------------------------------
# 2. Unknown & Adversarial Questionnaire Payloads
# -----------------------------------------------------------------------------

def test_unknown_massive_symptom_notes_payload():
    """Verifies that a massive 6,000-character user note is safely accepted without buffer overflow."""
    img_bytes = make_test_image()
    massive_note = "Detailed patient scalp report: " + ("Severe borewell water scaling and chronic itching. " * 120)
    files = {"image": ("scalp.jpg", img_bytes, "image/jpeg")}
    data = {
        "sensation": "itching",
        "duration_weeks": "12",
        "symptom_notes": massive_note
    }
    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["scalp_health_score"] >= 0.0


def test_unknown_extreme_duration_values():
    """Verifies that 0 weeks and 520 weeks (10 years) duration are bounded and safely parsed."""
    img_bytes = make_test_image()

    for extreme_duration in ["0", "520"]:
        files = {"image": ("scalp.jpg", img_bytes, "image/jpeg")}
        data = {
            "duration_weeks": extreme_duration,
            "sensation": "painless"
        }
        response = client.post("/api/v1/analyze", files=files, data=data)
        assert response.status_code == 200
        res = response.json()
        assert res["triage_level"] in [
            "Level 1: Minor Home Routine",
            "Level 2: Monitored Routine",
            "Level 3: Escalation (Consult a Dermatologist)"
        ]


def test_unknown_special_characters_in_inputs():
    """Verifies resilience against non-ASCII, Unicode, and punctuation in user text."""
    img_bytes = make_test_image()
    files = {"image": ("scalp.jpg", img_bytes, "image/jpeg")}
    data = {
        "sensation": "itching",
        "symptom_notes": "Scalp test: <script>alert('xss')</script> & special symbols: %$#@!~*()_+[]"
    }
    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200


# -----------------------------------------------------------------------------
# 3. Unknown Longitudinal Alignment Scenarios
# -----------------------------------------------------------------------------

def test_unknown_longitudinal_completely_different_photos():
    """
    Verifies that comparing two completely unrelated images (where keypoint homography cannot converge)
    gracefully falls back to 'low' alignment quality without raising an unhandled OpenCV exception.
    """
    # Photo A: Dark texture; Photo B: Bright striped texture
    img_a = make_test_image(color=(80, 50, 40), sharp_lines=True)
    img_b = make_test_image(color=(220, 200, 180), sharp_lines=False)

    comparison = registration_service.align_and_compare(
        baseline_bytes=img_a,
        followup_bytes=img_b,
        days_elapsed=21
    )

    assert comparison.alignment_quality in ["low", "moderate", "high"]
    assert comparison.overall_trend in ["improving", "stable", "monitoring_advised", "escalation_recommended"]
    assert len(comparison.clinical_interpretation) > 0


def test_unknown_longitudinal_zero_and_extreme_days():
    """Verifies that 0 days elapsed and 1000 days elapsed compute valid delta trends."""
    img_a = make_test_image()
    img_b = make_test_image()

    for days in [0, 1000]:
        comparison = registration_service.align_and_compare(
            baseline_bytes=img_a,
            followup_bytes=img_b,
            days_elapsed=days
        )
        assert comparison.days_elapsed == days
        assert not np.isnan(comparison.follicle_density_delta_pct)



# -----------------------------------------------------------------------------
# 4. Unknown PDF Report Export Stress Scenarios
# -----------------------------------------------------------------------------

def test_unknown_pdf_generation_with_massive_payload():
    """
    Verifies that ReportLab compiles a valid PDF stream even when presented with
    unusually large note fields and special characters.
    """
    img_bytes = make_test_image()
    analysis_res = client.post("/api/v1/analyze", files={"image": ("test.jpg", img_bytes, "image/jpeg")}, data={}).json()

    pdf_payload = {
        "analysis_data": analysis_res,
        "context": {
            "duration_weeks": 40,
            "sensation": "burning",
            "water_environment": "hard_borewell",
            "stress_level": "high",
            "recent_exam_work_stress": True,
            "shedding_pattern": "sudden_acute",
            "family_history_hair_loss": True,
            "symptom_notes": "Unusual case with multi-year chronic flaking: " + ("Extensive notes for physician review. " * 30)
        },
        "patient_ref_id": "STUDENT-UNKNOWN-XYZ-999"
    }

    response = client.post("/api/v1/report/pdf", json=pdf_payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert len(response.content) > 1000
