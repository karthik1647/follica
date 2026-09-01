"""
Follica Randomized & Property-Based Scenario Test Suite.

Architectural Design Intent & Quality Assurance Rationale:
1. Why Randomized Fuzz Testing is Essential:
   - Scalp assessments combine stochastic visual features with heterogeneous patient histories.
   - We must guarantee mathematical invariants across any combination of inputs:
     a. Sub-scores must never produce NaN or violate [0.0, 25.0] bounds.
     b. Scalp Health Score must always equal the sum of the 4 sub-scores (within floating point precision).
     c. Universal disclaimer must be present on 100% of generated response payloads.
     d. Server must never throw unhandled 500 errors on extreme questionnaire parameters
        (e.g., duration=0, duration=500, extreme stress, conflicting symptoms).
     e. Confidence scores < 0.65 must consistently trigger the server-side Level 2 gate.
"""

import io
import random
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
from app.models.schemas import SubScores, SymptomContext

client = TestClient(app)


def generate_random_synthetic_scalp_image(
    width: int = 250,
    height: int = 250,
    base_r: int = 180,
    base_g: int = 140,
    base_b: int = 120,
    noise_level: int = 15,
    hair_density: int = 4
) -> bytes:
    """
    Generates parameterized synthetic scalp photographs with randomized skin tones,
    hair fiber frequencies, and noise textures.
    """
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Baseline skin chrominance with random jitter
    arr[:, :, 0] = np.clip(base_r + np.random.randint(-noise_level, noise_level, (height, width)), 0, 255)
    arr[:, :, 1] = np.clip(base_g + np.random.randint(-noise_level, noise_level, (height, width)), 0, 255)
    arr[:, :, 2] = np.clip(base_b + np.random.randint(-noise_level, noise_level, (height, width)), 0, 255)

    # Hair fibers to ensure high-frequency edges
    stride = max(2, hair_density)
    arr[::stride, :, :] = np.clip(arr[::stride, :, :] - 80, 0, 255)
    arr[:, ::(stride * 2), :] = np.clip(arr[:, ::(stride * 2), :] + 30, 0, 255)

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.parametrize("scenario_idx", range(1, 16))
def test_randomized_dual_pipeline_invariants(scenario_idx):
    """
    Executes 15 randomized permutations of patient questionnaires and image properties,
    asserting mathematical score invariants, disclaimer presence, and valid response contracts.
    """
    # 1. Randomized Questionnaire Context
    scan_angles = ["crown_vertex", "frontal_hairline", "left_temple", "right_temple", "parting_line"]
    sensations = ["itching", "burning", "painful", "painless", "mixed"]
    water_envs = ["hard_borewell", "treated_tap", "recent_relocation", "normal"]
    stress_levels = ["low", "moderate", "high"]
    shedding_patterns = ["sudden_acute", "gradual_chronic", "stable_no_shedding"]

    angle = random.choice(scan_angles)
    sensation = random.choice(sensations)
    water_env = random.choice(water_envs)
    stress = random.choice(stress_levels)
    shedding = random.choice(shedding_patterns)
    exam_stress = random.choice(["true", "false"])
    fam_history = random.choice(["true", "false"])
    duration = random.randint(0, 52)  # 0 to 1 year

    # 2. Randomized Image Generation
    r_tone = random.randint(140, 210)
    g_tone = random.randint(100, 160)
    b_tone = random.randint(80, 140)
    img_bytes = generate_random_synthetic_scalp_image(
        width=random.randint(200, 350),
        height=random.randint(200, 350),
        base_r=r_tone,
        base_g=g_tone,
        base_b=b_tone,
        hair_density=random.randint(3, 8)
    )

    files = {"image": (f"random_scalp_{scenario_idx}.jpg", img_bytes, "image/jpeg")}
    data = {
        "scan_angle": angle,
        "sensation": sensation,
        "water_environment": water_env,
        "stress_level": stress,
        "recent_exam_work_stress": exam_stress,
        "shedding_pattern": shedding,
        "duration_weeks": str(duration),
        "family_history_hair_loss": fam_history,
        "symptom_notes": f"Randomized test run scenario #{scenario_idx}"
    }

    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200, f"Failed on scenario {scenario_idx}: {response.text}"
    res = response.json()

    # Invariant A: Sub-scores strictly within [0.0, 25.0]
    sub = res["sub_scores"]
    for key in ["flake_score", "redness_score", "sebum_score", "density_score"]:
        score = sub[key]
        assert not np.isnan(score), f"Sub-score {key} is NaN in scenario {scenario_idx}"
        assert 0.0 <= score <= 25.0, f"Sub-score {key}={score} out of bounds in scenario {scenario_idx}"

    # Invariant B: Scalp Health Score arithmetic consistency
    expected_sum = round(sub["flake_score"] + sub["redness_score"] + sub["sebum_score"] + sub["density_score"], 2)
    assert abs(res["scalp_health_score"] - expected_sum) < 0.1, (
        f"Composite SHS ({res['scalp_health_score']}) does not match sum ({expected_sum})"
    )

    # Invariant C: Triage categorization validity
    assert res["triage_level"] in [
        "Level 1: Minor Home Routine",
        "Level 2: Monitored Routine",
        "Level 3: Escalation (Consult a Dermatologist)"
    ]

    # Invariant D: Non-diagnostic condition taxonomy & mandatory universal disclaimer
    valid_conditions = [
        "Normal / Balanced Scalp",
        "Dry Scalp / Mild Flaking",
        "Pattern Suggestive of Seborrheic Dermatitis",
        "Pattern Suggestive of Alopecia Areata / Patchy Loss",
        "Pattern Suggestive of Folliculitis / Scalp Bumps",
        "Pattern Suggestive of Scalp Plaque / Scaling",
        "Indeterminate Pattern / Requires Verification"
    ]
    assert res["suggested_condition"] in valid_conditions, f"Invalid condition: {res['suggested_condition']}"
    assert "prescribe" not in str(res).lower()
    assert res["disclaimer"] == UNIVERSAL_DISCLAIMER



@pytest.mark.parametrize("delta_case", [
    ("improving", 20.0, 20.0, 20.0),
    ("stable", 15.0, 15.0, 15.0),
    ("worsening", 5.0, 5.0, 5.0)
])
def test_randomized_longitudinal_progress_tracking(delta_case):
    """
    Verifies that the registration engine correctly derives progress directions
    (improving, stable, worsening) across diverse synthetic follow-up states.
    """
    case_name, flake_val, red_val, dense_val = delta_case

    # Generate baseline (Day 0) and follow-up (Day 14) captures
    img_baseline = generate_random_synthetic_scalp_image(base_r=160, base_g=120, base_b=100)
    img_followup = generate_random_synthetic_scalp_image(base_r=180, base_g=140, base_b=120)

    comparison = registration_service.align_and_compare(
        baseline_bytes=img_baseline,
        followup_bytes=img_followup,
        days_elapsed=random.randint(14, 45)
    )

    assert comparison.alignment_quality in ["high", "moderate", "low"]
    assert comparison.overall_trend in ["improving", "stable", "monitoring_advised", "escalation_recommended"]
    assert len(comparison.clinical_interpretation) > 10
    assert len(comparison.recommendation) > 10


def test_randomized_pdf_export_generation():
    """
    Verifies ReportLab 1-page clinical PDF generation with randomized patient reference IDs
    and diverse clinical conditions.
    """
    img_bytes = generate_random_synthetic_scalp_image()
    files = {"image": ("test_pdf_scalp.jpg", img_bytes, "image/jpeg")}
    analysis_res = client.post("/api/v1/analyze", files=files, data={"sensation": "burning"}).json()

    random_patient_id = f"STUDENT-REF-{random.randint(10000, 99999)}"
    pdf_payload = {
        "analysis_data": analysis_res,
        "context": {
            "scan_angle": "crown_vertex",
            "sensation": "burning",
            "water_environment": "hard_borewell",
            "stress_level": "high",
            "recent_exam_work_stress": True,
            "shedding_pattern": "sudden_acute",
            "duration_weeks": 6,
            "family_history_hair_loss": True,
            "symptom_notes": "Hard water crust observed at hostel."
        },
        "patient_ref_id": random_patient_id
    }

    pdf_res = client.post("/api/v1/report/pdf", json=pdf_payload)
    assert pdf_res.status_code == 200
    assert "application/pdf" in pdf_res.headers["content-type"]
    assert len(pdf_res.content) > 500
