"""
Tests for SDASI/SALT Condition-Anchored Score Calibration and Layer-2 Survey Pipeline.

Verifies:
1. All 10 real demo clinical photographs produce calibrated non-perfect SHS (< 85)
   with non-uniform sub-scores (never 25/25 across all 4 axes).
2. Continuous photo-to-photo variation is preserved across different raw CV scores
   within the same condition (affine bracket mapping rather than constant clipping).
3. Layer-2 SurveyRiskEngine produces condition-specific trigger rankings, modifiable
   counts, priority action queues, standalone /api/v1/survey/analyze, and cohort summary.
"""

import glob
import os
from fastapi.testclient import TestClient

from app.main import app
from app.core.constants import (
    CONDITION_LICHEN_PLANUS,
    CONDITION_SEBORRHEIC_DERMATITIS,
    CONDITION_NORMAL,
    UNIVERSAL_DISCLAIMER,
)
from app.models.schemas import SubScores, SymptomContext
from app.services.triage_engine import TriageEngine

client = TestClient(app)


def test_real_demo_photos_calibrated_non_perfect_scores():
    """Real pathology demo samples must never return 100/100 or uniform 25/25 sub-scores."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    demo_files = sorted(glob.glob(os.path.join(repo_root, "training", "demo_samples", "*.*")))
    assert len(demo_files) == 10, f"Expected 10 demo samples, found {len(demo_files)}"

    shs_values = []
    for fpath in demo_files:
        with open(fpath, "rb") as fh:
            resp = client.post(
                "/api/v1/analyze",
                files={"image": (os.path.basename(fpath), fh.read(), "image/jpeg")},
                data={"primary_concern_zone": "routine_healthy_check", "duration_months": "0"},
            )
        assert resp.status_code == 200, f"Failed on {fpath}: {resp.text}"
        body = resp.json()
        shs = body["scalp_health_score"]
        sub = body["sub_scores"]
        shs_values.append(shs)

        assert shs < 85.0, f"{os.path.basename(fpath)} unexpectedly scored {shs}/100"
        axes = [sub["flake_score"], sub["redness_score"], sub["sebum_score"], sub["density_score"]]
        assert not all(v >= 24.5 for v in axes), f"{os.path.basename(fpath)} returned near-max axes: {axes}"
        assert body["survey_risk_profile"] is not None
        assert body["survey_risk_profile"]["disclaimer"] == UNIVERSAL_DISCLAIMER

    # Ensure scores vary meaningfully across the 10 clinical photos
    assert len(set(round(v, 1) for v in shs_values)) >= 6


def test_affine_score_mapping_preserves_within_condition_variation():
    """Two different photos of Lichen Planus with distinct raw CV values yield distinct SHS."""
    ctx = SymptomContext()
    mild_raw = SubScores(flake_score=25.0, redness_score=24.0, sebum_score=25.0, density_score=22.0)
    severe_raw = SubScores(flake_score=14.0, redness_score=11.0, sebum_score=16.0, density_score=8.0)

    res_mild = TriageEngine.evaluate(CONDITION_LICHEN_PLANUS, 0.94, mild_raw, ctx)
    res_severe = TriageEngine.evaluate(CONDITION_LICHEN_PLANUS, 0.94, severe_raw, ctx)

    assert res_mild.scalp_health_score < 70.0
    assert res_severe.scalp_health_score < res_mild.scalp_health_score - 5.0
    assert res_severe.sub_scores.density_score < res_mild.sub_scores.density_score


def test_standalone_survey_pipeline_and_cohort_endpoints():
    """Verifies POST /api/v1/survey/analyze and POST /api/v1/survey/cohort-summary."""
    single_payload = {
        "condition": CONDITION_SEBORRHEIC_DERMATITIS,
        "context": {
            "water_environment": "hard_borewell",
            "daily_helmet_wearer": True,
            "stress_level": "high",
            "diet_pattern": "low_protein_crash_diet",
            "family_history_hair_loss": True,
        },
    }
    res = client.post("/api/v1/survey/analyze", json=single_payload)
    assert res.status_code == 200
    prof = res.json()
    assert prof["condition"] == CONDITION_SEBORRHEIC_DERMATITIS
    assert prof["risk_band"] in ["elevated", "high"]
    assert prof["modifiable_count"] >= 3
    assert len(prof["intervention_queue"]) >= 3
    assert "prescribe" not in str(prof).lower()

    cohort_payload = {
        "records": [
            single_payload,
            {"condition": CONDITION_NORMAL, "context": {"water_environment": "normal", "stress_level": "low"}},
        ]
    }
    cres = client.post("/api/v1/survey/cohort-summary", json=cohort_payload)
    assert cres.status_code == 200
    cbody = cres.json()
    assert cbody["total_records"] == 2
    assert len(cbody["top_population_triggers"]) >= 1
    assert cbody["disclaimer"] == UNIVERSAL_DISCLAIMER
