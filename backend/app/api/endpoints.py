"""
Follica REST API Endpoints.

Routes:
- POST /api/v1/analyze : Comprehensive dual-pipeline inference (Image + Multi-Factor Clinical Context)
- POST /api/v1/quality-check : Standalone image pre-validation (Blur, Exposure, Dimensions, EXIF)
- POST /api/v1/progress/compare : Longitudinal progress tracker (Homography alignment + SSIM + Delta)
- POST /api/v1/consent : Consent and data retention contract
- POST /api/v1/report/pdf : Generic clinical 1-page PDF export
- GET  /api/v1/system/status : System warmup and cold-start polling hook
- GET  /health : Standard container health check
"""

from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Response

from app.core.config import settings
from app.core.constants import UNIVERSAL_DISCLAIMER
from app.models.schemas import (
    AnalysisResponse,
    SymptomContext,
    ConsentRequest,
    ConsentResponse,
    SystemStatusResponse,
    ProgressComparisonResponse
)
from app.services.classifier_service import classifier_service
from app.services.cv_pipeline import cv_pipeline, ImageDecodeError
from app.services.triage_engine import triage_engine
from app.services.pdf_service import pdf_service
from app.services.quality_gate import quality_gate
from app.services.registration import registration_service
from app.services.consent_store import consent_store, TERMS_VERSION

router = APIRouter()


@router.get("/system/status", response_model=SystemStatusResponse, tags=["Infra & Health"])
@router.get("/health", response_model=SystemStatusResponse, tags=["Infra & Health"])
async def get_system_status():
    """
    Polling endpoint for frontend to check backend readiness during cold starts.
    Free-tier instances (Render/Hugging Face) can be polled until is_ready=True.
    """
    return SystemStatusResponse(
        status="ready" if classifier_service.model_loaded else "warming_up",
        is_ready=classifier_service.model_loaded,
        model_loaded=classifier_service.model_loaded,
        model_architecture=settings.MODEL_ARCHITECTURE,
        confidence_gate_threshold=settings.CONFIDENCE_GATE_THRESHOLD,
        server_time=datetime.now(timezone.utc).isoformat()
    )


@router.get("/domains", tags=["DermaIQ Modular Architecture"])
async def list_registered_domains():
    """
    Returns registered modular clinical domains supported by the Follica/DermaIQ engine.
    """
    from app.core.domains import DERMA_IQ_DOMAINS
    return {"domains": list(DERMA_IQ_DOMAINS.values())}


@router.post("/quality/check", tags=["Image Quality Gate"])
@router.post("/quality-check", tags=["Image Quality Gate"])
async def check_image_quality(image: UploadFile = File(...)):
    """
    Fast pre-validation endpoint for client apps to verify image blur, exposure,
    and EXIF metadata before initiating heavier dual-pipeline evaluation.
    """
    try:
        contents = await image.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "IMAGE_CORRUPTED_OR_UNREADABLE",
                    "rejection_reason": "Uploaded image file is empty (0 bytes).",
                    "user_guidance": "Please upload a valid JPEG, PNG, or WebP photograph of your scalp."
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "IMAGE_CORRUPTED_OR_UNREADABLE",
                "rejection_reason": f"Failed to read image upload: {str(e)}",
                "user_guidance": "Please upload a valid JPEG, PNG, or WebP photograph of your scalp."
            }
        )

    try:
        is_valid, report = quality_gate.evaluate(contents)
    except ImageDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "IMAGE_CORRUPTED_OR_UNREADABLE",
                "rejection_reason": f"Corrupted or unreadable image: {str(e)}",
                "user_guidance": "Please upload a valid JPEG, PNG, or WebP photograph of your scalp."
            }
        )

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "IMAGE_QUALITY_CHECK_FAILED",
                "error_code": report.get("error_code"),
                "rejection_reason": report.get("rejection_reason"),
                "user_guidance": report.get("user_guidance"),
                "metrics": report.get("metrics", {})
            }
        )

    return {
        "status": "passed",
        "message": "Image quality satisfies trichoscopy requirements.",
        "metrics": report.get("metrics", {})
    }


@router.post("/consent", response_model=ConsentResponse, tags=["Consent & Privacy"])
async def record_user_consent(request: ConsentRequest):
    """
    Persists a consent record to SQLite (path: FOLLICA_CONSENT_DB env var).
    Stored fields: UUID, UTC timestamp, terms version, salted-hash client token,
    consent scopes, retention_days, revoked_at (NULL until revoked).
    NOT stored: images, questionnaire answers, analysis results, raw identifiers.
    See docs/privacy.md for the full data inventory.
    """
    if not request.biometric_image_consent:
        raise HTTPException(
            status_code=400,
            detail="Biometric image consent is required to process scalp scans."
        )

    record = consent_store.create(
        raw_client_identifier=request.user_id,
        scope_analysis=True,
        scope_progress=request.research_opt_in,
        retention_days=request.data_retention_days,
    )

    return ConsentResponse(
        status="success",
        consent_id=record["id"],
        user_id=request.user_id,
        consent_recorded=True,
        retention_days=record["retention_days"],
        terms_version=record["terms_version"],
        message="Consent recorded and will be retained for "
                f"{record['retention_days']} days. "
                "Use DELETE /api/v1/consent/{id} to revoke at any time."
    )


@router.delete("/consent/{consent_id}", tags=["Consent & Privacy"])
async def revoke_consent(consent_id: str):
    """
    Revokes a previously recorded consent by UUID.
    Sets revoked_at to the current UTC time; the record is purged on the next
    retention sweep. Does NOT delete analysis results already generated.
    """
    ok, record = consent_store.revoke(consent_id)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"Consent record '{consent_id}' not found."
        )
    return {
        "status": "revoked",
        "consent_id": consent_id,
        "revoked_at": record["revoked_at"],
        "message": "Consent revoked. The record will be purged on the next retention sweep."
    }


@router.get("/consent/{consent_id}", tags=["Consent & Privacy"])
async def get_consent_record(consent_id: str):
    """
    Retrieves a previously recorded consent record by UUID.
    Returns status, terms_version, retention_days, revoked_at, etc.
    Does not expose raw client identifiers, photos, or questionnaire answers.
    """
    record = consent_store.get(consent_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Consent record '{consent_id}' not found."
        )
    return {
        "status": "success",
        "consent": dict(record)
    }


@router.post("/analyze", response_model=AnalysisResponse, tags=["Scalp Analysis"])
async def analyze_scalp(
    image: UploadFile = File(..., description="Scalp photo (JPEG/PNG/WebP)"),
    # Demographics & Concern Zone
    age_group: str = Form(default="18_25"),
    biological_sex: str = Form(default="male"),
    primary_concern_zone: str = Form(default="crown_vertex"),
    # Sensation & Trichodynia
    sensation: str = Form(default="painless"),
    itching: bool = Form(default=False),
    burning_or_pain: bool = Form(default=False),
    scalp_pain_trichodynia: bool = Form(default=False),
    # Environmental & Habits
    water_environment: str = Form(default="normal"),
    hard_water_exposure: bool = Form(default=False),
    pollution_exposure: str = Form(default="high_campus_dust"),
    daily_helmet_wearer: bool = Form(default=False),
    washing_frequency: str = Form(default="2_3_per_week"),
    chemical_heat_styling: bool = Form(default=False),
    # Nutrition, Sleep & Genetics
    diet_pattern: str = Form(default="balanced_protein"),
    gut_health: str = Form(default="optimal"),
    metabolic_energy: str = Form(default="balanced"),
    hormonal_profile: str = Form(default="none"),
    sleep_quality: str = Form(default="good_7_8h"),
    stress_level: str = Form(default="moderate"),
    recent_exam_work_stress: bool = Form(default=False),
    family_history_hair_loss: bool = Form(default=False),
    deficiency_flags: Optional[str] = Form(default=None),
    # Follicular Dynamics & Duration
    hair_pull_test: str = Form(default="negative_under_3"),
    flake_type: str = Form(default="none"),
    shedding_pattern: str = Form(default="gradual_chronic"),
    duration_months: Optional[int] = Form(default=None, description="Duration in months (0-72 months)"),
    duration_weeks: Optional[int] = Form(default=None, description="Optional legacy duration in weeks"),
    symptom_notes: Optional[str] = Form(default=None)
):
    """
    Comprehensive Dual-Pipeline Scalp Assessment:
    0. Pre-inference quality gate validates blur, exposure, EXIF, and dimensions.
    1. Deep Learning Classifier predicts condition pattern and confidence score.
    2. Classical Computer Vision extracts 4 discrete sub-scores (0-25 each).
    3. Multi-Factor Correlation Engine derives lifestyle/physiological risk drivers.
    4. Server-side confidence gate reroutes low-confidence (<0.65) cases to Level 2.
    5. Applies non-diagnostic terminology and appends the universal disclaimer.
    """
    try:
        contents = await image.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "IMAGE_CORRUPTED_OR_UNREADABLE",
                    "rejection_reason": "Uploaded image file is empty (0 bytes).",
                    "user_guidance": "Please upload a valid JPEG, PNG, or WebP photograph of your scalp."
                }
            )
        if len(contents) > settings.MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 10MB limit.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "IMAGE_CORRUPTED_OR_UNREADABLE",
                "rejection_reason": f"Failed to read image upload: {str(e)}",
                "user_guidance": "Please upload a valid JPEG, PNG, or WebP photograph of your scalp."
            }
        )

    # 0. Pre-Inference Image Quality Gate
    if settings.QUALITY_GATE_ENABLED:
        is_valid, quality_report = quality_gate.evaluate(contents)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "IMAGE_QUALITY_CHECK_FAILED",
                    "error_code": quality_report.get("error_code"),
                    "rejection_reason": quality_report.get("rejection_reason"),
                    "user_guidance": quality_report.get("user_guidance"),
                    "metrics": quality_report.get("metrics", {})
                }
            )

    # Parse deficiency flags list
    deficiencies = [f.strip() for f in deficiency_flags.split(",")] if deficiency_flags else []

    # Parse multi-select sensations if passed as list or comma-separated
    sens_items = [s.strip() for s in sensation.split(",") if s.strip()] if isinstance(sensation, str) else ["painless"]
    if not sens_items:
        sens_items = ["painless"]

    # Normalize common form and mobile aliases
    zone_norm = primary_concern_zone.lower()
    if "diffuse" in zone_norm:
        primary_concern_zone = "diffuse_all_over"
    elif "patchy" in zone_norm or "spots" in zone_norm:
        primary_concern_zone = "patchy_spots"
    elif "hairline" in zone_norm or "temple" in zone_norm:
        primary_concern_zone = "hairline_temples"
    elif "parting" in zone_norm:
        primary_concern_zone = "central_parting"

    pull_norm = hair_pull_test.lower()
    if any(x in pull_norm for x in ["over_5", "positive", "acute_shed", ">5"]):
        hair_pull_test = "positive_over_5"
    elif any(x in pull_norm for x in ["3_5", "3-5", "borderline"]):
        hair_pull_test = "borderline_3_5"

    flake_norm = flake_type.lower()
    if any(x in flake_norm for x in ["plaque", "crust"]):
        flake_type = "plaque_crusty"
    elif any(x in flake_norm for x in ["greasy", "yellow"]):
        flake_type = "greasy_yellow_thick"
    elif any(x in flake_norm for x in ["dry", "white", "fine"]):
        flake_type = "dry_white_fine"

    water_norm = water_environment.lower()
    if any(x in water_norm for x in ["hard", "borewell"]):
        water_environment = "hard_borewell"
    elif any(x in water_norm for x in ["treated", "soft", "tap"]):
        water_environment = "treated_tap"

    # Assemble comprehensive symptom context
    valid_zones = ["routine_healthy_check", "crown_vertex", "hairline_temples", "central_parting", "diffuse_all_over", "patchy_spots"]
    
    # Calculate reconciled duration
    if duration_months is not None:
        dur_m_final = max(0, min(72, duration_months))
        dur_w_final = round(dur_m_final * 4.33)
    elif duration_weeks is not None:
        dur_w_final = max(0, duration_weeks)
        dur_m_final = max(0, min(72, round(dur_w_final / 4.33)))
    else:
        dur_m_final = 1
        dur_w_final = 4

    # Ignore 'painless' when searching for inflammatory descriptors
    active_pain_symptoms = [s.lower() for s in sens_items if s.lower() != "painless"]

    context = SymptomContext(
        age_group=age_group if age_group in ["under_18", "18_25", "26_35", "36_50", "over_50"] else "18_25",
        biological_sex=biological_sex if biological_sex in ["male", "female", "other"] else "male",
        primary_concern_zone=primary_concern_zone if primary_concern_zone in valid_zones else "crown_vertex",
        sensation=",".join(sens_items),
        sensations=sens_items,
        itching=itching or ("itching" in sens_items or "pruritus" in sens_items),
        burning_or_pain=burning_or_pain or any(any(w in s for w in ["burning", "tender", "painful", "bump", "pustule"]) for s in active_pain_symptoms),
        scalp_pain_trichodynia=scalp_pain_trichodynia or any(any(w in s for w in ["tender", "trichodynia", "burning"]) for s in active_pain_symptoms),
        water_environment=water_environment if water_environment in ["hard_borewell", "treated_tap", "recent_relocation", "normal"] else "normal",
        hard_water_exposure=hard_water_exposure or (water_environment == "hard_borewell"),
        pollution_exposure=pollution_exposure if pollution_exposure in ["low_rural", "moderate_suburban", "high_metro_industrial", "high_campus_dust"] else "high_campus_dust",
        daily_helmet_wearer=daily_helmet_wearer,
        washing_frequency=washing_frequency if washing_frequency in ["daily", "2_3_per_week", "once_per_week", "rarely"] else "2_3_per_week",
        chemical_heat_styling=chemical_heat_styling,
        diet_pattern=diet_pattern if diet_pattern in ["balanced_protein", "vegetarian_vegan", "low_protein_crash_diet"] else "balanced_protein",
        gut_health=gut_health if gut_health in ["optimal", "frequent_bloating_acidity", "chronic_constipation", "irregular_ibs"] else "optimal",
        metabolic_energy=metabolic_energy if metabolic_energy in ["balanced", "chronic_sluggish_fatigue", "sudden_weight_shift"] else "balanced",
        hormonal_profile=hormonal_profile if hormonal_profile in ["none", "thyroid_dysfunction", "pcos_irregular_cycles", "insulin_resistance"] else "none",
        sleep_quality=sleep_quality if sleep_quality in ["good_7_8h", "irregular_under_6h", "shift_work_disrupted"] else "good_7_8h",
        stress_level=stress_level if stress_level in ["low", "moderate", "high"] else "moderate",
        recent_exam_work_stress=recent_exam_work_stress,
        family_history_hair_loss=family_history_hair_loss,
        deficiency_flags=deficiencies,
        hair_pull_test=hair_pull_test if hair_pull_test in ["negative_under_3", "borderline_3_5", "positive_over_5"] else "negative_under_3",
        flake_type=flake_type if flake_type in ["none", "dry_white_fine", "greasy_yellow_thick", "plaque_crusty"] else "none",
        shedding_pattern=shedding_pattern if shedding_pattern in ["sudden_acute", "gradual_chronic", "stable_no_shedding"] else "gradual_chronic",
        duration_months=dur_m_final,
        duration_weeks=dur_w_final,
        symptom_notes=symptom_notes
    )

    # 1. Pipeline 1: Deep Learning Classification
    raw_condition, confidence = classifier_service.predict(contents, context)

    # 2. Pipeline 2: Classical Computer Vision Micro-Feature Extraction
    try:
        sub_scores = cv_pipeline.analyze_image(contents)
    except ImageDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "IMAGE_CORRUPTED_OR_UNREADABLE",
                "rejection_reason": f"Corrupted or unreadable image: {str(e)}",
                "user_guidance": "Please upload a valid JPEG, PNG, or WebP photograph of your scalp."
            }
        )

    # 3. Server-Side Triage, Multi-Factor Correlation & Confidence Gate Evaluation
    analysis_result = triage_engine.evaluate(
        raw_condition=raw_condition,
        confidence=confidence,
        sub_scores=sub_scores,
        context=context
    )

    return analysis_result


@router.post("/progress/compare", response_model=ProgressComparisonResponse, tags=["Longitudinal Progress Tracker"])
async def compare_progress_scans(
    baseline_image: UploadFile = File(..., description="Baseline scalp scan photo (Day 0)"),
    followup_image: UploadFile = File(..., description="Follow-up scalp scan photo (Day 14/30)"),
    days_elapsed: int = Form(default=14, ge=1, le=365)
):
    """
    Compares two longitudinal scalp assessments (Baseline vs Follow-up) and computes
    homography alignment quality, sub-score deltas, and overall treatment response trend.
    """
    try:
        base_bytes = await baseline_image.read()
        follow_bytes = await followup_image.read()
        if len(base_bytes) == 0 or len(follow_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "IMAGE_CORRUPTED_OR_UNREADABLE",
                    "rejection_reason": "One or both uploaded images are empty (0 bytes).",
                    "user_guidance": "Please upload valid baseline and follow-up scalp photographs."
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "IMAGE_CORRUPTED_OR_UNREADABLE",
                "rejection_reason": f"Failed to read image upload: {str(e)}",
                "user_guidance": "Please upload valid baseline and follow-up scalp photographs."
            }
        )

    try:
        return registration_service.align_and_compare(base_bytes, follow_bytes, days_elapsed)
    except ImageDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "IMAGE_CORRUPTED_OR_UNREADABLE",
                "rejection_reason": f"Corrupted or unreadable image: {str(e)}",
                "user_guidance": "Please upload valid JPEG, PNG, or WebP photographs for comparison."
            }
        )


@router.post("/report/pdf", tags=["Clinical Export"])
async def export_clinical_pdf(
    analysis_data: AnalysisResponse,
    context: SymptomContext,
    patient_ref_id: str = "ANON-PATIENT"
):
    """
    Generates a generic 1-page clinical summary PDF for hand-off to a treating physician/dermatologist.
    """
    pdf_bytes = pdf_service.generate_report(analysis_data, context, patient_ref_id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=follica_clinical_summary_{patient_ref_id}.pdf"
        }
    )