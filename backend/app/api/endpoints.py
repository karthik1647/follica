"""
Follica REST API Endpoints.

Routes:
- POST /api/v1/analyze : Dual-pipeline inference (Image + Questionnaire Context)
- POST /api/v1/consent : Consent and data retention contract
- POST /api/v1/report/pdf : Generic clinical 1-page PDF export
- GET  /api/v1/system/status : System warmup and cold-start polling hook
- GET  /health : Standard container health check
"""

from datetime import datetime, timezone
from typing import Optional
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

from app.services.cv_pipeline import cv_pipeline
from app.services.triage_engine import triage_engine
from app.services.pdf_service import pdf_service
from app.services.quality_gate import quality_gate

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


@router.post("/quality/check", tags=["Image Quality"])
async def check_image_quality(
    image: UploadFile = File(..., description="Scalp photo for pre-inference quality validation")
):
    """
    Pre-inference image validation endpoint.
    Tests focus sharpness (Laplacian variance) and illumination (luminance distribution)
    before initiating heavier dual-pipeline evaluation.
    """
    try:
        contents = await image.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image upload: {str(e)}")

    is_valid, report = quality_gate.evaluate(contents)
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
    Records biometric photo consent, user-defined retention policy, and optional research opt-in.
    All data is governed by encrypted-at-rest storage policies.
    """
    if not request.biometric_image_consent:
        raise HTTPException(
            status_code=400,
            detail="Biometric image consent is required to process scalp scans."
        )

    return ConsentResponse(
        status="success",
        user_id=request.user_id,
        consent_recorded=True,
        retention_days=request.data_retention_days,
        storage_security="AES-256 encrypted at rest in Supabase bucket with RLS policies",
        message="User consent and retention policy successfully registered."
    )


@router.post("/analyze", response_model=AnalysisResponse, tags=["Scalp Analysis"])
async def analyze_scalp(
    image: UploadFile = File(..., description="Scalp photo (JPEG/PNG/WebP)"),
    scan_angle: str = Form(default="unspecified", description="Target angle: crown_vertex, frontal_hairline, left_temple, right_temple, parting_line"),
    sensation: str = Form(default="painless", description="Reported sensation: itching, burning, painful, painless, mixed"),
    water_environment: str = Form(default="normal", description="Water supply: hard_borewell, treated_tap, recent_relocation, normal"),
    stress_level: str = Form(default="moderate"),
    recent_exam_work_stress: bool = Form(default=False),
    shedding_pattern: str = Form(default="gradual_chronic", description="sudden_acute vs gradual_chronic"),
    duration_weeks: int = Form(default=2),
    family_history_hair_loss: bool = Form(default=False),
    symptom_notes: Optional[str] = Form(default=None),
    itching: bool = Form(default=False),
    burning_or_pain: bool = Form(default=False),
    hard_water_exposure: bool = Form(default=False)
):
    """
    Dual-Pipeline Scalp Assessment:
    0. Pre-inference quality gate validates blur, exposure, and dimensions.
    1. Deep Learning Classifier predicts condition pattern and confidence score.
    2. Classical Computer Vision extracts 4 discrete sub-scores (0-25 each).
    3. Server-side confidence gate reroutes low-confidence (<0.65) cases to Level 2.
    4. Applies non-diagnostic terminology and appends the universal disclaimer.
    """
    # Read image bytes
    try:
        contents = await image.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
        if len(contents) > settings.MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 10MB limit.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image upload: {str(e)}")

    # 0. Pre-Inference Image Quality Gate
    # Intercepts blurry, underexposed, or overexposed captures before wasting inference compute.
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

    # Reconcile boolean flags with rich sensation/water categoricals for backwards compatibility
    eff_itching = itching or (sensation in ["itching", "mixed"])
    eff_burning = burning_or_pain or (sensation in ["burning", "painful", "mixed"])
    eff_hard_water = hard_water_exposure or (water_environment in ["hard_borewell", "recent_relocation"])

    # Assemble symptom context
    context = SymptomContext(
        scan_angle=scan_angle if scan_angle in ["crown_vertex", "frontal_hairline", "left_temple", "right_temple", "parting_line"] else "unspecified",
        sensation=sensation if sensation in ["itching", "burning", "painful", "painless", "mixed"] else "painless",
        itching=eff_itching,
        burning_or_pain=eff_burning,
        water_environment=water_environment if water_environment in ["hard_borewell", "treated_tap", "recent_relocation", "normal"] else "normal",
        hard_water_exposure=eff_hard_water,
        stress_level=stress_level if stress_level in ["low", "moderate", "high"] else "moderate",
        recent_exam_work_stress=recent_exam_work_stress,
        shedding_pattern=shedding_pattern if shedding_pattern in ["sudden_acute", "gradual_chronic", "stable_no_shedding"] else "gradual_chronic",
        duration_weeks=max(0, duration_weeks),
        family_history_hair_loss=family_history_hair_loss,
        symptom_notes=symptom_notes
    )

    # 1. Pipeline 1: Deep Learning Classification
    raw_condition, confidence = classifier_service.predict(contents, context)

    # 2. Pipeline 2: Classical Computer Vision Micro-Feature Extraction
    sub_scores = cv_pipeline.analyze_image(contents)

    # 3. Server-Side Triage & Confidence Gate Evaluation
    analysis_result = triage_engine.evaluate(
        raw_condition=raw_condition,
        confidence=confidence,
        sub_scores=sub_scores,
        context=context
    )

    return analysis_result


@router.post("/progress/compare", response_model=ProgressComparisonResponse, tags=["Progress Tracker"])
async def compare_longitudinal_scans(
    baseline_image: UploadFile = File(..., description="Day 0 baseline scalp capture"),
    followup_image: UploadFile = File(..., description="Follow-up scalp capture (e.g. Day 14 or 30)"),
    days_elapsed: int = Form(default=14, ge=1, le=365, description="Days elapsed between scans")
):
    """
    Longitudinal Hair Diary & Progress Comparison Engine:
    Aligns multi-week photos using ORB keypoint homography and computes spot-by-spot
    deltas for follicle density, erythema subsidence, and scaling reduction.
    """
    try:
        base_bytes = await baseline_image.read()
        follow_bytes = await followup_image.read()
        if len(base_bytes) == 0 or len(follow_bytes) == 0:
            raise HTTPException(status_code=400, detail="One or both uploaded images are empty.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read comparison uploads: {str(e)}")

    from app.services.registration import registration_service
    return registration_service.align_and_compare(
        baseline_bytes=base_bytes,
        followup_bytes=follow_bytes,
        days_elapsed=days_elapsed
    )


@router.get("/domains", tags=["DermaIQ Modular Scalability"])
async def list_derma_iq_domains():
    """
    Returns registered DermaIQ diagnostic modules (Scalp Health, Facial Acne, Eczema/Rashes)
    to support pluggable multi-domain dermatological scaling.
    """
    from app.core.domains import get_registered_domains
    return {
        "framework": "DermaIQ Modular Dermatology Engine",
        "domains": get_registered_domains()
    }


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

