"""
Follica Pydantic Data Models and API Contracts.

Includes:
- Analysis Request and Response schemas with separate CV sub-scores
- Strict Non-Diagnostic Terminology
- Universal Disclaimer enforcement
- Consent and Data Retention Contract
- System Health / Cold-Start Hook schemas
"""

from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field
from app.core.constants import UNIVERSAL_DISCLAIMER


# ---------------------------------------------------------
# Questionnaire & Context Models
# ---------------------------------------------------------
class SymptomContext(BaseModel):
    itching: bool = Field(default=False, description="Presence of scalp itching or pruritus")
    burning_or_pain: bool = Field(default=False, description="Presence of burning, tenderness, or painful bumps")
    hard_water_exposure: bool = Field(default=False, description="Exposure to hard borewell/tap water")
    stress_level: Literal["low", "moderate", "high"] = Field(default="moderate", description="Self-reported stress level")
    duration_weeks: int = Field(default=2, ge=0, description="Duration of symptoms in weeks")
    family_history_hair_loss: bool = Field(default=False, description="Family history of androgenetic alopecia")
    symptom_notes: Optional[str] = Field(default=None, description="Additional context or past shampoos tried")


# ---------------------------------------------------------
# Classical CV Sub-Scores (Returned as Separate Fields)
# ---------------------------------------------------------
class SubScores(BaseModel):
    flake_score: float = Field(
        ...,
        ge=0.0,
        le=25.0,
        description="Flake & Scaling Score (0=severe flaking/scaling, 25=completely clear scalp)"
    )
    redness_score: float = Field(
        ...,
        ge=0.0,
        le=25.0,
        description="Erythema & Redness Score (0=severe vascular erythema, 25=calm/non-inflamed scalp)"
    )
    sebum_score: float = Field(
        ...,
        ge=0.0,
        le=25.0,
        description="Sebum & Lipid Balance Score (0=extreme oiliness/severe dehydration, 25=optimal barrier balance)"
    )
    density_score: float = Field(
        ...,
        ge=0.0,
        le=25.0,
        description="Hair Follicle & Density Score (0=significant thinning/spacing, 25=dense follicular coverage)"
    )


class IngredientRecommendation(BaseModel):
    name: str = Field(..., description="Active ingredient molecule or formula type")
    frequency: str = Field(..., description="Suggested application cadence")
    purpose: str = Field(..., description="Functional mechanism of action")


class ConsultationReferral(BaseModel):
    required: bool = Field(default=False, description="Whether clinical consultation is recommended")
    urgency: Literal["routine", "recommended", "immediate"] = "routine"
    addressed_to: str = Field(
        default="For your treating physician / dermatologist",
        description="Generic recipient title for clinical hand-off"
    )
    search_url: str = Field(
        default="https://www.google.com/maps/search/dermatologist+near+me",
        description="Generic non-affiliated local clinic search"
    )
    clinical_hand_off_notes: str = Field(
        default="",
        description="Summary of patterns observed for review by healthcare professional"
    )


# ---------------------------------------------------------
# Core Analysis Response Schema
# ---------------------------------------------------------
class AnalysisResponse(BaseModel):
    suggested_condition: str = Field(
        ...,
        description="Non-diagnostic condition suggestion ('suggests', never 'diagnoses')"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classifier confidence score"
    )
    confidence_gated: bool = Field(
        default=False,
        description="True if low confidence (<0.65) rerouted output to Level 2 (Monitored)"
    )
    sub_scores: SubScores = Field(
        ...,
        description="Discrete classical-CV derived sub-scores in range [0, 25]"
    )
    scalp_health_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite Scalp Health Score (0-100, sum of sub-scores)"
    )
    triage_level: str = Field(
        ...,
        description="Triage categorization (Level 1 Minor, Level 2 Monitored, Level 3 Escalation)"
    )
    triage_reason: str = Field(
        ...,
        description="Non-diagnostic clinical rationale for triage assignment"
    )
    recommended_ingredients: List[IngredientRecommendation] = Field(
        default_factory=list,
        description="List of suggested active OTC ingredients"
    )
    lifestyle_guidance: List[str] = Field(
        default_factory=list,
        description="Supportive scalp care and hygiene guidance"
    )
    consultation_referral: ConsultationReferral = Field(
        default_factory=ConsultationReferral,
        description="Generic referral information if escalation is warranted"
    )
    disclaimer: str = Field(
        default=UNIVERSAL_DISCLAIMER,
        description="Mandatory medical disclaimer returned on all responses"
    )
    analysis_timestamp: str = Field(..., description="ISO 8601 timestamp of analysis")


# ---------------------------------------------------------
# Consent & Data Retention Contract Schemas
# ---------------------------------------------------------
class ConsentRequest(BaseModel):
    user_id: str = Field(..., description="Unique user or device pseudonym ID")
    biometric_image_consent: bool = Field(
        ...,
        description="Explicit consent to store and process scalp photos for health scoring"
    )
    research_opt_in: bool = Field(
        default=False,
        description="Optional consent to include anonymized photos in future calibration sets"
    )
    data_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="User-selected retention window before automatic purge (default 30 days)"
    )


class ConsentResponse(BaseModel):
    status: str = "success"
    user_id: str
    consent_recorded: bool
    retention_days: int
    storage_security: str = "AES-256 encrypted at rest in isolated Supabase storage bucket"
    message: str = "Consent preferences successfully recorded."
    disclaimer: str = UNIVERSAL_DISCLAIMER


# ---------------------------------------------------------
# System Status / Warmup Hook Schema (Cold-Start Polling)
# ---------------------------------------------------------
class SystemStatusResponse(BaseModel):
    status: Literal["ready", "warming_up", "standby"] = "ready"
    is_ready: bool = True
    model_loaded: bool = True
    model_architecture: str = "mobilenetv4"
    confidence_gate_threshold: float = 0.65
    server_time: str
    free_tier_notice: str = (
        "Free-tier instances (Render/Hugging Face) may take up to 45s on initial spin-up. "
        "Client apps should poll this endpoint before submitting scans."
    )
