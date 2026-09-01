"""
Follica Pydantic Data Models and API Contracts.

Architectural Design Intent and Clinical Rationale:
1. Multi-Factor Context Engine:
   - Evaluates real-world physiological and lifestyle variables:
     * Demographics (Age bracket, Biological Sex)
     * Environmental Stressors (PM2.5 Pollution, Daily Helmet/Sweat Trap, Water Source)
     * Hair Care Mechanics (Washing Frequency, Chemical/Heat Damage, Traction)
     * Metabolic & Nutritional Biomarkers (Protein intake, Vitamin D3/Iron flags, Sleep)
     * Clinical Follicular Dynamics (Hair Pull Test, Flake Consistency, Scalp Trichodynia)
   - These variables correlate with visual computer vision signals to form a comprehensive clinical picture.

2. Why Sub-Scores Are Kept Separate (Not Pre-Summed):
   - Flake, Redness, Sebum, and Density are distinct physiological manifestations.
   - Exposing the 4 sub-scores as separate [0.0, 25.0] fields enables granular progress
     tracking across longitudinal scans.
"""

from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator
from app.core.constants import UNIVERSAL_DISCLAIMER


# ---------------------------------------------------------------------------
# Comprehensive Clinical Questionnaire Context Model
# ---------------------------------------------------------------------------
class SymptomContext(BaseModel):
    """
    Comprehensive self-reported patient symptom parameters and lifestyle context.
    Captures multi-angle orientation, demographic age, pollution, helmet usage,
    metabolic/dietary markers, multi-select sensations, and shedding dynamics.
    Supports healthy preventative baseline checkups as well as active symptoms.
    """
    # 1. Demographics & Concern Zone
    age_group: Literal["under_18", "18_25", "26_35", "36_50", "over_50"] = Field(
        default="18_25",
        description="Biological age bracket"
    )
    biological_sex: Literal["male", "female", "other"] = Field(
        default="male",
        description="Biological sex for hormonal pattern mapping (Norwood vs Ludwig)"
    )
    primary_concern_zone: Literal[
        "routine_healthy_check",
        "crown_vertex",
        "hairline_temples",
        "central_parting",
        "diffuse_all_over",
        "patchy_spots"
    ] = Field(default="crown_vertex", description="Primary anatomical area of concern or routine check")
    scan_angle: str = Field(default="crown_vertex", description="Targeted anatomical scalp angle")

    # 2. Multi-Select Sensations & Trichodynia
    sensation: str = Field(
        default="painless",
        description="Primary scalp sensation string (comma-separated if multiple)"
    )
    sensations: List[str] = Field(
        default_factory=lambda: ["painless"],
        description="List of multi-select scalp sensations (e.g. ['itching', 'burning_or_tender'])"
    )
    itching: bool = Field(default=False, description="Reported scalp itching or pruritus")
    burning_or_pain: bool = Field(default=False, description="Reported burning sensation, tenderness, or painful bumps")
    scalp_pain_trichodynia: bool = Field(
        default=False,
        description="Scalp hurts or burns when touching, brushing, or moving hair"
    )

    # 3. Environmental & Physical Stressors
    water_environment: Literal["hard_borewell", "treated_tap", "recent_relocation", "normal"] = Field(
        default="normal",
        description="Hostel or residence water supply and environmental changes"
    )
    hard_water_exposure: bool = Field(default=False, description="Exposure to hard borewell/tap water in living quarters")
    pollution_exposure: Literal["low_rural", "moderate_suburban", "high_metro_industrial", "high_campus_dust"] = Field(
        default="high_campus_dust",
        description="Atmospheric PM2.5 and urban dust exposure"
    )
    daily_helmet_wearer: bool = Field(
        default=False,
        description="Daily use of motorcycle helmet or tight headwear (friction & sweat occlusion)"
    )
    washing_frequency: Literal["daily", "2_3_per_week", "once_per_week", "rarely"] = Field(
        default="2_3_per_week",
        description="Scalp washing cadence"
    )
    chemical_heat_styling: bool = Field(
        default=False,
        description="Frequent heat blow-drying, chemical straightening, bleaching, or dyeing"
    )

    # 4. Metabolic, Dietary & Stress Factors
    diet_pattern: Literal["balanced_protein", "vegetarian_vegan", "low_protein_crash_diet"] = Field(
        default="balanced_protein",
        description="Nutritional protein and macro-nutrient balance"
    )
    sleep_quality: Literal["good_7_8h", "irregular_under_6h", "shift_work_disrupted"] = Field(
        default="good_7_8h",
        description="Circadian rhythm and sleep duration"
    )
    stress_level: Literal["low", "moderate", "high"] = Field(
        default="moderate",
        description="Self-reported psychological stress"
    )
    recent_exam_work_stress: bool = Field(
        default=False,
        description="Intense exam pressure, academic deadlines, or work stress in past 3 months"
    )
    family_history_hair_loss: bool = Field(
        default=False,
        description="Parental or sibling history of androgenetic alopecia / thinning"
    )
    deficiency_flags: List[str] = Field(
        default_factory=list,
        description="Known deficiencies (e.g., 'vitamin_d3', 'iron_ferritin', 'thyroid')"
    )

    # 5. Follicular Dynamics & Timeline
    hair_pull_test: Literal["negative_under_3", "borderline_3_5", "positive_over_5"] = Field(
        default="negative_under_3",
        description="Self-administered hair pull test outcome from scalp root"
    )
    flake_type: Literal["none", "dry_white_fine", "greasy_yellow_thick", "plaque_crusty"] = Field(
        default="none",
        description="Characteristic physical presentation of scalp scaling"
    )
    shedding_pattern: Literal["sudden_acute", "gradual_chronic", "stable_no_shedding"] = Field(
        default="gradual_chronic",
        description="Acute shedding (telogen effluvium pattern) vs gradual chronic miniaturization"
    )
    duration_weeks: int = Field(default=2, ge=0, description="Duration of symptoms in weeks (0 for routine healthy check)")
    symptom_notes: Optional[str] = Field(default=None, description="Free-text notes on past products or treatments tried")

    @model_validator(mode="before")
    @classmethod
    def synchronize_sensations_and_context(cls, data: Any) -> Any:
        """
        Automatically reconciles multi-select sensations with legacy single-string fields
        and boolean clinical flags.
        """
        if isinstance(data, dict):
            explicit_itching = str(data.get("itching", "")).lower() in ["true", "1"] or data.get("itching") is True
            explicit_pain = str(data.get("burning_or_pain", "")).lower() in ["true", "1"] or data.get("burning_or_pain") is True
            explicit_trichodynia = str(data.get("scalp_pain_trichodynia", "")).lower() in ["true", "1"] or data.get("scalp_pain_trichodynia") is True

            sens_list = []
            if "sensations" in data and isinstance(data["sensations"], list):
                sens_list.extend(data["sensations"])
            elif "sensations" in data and isinstance(data["sensations"], str):
                sens_list.extend([s.strip() for s in data["sensations"].split(",") if s.strip()])

            if "sensation" in data and isinstance(data["sensation"], str):
                for item in data["sensation"].split(","):
                    item_clean = item.strip()
                    if item_clean and item_clean not in sens_list:
                        sens_list.append(item_clean)

            if explicit_itching and "itching" not in sens_list:
                sens_list.append("itching")
            if explicit_pain and not any(s in sens_list for s in ["burning", "burning_or_tender", "painful", "painful_bumps"]):
                sens_list.append("burning")
            if explicit_trichodynia and "burning_or_tender" not in sens_list:
                sens_list.append("burning_or_tender")

            has_active_symptoms = explicit_itching or explicit_pain or explicit_trichodynia or any(
                s in sens_list for s in ["itching", "pruritus", "burning", "burning_or_tender", "painful", "painful_bumps", "tightness"]
            )
            if has_active_symptoms and "painless" in sens_list:
                sens_list = [s for s in sens_list if s != "painless"]

            if not sens_list:
                sens_list = ["painless"]

            data["sensations"] = sens_list
            data["sensation"] = ",".join(sens_list)

            sens_set = set(sens_list)
            data["itching"] = "itching" in sens_set or "pruritus" in sens_set or explicit_itching
            data["burning_or_pain"] = any(s in sens_set for s in ["burning", "burning_or_tender", "tenderness", "painful", "painful_bumps"]) or explicit_pain
            data["scalp_pain_trichodynia"] = "burning_or_tender" in sens_set or "tenderness" in sens_set or explicit_trichodynia

            if not has_active_symptoms and "painless" in sens_set:
                data["itching"] = False
                data["burning_or_pain"] = False
                data["scalp_pain_trichodynia"] = False

            if data.get("water_environment") == "hard_borewell" or str(data.get("hard_water_exposure", "")).lower() in ["true", "1"] or data.get("hard_water_exposure") is True:
                data["hard_water_exposure"] = True

        return data



# ---------------------------------------------------------------------------
# Classical CV Sub-Scores Schema
# ---------------------------------------------------------------------------
class SubScores(BaseModel):
    """
    Discrete micro-feature sub-scores generated by the classical CV pipeline.
    Each sub-score is calibrated to a [0.0, 25.0] range, where 25.0 represents optimal health.
    """
    flake_score: float = Field(
        ...,
        ge=0.0,
        le=25.0,
        description="Flake & Scaling Score (0=severe flaking/scaling, 25=clear scalp)"
    )
    redness_score: float = Field(
        ...,
        ge=0.0,
        le=25.0,
        description="Erythema & Redness Score (0=severe erythema/vascular inflammation, 25=calm scalp)"
    )
    sebum_score: float = Field(
        ...,
        ge=0.0,
        le=25.0,
        description="Sebum & Lipid Balance Score (0=excessive oiliness or extreme dehydration, 25=balanced barrier)"
    )
    density_score: float = Field(
        ...,
        ge=0.0,
        le=25.0,
        description="Hair Follicle & Density Score (0=significant thinning/spacing, 25=dense follicular coverage)"
    )
    stage_comparison: Optional[str] = Field(
        default="Within baseline density boundaries (Norwood 1-2 / Ludwig 1)",
        description="Pattern stage reference against standardized hair loss progression scales"
    )


class IngredientRecommendation(BaseModel):
    """Structured OTC active molecule recommendation."""
    name: str = Field(..., description="Active chemical molecule (e.g., Ketoconazole 2%, Salicylic Acid 2%)")
    frequency: str = Field(..., description="Suggested application cadence (e.g., 2x per week)")
    purpose: str = Field(..., description="Mechanism of action explaining why this molecule is recommended")


class ClinicalCorrelationFactor(BaseModel):
    """Correlated lifestyle / demographic risk driver."""
    factor: str = Field(..., description="Identified lifestyle or clinical risk factor")
    impact_level: Literal["high", "moderate", "low"] = "moderate"
    clinical_note: str = Field(..., description="Physiological explanation of how this factor affects hair/scalp")


class ConsultationReferral(BaseModel):
    """Generic clinical escalation data model."""
    required: bool = Field(default=False, description="Whether consultation with a dermatologist is recommended")
    urgency: Literal["routine", "recommended", "immediate"] = "routine"
    addressed_to: str = Field(
        default="For your treating physician / dermatologist",
        description="Generic recipient title for clinical hand-off"
    )
    search_url: str = Field(
        default="https://www.google.com/maps/search/dermatologist+near+me",
        description="Generic self-triggered local clinic search query"
    )
    clinical_hand_off_notes: str = Field(
        default="",
        description="Concise summary of visual patterns and context for review by a healthcare professional"
    )


# ---------------------------------------------------------------------------
# Primary Analysis Response Schema
# ---------------------------------------------------------------------------
class AnalysisResponse(BaseModel):
    """
    Full dual-pipeline response contract.
    Combines DL classification, CV sub-scores, multi-factor correlations, server-gated triage, and disclaimers.
    """
    suggested_condition: str = Field(
        ...,
        description="Non-diagnostic condition suggestion ('suggests', never 'diagnoses')"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classifier probability score for the suggested condition"
    )
    confidence_gated: bool = Field(
        default=False,
        description="True if low classifier confidence (<0.65) forced rerouting to Level 2"
    )
    sub_scores: SubScores = Field(
        ...,
        description="Discrete classical-CV derived sub-scores in range [0, 25]"
    )
    scalp_health_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite Scalp Health Score (0-100, arithmetic sum of the 4 sub-scores)"
    )
    triage_level: str = Field(
        ...,
        description="Triage categorization (Level 1 Minor, Level 2 Monitored, Level 3 Escalation)"
    )
    triage_reason: str = Field(
        ...,
        description="Clinical rationale explaining the triage categorization in non-diagnostic language"
    )
    clinical_correlations: List[ClinicalCorrelationFactor] = Field(
        default_factory=list,
        description="Correlated multi-factor lifestyle and physiological risk drivers"
    )
    recommended_ingredients: List[IngredientRecommendation] = Field(
        default_factory=list,
        description="List of suggested active OTC ingredients"
    )
    lifestyle_guidance: List[str] = Field(
        default_factory=list,
        description="Supportive scalp care, hygiene, and environmental recommendations"
    )
    consultation_referral: ConsultationReferral = Field(
        default_factory=ConsultationReferral,
        description="Generic referral information if escalation is warranted"
    )
    disclaimer: str = Field(
        default=UNIVERSAL_DISCLAIMER,
        description="Universal medical disclaimer mandatory across all response payloads"
    )
    analysis_timestamp: str = Field(..., description="ISO 8601 UTC timestamp of analysis")


# ---------------------------------------------------------------------------
# Consent and Data Retention Contract
# ---------------------------------------------------------------------------
class ConsentRequest(BaseModel):
    user_id: str = Field(..., description="Pseudonymous user or device identifier")
    biometric_image_consent: bool = Field(
        ...,
        description="Explicit user consent to store and process scalp photos for health scoring"
    )
    research_opt_in: bool = Field(
        default=False,
        description="Optional consent to include anonymized photos in future model calibration sets"
    )
    data_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="User-selected retention period before automatic deletion (default 30 days)"
    )


class ConsentResponse(BaseModel):
    status: str = "success"
    user_id: str
    consent_recorded: bool
    retention_days: int
    storage_security: str = "AES-256 encrypted at rest in Supabase bucket with RLS policies"
    message: str = "Consent preferences and retention window successfully registered."
    disclaimer: str = UNIVERSAL_DISCLAIMER


# ---------------------------------------------------------------------------
# Cold-Start Warmup & Health Hook Schema
# ---------------------------------------------------------------------------
class SystemStatusResponse(BaseModel):
    status: Literal["ready", "warming_up", "standby"] = "ready"
    is_ready: bool = True
    model_loaded: bool = True
    model_architecture: str = "mobilenetv4"
    confidence_gate_threshold: float = 0.65
    server_time: str
    free_tier_notice: str = (
        "Free-tier cloud containers (Render/Hugging Face) may take up to 45 seconds on cold start. "
        "Client applications should poll this endpoint before submitting scan payloads."
    )

# ---------------------------------------------------------------------------
# Longitudinal Progress Tracking Contract
# ---------------------------------------------------------------------------
class ProgressComparisonResponse(BaseModel):
    days_elapsed: int = Field(default=14, description="Days elapsed between baseline and follow-up scan")
    alignment_quality: str = Field(..., description="Homography alignment confidence (high, moderate, low)")
    follicle_density_delta_pct: float = Field(..., description="Percentage change in follicular coverage index")
    redness_reduction_delta_pct: float = Field(..., description="Percentage change in erythema status (positive is improved)")
    flake_reduction_delta_pct: float = Field(..., description="Percentage change in scaling status (positive is improved)")
    overall_trend: Literal["improving", "stable", "monitoring_advised", "escalation_recommended"] = Field(...)
    clinical_interpretation: str = Field(..., description="Objective clinical progress summary")
    recommendation: str = Field(..., description="Actionable next step recommendation")
    disclaimer: str = Field(default=UNIVERSAL_DISCLAIMER, description="Medical disclaimer")

# ---------------------------------------------------------------------------
# DermaIQ Modular Domain Architecture Schema
# ---------------------------------------------------------------------------
class DermaIQDomainInfo(BaseModel):
    domain_id: str
    display_name: str
    description: str
    status: str
    supported_models: List[str]
    active_sub_scores: List[str]