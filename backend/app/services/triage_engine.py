"""
Follica Server-Side Triage and Clinical Scoring Engine.

Architectural Design Intent & Rationale:
1. Why the Confidence Gate Reroutes to Level 2 (Monitored Routine):
   - When a deep learning model produces a low-confidence classification (< 0.65),
     the underlying pattern is ambiguous (e.g. poor photo lighting, blurry focus, or early-stage symptoms).
   - Why not Level 1 (Minor)?
     If an actual inflammatory infection (Folliculitis) or autoimmune loss (Alopecia Areata)
     is present but poorly captured, routing to Level 1 would falsely reassure the user,
     risking disease progression.
   - Why not Level 3 (Doctor Escalation)?
     If ordinary dandruff is captured poorly, routing to Level 3 causes unnecessary panic
     and places undue burden on clinical dermatology queues.
   - Level 2 (Monitored Routine) is the optimal medical-adjacent triage: it guides the user
     to use non-irritating base care, improve scan conditions, and track changes over 7-14 days.

2. Transparent Composite Health Score:
   - Scalp Health Score = Flake + Redness + Sebum + Density in range [0.0, 100.0].
   - Each component is bounded to [0.0, 25.0], ensuring clear visibility into what drives the total score.
"""

from datetime import datetime, timezone
from app.core.config import settings
from app.core.constants import (
    UNIVERSAL_DISCLAIMER,
    GENERIC_DERMATOLOGIST_SEARCH_URL,
    CONDITION_NORMAL,
    CONDITION_DRY_SCALP,
    CONDITION_SEBORRHEIC_DERMATITIS,
    CONDITION_ALOPECIA_AREATA,
    CONDITION_FOLLICULITIS,
    CONDITION_SCALP_PSORIASIS,
    CONDITION_UNVERIFIED_MONITORED,
    TRIAGE_LEVEL_1_MINOR,
    TRIAGE_LEVEL_2_MONITORED,
    TRIAGE_LEVEL_3_ESCALATION,
    INGREDIENT_GUIDES
)
from app.models.schemas import (
    SymptomContext,
    SubScores,
    AnalysisResponse,
    IngredientRecommendation,
    ConsultationReferral
)


class TriageEngine:
    """
    Evaluates multimodal signals, enforces server-side confidence gates,
    and constructs non-diagnostic, safety-compliant triage recommendations.
    """

    @classmethod
    def evaluate(
        cls,
        raw_condition: str,
        confidence: float,
        sub_scores: SubScores,
        context: SymptomContext
    ) -> AnalysisResponse:
        """
        Main triage evaluation method.
        """
        # Calculate composite Scalp Health Score (0.0 to 100.0) from discrete CV sub-scores
        total_shs = round(
            sub_scores.flake_score +
            sub_scores.redness_score +
            sub_scores.sebum_score +
            sub_scores.density_score,
            2
        )
        total_shs = float(max(0.0, min(100.0, total_shs)))

        # -------------------------------------------------------------------
        # 1. Server-Side Confidence Gate Enforcement
        # -------------------------------------------------------------------
        confidence_gated = False
        final_condition = raw_condition

        if confidence < settings.CONFIDENCE_GATE_THRESHOLD:
            # Low classifier confidence: Reroute strictly to Level 2 (Monitored)
            confidence_gated = True
            final_condition = CONDITION_UNVERIFIED_MONITORED
            triage_level = TRIAGE_LEVEL_2_MONITORED
            triage_reason = (
                f"Model classification confidence ({confidence:.2f}) is below the clinical "
                f"threshold ({settings.CONFIDENCE_GATE_THRESHOLD:.2f}). To ensure safety, this case is "
                f"rerouted to Level 2 (Monitored Routine) for self-tracking before any escalation."
            )
            ingredients = [
                IngredientRecommendation(
                    name="Gentle Sulfate-Free Daily Cleanser",
                    frequency="2-3x per week",
                    purpose="Maintains baseline cleanliness while tracking scalp symptoms."
                ),
                IngredientRecommendation(
                    name="Scalp Soothing Serum (Aloe Vera / Panthenol)",
                    frequency="Daily",
                    purpose="Supports barrier comfort without altering clinical presentation."
                )
            ]
            lifestyle = [
                "Track symptoms daily and retake photo in 7-14 days with improved direct lighting.",
                "Avoid harsh physical exfoliation or untested chemical treatments."
            ]
            referral = ConsultationReferral(
                required=False,
                urgency="routine",
                clinical_hand_off_notes="Inconclusive visual classifier confidence. Monitoring recommended."
            )

        # -------------------------------------------------------------------
        # 2. Confident Prediction Triage Evaluation (Confidence >= 0.65)
        # -------------------------------------------------------------------
        else:
            # Check for Urgent Red Flags / Level 3 Escalation
            is_level_3_condition = raw_condition in [
                CONDITION_ALOPECIA_AREATA,
                CONDITION_FOLLICULITIS,
                CONDITION_SCALP_PSORIASIS
            ]
            has_severe_pain = context.burning_or_pain
            has_severe_erythema = sub_scores.redness_score < 8.0

            if is_level_3_condition or has_severe_pain or has_severe_erythema:
                triage_level = TRIAGE_LEVEL_3_ESCALATION
                triage_reason = (
                    "Visual patterns and symptom indicators suggest potential inflammatory or "
                    "localized follicular involvement that warrants direct clinical assessment."
                )
                guide = INGREDIENT_GUIDES.get(raw_condition, INGREDIENT_GUIDES[CONDITION_FOLLICULITIS])
                ingredients = [
                    IngredientRecommendation(**item) for item in guide.get("ingredients", [])
                ]
                lifestyle = guide.get("lifestyle", [])
                referral = ConsultationReferral(
                    required=True,
                    urgency="recommended" if not has_severe_pain else "immediate",
                    addressed_to="For your treating physician / dermatologist",
                    search_url=GENERIC_DERMATOLOGIST_SEARCH_URL,
                    clinical_hand_off_notes=(
                        f"Visual analysis suggests {raw_condition}. "
                        f"Reported symptoms: Burning/Pain={context.burning_or_pain}, "
                        f"Duration={context.duration_weeks} weeks. Scalp Health Score={total_shs}/100."
                    )
                )

            # Check for Level 2 Monitored Routine
            elif raw_condition == CONDITION_SEBORRHEIC_DERMATITIS or context.duration_weeks > 4:
                triage_level = TRIAGE_LEVEL_2_MONITORED
                triage_reason = (
                    "Visual indications suggest active sebum/microbial flaking. Recommends an "
                    "active ingredient schedule and structured 14-day progress tracking."
                )
                guide = INGREDIENT_GUIDES.get(raw_condition, INGREDIENT_GUIDES[CONDITION_SEBORRHEIC_DERMATITIS])
                ingredients = [
                    IngredientRecommendation(**item) for item in guide.get("ingredients", [])
                ]
                lifestyle = guide.get("lifestyle", [])
                referral = ConsultationReferral(
                    required=False,
                    urgency="routine",
                    addressed_to="For your treating physician / dermatologist",
                    search_url=GENERIC_DERMATOLOGIST_SEARCH_URL,
                    clinical_hand_off_notes="Monitored active ingredient regimen in progress. Review if unresolved after 4 weeks."
                )

            # Level 1 Minor Routine
            else:
                triage_level = TRIAGE_LEVEL_1_MINOR
                triage_reason = (
                    "Visual indications suggest mild surface dryness or balanced scalp state. "
                    "Standard supportive hygiene and barrier hydration recommended."
                )
                guide = INGREDIENT_GUIDES.get(raw_condition, INGREDIENT_GUIDES[CONDITION_DRY_SCALP])
                ingredients = [
                    IngredientRecommendation(**item) for item in guide.get("ingredients", [])
                ]
                lifestyle = guide.get("lifestyle", [])
                referral = ConsultationReferral(
                    required=False,
                    urgency="routine",
                    clinical_hand_off_notes="Self-managed mild condition."
                )

        return AnalysisResponse(
            suggested_condition=final_condition,
            confidence_score=confidence,
            confidence_gated=confidence_gated,
            sub_scores=sub_scores,
            scalp_health_score=total_shs,
            triage_level=triage_level,
            triage_reason=triage_reason,
            recommended_ingredients=ingredients,
            lifestyle_guidance=lifestyle,
            consultation_referral=referral,
            disclaimer=UNIVERSAL_DISCLAIMER,
            analysis_timestamp=datetime.now(timezone.utc).isoformat()
        )


triage_engine = TriageEngine()
