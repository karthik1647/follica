"""
Follica Server-Side Triage and Clinical Scoring Engine.

Architectural Design Intent and Clinical Rationale:
1. Multi-Factor Clinical Correlation:
   - In addition to visual computer vision feature extraction and classifier gating,
     this engine correlates patient lifestyle and demographic risk factors:
     * Age + Family History -> Androgenetic Alopecia (AGA) miniaturization risk
     * Sudden Shedding + Positive Pull Test + Stress/Deficiencies -> Acute Telogen Effluvium (TE)
     * Greasy Scaling + Daily Helmet + Hard Water -> Malassezia yeast overgrowth and sebum occlusion
     * Hard Water + Dry Powder Flakes -> Stratum corneum mineral buildup
     * Scalp Burning/Trichodynia -> Perifollicular inflammation risk
   - These correlations are returned in the `clinical_correlations` array.

2. Server-Side Confidence Gate (< 0.65):
   - Inconclusive visual confidence is strictly diverted to Level 2 (Monitored Routine).
"""

from datetime import datetime, timezone
from typing import List

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
    CONDITION_ANDROGENETIC_ALOPECIA,
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
    ConsultationReferral,
    ClinicalCorrelationFactor,
    RiskIndicatorItem,
    RiskIndicators
)


class TriageEngine:
    """
    Evaluates multimodal signals, correlates multi-factor lifestyle data,
    enforces server-side confidence gates, and constructs non-diagnostic triage plans.
    """

    @classmethod
    def compute_correlations(cls, context: SymptomContext, sub_scores: SubScores) -> List[ClinicalCorrelationFactor]:
        """
        Derives physiological correlations between self-reported context and scalp state.
        """
        correlations = []

        # 1. Hard Water & Mineral Occlusion
        if context.hard_water_exposure or context.water_environment == "hard_borewell":
            correlations.append(ClinicalCorrelationFactor(
                factor="Hard Water Mineral Deposition",
                impact_level="high",
                clinical_note="High calcium/magnesium ion concentration binds with fatty acids in sebum, forming insoluble soap scum that disrupts the stratum corneum barrier."
            ))

        # 2. Daily Helmet / Sweat Occlusion
        if context.daily_helmet_wearer:
            correlations.append(ClinicalCorrelationFactor(
                factor="Daily Helmet / Occlusive Friction",
                impact_level="high" if context.itching else "moderate",
                clinical_note="Prolonged helmet wear traps heat, perspiration, and sebum, creating a warm micro-environment that accelerates Malassezia fungal proliferation and mechanical traction."
            ))

        # 3. High PM2.5 Pollution & Particulate Settling
        if context.pollution_exposure in ["high_metro_industrial", "high_campus_dust"]:
            correlations.append(ClinicalCorrelationFactor(
                factor="Atmospheric Particulate (PM2.5) Load",
                impact_level="moderate",
                clinical_note="Fine particulate matter adheres to sebum-coated hair shafts and oxidizes surface lipids, triggering low-grade micro-inflammation."
            ))

        # 4. Acute Telogen Effluvium Correlation (Pull Test + Sudden Shedding + Stress/Diet)
        is_te_profile = (
            context.shedding_pattern == "sudden_acute" or
            context.hair_pull_test == "positive_over_5" or
            context.recent_exam_work_stress or
            context.stress_level == "high" or
            context.diet_pattern == "low_protein_crash_diet"
        )
        if is_te_profile and context.hair_pull_test != "negative_under_3":
            correlations.append(ClinicalCorrelationFactor(
                factor="Acute Telogen Effluvium Risk Dynamics",
                impact_level="high",
                clinical_note="Sudden shedding combined with a positive hair pull test and physiological stressors (exam stress, low protein intake, or nutritional gap) suggests premature follicle transition from anagen to telogen."
            ))

        # 5. Androgenetic Miniaturization Correlation (Family History + Gradual Thinning + Age)
        if context.family_history_hair_loss and (context.primary_concern_zone in ["crown_vertex", "hairline_temples"] or context.shedding_pattern == "gradual_chronic"):
            correlations.append(ClinicalCorrelationFactor(
                factor="Familial Androgenetic Miniaturization Predisposition",
                impact_level="high",
                clinical_note="Genetic sensitivity to dihydrotestosterone (DHT) progressively shortens the anagen growth cycle and miniaturizes hair shaft diameter along the frontal hairline and vertex."
            ))

        # 6. Scalp Trichodynia / Inflammatory Hyperalgesia
        if context.scalp_pain_trichodynia or context.burning_or_pain:
            correlations.append(ClinicalCorrelationFactor(
                factor="Perifollicular Trichodynia & Nerve Sensitization",
                impact_level="high",
                clinical_note="Scalp tenderness or burning upon hair movement indicates localized perifollicular inflammation, release of substance P, and vascular irritation."
            ))

        # 7. Nutritional & Blood Marker Deficiencies
        if context.deficiency_flags or context.diet_pattern == "low_protein_crash_diet":
            flags_str = ", ".join(context.deficiency_flags) if context.deficiency_flags else "low dietary protein"
            correlations.append(ClinicalCorrelationFactor(
                factor=f"Nutritional Synthesis Gap ({flags_str})",
                impact_level="moderate",
                clinical_note="Keratin synthesis and follicle matrix division depend heavily on adequate serum ferritin, 25-hydroxyvitamin D3, and amino acid availability."
            ))

        # 8. Gastrointestinal Absorption & Gut-Skin-Hair Axis (Traya 3-Science Model)
        if context.gut_health in ["frequent_bloating_acidity", "chronic_constipation", "irregular_ibs"]:
            gut_label = context.gut_health.replace('_', ' ').title()
            correlations.append(ClinicalCorrelationFactor(
                factor=f"Gastrointestinal Absorption Barrier ({gut_label})",
                impact_level="high" if context.gut_health == "irregular_ibs" else "moderate",
                clinical_note="Frequent gastrointestinal distress, dysbiosis, or acidity impairs duodenal and jejunal micronutrient absorption (ferritin, zinc, biotin, amino acids), starving hair matrix keratinocytes during anagen synthesis."
            ))

        # 9. Systemic Metabolic Energy & Fatigue Indicator
        if context.metabolic_energy in ["chronic_sluggish_fatigue", "sudden_weight_shift"]:
            meta_label = context.metabolic_energy.replace('_', ' ').title()
            correlations.append(ClinicalCorrelationFactor(
                factor=f"Systemic Metabolic Load ({meta_label})",
                impact_level="moderate",
                clinical_note="Chronic fatigue or sudden weight shifts signal systemic metabolic deceleration or adrenal-cortisol strain, compromising follicular micro-circulation."
            ))

        # 10. Endocrine & Hormonal Profile
        if context.hormonal_profile in ["thyroid_dysfunction", "pcos_irregular_cycles", "insulin_resistance"]:
            endo_label = context.hormonal_profile.replace('_', ' ').title()
            correlations.append(ClinicalCorrelationFactor(
                factor=f"Endocrine Axis Imbalance ({endo_label})",
                impact_level="high",
                clinical_note="Endocrine dysregulation directly alters follicular cycling: thyroid slowdown prolongs telogen resting, while hyperandrogenism/insulin resistance elevates local follicular DHT sensitivity."
            ))

        # 11. Optimal Homeostasis Factor for Healthy Users
        if (
            not context.itching
            and not context.burning_or_pain
            and context.flake_type == "none"
            and context.hair_pull_test == "negative_under_3"
            and context.gut_health == "optimal"
            and context.metabolic_energy == "balanced"
            and context.hormonal_profile == "none"
        ):
            correlations.append(ClinicalCorrelationFactor(
                factor="Optimal Scalp & Systemic Homeostasis",
                impact_level="low",
                clinical_note="No active inflammation, microbial scaling, gut dysbiosis, or pathological shedding reported. Epidermal barrier function, metabolic balance, and follicle retention are intact."
            ))
        elif not context.itching and not context.burning_or_pain and context.flake_type == "none" and context.hair_pull_test == "negative_under_3":
            correlations.append(ClinicalCorrelationFactor(
                factor="Optimal Scalp Homeostasis & Protective Acid Mantle",
                impact_level="low",
                clinical_note="No active cutaneous inflammation or microbial scaling detected on the scalp surface."
            ))

        # 12. Demographic Presentation Profile (Age Group + Biological Sex)
        # Informational explanatory text only; no numeric score weights.
        pattern_desc = ""
        if context.biological_sex == "male":
            pattern_desc = "In biological males, androgenetic sensitivity typically manifests as bitemporal recession and vertex thinning (Norwood pattern)."
        elif context.biological_sex == "female":
            pattern_desc = "In biological females, pattern thinning commonly presents as diffuse widening along the central parting with frontal hairline preservation (Ludwig pattern)."
        else:
            pattern_desc = "Pattern hair thinning may present with either diffuse central widening or frontal edge recession."

        age_desc = ""
        if context.age_group == "under_18":
            age_desc = "In individuals under 18, androgenetic miniaturization is relatively rare; shedding is more commonly linked to nutritional demands, traction, or acute telogen shifts. Note: This assessment tool is not intended for unsupervised use by minors; involvement of a parent or guardian and a paediatric dermatologist is recommended."
        elif context.age_group in ["18_25", "26_35"]:
            age_desc = f"In early adulthood ({context.age_group.replace('_', '-')}), hair cycling dynamics frequently reflect genetic predisposition, lifestyle transitions, or nutritional demands."
        else:
            age_desc = f"In mature age brackets ({context.age_group.replace('_', '-')}), progressive age-related follicular diameter reduction and hormonal shifts commonly influence overall visible density."

        correlations.append(ClinicalCorrelationFactor(
            factor="Demographic Presentation Profile",
            impact_level="moderate" if context.age_group == "under_18" else "low",
            clinical_note=f"{pattern_desc} {age_desc}"
        ))

        return correlations

    @classmethod
    def compute_risk_indicators(cls, context: SymptomContext) -> RiskIndicators:
        """
        Derives self-reported questionnaire risk indicators using transparent count-based rules.
        Does NOT blend into the image-derived Scalp Health Score (provisional / uncalibrated).
        """
        markers: List[RiskIndicatorItem] = []

        # 1. Hair pull test result
        if context.hair_pull_test == "positive_over_5":
            markers.append(RiskIndicatorItem(
                name="Hair Pull Test: Positive (>5 hairs)",
                detail="Active shedding detected during gentle traction test (>5 hairs released)."
            ))
        elif context.hair_pull_test == "borderline_3_5":
            markers.append(RiskIndicatorItem(
                name="Hair Pull Test: Borderline (3-5 hairs)",
                detail="Elevated shedding observed during traction test (3-5 hairs released)."
            ))

        # 2. Shedding pattern
        if context.shedding_pattern == "sudden_acute":
            markers.append(RiskIndicatorItem(
                name="Shedding Pattern: Sudden Acute",
                detail="Rapid-onset acute shedding reported (characteristic of telogen effluvium dynamics)."
            ))
        elif context.shedding_pattern == "gradual_chronic":
            markers.append(RiskIndicatorItem(
                name="Shedding Pattern: Gradual Chronic",
                detail="Gradual progressive shedding or thinning reported over time."
            ))

        # 3. Scalp pain / trichodynia
        if context.scalp_pain_trichodynia:
            markers.append(RiskIndicatorItem(
                name="Scalp Pain / Trichodynia",
                detail="Scalp pain or tenderness upon hair movement (indicative of perifollicular nerve irritation)."
            ))

        # 4. Burning or pain
        if context.burning_or_pain:
            markers.append(RiskIndicatorItem(
                name="Burning Sensation or Pain",
                detail="Active burning or painful tenderness reported across the scalp."
            ))

        # 5. Deficiency flags
        if context.deficiency_flags:
            flags_str = ", ".join(context.deficiency_flags)
            markers.append(RiskIndicatorItem(
                name=f"Known Deficiencies ({flags_str})",
                detail=f"Self-reported nutritional or blood biomarker deficiencies: {flags_str}."
            ))

        # 6. Diet pattern
        if context.diet_pattern == "low_protein_crash_diet":
            markers.append(RiskIndicatorItem(
                name="Diet Pattern: Low-Protein / Crash Diet",
                detail="Inadequate protein intake restricts amino acid substrate required for keratin matrix synthesis."
            ))

        # 7. Gut health
        if context.gut_health in ["frequent_bloating_acidity", "chronic_constipation", "irregular_ibs"]:
            gut_lbl = context.gut_health.replace("_", " ").title()
            markers.append(RiskIndicatorItem(
                name=f"Gut Health: {gut_lbl}",
                detail="Digestive distress or gut dysbiosis can impair intestinal absorption of micronutrients."
            ))

        # 8. Metabolic energy
        if context.metabolic_energy in ["chronic_sluggish_fatigue", "sudden_weight_shift"]:
            meta_lbl = context.metabolic_energy.replace("_", " ").title()
            markers.append(RiskIndicatorItem(
                name=f"Metabolic Energy: {meta_lbl}",
                detail="Chronic fatigue or sudden weight shifts signal systemic metabolic or adrenal load."
            ))

        # 9. Hormonal profile
        if context.hormonal_profile in ["thyroid_dysfunction", "pcos_irregular_cycles", "insulin_resistance"]:
            endo_lbl = context.hormonal_profile.replace("_", " ").title()
            markers.append(RiskIndicatorItem(
                name=f"Hormonal Profile: {endo_lbl}",
                detail="Endocrine imbalance directly impacts follicle cycling and androgen receptor sensitivity."
            ))

        # 10. Duration (provisional threshold: >=3 months or >=12 weeks indicates prolonged/chronic involvement)
        if context.duration_months >= 3 or context.duration_weeks >= 12:
            markers.append(RiskIndicatorItem(
                name=f"Prolonged Duration ({context.duration_months} mo / {context.duration_weeks} wk)",
                detail="Symptoms have persisted for 3 months or longer, suggesting non-transient progression."
            ))

        # 11. Family history of hair loss
        if context.family_history_hair_loss:
            markers.append(RiskIndicatorItem(
                name="Family History: Hair Loss",
                detail="Genetic predisposition to androgenetic pattern hair loss."
            ))

        marker_count = len(markers)
        has_pain = context.scalp_pain_trichodynia or context.burning_or_pain

        # Transparent count-based rule with clinical pain escalation (provisional / uncalibrated):
        # 0 markers: low
        # 1-2 markers (and no severe inflammatory pain): moderate
        # 3+ markers OR presence of acute inflammatory pain: elevated
        if marker_count == 0:
            overall_level = "low"
        elif marker_count <= 2 and not has_pain:
            overall_level = "moderate"
        else:
            overall_level = "elevated"

        return RiskIndicators(
            overall_level=overall_level,
            marker_count=marker_count,
            triggered_markers=markers,
            note="Self-reported, not clinically validated. Kept separate from image-derived Scalp Health Score."
        )

    @classmethod
    def evaluate(
        cls,
        raw_condition: str,
        confidence: float,
        sub_scores: SubScores,
        context: SymptomContext
    ) -> AnalysisResponse:
        """
        Main triage evaluation method with multi-factor correlation synthesis.
        Supports healthy routine baseline checkups as well as active clinical triage.
        """
        total_shs = round(
            sub_scores.flake_score +
            sub_scores.redness_score +
            sub_scores.sebum_score +
            sub_scores.density_score,
            2
        )
        total_shs = float(max(0.0, min(100.0, total_shs)))

        correlations = cls.compute_correlations(context, sub_scores)
        risk_indicators = cls.compute_risk_indicators(context)

        # -------------------------------------------------------------------
        # 1. Server-Side Confidence Gate Enforcement
        # -------------------------------------------------------------------
        confidence_gated = False
        final_condition = raw_condition

        if confidence < settings.CONFIDENCE_GATE_THRESHOLD:
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
            is_level_3_condition = raw_condition in [
                CONDITION_ALOPECIA_AREATA,
                CONDITION_FOLLICULITIS,
                CONDITION_SCALP_PSORIASIS
            ]
            has_severe_pain = context.burning_or_pain or context.scalp_pain_trichodynia
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
                        f"Trichodynia={context.scalp_pain_trichodynia}, "
                        f"Duration={context.duration_months} months ({context.duration_weeks} weeks). Scalp Health Score={total_shs}/100."
                    )
                )

            elif raw_condition == CONDITION_ANDROGENETIC_ALOPECIA:
                triage_level = TRIAGE_LEVEL_2_MONITORED
                triage_reason = (
                    "Visual patterns and reported intake indicate progressive follicular miniaturization consistent with pattern hair loss (androgenetic alopecia). "
                    "Structured routine monitoring and consultation with a dermatologist are recommended to discuss medical management options."
                )
                guide = INGREDIENT_GUIDES.get(raw_condition, INGREDIENT_GUIDES[CONDITION_NORMAL])
                ingredients = [
                    IngredientRecommendation(**item) for item in guide.get("ingredients", [])
                ]
                lifestyle = guide.get("lifestyle", [])
                referral = ConsultationReferral(
                    required=True,
                    urgency="recommended",
                    addressed_to="For your treating physician / dermatologist",
                    search_url=GENERIC_DERMATOLOGIST_SEARCH_URL,
                    clinical_hand_off_notes=(
                        f"Pattern hair loss assessment suggests {raw_condition}. "
                        f"Duration={context.duration_months} months ({context.duration_weeks} weeks). "
                        f"Scalp Health Score={total_shs}/100. Dermatologist evaluation recommended for clinical trichoscopy."
                    )
                )

            elif raw_condition == CONDITION_SEBORRHEIC_DERMATITIS or context.flake_type == "greasy_yellow_thick" or (context.itching and (context.duration_months >= 1 or context.duration_weeks > 4)):
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

            elif raw_condition == CONDITION_NORMAL or (total_shs >= 75.0 and not context.itching and not context.burning_or_pain and context.flake_type == "none"):
                triage_level = TRIAGE_LEVEL_1_MINOR
                final_condition = CONDITION_NORMAL
                triage_reason = (
                    "Visual signals and clinical intake confirm optimal scalp homeostasis, intact "
                    "protective acid mantle, and balanced sebum levels. Preventive care regimen recommended."
                )
                guide = INGREDIENT_GUIDES.get(CONDITION_NORMAL, INGREDIENT_GUIDES[CONDITION_DRY_SCALP])
                ingredients = [
                    IngredientRecommendation(**item) for item in guide.get("ingredients", [])
                ]
                lifestyle = guide.get("lifestyle", [])
                referral = ConsultationReferral(
                    required=False,
                    urgency="routine",
                    addressed_to="For your treating physician / dermatologist",
                    search_url=GENERIC_DERMATOLOGIST_SEARCH_URL,
                    clinical_hand_off_notes="Routine healthy scalp checkup. No active clinical intervention indicated."
                )

            else:
                triage_level = TRIAGE_LEVEL_1_MINOR
                triage_reason = (
                    "Visual indications suggest mild surface dryness or minor flaking. "
                    "Standard supportive barrier hydration and gentle cleansing recommended."
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

        # -------------------------------------------------------------------
        # 3. Systemic Biomarker Guidance & Physician Hand-off Enrichment
        # -------------------------------------------------------------------
        lifestyle_enriched = list(lifestyle)
        if context.gut_health in ["frequent_bloating_acidity", "chronic_constipation", "irregular_ibs"]:
            lifestyle_enriched.append(
                "Support the gut-skin-hair axis: prioritize dietary fiber, fermented foods, and hydration to optimize micronutrient (iron, zinc, amino acid) bioavailability."
            )
        if context.hormonal_profile in ["thyroid_dysfunction", "pcos_irregular_cycles", "insulin_resistance"] or context.metabolic_energy in ["chronic_sluggish_fatigue", "sudden_weight_shift"]:
            lifestyle_enriched.append(
                "Discuss targeted blood work (TSH, Free T3/T4, Serum Ferritin, Vitamin D3, Fasting Glucose) with your physician to evaluate systemic metabolic drivers."
            )

        # Age group guardian advisory
        if context.age_group == "under_18":
            lifestyle_enriched.append(
                "Notice for users under 18: Follica is an informational tool not intended for unsupervised use by minors. Please review your symptoms with a parent or legal guardian and seek evaluation from a qualified paediatric physician or dermatologist before starting any new scalp treatments."
            )

        # Washing frequency guidance
        if context.washing_frequency == "daily":
            lifestyle_enriched.append(
                "Washing frequency advice: Daily washing with standard surfactants can strip the scalp's protective lipid barrier and provoke compensatory sebum rebound. Consider alternating washes with lukewarm water or switching to an ultra-gentle, sulfate-free daily formulation."
            )
        elif context.washing_frequency in ["once_per_week", "rarely"]:
            lifestyle_enriched.append(
                "Washing frequency advice: Infrequent scalp cleansing allows sebum, environmental debris, and shed keratinocytes to accumulate, fostering an environment for microbial and Malassezia yeast overgrowth. Aim for a regular cadence of 2-3 washes per week."
            )

        # Chemical / thermal heat styling guidance
        if context.chemical_heat_styling:
            lifestyle_enriched.append(
                "Thermal and chemical styling advisory: Frequent high-heat blow drying, straightening, or chemical treatments can damage the protective hair cuticle and weaken tensile strength; use thermal protection sprays and minimize heat exposure during recovery."
            )

        # Stress level guidance
        if context.stress_level == "high":
            lifestyle_enriched.append(
                "Stress management: Elevated psychological stress increases circulating cortisol levels, which can prematurely trigger follicle transition into the telogen resting phase. Prioritize stress-reduction practices and regular physical activity."
            )

        # Sleep quality guidance
        if context.sleep_quality in ["irregular_under_6h", "shift_work_disrupted"]:
            lifestyle_enriched.append(
                "Sleep hygiene: Irregular or inadequate sleep (<6 hours) disrupts nocturnal cellular repair cycles and hormone regulation essential for sustained follicular anagen growth. Aim for consistent, restorative sleep windows."
            )

        if context.gut_health != "optimal" or context.metabolic_energy != "balanced" or context.hormonal_profile != "none":
            systemic_tag = f" Systemic Profile: Gut={context.gut_health}, Energy={context.metabolic_energy}, Hormonal={context.hormonal_profile}."
            referral.clinical_hand_off_notes = (referral.clinical_hand_off_notes.rstrip(".") + "." + systemic_tag).strip()

        return AnalysisResponse(
            suggested_condition=final_condition,
            confidence_score=confidence,
            confidence_gated=confidence_gated,

            sub_scores=sub_scores,
            scalp_health_score=total_shs,
            triage_level=triage_level,
            triage_reason=triage_reason,
            clinical_correlations=correlations,
            recommended_ingredients=ingredients,
            lifestyle_guidance=lifestyle_enriched,
            consultation_referral=referral,
            disclaimer=UNIVERSAL_DISCLAIMER,
            risk_indicators=risk_indicators,
            analysis_timestamp=datetime.now(timezone.utc).isoformat()
        )


triage_engine = TriageEngine()