"""
Follica Post-Identification Survey Risk & Cohort Analytics Engine.

Architectural Design Intent:
1. Why a Separate Post-Diagnosis Survey Pipeline:
   - Before condition identification, self-reported lifestyle answers (hard water,
     helmet wear, stress, gut health, diet) are noisy and non-specific because the
     same stressor amplifies very different conditions in different people.
   - AFTER the visual classifier identifies the condition pattern (Layer 1), those
     same survey variables become high-signal clinical refiners (Layer 2):
       * Seborrheic Dermatitis + Hard Water + Helmet -> Malassezia occlusion cascade
       * Telogen Effluvium + Exam Stress + Low Protein -> Acute anagen-to-telogen shift
       * Androgenetic Alopecia + Family History + Diet -> Miniaturisation velocity
       * Folliculitis + Helmet + Infrequent Wash -> Bacterial ostial occlusion
   - Separating visual disease identification from post-identification survey
     stratification keeps the Scalp Health Score image-grounded while turning
     survey data into a personalised Trigger Ranking & Priority Action Queue.

2. Cohort Analytics (`analyze_cohort`):
   - Aggregates batch survey datasets collected from user populations to quantify
     top modifiable vs non-modifiable triggers per condition and average risk load.

All weights and risk-band cut-offs are PROVISIONAL / UNCALIBRATED engineering heuristics.
"""

from __future__ import annotations

from typing import Dict, Any, List, Literal
from collections import Counter

from app.core.constants import (
    CONDITION_NORMAL,
    CONDITION_DRY_SCALP,
    CONDITION_SEBORRHEIC_DERMATITIS,
    CONDITION_ALOPECIA_AREATA,
    CONDITION_FOLLICULITIS,
    CONDITION_ANDROGENETIC_ALOPECIA,
    CONDITION_SCALP_PSORIASIS,
    CONDITION_CONTACT_DERMATITIS,
    CONDITION_HEAD_LICE,
    CONDITION_LICHEN_PLANUS,
    CONDITION_TELOGEN_EFFLUVIUM,
    CONDITION_TINEA_CAPITIS,
    UNIVERSAL_DISCLAIMER,
)
from app.models.schemas import (
    SymptomContext,
    TriggerFactor,
    InterventionStep,
    SurveyRiskProfile,
    SurveyCohortRequest,
    SurveyCohortSummary,
)


class SurveyRiskEngine:
    """
    Post-identification survey analysis engine (Layer 2).
    Evaluates SymptomContext conditioned on the identified scalp pattern.
    """

    # Provisional / uncalibrated severity weights
    SEVERITY_WEIGHTS = {"high": 22, "moderate": 12, "low": 6}

    @classmethod
    def analyze(cls, condition: str, context: SymptomContext) -> SurveyRiskProfile:
        """Produces a condition-specific SurveyRiskProfile from intake survey answers."""
        triggers = cls._evaluate_triggers(condition, context)
        non_modifiable = cls._identify_non_modifiable(condition, context)

        # Normalize raw weight sum into [0, 100] (provisional / uncalibrated ceiling = 110 raw pts)
        raw = sum(cls.SEVERITY_WEIGHTS.get(t.severity_contribution, 6) for t in triggers)
        risk_score = round(min(100.0, (raw / 110.0) * 100.0), 1)

        if risk_score < 20.0:
            risk_band: Literal["low", "moderate", "elevated", "high"] = "low"
        elif risk_score < 45.0:
            risk_band = "moderate"
        elif risk_score < 70.0:
            risk_band = "elevated"
        else:
            risk_band = "high"

        queue = cls._build_intervention_queue(condition, triggers)
        summary = cls._build_summary(condition, risk_band, triggers)

        return SurveyRiskProfile(
            condition=condition,
            risk_score=risk_score,
            risk_band=risk_band,
            modifiable_count=sum(1 for t in triggers if t.modifiable),
            non_modifiable_count=len(non_modifiable),
            top_triggers=triggers,
            intervention_queue=queue,
            non_modifiable_factors=non_modifiable,
            summary=summary,
            disclaimer=UNIVERSAL_DISCLAIMER,
        )

    @classmethod
    def analyze_cohort(cls, batch: SurveyCohortRequest) -> SurveyCohortSummary:
        """
        Aggregates a batch of collected survey responses (optionally tagged with
        identified conditions) to extract population-level trigger prevalence,
        risk band distribution, and per-condition mean risk scores.
        """
        if not batch.records:
            return SurveyCohortSummary(
                total_records=0,
                mean_risk_score=0.0,
                risk_band_distribution={"low": 0, "moderate": 0, "elevated": 0, "high": 0},
                top_population_triggers=[],
                condition_breakdown={},
                disclaimer=UNIVERSAL_DISCLAIMER,
            )

        scores: List[float] = []
        bands: Counter = Counter({"low": 0, "moderate": 0, "elevated": 0, "high": 0})
        trigger_counts: Counter = Counter()
        cond_Buckets: Dict[str, List[float]] = {}

        for item in batch.records:
            cond = item.condition or CONDITION_NORMAL
            prof = cls.analyze(cond, item.context)
            scores.append(prof.risk_score)
            bands[prof.risk_band] += 1
            cond_Buckets.setdefault(cond, []).append(prof.risk_score)
            for trig in prof.top_triggers:
                if trig.modifiable:
                    trigger_counts[trig.name] += 1

        n = len(scores)
        top_trigs = [
            {
                "trigger": name,
                "count": cnt,
                "prevalence_pct": round((cnt / n) * 100.0, 1),
            }
            for name, cnt in trigger_counts.most_common(8)
        ]
        cond_summary = {
            c: {
                "count": len(vals),
                "mean_risk_score": round(sum(vals) / len(vals), 1),
            }
            for c, vals in cond_Buckets.items()
        }

        return SurveyCohortSummary(
            total_records=n,
            mean_risk_score=round(sum(scores) / n, 1),
            risk_band_distribution=dict(bands),
            top_population_triggers=top_trigs,
            condition_breakdown=cond_summary,
            disclaimer=UNIVERSAL_DISCLAIMER,
        )

    @classmethod
    def _evaluate_triggers(cls, condition: str, ctx: SymptomContext) -> List[TriggerFactor]:
        triggers: List[TriggerFactor] = []
        triggers.extend(cls._universal_factors(ctx))

        dispatch = {
            CONDITION_SEBORRHEIC_DERMATITIS: cls._factors_seborrheic_dermatitis,
            CONDITION_SCALP_PSORIASIS: cls._factors_psoriasis,
            CONDITION_FOLLICULITIS: cls._factors_folliculitis,
            CONDITION_ANDROGENETIC_ALOPECIA: cls._factors_aga,
            CONDITION_TELOGEN_EFFLUVIUM: cls._factors_telogen_effluvium,
            CONDITION_ALOPECIA_AREATA: cls._factors_alopecia_areata,
            CONDITION_CONTACT_DERMATITIS: cls._factors_contact_dermatitis,
            CONDITION_DRY_SCALP: cls._factors_dry_scalp,
            CONDITION_LICHEN_PLANUS: cls._factors_lichen_planus,
            CONDITION_TINEA_CAPITIS: cls._factors_tinea_capitis,
            CONDITION_HEAD_LICE: cls._factors_head_lice,
            CONDITION_NORMAL: lambda _c: [],
        }
        fn = dispatch.get(condition, lambda _c: [])
        triggers.extend(fn(ctx))

        # Deduplicate by factor name while keeping highest severity version
        order = {"high": 0, "moderate": 1, "low": 2}
        dedup: Dict[str, TriggerFactor] = {}
        for t in triggers:
            existing = dedup.get(t.name)
            if existing is None or order[t.severity_contribution] < order[existing.severity_contribution]:
                dedup[t.name] = t

        ranked = list(dedup.values())
        ranked.sort(key=lambda t: (order[t.severity_contribution], not t.modifiable))
        return ranked

    @classmethod
    def _universal_factors(cls, ctx: SymptomContext) -> List[TriggerFactor]:
        factors: List[TriggerFactor] = []

        if ctx.hard_water_exposure or ctx.water_environment == "hard_borewell":
            factors.append(TriggerFactor(
                name="Hard Water Mineral Load",
                category="environmental",
                modifiable=True,
                severity_contribution="high",
                rationale=(
                    "Calcium and magnesium ions in hard borewell water bind with sebum fatty acids "
                    "to form insoluble residue, elevating scalp pH and weakening barrier integrity."
                ),
                action=(
                    "Install a shower softening filter or incorporate a weekly chelating rinse "
                    "to clear mineral buildup from the stratum corneum."
                ),
            ))

        if ctx.daily_helmet_wearer:
            factors.append(TriggerFactor(
                name="Daily Helmet / Sweat Occlusion",
                category="mechanical",
                modifiable=True,
                severity_contribution="high",
                rationale=(
                    "Prolonged helmet wear traps heat, sweat, and sebum, accelerating microbial "
                    "overgrowth and mechanical friction at follicular openings."
                ),
                action=(
                    "Use a washable cotton or antibacterial helmet liner, wash it every 2-3 days, "
                    "and air-dry the scalp after rides."
                ),
            ))

        if ctx.pollution_exposure in ["high_metro_industrial", "high_campus_dust"]:
            factors.append(TriggerFactor(
                name="Atmospheric Particulate (PM2.5) Deposition",
                category="environmental",
                modifiable=True,
                severity_contribution="moderate",
                rationale=(
                    "Fine particulate dust settles on sebum-coated shafts and oxidizes surface "
                    "lipids, contributing to low-grade perifollicular irritation."
                ),
                action="Cleanse scalp regularly after heavy outdoor dust exposure and use breathable headwear.",
            ))

        if ctx.washing_frequency in ["rarely", "once_per_week"]:
            factors.append(TriggerFactor(
                name="Infrequent Scalp Cleansing",
                category="lifestyle",
                modifiable=True,
                severity_contribution="moderate",
                rationale="Extended intervals between washes allow sebum, corneocytes, and debris to accumulate.",
                action="Increase cleansing cadence to 2-3 times per week using a gentle pH-balanced cleanser.",
            ))
        elif ctx.washing_frequency == "daily":
            factors.append(TriggerFactor(
                name="Daily Surfactant Stripping",
                category="lifestyle",
                modifiable=True,
                severity_contribution="low",
                rationale="Daily harsh surfactant cleansing can deplete barrier lipids and trigger rebound oiliness.",
                action="Switch to a mild sulfate-free daily wash or alternate with lukewarm water rinses.",
            ))

        if ctx.chemical_heat_styling:
            factors.append(TriggerFactor(
                name="Thermal & Chemical Cuticle Stress",
                category="mechanical",
                modifiable=True,
                severity_contribution="moderate",
                rationale="High-heat blow-drying, bleaching, or straightening weakens shaft tensile strength and irritates skin.",
                action="Pause chemical treatments and high-heat styling during active scalp recovery.",
            ))

        if ctx.sleep_quality in ["irregular_under_6h", "shift_work_disrupted"]:
            factors.append(TriggerFactor(
                name="Circadian & Sleep Disruption",
                category="metabolic",
                modifiable=True,
                severity_contribution="moderate",
                rationale="Sleep under 6 hours elevates nocturnal cortisol and impairs follicular cellular repair.",
                action="Establish a consistent 7-8 hour sleep window to support anagen cycling.",
            ))

        if ctx.stress_level == "high" or ctx.recent_exam_work_stress:
            factors.append(TriggerFactor(
                name="Elevated Psychological / Academic Stress",
                category="lifestyle",
                modifiable=True,
                severity_contribution="high",
                rationale="Sustained cortisol and neurogenic mediators promote premature telogen shift and scalp sensitivity.",
                action="Incorporate daily 20-minute aerobic movement, structured breaks, and sleep recovery.",
            ))

        if ctx.deficiency_flags:
            flags_str = ", ".join(ctx.deficiency_flags)
            factors.append(TriggerFactor(
                name=f"Reported Micronutrient Gap ({flags_str})",
                category="nutritional",
                modifiable=True,
                severity_contribution="high",
                rationale="Iron ferritin and Vitamin D3 are key cofactors for hair matrix division and immune balance.",
                action=f"Review targeted blood labs and physician-guided correction for {flags_str}.",
            ))

        if ctx.diet_pattern == "low_protein_crash_diet":
            factors.append(TriggerFactor(
                name="Low-Protein / Caloric Restriction",
                category="nutritional",
                modifiable=True,
                severity_contribution="high",
                rationale="Keratin synthesis depends on continuous dietary sulfur-containing amino acids.",
                action="Increase complete dietary protein intake across main meals (legumes, eggs, dairy, fish, or soy).",
            ))

        if ctx.gut_health in ["frequent_bloating_acidity", "chronic_constipation", "irregular_ibs"]:
            factors.append(TriggerFactor(
                name=f"Gut-Skin-Hair Absorption Barrier ({ctx.gut_health.replace('_', ' ').title()})",
                category="metabolic",
                modifiable=True,
                severity_contribution="moderate",
                rationale="Digestive dysbiosis can impair intestinal uptake of iron, zinc, biotin, and amino acids.",
                action="Prioritise hydration, soluble dietary fibre, and fermented foods; seek medical review if chronic.",
            ))

        if ctx.hormonal_profile in ["thyroid_dysfunction", "pcos_irregular_cycles", "insulin_resistance"]:
            factors.append(TriggerFactor(
                name=f"Endocrine Axis Factor ({ctx.hormonal_profile.replace('_', ' ').title()})",
                category="hormonal",
                modifiable=False,
                severity_contribution="high",
                rationale="Endocrine shifts directly alter follicular cycling speed, sebum output, and androgen sensitivity.",
                action="Coordinate systemic endocrine evaluation (TSH, metabolic/androgen panel) with your physician.",
            ))

        return factors

    @classmethod
    def _factors_seborrheic_dermatitis(cls, ctx: SymptomContext) -> List[TriggerFactor]:
        out: List[TriggerFactor] = []
        if ctx.daily_helmet_wearer and (ctx.hard_water_exposure or ctx.water_environment == "hard_borewell"):
            out.append(TriggerFactor(
                name="Compound Trigger: Hard Water x Helmet Sebum Trap",
                category="microbial",
                modifiable=True,
                severity_contribution="high",
                rationale=(
                    "Alkaline mineral soap-scum combined with occluded helmet humidity creates peak "
                    "conditions for Malassezia lipid hydrolysis and yellow scaling flares."
                ),
                action="Pair antifungal wash cadence (2x/week, 5-min contact) with a clean helmet liner and filtered rinse.",
            ))
        if ctx.metabolic_energy == "chronic_sluggish_fatigue":
            out.append(TriggerFactor(
                name="Systemic Fatigue & Barrier Sensitivity",
                category="metabolic",
                modifiable=True,
                severity_contribution="moderate",
                rationale="Chronic fatigue often accompanies heightened cutaneous reactivity to commensal yeast metabolites.",
                action="Support recovery with regular sleep, balanced zinc-rich nutrition, and consistent cleansing.",
            ))
        return out

    @classmethod
    def _factors_psoriasis(cls, ctx: SymptomContext) -> List[TriggerFactor]:
        out: List[TriggerFactor] = []
        if ctx.stress_level == "high" or ctx.recent_exam_work_stress:
            out.append(TriggerFactor(
                name="Neuro-Inflammatory Plaque Flare Driver",
                category="lifestyle",
                modifiable=True,
                severity_contribution="high",
                rationale="Acute stress amplifies inflammatory cytokine signalling linked to hyperkeratotic scalp plaques.",
                action="Avoid scratching/picking scales (Koebner trauma risk) and combine gentle keratolytics with dermatologist care.",
            ))
        return out

    @classmethod
    def _factors_folliculitis(cls, ctx: SymptomContext) -> List[TriggerFactor]:
        out: List[TriggerFactor] = []
        if ctx.daily_helmet_wearer or ctx.washing_frequency in ["rarely", "once_per_week"]:
            out.append(TriggerFactor(
                name="Follicular Ostia Sweat & Friction Occlusion",
                category="microbial",
                modifiable=True,
                severity_contribution="high",
                rationale="Sweat occlusion and mechanical rubbing traumatize follicular openings and encourage bacterial bumps.",
                action="Cleanse promptly after sweating, sanitize headwear, and avoid squeezing tender scalp bumps.",
            ))
        return out

    @classmethod
    def _factors_aga(cls, ctx: SymptomContext) -> List[TriggerFactor]:
        out: List[TriggerFactor] = []
        if ctx.family_history_hair_loss:
            out.append(TriggerFactor(
                name="Familial Androgenetic Sensitivity",
                category="genetic",
                modifiable=False,
                severity_contribution="high",
                rationale="Inherited follicular sensitivity to dihydrotestosterone progressively miniaturizes terminal shafts.",
                action="Schedule a dermatologist trichoscopy consult early to discuss evidence-based long-term management.",
            ))
        return out

    @classmethod
    def _factors_telogen_effluvium(cls, ctx: SymptomContext) -> List[TriggerFactor]:
        out: List[TriggerFactor] = []
        if ctx.shedding_pattern == "sudden_acute" or ctx.hair_pull_test in ["borderline_3_5", "positive_over_5"]:
            out.append(TriggerFactor(
                name="Active Synchronous Telogen Shedding Window",
                category="metabolic",
                modifiable=True,
                severity_contribution="high",
                rationale="Acute diffuse shedding typically trails a metabolic, nutritional, or stress trigger by 6-12 weeks.",
                action="Check serum ferritin, Vitamin D3, and TSH with a clinician and maintain steady protein intake over 3-6 months.",
            ))
        return out

    @classmethod
    def _factors_alopecia_areata(cls, ctx: SymptomContext) -> List[TriggerFactor]:
        return [TriggerFactor(
            name="Focal Autoimmune Follicular Involvement",
            category="genetic",
            modifiable=False,
            severity_contribution="high",
            rationale="Patchy non-scarring hair loss involves localized immune reactivity around anagen hair bulbs.",
            action="Seek timely dermatologist evaluation; avoid harsh traction or irritant remedies on bare patches.",
        )]

    @classmethod
    def _factors_contact_dermatitis(cls, ctx: SymptomContext) -> List[TriggerFactor]:
        out: List[TriggerFactor] = []
        if ctx.chemical_heat_styling:
            out.append(TriggerFactor(
                name="Sensitising Hair Dye / Chemical Exposure",
                category="environmental",
                modifiable=True,
                severity_contribution="high",
                rationale="Oxidative hair dyes, fragrances, and preservatives are primary triggers of scalp contact reactions.",
                action="Discontinue newly introduced hair products immediately and discuss patch testing with a dermatologist.",
            ))
        return out

    @classmethod
    def _factors_dry_scalp(cls, ctx: SymptomContext) -> List[TriggerFactor]:
        out: List[TriggerFactor] = []
        if ctx.washing_frequency == "daily" or ctx.hard_water_exposure:
            out.append(TriggerFactor(
                name="Transepidermal Moisture Depletion",
                category="environmental",
                modifiable=True,
                severity_contribution="moderate",
                rationale="Frequent washing and hard water strip intercellular ceramides, causing fine dry white flaking.",
                action="Use lukewarm water, humectant scalp serums (panthenol/glycerin), and mild sulfate-free cleansers.",
            ))
        return out

    @classmethod
    def _factors_lichen_planus(cls, ctx: SymptomContext) -> List[TriggerFactor]:
        return [TriggerFactor(
            name="Scarring Follicular Inflammation Risk",
            category="metabolic",
            modifiable=False,
            severity_contribution="high",
            rationale="Lichenoid perifollicular inflammation can permanently affect follicles if clinical care is delayed.",
            action="Arrange prompt in-person dermatologist assessment for trichoscopy and targeted anti-inflammatory care.",
        )]

    @classmethod
    def _factors_tinea_capitis(cls, ctx: SymptomContext) -> List[TriggerFactor]:
        return [TriggerFactor(
            name="Dermatophyte Fungal Transmission & Fomite Load",
            category="microbial",
            modifiable=True,
            severity_contribution="high",
            rationale="Fungal scalp involvement spreads through shared combs, towels, helmets, and pillowcases.",
            action="Consult a physician promptly for systemic clinical evaluation and wash bedding/towels in hot water.",
        )]

    @classmethod
    def _factors_head_lice(cls, ctx: SymptomContext) -> List[TriggerFactor]:
        return [TriggerFactor(
            name="Close-Contact & Shared Headwear Transfer",
            category="microbial",
            modifiable=True,
            severity_contribution="high",
            rationale="Pediculosis spreads rapidly via head-to-head contact, helmets, pillows, and brushes.",
            action="Use targeted OTC pediculicide lotion plus wet fine-tooth nit combing every 3-4 days for 2 weeks.",
        )]

    @classmethod
    def _identify_non_modifiable(cls, condition: str, ctx: SymptomContext) -> List[str]:
        nm: List[str] = []
        if ctx.family_history_hair_loss:
            nm.append("Family history of pattern hair thinning (genetic androgen sensitivity)")
        if ctx.hormonal_profile != "none":
            nm.append(f"Endocrine background: {ctx.hormonal_profile.replace('_', ' ').title()} (physician-managed)")
        if condition in [CONDITION_ALOPECIA_AREATA, CONDITION_LICHEN_PLANUS]:
            nm.append(f"{condition}: autoimmune/inflammatory pathway requiring dermatologist oversight")
        nm.append(f"Demographic bracket: {ctx.biological_sex}, age {ctx.age_group.replace('_', '-')}")
        return nm

    @classmethod
    def _build_intervention_queue(
        cls, condition: str, triggers: List[TriggerFactor]
    ) -> List[InterventionStep]:
        steps: List[InterventionStep] = []
        priority = 1

        if condition in [
            CONDITION_LICHEN_PLANUS,
            CONDITION_ALOPECIA_AREATA,
            CONDITION_TINEA_CAPITIS,
            CONDITION_HEAD_LICE,
            CONDITION_SCALP_PSORIASIS,
            CONDITION_FOLLICULITIS,
        ]:
            steps.append(InterventionStep(
                priority=priority,
                title="Book Certified Dermatologist Assessment",
                detail=(
                    f"Visual pattern ({condition}) warrants clinical confirmation before relying on "
                    "cosmetic routines alone."
                ),
                expected_timeline="Within 1-2 weeks",
            ))
            priority += 1

        for trig in [t for t in triggers if t.modifiable and t.severity_contribution == "high"][:3]:
            steps.append(InterventionStep(
                priority=priority,
                title=f"Eliminate High-Impact Trigger: {trig.name}",
                detail=trig.action,
                expected_timeline="2-4 weeks",
            ))
            priority += 1

        for trig in [t for t in triggers if t.modifiable and t.severity_contribution == "moderate"][:2]:
            steps.append(InterventionStep(
                priority=priority,
                title=f"Optimize Routine: {trig.name}",
                detail=trig.action,
                expected_timeline="3-6 weeks",
            ))
            priority += 1

        steps.append(InterventionStep(
            priority=priority,
            title="Longitudinal Follow-Up Scan & Survey Re-Score",
            detail="Capture a standardized follow-up photo in 2-4 weeks under identical lighting to measure sub-score recovery.",
            expected_timeline="Day 14 to Day 28",
        ))
        return steps

    @classmethod
    def _build_summary(
        cls, condition: str, risk_band: str, triggers: List[TriggerFactor]
    ) -> str:
        mod_triggers = [t for t in triggers if t.modifiable]
        top_names = ", ".join(t.name for t in mod_triggers[:3]) if mod_triggers else "minimal external lifestyle load"
        return (
            f"Condition-anchored survey analysis for '{condition}' indicates a {risk_band.upper()} "
            f"modifiable trigger load ({len(mod_triggers)} actionable factors out of {len(triggers)} total). "
            f"Primary addressable drivers: {top_names}."
        )


survey_risk_engine = SurveyRiskEngine()
