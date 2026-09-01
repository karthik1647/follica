"""
Follica Core Constants and Non-Diagnostic Terminology Definitions.

Safety and Medical Compliance Rules:
1. NEVER use diagnostic verbs ("diagnoses", "detects with certainty", "prescribes").
2. ALWAYS use probabilistic/suggestive phrasing ("suggests", "patterns consistent with").
3. Level 2 recommendations state "recommends an active ingredient schedule" (never "prescribes").
4. Universal disclaimer is included on EVERY API response across all triage levels.
5. Escalation referrals and PDF hand-offs are strictly generic without naming specific clinics or institutions.
"""

# Universal Medical Disclaimer (Mandatory in every response)
UNIVERSAL_DISCLAIMER: str = (
    "This assessment is for informational and educational purposes only and suggests potential scalp "
    "care routines. It does not constitute medical diagnosis, formal medical advice, or treatment. "
    "If symptoms persist, cause discomfort, or worsen, please consult a certified dermatologist."
)

# Generic Dermatologist Search URL (No institutional affiliation)
GENERIC_DERMATOLOGIST_SEARCH_URL: str = "https://www.google.com/maps/search/dermatologist+near+me"

# Standard Non-Diagnostic Condition Names
CONDITION_NORMAL = "Normal / Balanced Scalp"
CONDITION_DRY_SCALP = "Dry Scalp / Mild Flaking"
CONDITION_SEBORRHEIC_DERMATITIS = "Pattern Suggestive of Seborrheic Dermatitis"
CONDITION_ALOPECIA_AREATA = "Pattern Suggestive of Alopecia Areata / Patchy Loss"
CONDITION_FOLLICULITIS = "Pattern Suggestive of Folliculitis / Scalp Bumps"
CONDITION_SCALP_PSORIASIS = "Pattern Suggestive of Scalp Plaque / Scaling"
CONDITION_UNVERIFIED_MONITORED = "Indeterminate Pattern / Requires Verification"

# Triage Levels
TRIAGE_LEVEL_1_MINOR = "Level 1: Minor Home Routine"
TRIAGE_LEVEL_2_MONITORED = "Level 2: Monitored Routine"
TRIAGE_LEVEL_3_ESCALATION = "Level 3: Escalation (Consult a Dermatologist)"

# Recommended Active Ingredient Catalog
INGREDIENT_GUIDES = {
    CONDITION_NORMAL: {
        "summary": "Maintains balanced scalp biome and protective lipid barrier.",
        "ingredients": [
            {"name": "Gentle Sulfate-Free Cleanser", "frequency": "2-3x per week", "purpose": "Removes daily pollutants without stripping natural sebum."},
            {"name": "Lightweight Hydrating Scalp Serum (Hyaluronic / Panthenol)", "frequency": "As needed", "purpose": "Supports moisture retention."}
        ],
        "lifestyle": [
            "Avoid excessively hot shower water on the scalp.",
            "Ensure regular washing after intense workouts or sweat accumulation."
        ]
    },
    CONDITION_DRY_SCALP: {
        "summary": "Recommends an active ingredient schedule focused on epidermal barrier rehydration and gentle exfoliation.",
        "ingredients": [
            {"name": "Salicylic Acid (0.5% - 1.0%) or Lactic Acid Scalp Exfoliant", "frequency": "1x per week", "purpose": "Gently loosens dry adherent flakes."},
            {"name": "Squalane / Ceramide Barrier Scalp Oil", "frequency": "20 mins before wash", "purpose": "Restores stratum corneum lipids."},
            {"name": "EDTA Chelating Cleanser", "frequency": "1x every 2 weeks", "purpose": "Removes hard water mineral crust."}
        ],
        "lifestyle": [
            "If using campus/hostel hard water, consider an attachable tap/shower filter.",
            "Reduce washing frequency to prevent lipid barrier depletion."
        ]
    },
    CONDITION_SEBORRHEIC_DERMATITIS: {
        "summary": "Recommends an active ingredient schedule targeting Malassezia yeast overgrowth and sebum balance.",
        "ingredients": [
            {"name": "Ketoconazole 2% or Selenium Sulfide 2.5% Wash", "frequency": "2x per week (leave on 5 mins before rinsing)", "purpose": "Targets fungal colonization."},
            {"name": "Zinc Pyrithione (1%) or Piroctone Olamine Cleanser", "frequency": "Alternate wash days", "purpose": "Controls microbial balance and flaking."},
            {"name": "Salicylic Acid 2% Scalp Solution", "frequency": "1x per week", "purpose": "Breaks down oily crusts and sebum plugs."}
        ],
        "lifestyle": [
            "Do not leave hair damp for prolonged periods after washing.",
            "Avoid heavy botanical oils (like coconut oil) directly on active flare-ups, as they can feed Malassezia."
        ]
    },
    CONDITION_ALOPECIA_AREATA: {
        "summary": "Visual analysis suggests localized patch thinning. Requires formal clinical evaluation.",
        "ingredients": [
            {"name": "Mild, Non-Irritating Peptide / Caffeine Serum", "frequency": "Daily", "purpose": "Gentle follicle stimulation while awaiting clinical consult."}
        ],
        "lifestyle": [
            "Avoid aggressive brushing, tight hairstyles, or pulling tension.",
            "Consult a certified dermatologist promptly for dermoscopy and autoimmune evaluation."
        ]
    },
    CONDITION_FOLLICULITIS: {
        "summary": "Visual analysis suggests follicular inflammation / pustular pattern. Requires clinical evaluation.",
        "ingredients": [
            {"name": "Antibacterial Benzoyl Peroxide (2.5%) or Hypochlorous Acid Spray", "frequency": "As tolerated", "purpose": "Reduces surface bacterial load."}
        ],
        "lifestyle": [
            "Do not squeeze or scratch inflamed scalp bumps.",
            "Wash pillowcases frequently in hot water.",
            "Seek evaluation from a dermatologist for appropriate antimicrobial care."
        ]
    }
}
