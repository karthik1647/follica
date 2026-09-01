"""
Follica Core Constants and Non-Diagnostic Terminology.

Architectural Design Intent & Safety Compliance:
1. Strict Non-Diagnostic Terminology:
   - Rationale: Software applications evaluating user-uploaded photos must not make
     authoritative diagnostic claims under medical device and consumer safety standards.
   - Rule: Always use probabilistic terms such as "suggests", "patterns consistent with",
     or "visual indications of".
   - Rule: Never use diagnostic verbs such as "diagnoses", "detects with certainty", or "confirms".
   - Rule: Level 2 outputs state "recommends an active ingredient schedule", never "prescribes medication".

2. Institutional Neutrality (No Named Clinics / Hospitals):
   - Rationale: The system is designed to be universally functional regardless of geographic
     location or campus context. Hardcoding references to specific institutions (such as campus
     health centers or specific private clinics) restricts portability, creates legal affiliation
     assumptions, and breaks user trust outside of a single location.
   - Escalations provide generic guidance ("consult a certified dermatologist") with a self-triggered
     local search link rather than a hardcoded booking flow.

3. Universal Medical Disclaimer:
   - Rationale: Displayed on EVERY API response (Levels 1, 2, and 3). Users must be continuously
     reminded that algorithmic photo assessment is an informational aid, not formal medical care.

4. Ingredient-First Formulation Philosophy:
   - Rationale: Generic commercial anti-dandruff products often fail because users buy branded
     shampoos without understanding the active biochemical molecule needed for their specific scalp state
     (e.g., fungal overgrowth vs dry epidermal barrier dehydration vs hard water mineral crust).
     We recommend scientifically supported active molecules (e.g. Ketoconazole 2%, Salicylic Acid 2%).
"""

# Universal Medical Disclaimer (Mandatory on every API response payload)
UNIVERSAL_DISCLAIMER: str = (
    "This assessment is for informational and educational purposes only and suggests potential scalp "
    "care routines. It does not constitute medical diagnosis, formal medical advice, or treatment. "
    "If symptoms persist, cause discomfort, or worsen, please consult a certified dermatologist."
)

# Generic Dermatologist Search URL (Self-triggered, no institutional lock-in)
GENERIC_DERMATOLOGIST_SEARCH_URL: str = "https://www.google.com/maps/search/dermatologist+near+me"

# Standard Non-Diagnostic Condition Taxonomy
CONDITION_NORMAL = "Normal / Balanced Scalp"
CONDITION_DRY_SCALP = "Dry Scalp / Mild Flaking"
CONDITION_SEBORRHEIC_DERMATITIS = "Pattern Suggestive of Seborrheic Dermatitis"
CONDITION_ALOPECIA_AREATA = "Pattern Suggestive of Alopecia Areata / Patchy Loss"
CONDITION_FOLLICULITIS = "Pattern Suggestive of Folliculitis / Scalp Bumps"
CONDITION_SCALP_PSORIASIS = "Pattern Suggestive of Scalp Plaque / Scaling"
CONDITION_UNVERIFIED_MONITORED = "Indeterminate Pattern / Requires Verification"

# Triage Categories
TRIAGE_LEVEL_1_MINOR = "Level 1: Minor Home Routine"
TRIAGE_LEVEL_2_MONITORED = "Level 2: Monitored Routine"
TRIAGE_LEVEL_3_ESCALATION = "Level 3: Escalation (Consult a Dermatologist)"

# Molecule-First Ingredient Guides (Scientific OTC Actives)
INGREDIENT_GUIDES = {
    CONDITION_NORMAL: {
        "summary": "Maintains balanced scalp biome and protective lipid barrier.",
        "ingredients": [
            {
                "name": "Gentle Sulfate-Free Cleanser",
                "frequency": "2-3x per week",
                "purpose": "Removes environmental pollutants without stripping protective sebum lipids."
            },
            {
                "name": "Hydrating Scalp Serum (Panthenol / Hyaluronic Acid)",
                "frequency": "As needed",
                "purpose": "Maintains stratum corneum hydration."
            }
        ],
        "lifestyle": [
            "Avoid excessively hot water during hair washing.",
            "Cleanse promptly following intense physical workouts or excessive perspiration."
        ]
    },
    CONDITION_DRY_SCALP: {
        "summary": "Recommends an active ingredient schedule focused on epidermal barrier rehydration and gentle keratolytic exfoliation.",
        "ingredients": [
            {
                "name": "Salicylic Acid (0.5% - 1.0%) Scalp Solution",
                "frequency": "1x per week",
                "purpose": "Gently loosens dry, adherent stratum corneum flakes."
            },
            {
                "name": "Squalane / Ceramide Scalp Barrier Oil",
                "frequency": "20 minutes before wash",
                "purpose": "Replenishes essential epidermal intercellular lipids."
            },
            {
                "name": "EDTA Chelating Cleanser",
                "frequency": "1x every 2 weeks",
                "purpose": "Binds and removes calcium and magnesium mineral deposits caused by hard tap water."
            }
        ],
        "lifestyle": [
            "If exposed to hard tap/borewell water, consider using an attachable tap or shower filter.",
            "Reduce washing frequency to avoid stripping the natural acid mantle."
        ]
    },
    CONDITION_SEBORRHEIC_DERMATITIS: {
        "summary": "Recommends an active ingredient schedule targeting Malassezia yeast overgrowth and hyper-seborrhea.",
        "ingredients": [
            {
                "name": "Ketoconazole 2% or Selenium Sulfide 2.5% Wash",
                "frequency": "2x per week (leave on scalp for 5 minutes prior to rinsing)",
                "purpose": "Inhibits Malassezia fungal proliferation and reduces inflammatory cytokine release."
            },
            {
                "name": "Zinc Pyrithione (1%) or Piroctone Olamine Cleanser",
                "frequency": "Alternate wash days",
                "purpose": "Maintains antimicrobial suppression between primary active washes."
            },
            {
                "name": "Salicylic Acid 2% Scalp Solution",
                "frequency": "1x per week",
                "purpose": "Breaks down oily follicular plaques and sebum plugs."
            }
        ],
        "lifestyle": [
            "Ensure hair and scalp are thoroughly dried after washing; do not sleep with damp hair.",
            "Avoid applying heavy plant oils (e.g., pure coconut or olive oil) directly to active flare-ups, as lipids can nourish Malassezia species."
        ]
    },
    CONDITION_ALOPECIA_AREATA: {
        "summary": "Visual analysis suggests localized patch thinning. Requires formal clinical evaluation.",
        "ingredients": [
            {
                "name": "Non-Irritating Peptide / Caffeine Scalp Serum",
                "frequency": "Daily",
                "purpose": "Provides gentle micro-environmental support while awaiting clinical consultation."
            }
        ],
        "lifestyle": [
            "Avoid mechanical traction, aggressive brushing, or tight hair styling.",
            "Consult a certified dermatologist promptly for dermoscopy, trichoscopy, and formal assessment."
        ]
    },
    CONDITION_FOLLICULITIS: {
        "summary": "Visual analysis suggests follicular inflammation or pustular presentation. Requires clinical evaluation.",
        "ingredients": [
            {
                "name": "Hypochlorous Acid Scalp Spray or Dilute Benzoyl Peroxide (2.5%) Wash",
                "frequency": "As tolerated",
                "purpose": "Reduces surface bacterial colonization."
            }
        ],
        "lifestyle": [
            "Do not squeeze, pick, or mechanically scratch inflamed follicular bumps.",
            "Launder pillowcases frequently in warm water.",
            "Seek evaluation from a dermatologist for appropriate clinical management."
        ]
    }
}
