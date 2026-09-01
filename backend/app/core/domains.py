"""
Follica DermaIQ Multi-Domain Architecture Module.

Architectural Design Intent & Scalability Rationale:
1. Why Modular Domain Architecture (DermaIQ):
   - Dermatological computer vision workflows share a common architectural pattern:
     a. Pre-inference capture quality validation (blur, lighting, resolution)
     b. Dual-pipeline inference (Deep Learning classification + classical CV micro-features)
     c. Contextual questionnaire gating (symptoms, environmental exposure, duration)
     d. Transparent sub-score summation (0-100 composite index)
     e. Molecule-first active ingredient routing and clinical PDF export
   - Standardizing these domain contracts enables the platform to expand from Scalp Health
     to Facial Acne and Fungal/Eczema skin conditions with zero breaking changes to the core engine.

2. Registered Domains:
   - "scalp" (Production): Scalp health, Seborrheic Dermatitis, Alopecia Areata, Folliculitis, Hard Water Crust.
   - "facial_acne" (Modular Extension): Comedones, Inflammatory Papules, Pustules, Sebum balance, Post-Inflammatory Erythema.
   - "eczema_rashes" (Modular Extension): Atopic dermatitis, Contact dermatitis, Superficial fungal tinea rashes.
"""

from typing import Dict, List
from app.models.schemas import DermaIQDomainInfo


DERMA_IQ_DOMAINS: Dict[str, DermaIQDomainInfo] = {
    "scalp": DermaIQDomainInfo(
        domain_id="scalp",
        display_name="Scalp & Hair Health Diagnostic Engine",
        description="Dual-pipeline assessment of scalp barrier, fungal flaking, vascular erythema, and follicular density.",
        status="production",
        supported_models=["mobilenetv4", "efficientnet_b0"],
        active_sub_scores=[
            "flake_scaling_score",
            "erythema_redness_score",
            "sebum_lipid_score",
            "follicle_density_score"
        ]
    ),
    "facial_acne": DermaIQDomainInfo(
        domain_id="facial_acne",
        display_name="Facial Acne & Barrier Assessment (DermaIQ Module)",
        description="Quantifies non-inflammatory comedones vs inflammatory papules, pore congestion, and barrier lipid health.",
        status="beta",
        supported_models=["mobilenetv4", "convnext_nano"],
        active_sub_scores=[
            "comedone_density_score",
            "inflammatory_papule_score",
            "erythema_hyperpigmentation_score",
            "barrier_hydration_score"
        ]
    ),
    "eczema_rashes": DermaIQDomainInfo(
        domain_id="eczema_rashes",
        display_name="Eczema & Fungal Dermatitis Engine (DermaIQ Module)",
        description="Evaluates localized epidermal lichenification, barrier xerosis, and circular active erythematous margins.",
        status="planned",
        supported_models=["mobilenetv4"],
        active_sub_scores=[
            "lichenification_roughness_score",
            "erythema_intensity_score",
            "excoriation_pruritus_score",
            "barrier_xerosis_score"
        ]
    )
}


def get_registered_domains() -> List[DermaIQDomainInfo]:
    """Returns the list of active and pluggable DermaIQ domain configurations."""
    return list(DERMA_IQ_DOMAINS.values())
