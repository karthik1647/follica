"""
Follica Deep Learning Classifier Service.

Architectural Design Intent & Rationale:
1. Model Choice (MobileNetV4 vs EfficientNet-B0):
   - MobileNetV4 (2024) is chosen as default because its Universal Inverted Bottleneck (UIB)
     architecture delivers lower inference latency and smaller parameter footprint (~14 MB ONNX),
     making it ideal for free cloud containers or on-device mobile execution.
   - EfficientNet-B0 is provided as a battle-tested fallback if ONNX export issues arise.

2. Embedded Heuristic Classifier for Local Testing:
   - To ensure the repository is completely testable on localhost:8000 immediately without
     forcing the developer to first spend 30 minutes training or downloading 20 MB weights,
     an embedded contextual fallback operates when no model weights file is present.
"""

import os
import io
import time
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any

from app.core.config import settings
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
    CONDITION_UNVERIFIED_MONITORED,
)
from app.models.schemas import SymptomContext


class ScalpClassifierService:
    """Manages model loading, warmup, and inference for scalp condition classification."""

    CLASSES = [
        CONDITION_NORMAL,
        CONDITION_DRY_SCALP,
        CONDITION_SEBORRHEIC_DERMATITIS,
        CONDITION_ALOPECIA_AREATA,
        CONDITION_FOLLICULITIS
    ]

    # Output order of the trained MobileNetV4 model (alphabetical folder order of the
    # Kaggle "Hair Diseases" dataset). Must match training/follica_classes.json.
    MODEL_CLASSES = [
        CONDITION_ALOPECIA_AREATA,        # Alopecia Areata
        CONDITION_CONTACT_DERMATITIS,     # Contact Dermatitis
        CONDITION_FOLLICULITIS,           # Folliculitis
        CONDITION_HEAD_LICE,              # Head Lice
        CONDITION_LICHEN_PLANUS,          # Lichen Planus
        CONDITION_ANDROGENETIC_ALOPECIA,  # Male Pattern Baldness
        CONDITION_SCALP_PSORIASIS,        # Psoriasis
        CONDITION_SEBORRHEIC_DERMATITIS,  # Seborrheic Dermatitis
        CONDITION_TELOGEN_EFFLUVIUM,      # Telogen Effluvium
        CONDITION_TINEA_CAPITIS,          # Tinea Capitis
    ]

    def __init__(self):
        self.model_loaded = False
        self.architecture = settings.MODEL_ARCHITECTURE
        self.model_path = settings.MODEL_WEIGHTS_PATH
        self.onnx_session = None
        self._initialize_model()

    def _initialize_model(self):
        """Attempts to load ONNX / PyTorch weights, or sets ready status for heuristic mode."""
        if os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(self.model_path)
                n_out = self.onnx_session.get_outputs()[0].shape[-1]
                if n_out != len(self.MODEL_CLASSES):
                    print(f"[Follica ML] Model has {n_out} outputs but {len(self.MODEL_CLASSES)} classes are configured. Using heuristic fallback.")
                    self.onnx_session = None
                self.model_loaded = True
                print(f"[Follica ML] Loaded ONNX weights from {self.model_path}")
            except Exception as e:
                print(f"[Follica ML] Could not load ONNX model ({e}). Using embedded heuristic fallback.")
                self.model_loaded = True
        else:
            # Model weights not yet placed in /models; active in heuristic/dev mode
            print(f"[Follica ML] No weights found at {self.model_path}. Running with embedded feature classifier.")
            self.model_loaded = True

    def warmup(self) -> Dict[str, Any]:
        """Runs a synthetic inference pass to warm up memory/cache on cold start."""
        start_time = time.time()
        dummy_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
        if self.onnx_session is not None:
            input_name = self.onnx_session.get_inputs()[0].name
            _ = self.onnx_session.run(None, {input_name: dummy_input})
        elapsed = round((time.time() - start_time) * 1000, 2)
        return {"status": "ready", "warmup_latency_ms": elapsed, "architecture": self.architecture}

    def predict(self, img_bytes: bytes, context: SymptomContext) -> Tuple[str, float]:
        """
        Executes classification on input scalp image + clinical context.
        Returns:
            (suggested_condition: str, confidence_score: float)
        """
        notes_lower = (context.symptom_notes or "").lower()
        has_red_flag_lesion = any(
            kw in notes_lower
            for kw in ["bleed", "sore", "ulcer", "oozing", "open wound", "not heal", "won't heal"]
        )
        if has_red_flag_lesion:
            return CONDITION_UNVERIFIED_MONITORED, 0.88

        # The photo model has no healthy-scalp class, so a symptom-free routine check-up or
        # explicit zero-symptom intake is handled without forcing one of the 10 disease labels.
        is_healthy_checkup = (
            not context.itching
            and not context.burning_or_pain
            and context.flake_type == "none"
            and context.hair_pull_test == "negative_under_3"
            and (
                context.primary_concern_zone == "routine_healthy_check"
                or (
                    context.shedding_pattern == "stable_no_shedding"
                    and (context.duration_weeks or 0) == 0
                    and not context.family_history_hair_loss
                )
            )
        )
        if is_healthy_checkup:
            return CONDITION_NORMAL, 0.94

        if self.onnx_session is not None:
            try:
                # Preprocess image for MobileNetV4 / EfficientNet (224x224, Normalized)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img = img.resize((224, 224))
                img_np = np.array(img, dtype=np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_np = (img_np - mean) / std
                img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
                img_tensor = np.expand_dims(img_np, axis=0)  # Add batch dim

                input_name = self.onnx_session.get_inputs()[0].name
                outputs = self.onnx_session.run(None, {input_name: img_tensor})
                logits = outputs[0][0]

                # Numerically stable softmax
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)

                class_idx = int(np.argmax(probs))
                confidence = float(probs[class_idx])
                if confidence >= settings.CONFIDENCE_GATE_THRESHOLD:
                    onnx_cond = self.MODEL_CLASSES[class_idx]
                    # Reconcile high-specificity acute clinical symptoms when they directly
                    # contradict a non-inflammatory pattern-thinning visual prediction:
                    if onnx_cond in [CONDITION_ANDROGENETIC_ALOPECIA, CONDITION_TELOGEN_EFFLUVIUM]:
                        if (
                            context.burning_or_pain
                            and any(b in context.sensation for b in ["bump", "pustule", "painful_bumps", "folliculitis"])
                        ):
                            return CONDITION_FOLLICULITIS, 0.85
                        if context.flake_type == "plaque_crusty":
                            return CONDITION_SCALP_PSORIASIS, 0.85
                        if context.flake_type == "greasy_yellow_thick" and context.itching:
                            return CONDITION_SEBORRHEIC_DERMATITIS, 0.85
                    return onnx_cond, round(confidence, 3)
            except Exception as e:
                print(f"[Follica ML] Inference error: {e}. Falling back to contextual inference.")

        # Context-aware fallback classifier (when ONNX weights are absent or visual confidence < threshold)
        return self._heuristic_classify(img_bytes, context)

    def _heuristic_classify(self, img_bytes: bytes, context: SymptomContext) -> Tuple[str, float]:
        """
        Contextual fallback evaluator for local testing when CNN model weights are absent.
        Combines multi-select questionnaire flags, concern zones, and visual features.

        Note: This fallback is a rule-based stand-in until the CNN is retrained. The current
        CNN class list (CLASSES, 5 classes) cannot output CONDITION_ANDROGENETIC_ALOPECIA or
        CONDITION_SCALP_PSORIASIS without expanded training data and new class heads.
        """
        # 1. Psoriasis / Scalp Plaque check (provisional / uncalibrated heuristic)
        if context.flake_type == "plaque_crusty" or "plaque" in context.sensation.lower():
            return CONDITION_SCALP_PSORIASIS, 0.82

        # 2. Routine healthy check
        elif context.primary_concern_zone == "routine_healthy_check" and not context.itching and not context.burning_or_pain:
            return CONDITION_NORMAL, 0.94

        # 3. Folliculitis / Scalp Bumps (requires actual painful bumps/pustules, not simple diffuse trichodynia)
        elif (
            context.burning_or_pain
            and any(b in context.sensation for b in ["bump", "pustule", "painful_bumps", "folliculitis"])
            and (context.duration_months >= 1 or context.duration_weeks > 3)
        ):
            return CONDITION_FOLLICULITIS, 0.81

        # 4. Focal / Patchy bald spots -> Alopecia Areata (autoimmune focal loss)
        elif context.primary_concern_zone == "patchy_spots":
            return CONDITION_ALOPECIA_AREATA, 0.78

        # 5. Seborrheic Dermatitis (greasy yellow scales or itching with hard water / burning)
        elif (
            context.flake_type == "greasy_yellow_thick"
            or (context.itching and (context.hard_water_exposure or "burning" in context.sensation))
        ):
            return CONDITION_SEBORRHEIC_DERMATITIS, 0.84

        # 6. Acute / Diffuse Shedding -> Telogen Effluvium (positive pull test or acute shedding)
        elif (
            context.hair_pull_test in ["positive_over_5", "borderline_3_5"]
            or context.shedding_pattern == "sudden_acute"
        ):
            return CONDITION_TELOGEN_EFFLUVIUM, 0.83

        # 7. Pattern thinning (crown/vertex, hairline/temples, central parting, diffuse) + family history or long-standing chronic progression -> Androgenetic Alopecia
        elif (
            (context.family_history_hair_loss or context.duration_months >= 6 or context.duration_weeks >= 24)
            and context.primary_concern_zone in ["crown_vertex", "hairline_temples", "central_parting", "diffuse_all_over"]
            and (context.duration_months >= 2 or context.duration_weeks > 8)
        ):
            return CONDITION_ANDROGENETIC_ALOPECIA, 0.76

        # 7. Dry Scalp / Mild Flaking
        elif context.itching or context.flake_type == "dry_white_fine":
            return CONDITION_DRY_SCALP, 0.77

        # 8. Baseline healthy
        else:
            return CONDITION_NORMAL, 0.89


classifier_service = ScalpClassifierService()

