"""
Follica Backend Configuration Module.

Architectural Design Intent & Operational Constraints:
1. Zero-Cost Infrastructure Constraint:
   - Render.com Free Tier: Backend container spins down after 15 minutes of inactivity.
     Cold start requires 40-50 seconds to reallocate memory and initialize Python runtime.
     Rationale: We expose /api/v1/system/status so the client app can poll and display a
     transparent "Waking up model" indicator rather than timing out abruptly.
   - Supabase Free Tier: 500 MB database, 1 GB storage bucket cap.
     Rationale: Photos must not be stored indefinitely. We enforce an automated 30-day
     retention purge and client-side downscaling (max 1024px) to preserve free tier quota.
   - Google Colab / Kaggle Compute: Sessions terminate after 6-12 hours.
     Rationale: The training pipeline implements per-epoch checkpointing directly to
     Google Drive so disconnects never result in lost training progress.

2. Confidence Threshold Decision (0.65):
   - A threshold of 0.65 (65%) represents the boundary below which deep learning
     probabilistic predictions exhibit high entropy across scalp disease boundaries
     (e.g., distinguishing early seborrheic dermatitis from simple dry scalp).
   - Any prediction with confidence < 0.65 is automatically intercepted server-side
     and diverted to Level 2 (Monitored Routine) to prevent improper self-treatment
     or unnecessary clinical panic.
"""

import os
from pydantic import BaseModel


class Settings(BaseModel):
    APP_NAME: str = "Follica API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = True

    # Server-Side Confidence Gate Threshold:
    # Rationale: Threshold set to 0.65 to ensure high-entropy predictions are never
    # treated as definitive classifications. Low confidence forces Level 2 monitoring.
    CONFIDENCE_GATE_THRESHOLD: float = 0.65

    # Storage and Data Retention Defaults:
    # Rationale: 30 days provides sufficient time for a user to complete a 14-day or
    # 28-day re-scan comparison cycle while preventing unbounded database growth.
    DEFAULT_DATA_RETENTION_DAYS: int = 30
    MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB upload ceiling

    # Model Architecture Selection:
    # Supported: "mobilenetv4" (default) or "efficientnet_b0" (fallback)
    MODEL_ARCHITECTURE: str = "mobilenetv4"
    MODEL_WEIGHTS_PATH: str = os.getenv("FOLLICA_MODEL_PATH", "models/follica_mobilenetv4.onnx")

    # Image Quality Pre-Inference Gate Configuration:
    # Note: These optical gating thresholds are provisional / uncalibrated baseline thresholds
    # pending large-scale clinical dataset fitting.
    # - Laplacian variance < 60.0 indicates significant motion or optical blur.
    # - Luminance < 35.0 indicates severe underexposure.
    # - Luminance > 225.0 indicates blown-out specular highlights or flashlight wash.
    QUALITY_GATE_ENABLED: bool = True
    QUALITY_MIN_BLUR_SCORE: float = 60.0
    QUALITY_MIN_BRIGHTNESS: float = 35.0
    QUALITY_MAX_BRIGHTNESS: float = 225.0
    QUALITY_MIN_DIMENSION: int = 150

    # Storage Bucket Name
    SUPABASE_STORAGE_BUCKET: str = "follica-scalp-scans-encrypted"

    # CORS Configuration:
    # Comma-separated list or list of allowed origin URLs.
    # Default is restricted to local development environments.
    CORS_ALLOWED_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if origin.strip()
    ]


settings = Settings()

