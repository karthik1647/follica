"""
Follica Backend Configuration Module.

Includes settings for local execution (localhost:8000) and cloud deployments.
Free-tier operational limits and considerations:
- Render.com: Spins down after 15 mins of inactivity. Cold start takes 40-50 seconds.
  Frontend must poll /api/v1/system/status before firing heavy scan requests.
- Supabase Free Tier: 500 MB database, 1 GB storage cap. Photos should be compressed
  client-side (max 1024px WebP/JPEG) and purged per retention policy (default 30 days).
- Google Colab / Kaggle: Free GPU sessions expire after 6-12 hours. The training
  pipeline uses per-epoch checkpointing to prevent progress loss.
"""

import os
from pydantic import BaseModel


class Settings(BaseModel):
    APP_NAME: str = "Follica API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = True

    # Server-side confidence gate threshold (0.0 to 1.0)
    # If classifier confidence < CONFIDENCE_GATE_THRESHOLD, the response is
    # forced to Triage Level 2 (Monitored / Requires Verification).
    CONFIDENCE_GATE_THRESHOLD: float = 0.65

    # Storage and retention settings
    DEFAULT_DATA_RETENTION_DAYS: int = 30
    MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    # Model configuration
    # Supported: "mobilenetv4" (default) or "efficientnet_b0"
    MODEL_ARCHITECTURE: str = "mobilenetv4"
    MODEL_WEIGHTS_PATH: str = os.getenv("FOLLICA_MODEL_PATH", "models/follica_mobilenetv4.onnx")

    # Supabase Settings (Placeholder for cloud deployment)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    SUPABASE_STORAGE_BUCKET: str = "follica-scalp-scans-encrypted"


settings = Settings()
