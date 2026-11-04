import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.endpoints import router as api_router
from app.services.classifier_service import classifier_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown hooks."""
    print(f"[{settings.APP_NAME}] Initializing inference models and warmup...")
    try:
        classifier_service.warmup()
        print(f"[{settings.APP_NAME}] Model warmup completed successfully.")
    except Exception as e:
        print(f"[{settings.APP_NAME}] Warmup warning: {e}")

    try:
        from app.services.consent_store import consent_store
        print(f"[{settings.APP_NAME}] Consent DB ready at: {settings.CONSENT_DB_PATH}")
    except Exception as e:
        print(f"[{settings.APP_NAME}] Consent DB init warning: {e}")

    yield
    print(f"[{settings.APP_NAME}] Shutting down gracefully.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Powered Hair & Scalp Health Triage & Classical CV Inference Engine",
    lifespan=lifespan
)

# Configure CORS with explicit allowed origins and safe headers/methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-Requested-With"],
)

# Static directory path
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Register API Router
app.include_router(api_router, prefix=settings.API_PREFIX)
app.include_router(api_router)  # Also expose directly for /health


@app.get("/", response_class=FileResponse)
def root():
    """
    Serves the interactive single-page web interface for browser users.
    Enables immediate local testing of photo uploads, pre-inference quality checks,
    symptom forms, and clinical PDF downloads.
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs_url": "/docs",
        "health_check": "/api/v1/system/status"
    }


@app.get("/api")
def api_root():
    """Lightweight API metadata discovery endpoint."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs_url": "/docs",
        "health_check": "/api/v1/system/status"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
