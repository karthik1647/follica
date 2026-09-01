"""
Follica FastAPI Main Application Entrypoint.

Starts the REST API server on localhost:8000 or cloud container.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
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
    yield
    print(f"[{settings.APP_NAME}] Shutting down gracefully.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Powered Hair & Scalp Health Triage & Classical CV Inference Engine",
    lifespan=lifespan
)

# Enable CORS for Expo Mobile App, Localhost Web, and Cloud clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router, prefix=settings.API_PREFIX)
app.include_router(api_router)  # Also expose directly for /health


@app.get("/")
def root():
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
