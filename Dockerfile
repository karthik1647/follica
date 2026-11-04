# Multi-stage production Dockerfile for Follica Scalp Health Triage API & Web Studio
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend

# Install required system packages for OpenCV headless image processing and curl health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition and install packages
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application backend, static web assets, and documentation
COPY backend /app/backend
COPY docs /app/docs

# Configure secure unprivileged user and persistable data directory for SQLite consent store
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/backend/data && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Health check monitoring system ready status
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/system/status || exit 1

# Start ASGI application server on port 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
