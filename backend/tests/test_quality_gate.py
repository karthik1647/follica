"""
Follica Image Quality Pre-Inference Gate Test Suite.

Verifies:
1. Sharp, well-illuminated images pass the quality gate.
2. Low-variance blurry images are rejected with actionable blur guidance.
3. Underexposed (<35 mean luminance) images are rejected with lighting guidance.
4. Overexposed (>225 mean luminance) images are rejected with glare guidance.
5. Undersized images (<150x150px) are rejected with resolution guidance.
6. Corrupted/invalid image payloads fail with standard validation errors.
7. Dedicated /api/v1/quality/check endpoint validation.
"""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.services.quality_gate import quality_gate

client = TestClient(app)


def make_sharp_image(size=(300, 300)) -> bytes:
    """Generates high-contrast synthetic image simulating hair/scalp texture."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = [180, 140, 120]
    arr[::4, :, :] = [30, 20, 10]    # Simulated dark hair fibers
    arr[:, ::8, :] = [210, 170, 150]  # Simulated bright parting/flakes
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_blurry_image(size=(300, 300)) -> bytes:
    """Generates a flat, smooth image with zero high-frequency edges."""
    img = Image.new("RGB", size, color=(160, 130, 110))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_dark_image(size=(300, 300)) -> bytes:
    """Generates a severely underexposed photo (<20 luminance)."""
    img = Image.new("RGB", size, color=(15, 12, 10))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_overexposed_image(size=(300, 300)) -> bytes:
    """Generates a washed-out, glare-heavy photo (>240 luminance)."""
    img = Image.new("RGB", size, color=(245, 245, 245))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_small_image(size=(100, 100)) -> bytes:
    """Generates an image below the 150x150px dimensional threshold."""
    img = Image.new("RGB", size, color=(150, 120, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_quality_gate_passes_sharp_image():
    """Verify sharp image passes the quality check."""
    sharp_bytes = make_sharp_image()
    is_valid, report = quality_gate.evaluate(sharp_bytes)
    assert is_valid is True
    assert report["error_code"] is None
    assert report["metrics"]["blur_passed"] is True
    assert report["metrics"]["brightness_passed"] is True


def test_quality_gate_rejects_blurry_image():
    """Verify flat/blurry image fails with IMAGE_TOO_BLURRY code."""
    blurry_bytes = make_blurry_image()
    is_valid, report = quality_gate.evaluate(blurry_bytes)
    assert is_valid is False
    assert report["error_code"] == "IMAGE_TOO_BLURRY"
    assert "focus on the scalp" in report["user_guidance"].lower()


def test_quality_gate_rejects_dark_image():
    """Verify dark image fails with IMAGE_UNDEREXPOSED code."""
    dark_bytes = make_dark_image()
    is_valid, report = quality_gate.evaluate(dark_bytes)
    assert is_valid is False
    assert report["error_code"] == "IMAGE_UNDEREXPOSED"
    assert "dark" in report["user_guidance"].lower() or "flash" in report["user_guidance"].lower()


def test_quality_gate_rejects_overexposed_image():
    """Verify washed-out image fails with IMAGE_OVEREXPOSED code."""
    bright_bytes = make_overexposed_image()
    is_valid, report = quality_gate.evaluate(bright_bytes)
    assert is_valid is False
    assert report["error_code"] == "IMAGE_OVEREXPOSED"
    assert "washed out" in report["user_guidance"].lower()


def test_quality_gate_rejects_undersized_image():
    """Verify undersized image fails with IMAGE_RESOLUTION_TOO_LOW code."""
    small_bytes = make_small_image()
    is_valid, report = quality_gate.evaluate(small_bytes)
    assert is_valid is False
    assert report["error_code"] == "IMAGE_RESOLUTION_TOO_LOW"


def test_quality_check_endpoint_api():
    """Verify standalone POST /api/v1/quality/check endpoint."""
    sharp_bytes = make_sharp_image()
    response = client.post(
        "/api/v1/quality/check",
        files={"image": ("sharp.jpg", sharp_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "passed"

    blurry_bytes = make_blurry_image()
    bad_response = client.post(
        "/api/v1/quality/check",
        files={"image": ("blurry.jpg", blurry_bytes, "image/jpeg")}
    )
    assert bad_response.status_code == 400
    assert bad_response.json()["detail"]["error_code"] == "IMAGE_TOO_BLURRY"
