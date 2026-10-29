"""
Test Suite for ScalpCVPipeline.compute_redness_score Erythema Scoring.

Verifies:
1. Flat erythema image with colour (225, 120, 110) plus mild noise scores clearly
   below 25.0 (specifically <= 18.0). Previously, Gray-World white balancing scaled
   R, G, B channel means to equality, resulting in a false-healthy score of 25.0.
2. Neutral skin-tone image under a mild warm ambient tint still scores >= 22.0.
3. Global strong red cast exposes low confidence indicator / note due to the
   inherent physical ambiguity between warm lighting and widespread erythema.

NOTE: All thresholds in this module are provisional / uncalibrated heuristics.
Progress comparison (/progress/compare) against the user's own baseline scan is
the clinically recommended approach for tracking vascular erythema.
"""

import numpy as np
import pytest

from app.services.cv_pipeline import ScalpCVPipeline


def make_noisy_color_image(color: tuple[int, int, int], noise_sd: float = 3.0, seed: int = 42, size: int = 512) -> np.ndarray:
    """Creates a 512x512 synthetic RGB image with specified base color and Gaussian pixel noise."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), color, dtype=np.float32)
    if noise_sd > 0:
        img += rng.normal(0, noise_sd, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_flat_erythema_scores_clearly_below_25():
    """
    Acceptance Test 1:
    A flat image with colour (225, 120, 110) plus mild noise represents widespread
    vascular erythema and must score clearly below 25.0 (specifically <= 18.0).
    Before the fix, Gray-World white balance fully normalized R/G/B means and produced 25.0.
    """
    erythema_img = make_noisy_color_image(color=(225, 120, 110), noise_sd=3.0, seed=42)
    score = ScalpCVPipeline.compute_redness_score(erythema_img)

    assert score <= 18.0, (
        f"Widespread erythema image (225, 120, 110) scored {score:.2f}, expected <= 18.0. "
        f"Gray-World white balancing must not fully cancel a global red inflammation cast."
    )


def test_neutral_skin_under_mild_warm_tint_scores_at_least_22():
    """
    Acceptance Test 2:
    A neutral skin tone (e.g. Fitzpatrick III ~(210, 160, 135)) photographed under
    mild warm ambient lighting (~2700-3000K incandescent, shifting chromaticity to ~(222, 160, 125))
    must still score >= 22.0 (retaining calm/healthy scalp status).
    """
    warm_img = make_noisy_color_image(color=(222, 160, 125), noise_sd=3.0, seed=42)
    score = ScalpCVPipeline.compute_redness_score(warm_img)

    assert score >= 22.0, (
        f"Neutral skin under mild warm tint scored {score:.2f}, expected >= 22.0. "
        f"White balance must tolerate reasonable ambient lighting shifts."
    )


def test_redness_details_and_low_confidence_flag():
    """
    Acceptance Test 3:
    When an image has an extreme global red cast (225, 120, 110), compute_redness_score(return_details=True)
    must flag low confidence due to the unavoidable ambiguity between warm lighting and erythema.
    """
    erythema_img = make_noisy_color_image(color=(225, 120, 110), noise_sd=3.0, seed=42)
    score, low_conf, note = ScalpCVPipeline.compute_redness_score(erythema_img, return_details=True)

    assert score <= 18.0
    assert low_conf is True, "Expected low_confidence=True for global red cast image"
    assert "lighting" in note.lower() or "erythema" in note.lower()
