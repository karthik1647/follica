"""
Test Suite for ScalpCVPipeline.compute_sebum_score Sebum Balance Scoring.

Verifies:
1. Zero-shine (matte) image scores >= 24.0 (previously docked to 19.0).
2. Exactly 3.5% shine scores 25.0.
3. 8.0% shine scores clearly lower than 25.0 (e.g. ~13.75).
4. Score is monotonically non-increasing for shine ratios >= 3.5%.
5. Dryness is not double-counted (xerosis/flaking measured in flake score).

NOTE: All thresholds in this module are provisional / uncalibrated heuristics.
"""

import numpy as np
import pytest

from app.services.cv_pipeline import ScalpCVPipeline


def make_image_with_shine_ratio(shine_pct: float, size: int = 512) -> np.ndarray:
    """
    Creates a synthetic RGB image with an exact percentage of shine pixels.
    Base skin: (210, 160, 135) - matte, no shine (brightness=210 <= 220).
    Shine pixels: (240, 240, 240) - brightness=240 > 220, color_diff=0 < 30.
    """
    img = np.full((size, size, 3), (210, 160, 135), dtype=np.uint8)
    total_pixels = size * size
    num_shine = int(round((shine_pct / 100.0) * total_pixels))
    if num_shine > 0:
        flat = img.reshape(-1, 3)
        flat[:num_shine] = [240, 240, 240]
    return img


def test_zero_shine_matte_image_scores_at_least_24():
    """
    Acceptance Test 1:
    A matte, non-shiny scalp with zero shine should score 24-25.
    Previously, dryness subtraction (1 - ratio) * 6 docked all zero-shine
    images to exactly 19.0.
    """
    matte_img = make_image_with_shine_ratio(0.0)
    score = ScalpCVPipeline.compute_sebum_score(matte_img)
    assert score >= 24.0, (
        f"Matte image with zero shine scored {score:.2f}, expected >= 24.0. "
        f"Dryness must not be double-penalized in compute_sebum_score."
    )


def test_optimal_shine_scores_25():
    """
    Acceptance Test 2:
    Images with shine ratio <= 3.5% (e.g. 1.0%, 2.5%, 3.5%) score 25.0.
    """
    for pct in [0.5, 1.5, 2.5, 3.5]:
        img = make_image_with_shine_ratio(pct)
        score = ScalpCVPipeline.compute_sebum_score(img)
        assert score == 25.0, (
            f"Image with {pct}% shine scored {score:.2f}, expected 25.0."
        )


def test_excessive_shine_scores_clearly_lower():
    """
    Acceptance Test 3:
    8% shine scores clearly lower than 25.0.
    """
    img_8pct = make_image_with_shine_ratio(8.0)
    score_8pct = ScalpCVPipeline.compute_sebum_score(img_8pct)
    assert score_8pct < 20.0, (
        f"Image with 8% shine scored {score_8pct:.2f}, expected < 20.0."
    )
    # At slope 2.5: 25.0 - (8.0 - 3.5) * 2.5 = 13.75
    assert abs(score_8pct - 13.75) < 0.2, (
        f"Expected score ~13.75 for 8% shine, got {score_8pct:.2f}"
    )


def test_sebum_score_monotonic_non_increasing_above_optimal():
    """
    Acceptance Test 4:
    Function is monotonically non-increasing for shine ratios >= 3.5%.
    """
    shine_levels = [3.5, 4.0, 5.0, 6.5, 8.0, 10.0, 12.0, 15.0]
    scores = []
    for pct in shine_levels:
        img = make_image_with_shine_ratio(pct)
        scores.append(ScalpCVPipeline.compute_sebum_score(img))

    for i in range(1, len(scores)):
        assert scores[i] <= scores[i - 1], (
            f"Monotonicity violation at {shine_levels[i]}% shine: "
            f"score {scores[i]:.2f} > previous {scores[i - 1]:.2f}"
        )
