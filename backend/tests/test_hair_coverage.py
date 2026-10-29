"""
Test Suite for ScalpCVPipeline.compute_density_score (Hair Coverage Indicative).

Verifies:
1. Optical blur resilience: Gaussian blur (sigma=3.0) changes the score by < 20%
   (previously gradient edge energy collapsed by > 50-98%).
2. Monotonicity: Increasing synthetic dark hair strands produces higher coverage scores.
3. Strand-free images: A bare scalp or skin with heavy pixel noise scores low (<= 2.0),
   not falsely elevated by noise sharpness.
4. Low-contrast handling: Low-contrast inputs (e.g. dark skin with dark hair, or grey hair
   on skin) return a neutral value (12.50) with the low-confidence flag set.

NOTE: All thresholds in this module are provisional / uncalibrated heuristics.
Optical hair coverage indicates visible surface fiber area ratio, NOT follicular unit density.
"""

import cv2
import numpy as np
import pytest

from app.services.cv_pipeline import ScalpCVPipeline

SKIN_COLOR = (210, 160, 135)
DARK_HAIR_COLOR = (35, 28, 22)
IMG_SIZE = 512


def make_strand_image(n_strands: int, hair_color=DARK_HAIR_COLOR, skin_color=SKIN_COLOR, seed: int = 42) -> np.ndarray:
    """Creates synthetic image with n random linear strands drawn over skin background."""
    img = np.full((IMG_SIZE, IMG_SIZE, 3), skin_color, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(n_strands):
        x0, y0 = rng.integers(0, IMG_SIZE, size=2)
        x1, y1 = rng.integers(0, IMG_SIZE, size=2)
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), hair_color, 3)
    return img


def test_blur_does_not_collapse_score():
    """
    Acceptance Test 1:
    Blurring a synthetic photo (sigma=3.0) must change the score by less than 20%.
    Before the fix, edge_energy/18 collapsed by > 50% under mild blur.
    """
    for n in [50, 100, 150]:
        img = make_strand_image(n_strands=n, seed=42)
        score_orig = ScalpCVPipeline.compute_density_score(img)
        score_blur = ScalpCVPipeline.compute_density_score(cv2.GaussianBlur(img, (0, 0), sigmaX=3.0))

        diff_pct = abs(score_orig - score_blur) / score_orig * 100.0
        assert diff_pct < 20.0, (
            f"Blur collapse at n={n} strands: orig={score_orig:.2f}, blur={score_blur:.2f}, "
            f"diff={diff_pct:.1f}% (expected < 20%)"
        )


def test_more_strands_give_higher_coverage_score():
    """
    Acceptance Test 2:
    More synthetic dark strands must produce monotonically non-decreasing coverage scores.
    """
    strand_counts = [0, 10, 30, 75, 120, 180]
    scores = []
    for n in strand_counts:
        img = make_strand_image(n_strands=n, seed=42)
        score = ScalpCVPipeline.compute_density_score(img)
        scores.append(score)

    # Strand-free must score low
    assert scores[0] <= 2.0, f"Strand-free image scored {scores[0]:.2f}, expected <= 2.0"

    # Monotonically non-decreasing
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i - 1], (
            f"Monotonicity violation: {strand_counts[i]} strands scored {scores[i]:.2f} "
            f"< previous {strand_counts[i-1]} strands ({scores[i-1]:.2f})"
        )


def test_strand_free_noisy_image_scores_low():
    """
    Acceptance Test 3:
    A strand-free skin image with sharp Gaussian noise must score low (<= 2.0).
    Before the fix, high-frequency noise yielded high edge energy and a false 25.0 score.
    """
    rng = np.random.default_rng(42)
    img_noise = np.full((IMG_SIZE, IMG_SIZE, 3), SKIN_COLOR, dtype=np.float32)
    img_noise += rng.normal(0, 15, img_noise.shape)
    img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)

    score = ScalpCVPipeline.compute_density_score(img_noise)
    assert score <= 2.0, (
        f"Strand-free noisy image scored {score:.2f}, expected <= 2.0. "
        f"Pixel noise sharpness must not masquerade as hair coverage."
    )


def test_low_contrast_input_returns_neutral_value_and_flag():
    """
    Acceptance Test 4:
    Low-contrast inputs (dark skin with dark hair, or grey hair on skin) must return
    a neutral value (12.50) with the low-confidence flag set.
    """
    # 1. Dark skin with dark hair (phototype V-VI)
    img_dark_skin = make_strand_image(n_strands=50, hair_color=(35, 28, 22), skin_color=(55, 45, 40), seed=42)
    score_ds, low_ds, note_ds = ScalpCVPipeline.compute_density_score(img_dark_skin, return_details=True)
    assert score_ds == 12.50, f"Dark skin scored {score_ds}, expected neutral 12.50"
    assert low_ds is True, "Expected low_confidence=True for dark skin phototype"
    assert "contrast" in note_ds.lower()

    # 2. Grey hair on light skin
    img_grey = make_strand_image(n_strands=50, hair_color=(190, 185, 180), skin_color=(210, 160, 135), seed=42)
    score_gh, low_gh, note_gh = ScalpCVPipeline.compute_density_score(img_grey, return_details=True)
    assert score_gh == 12.50, f"Grey hair scored {score_gh}, expected neutral 12.50"
    assert low_gh is True, "Expected low_confidence=True for grey hair on skin"
    assert "contrast" in note_gh.lower()
