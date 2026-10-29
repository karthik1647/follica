"""
Regression & Stability Test Suite for ScalpCVPipeline Flake Detector.

Addresses Prompt 5 / Flake detector instability:
1. Verifies no-flake images with Gaussian noise sd in [0, 25] all score >= 22.0.
2. Verifies images with dark hair strands (colour ~(35, 28, 22), 3px width) but no flakes score >= 20.0.
3. Verifies score is monotonically non-increasing as synthetic flake particle coverage increases.

NOTE: All thresholds in this pipeline and test suite are provisional / uncalibrated
engineering heuristics. None have been fitted to or validated against clinical datasets.
"""

import cv2
import numpy as np
import pytest

from app.services.cv_pipeline import ScalpCVPipeline

# Constants matching benchmark specifications
SKIN_COLOR = (210, 160, 135)   # RGB base skin tone
HAIR_COLOR = (35, 28, 22)      # RGB dark hair fiber tone
FLAKE_COLOR = (245, 240, 220)  # RGB slightly yellowish-white keratin squames
IMG_SIZE = 512


def test_no_flake_gaussian_noise_stability():
    """
    Acceptance Test 1:
    No-flake images with Gaussian noise sd in range [0, 25] must all score >= 22.0.
    Prevents noise-induced false positives where CLAHE and Top-Hat previously
    amplified sub-pixel Gaussian noise into phantom flakes.

    Note: Pipeline thresholds used here are provisional / uncalibrated.
    """
    noise_levels = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]
    fixed_seeds = [42, 101, 777]

    for seed in fixed_seeds:
        for sd in noise_levels:
            rng = np.random.default_rng(seed)
            img = np.full((IMG_SIZE, IMG_SIZE, 3), SKIN_COLOR, dtype=np.float32)
            if sd > 0:
                img += rng.normal(0, sd, img.shape)
            img = np.clip(img, 0, 255).astype(np.uint8)

            score = ScalpCVPipeline.compute_flake_score(img)
            assert score >= 22.0, (
                f"Noise stability failure at seed={seed}, sd={sd}: "
                f"expected score >= 22.0, got {score:.2f}"
            )


def test_dark_hair_strands_no_false_flakes():
    """
    Acceptance Test 2:
    Images with dark hair strands on skin but no flakes must score >= 20.0.
    Verifies that inter-hair skin gaps are masked out and not misclassified as flakes.

    Note: Pipeline thresholds used here are provisional / uncalibrated.
    """
    strand_counts = [10, 30, 75, 150]
    fixed_seeds = [42, 2026]

    for seed in fixed_seeds:
        for n_strands in strand_counts:
            rng = np.random.default_rng(seed)
            img = np.full((IMG_SIZE, IMG_SIZE, 3), SKIN_COLOR, dtype=np.uint8)
            for _ in range(n_strands):
                x0, y0 = rng.integers(0, IMG_SIZE, size=2)
                x1, y1 = rng.integers(0, IMG_SIZE, size=2)
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), HAIR_COLOR, 3)

            score = ScalpCVPipeline.compute_flake_score(img)
            assert score >= 20.0, (
                f"Hair strands false-positive failure at seed={seed}, strands={n_strands}: "
                f"expected score >= 20.0, got {score:.2f}"
            )


def test_synthetic_flake_score_monotonicity():
    """
    Acceptance Test 3:
    Score must be monotonically non-increasing as synthetic flake coverage increases.
    Also verifies that real flakes are detected and degrade the score accordingly.

    Note: Pipeline thresholds used here are provisional / uncalibrated.
    """
    flake_counts = [0, 50, 150, 350, 700, 1200]
    fixed_seeds = [42, 123]

    for seed in fixed_seeds:
        scores = []
        for n_flakes in flake_counts:
            rng = np.random.default_rng(seed)
            img = np.full((IMG_SIZE, IMG_SIZE, 3), SKIN_COLOR, dtype=np.uint8)
            for _ in range(n_flakes):
                cx, cy = rng.integers(10, IMG_SIZE - 10, size=2)
                radius = rng.integers(2, 5)
                cv2.circle(img, (int(cx), int(cy)), int(radius), FLAKE_COLOR, -1)

            score = ScalpCVPipeline.compute_flake_score(img)
            scores.append(score)

        # 1. Monotonically non-increasing: score[i] <= score[i-1]
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1], (
                f"Monotonicity violation at seed={seed}, count={flake_counts[i]}: "
                f"score {scores[i]:.2f} > previous {scores[i - 1]:.2f} (counts={flake_counts}, scores={scores})"
            )

        # 2. Significant detection: 1200 flakes must push score below 10.0
        assert scores[-1] < 10.0, (
            f"Under-detection: 1200 flakes expected score < 10.0, got {scores[-1]:.2f}"
        )
