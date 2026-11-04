"""
Unit and Integration Tests for backend/scripts/evaluate_cv.py

Verifies:
1. Graceful execution on empty or missing folder (ACCEPTANCE requirement)
2. Evaluation over structured folder of generated images (data/eval/<label>/*.jpg)
3. CSV generation with per-image sub-scores, SHS, and optical metrics
4. Diagnostic checks: SHS clustering (>15%), saturation (>60%), and quality confounding
"""

import csv
import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

import pytest
from scripts.evaluate_cv import (
    discover_images,
    evaluate_single_image,
    generate_console_report,
    run_evaluation,
    compute_pearson_r,
)


def _create_synthetic_scalp_image(path: Path, mean_r: int, mean_g: int, mean_b: int, add_texture: bool = True):
    """Helper to synthesize test scalp images with known color/texture properties."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 256, 256
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(mean_r, 0, 255)
    arr[:, :, 1] = np.clip(mean_g, 0, 255)
    arr[:, :, 2] = np.clip(mean_b, 0, 255)

    if add_texture:
        # Add high-frequency noise to simulate sharpness/follicles
        noise = np.random.randint(-20, 20, (h, w, 3))
        arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    img.save(path, format="JPEG", quality=90)


def test_evaluate_cv_empty_and_missing_folder():
    """ACCEPTANCE: Runs on empty or missing folder without crashing and returns clear message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_path = Path(tmpdir) / "empty_eval"
        empty_path.mkdir()

        # 1. Empty folder
        result_empty = run_evaluation(data_dir=empty_path, print_report=False)
        assert result_empty["status"] == "empty"
        assert result_empty["total_images"] == 0
        assert "Notice: No evaluation images found" in result_empty["message"]

        # 2. Non-existent folder
        missing_path = Path(tmpdir) / "does_not_exist"
        result_missing = run_evaluation(data_dir=missing_path, print_report=False)
        assert result_missing["status"] == "empty"
        assert result_missing["total_images"] == 0
        assert "Notice: No evaluation images found" in result_missing["message"]


def test_evaluate_cv_generated_sample_images_dataset():
    """ACCEPTANCE: Runs on folder of generated sample images in unit test, produces CSV and report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data" / "eval"
        csv_path = Path(tmpdir) / "output" / "eval_report.csv"

        # Generate a structured dataset: 3 labels with 3 images each = 9 images
        labels = ["normal", "seborrheic_dermatitis", "erythema_active"]
        total_created = 0

        for label in labels:
            for idx in range(3):
                img_path = data_dir / label / f"sample_{idx}.jpg"
                if label == "normal":
                    _create_synthetic_scalp_image(img_path, mean_r=200, mean_g=160, mean_b=140, add_texture=True)
                elif label == "seborrheic_dermatitis":
                    _create_synthetic_scalp_image(img_path, mean_r=220, mean_g=140, mean_b=120, add_texture=True)
                else:
                    _create_synthetic_scalp_image(img_path, mean_r=240, mean_g=110, mean_b=100, add_texture=False)
                total_created += 1

        assert total_created == 9

        # Run evaluation
        eval_res = run_evaluation(data_dir=data_dir, output_csv=csv_path, print_report=False)

        assert eval_res["status"] == "success"
        assert eval_res["total_images"] == 9
        assert csv_path.exists()

        # Verify CSV contents
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 9

            for row in reader:
                assert row["label"] in labels
                assert Path(row["image_path"]).exists()

                flake = float(row["flake_score"])
                redness = float(row["redness_score"])
                sebum = float(row["sebum_score"])
                density = float(row["density_score"])
                shs = float(row["scalp_health_score"])

                # Check bounds
                assert 0.0 <= flake <= 25.0
                assert 0.0 <= redness <= 25.0
                assert 0.0 <= sebum <= 25.0
                assert 0.0 <= density <= 25.0
                assert 0.0 <= shs <= 100.0

                # Check optical metrics are present
                assert float(row["sharpness_laplacian"]) >= 0.0
                assert 0.0 <= float(row["mean_brightness"]) <= 255.0

        # Verify report text structure
        report_text = eval_res["report_text"]
        assert "FOLLICA SCALP CV PIPELINE EVALUATION REPORT" in report_text
        assert "PER-LABEL DESCRIPTIVE STATISTICS" in report_text
        assert "SCALP HEALTH SCORE (SHS) VALUE CLUSTERING" in report_text
        assert "SUB-SCORE BOUNDARY SATURATION" in report_text
        assert "PHOTO QUALITY CONFOUNDING" in report_text


def test_diagnostic_shs_clustering_flag():
    """Verify warning triggers when >15% of images share an identical rounded SHS."""
    results = []
    # 20 images total: 10 images share SHS = 85 (50% > 15%)
    for i in range(10):
        results.append({
            "label": "test",
            "flake_score": 20.0,
            "redness_score": 20.0,
            "sebum_score": 20.0,
            "density_score": 25.0,
            "scalp_health_score": 85.0,
            "sharpness_laplacian": 150.0 + i,
            "mean_brightness": 120.0 + i,
        })
    for i in range(10):
        results.append({
            "label": "test",
            "flake_score": 10.0 + i,
            "redness_score": 15.0,
            "sebum_score": 12.0,
            "density_score": 18.0,
            "scalp_health_score": 55.0 + i * 2,
            "sharpness_laplacian": 100.0 + i * 5,
            "mean_brightness": 90.0 + i * 3,
        })

    report_text, warnings, suggestions = generate_console_report(results)
    assert any("SHS Clustering" in w and "85" in w for w in warnings)
    assert any("SHS=85" in s for s in suggestions)


def test_diagnostic_subscore_saturation_flags():
    """Verify warning triggers when sub-score is stuck at cap (25) or floor (0) for >60% of images."""
    results = []
    # 10 images total: flake_score at cap 25 for 8 images (80% > 60%), sebum at floor 0 for 7 images (70% > 60%)
    for i in range(10):
        results.append({
            "label": "test",
            "flake_score": 25.0 if i < 8 else 15.0,
            "redness_score": 18.0,
            "sebum_score": 0.0 if i < 7 else 12.0,
            "density_score": 15.0,
            "scalp_health_score": 58.0 + i,
            "sharpness_laplacian": 120.0 + i,
            "mean_brightness": 110.0 + i,
        })

    report_text, warnings, suggestions = generate_console_report(results)
    assert any("Cap Saturation: 'flake_score'" in w for w in warnings)
    assert any("Floor Saturation: 'sebum_score'" in w for w in warnings)


def test_diagnostic_quality_correlation_flag():
    """Verify warning triggers when a sub-score has strong correlation with sharpness or brightness."""
    results = []
    # Create strong correlation between density_score and sharpness_laplacian (r ~ 1.0)
    for i in range(10):
        results.append({
            "label": "test",
            "flake_score": 15.0 + (i % 2),
            "redness_score": 15.0,
            "sebum_score": 15.0,
            "density_score": 10.0 + (i * 1.5),             # strictly increasing
            "scalp_health_score": 55.0 + i,
            "sharpness_laplacian": 50.0 + (i * 20.0),      # strictly increasing with density
            "mean_brightness": 100.0,
        })

    report_text, warnings, suggestions = generate_console_report(results)
    assert any("Quality Confounding: 'density_score'" in w and "sharpness" in w for w in warnings)


def test_compute_pearson_r_handles_constants_and_edge_cases():
    """Verify Pearson correlation calculation does not raise on constant inputs or small arrays."""
    # Constant input (zero variance)
    assert compute_pearson_r(np.array([5.0, 5.0, 5.0]), np.array([1.0, 2.0, 3.0])) == 0.0
    # Array with <2 points
    assert compute_pearson_r(np.array([1.0]), np.array([2.0])) == 0.0
    # Perfect positive correlation
    r_pos = compute_pearson_r(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    assert pytest.approx(r_pos, 0.001) == 1.0
    # Perfect negative correlation
    r_neg = compute_pearson_r(np.array([1.0, 2.0, 3.0]), np.array([6.0, 4.0, 2.0]))
    assert pytest.approx(r_neg, 0.001) == -1.0
