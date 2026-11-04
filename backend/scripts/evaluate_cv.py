#!/usr/bin/env python3
"""
Follica ScalpCVPipeline Dataset Evaluation & Quality Diagnostic Script.

Purpose:
  Evaluates the classical computer vision pipeline across a directory of scalp photos
  (organized as `data/eval/<label>/*.jpg` or flat folder). Generates:
  1. A structured CSV with per-image sub-scores, SHS, and optical metrics.
  2. A console report with per-label descriptive statistics (mean, std, min, max).
  3. Diagnostic warnings flagging:
     - SHS value clustering (>15% of images sharing an identical rounded score)
     - Score saturation (sub-scores stuck at cap 25 or floor 0 for >60% of images)
     - Quality confounding (correlation between sub-scores and sharpness/brightness)
  4. Diagnostic suggestions (no automatic threshold tuning).

Threshold Notice:
  Diagnostic thresholds (15% clustering, 60% saturation, 0.40 correlation) are
  provisional / uncalibrated heuristic guidelines designed for dataset quality checks.
"""

import argparse
import csv
import io
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Ensure backend root is in sys.path when executed as a script
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.services.cv_pipeline import ScalpCVPipeline, cv_pipeline
from app.services.quality_gate import quality_gate

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Provisional / uncalibrated diagnostic thresholds
SHS_CLUSTERING_THRESHOLD_PCT = 15.0      # Flag if any single rounded SHS represents >15% of dataset
SCORE_SATURATION_THRESHOLD_PCT = 60.0    # Flag if a sub-score is at floor (0) or cap (25) for >60% of dataset
QUALITY_CORRELATION_WARN_THRESHOLD = 0.40  # Flag if Pearson |r| between metric and quality is >0.40


def discover_images(data_dir: Path) -> List[Tuple[Path, str]]:
    """
    Scans data_dir for scalp images.
    If images are organized in subdirectories (`data/eval/<label>/*.jpg`), the subfolder
    name is treated as the clinical/ground-truth label.
    If images are in the root directory, label is marked as 'unlabeled'.
    """
    discovered: List[Tuple[Path, str]] = []
    if not data_dir.exists():
        return discovered

    # 1. Subdirectory discovery (preferred layout: data_dir/<label>/*.jpg)
    for entry in sorted(data_dir.iterdir()):
        if entry.is_dir():
            label = entry.name
            for img_file in sorted(entry.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    discovered.append((img_file, label))

    # 2. Top-level images (flat layout)
    for img_file in sorted(data_dir.iterdir()):
        if img_file.is_file() and img_file.suffix.lower() in SUPPORTED_EXTENSIONS:
            discovered.append((img_file, "unlabeled"))

    return discovered


def compute_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes Pearson correlation coefficient between two arrays.
    Returns 0.0 if either array has zero variance or if sample size < 2.
    """
    if len(x) < 2 or len(y) < 2:
        return 0.0
    std_x = float(np.std(x))
    std_y = float(np.std(y))
    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    corr_mat = np.corrcoef(x, y)
    r = float(corr_mat[0, 1])
    return 0.0 if np.isnan(r) else r


def evaluate_single_image(
    image_path: Path,
    label: str,
    pipeline: Optional[ScalpCVPipeline] = None,
) -> Dict[str, Any]:
    """
    Runs ScalpCVPipeline on an image file, computing sub-scores, SHS,
    and optical quality metrics (Laplacian sharpness, mean brightness).
    """
    pipe = pipeline or cv_pipeline
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # 1. Classical CV Pipeline Evaluation
    sub_scores = pipe.analyze_image(img_bytes)

    # 2. Total Scalp Health Score (clamped [0.0, 100.0])
    total_shs = round(
        sub_scores.flake_score
        + sub_scores.redness_score
        + sub_scores.sebum_score
        + sub_scores.density_score,
        2,
    )
    total_shs = float(max(0.0, min(100.0, total_shs)))

    # 3. Optical Quality Metrics
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_rgb = np.array(pil_img)
    sharpness = quality_gate.calculate_laplacian_variance(img_rgb)
    brightness = quality_gate.calculate_brightness(img_rgb)

    return {
        "image_path": str(image_path.resolve()),
        "filename": image_path.name,
        "label": label,
        "flake_score": float(round(sub_scores.flake_score, 2)),
        "redness_score": float(round(sub_scores.redness_score, 2)),
        "sebum_score": float(round(sub_scores.sebum_score, 2)),
        "density_score": float(round(sub_scores.density_score, 2)),
        "scalp_health_score": float(round(total_shs, 2)),
        "sharpness_laplacian": float(round(sharpness, 2)),
        "mean_brightness": float(round(brightness, 2)),
    }


def write_csv_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Writes per-image evaluation results to CSV."""
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "filename",
        "label",
        "flake_score",
        "redness_score",
        "sebum_score",
        "density_score",
        "scalp_health_score",
        "sharpness_laplacian",
        "mean_brightness",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def generate_console_report(results: List[Dict[str, Any]]) -> Tuple[str, List[str], List[str]]:
    """
    Generates a comprehensive diagnostic report from evaluation results.
    Returns:
      (report_text, list_of_warnings, list_of_suggestions)
    """
    lines: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

    n_images = len(results)
    if n_images == 0:
        return "[Follica CV Eval] No images analyzed. Report is empty.\n", [], []

    lines.append("=" * 80)
    lines.append(f"FOLLICA SCALP CV PIPELINE EVALUATION REPORT (Total Images: {n_images})")
    lines.append("=" * 80)

    # 1. Group by Label
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_label[r["label"]].append(r)

    score_keys = [
        "flake_score",
        "redness_score",
        "sebum_score",
        "density_score",
        "scalp_health_score",
    ]

    lines.append("\n[1] PER-LABEL DESCRIPTIVE STATISTICS (Mean +/- Std [Min, Max])")
    lines.append("-" * 80)

    for label, group in sorted(by_label.items()):
        lines.append(f"\n* Label: '{label}' (n = {len(group)})")
        for key in score_keys:
            vals = np.array([r[key] for r in group], dtype=float)
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            v_min = float(np.min(vals))
            v_max = float(np.max(vals))
            lines.append(
                f"    {key:<20}: mean = {mean:6.2f} +/- {std:5.2f} | range = [{v_min:5.2f}, {v_max:5.2f}]"
            )

    # Overall summary
    lines.append("\n* ALL LABELS COMBINED (Overall Dataset)")
    for key in score_keys:
        vals = np.array([r[key] for r in results], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        v_min = float(np.min(vals))
        v_max = float(np.max(vals))
        lines.append(
            f"    {key:<20}: mean = {mean:6.2f} +/- {std:5.2f} | range = [{v_min:5.2f}, {v_max:5.2f}]"
        )

    # 2. Identical SHS Value Clustering Check
    lines.append("\n" + "-" * 80)
    lines.append("[2] SCALP HEALTH SCORE (SHS) VALUE CLUSTERING")
    lines.append("-" * 80)

    rounded_shs = [int(round(r["scalp_health_score"])) for r in results]
    shs_counts = Counter(rounded_shs)
    most_common = shs_counts.most_common(5)

    lines.append("Top 5 most frequent rounded SHS values:")
    for score_val, count in most_common:
        pct = (count / n_images) * 100.0
        lines.append(f"  SHS = {score_val:3d} : {count:4d} images ({pct:5.1f}%)")
        if pct > SHS_CLUSTERING_THRESHOLD_PCT and n_images >= 5:
            warn_msg = (
                f"SHS Clustering: Rounded score {score_val} covers {pct:.1f}% of images "
                f"({count}/{n_images}), exceeding the {SHS_CLUSTERING_THRESHOLD_PCT:.1f}% threshold."
            )
            warnings.append(warn_msg)
            sugg_msg = (
                f"Suggestion for SHS={score_val}: Evaluate if default fallback weights or discrete step "
                f"functions in sub-scores are artificially attracting results to this score."
            )
            if sugg_msg not in suggestions:
                suggestions.append(sugg_msg)

    # 3. Sub-Score Cap (25) and Floor (0) Saturation Check
    lines.append("\n" + "-" * 80)
    lines.append("[3] SUB-SCORE BOUNDARY SATURATION (Cap = 25.0, Floor = 0.0)")
    lines.append("-" * 80)

    sub_score_keys = ["flake_score", "redness_score", "sebum_score", "density_score"]
    for key in sub_score_keys:
        vals = np.array([r[key] for r in results], dtype=float)
        floor_count = int(np.sum(vals <= 0.01))
        cap_count = int(np.sum(vals >= 24.99))
        floor_pct = (floor_count / n_images) * 100.0
        cap_pct = (cap_count / n_images) * 100.0

        lines.append(
            f"  {key:<18}: Floor (0.0) = {floor_count:3d} ({floor_pct:5.1f}%) | "
            f"Cap (25.0) = {cap_count:3d} ({cap_pct:5.1f}%)"
        )

        if floor_pct > SCORE_SATURATION_THRESHOLD_PCT and n_images >= 5:
            warn_msg = (
                f"Floor Saturation: '{key}' is stuck at 0.0 for {floor_pct:.1f}% of images "
                f"({floor_count}/{n_images}), exceeding {SCORE_SATURATION_THRESHOLD_PCT:.1f}% threshold."
            )
            warnings.append(warn_msg)
            suggestions.append(
                f"Suggestion for '{key}': Penalty threshold may be too severe. Consider reviewing "
                f"the baseline curve slope in cv_pipeline."
            )

        if cap_pct > SCORE_SATURATION_THRESHOLD_PCT and n_images >= 5:
            warn_msg = (
                f"Cap Saturation: '{key}' is stuck at 25.0 for {cap_pct:.1f}% of images "
                f"({cap_count}/{n_images}), exceeding {SCORE_SATURATION_THRESHOLD_PCT:.1f}% threshold."
            )
            warnings.append(warn_msg)
            suggestions.append(
                f"Suggestion for '{key}': Detection threshold may be too conservative or missing subtle signs. "
                f"Consider adjusting sensitivity."
            )

    # 4. Correlation with Image Sharpness & Mean Brightness
    lines.append("\n" + "-" * 80)
    lines.append("[4] PHOTO QUALITY CONFOUNDING (Pearson Correlation r)")
    lines.append("-" * 80)
    lines.append(f"{'Metric':<22} {'Sharpness (Laplacian)':<24} {'Mean Brightness':<20}")
    lines.append("-" * 80)

    sharpness_arr = np.array([r["sharpness_laplacian"] for r in results], dtype=float)
    brightness_arr = np.array([r["mean_brightness"] for r in results], dtype=float)

    for key in score_keys:
        metric_arr = np.array([r[key] for r in results], dtype=float)
        r_sharp = compute_pearson_r(metric_arr, sharpness_arr)
        r_bright = compute_pearson_r(metric_arr, brightness_arr)

        sharp_flag = " (!)" if abs(r_sharp) >= QUALITY_CORRELATION_WARN_THRESHOLD else ""
        bright_flag = " (!)" if abs(r_bright) >= QUALITY_CORRELATION_WARN_THRESHOLD else ""

        lines.append(
            f"{key:<22} r = {r_sharp:+6.3f}{sharp_flag:<5}          r = {r_bright:+6.3f}{bright_flag:<5}"
        )

        if abs(r_sharp) >= QUALITY_CORRELATION_WARN_THRESHOLD and n_images >= 5:
            warnings.append(
                f"Quality Confounding: '{key}' has high correlation with sharpness (|r| = {abs(r_sharp):.3f} >= {QUALITY_CORRELATION_WARN_THRESHOLD:.2f})."
            )
            suggestions.append(
                f"Suggestion for '{key}': Decouple edge-detection or texture energy from blur/focus differences."
            )

        if abs(r_bright) >= QUALITY_CORRELATION_WARN_THRESHOLD and n_images >= 5:
            warnings.append(
                f"Quality Confounding: '{key}' has high correlation with brightness (|r| = {abs(r_bright):.3f} >= {QUALITY_CORRELATION_WARN_THRESHOLD:.2f})."
            )
            suggestions.append(
                f"Suggestion for '{key}': Check exposure invariance, white balancing (Gray-World), or illumination normalization."
            )

    # 5. Summary of Diagnostic Flags & Suggestions
    lines.append("\n" + "=" * 80)
    lines.append("[5] DIAGNOSTIC FLAGS & SUGGESTIONS")
    lines.append("=" * 80)

    if warnings:
        lines.append(f"\n[!] {len(warnings)} WARNING(S) IDENTIFIED:")
        for w in warnings:
            lines.append(f"  - {w}")
    else:
        lines.append("\n[*] No major score saturation, clustering, or quality confounding issues detected.")

    if suggestions:
        lines.append("\n[i] RECOMMENDATIONS & NEXT STEPS (Manual review only - no auto-tuning):")
        for s in suggestions:
            lines.append(f"  * {s}")

    lines.append("\n" + "=" * 80 + "\n")
    report_text = "\n".join(lines)
    return report_text, warnings, suggestions


def run_evaluation(
    data_dir: Path,
    output_csv: Optional[Path] = None,
    print_report: bool = True,
    pipeline: Optional[ScalpCVPipeline] = None,
) -> Dict[str, Any]:
    """
    Main evaluation routine.
    Scans data_dir, evaluates images, writes CSV if path provided, and prints report.
    Returns a dictionary of execution results and diagnostic metrics.
    """
    images = discover_images(data_dir)

    if not images:
        msg = (
            f"[Follica CV Eval] Notice: No evaluation images found in '{data_dir}'.\n"
            f"Expected layout: '{data_dir}/<label>/*.jpg' (supports .jpg, .jpeg, .png, .webp).\n"
            f"Exiting cleanly without errors."
        )
        if print_report:
            print(msg)
        return {
            "status": "empty",
            "total_images": 0,
            "results": [],
            "warnings": [],
            "suggestions": [],
            "message": msg,
        }

    results: List[Dict[str, Any]] = []
    for img_path, label in images:
        try:
            res = evaluate_single_image(img_path, label, pipeline=pipeline)
            results.append(res)
        except Exception as e:
            if print_report:
                print(f"[Follica CV Eval] Warning: Failed to process {img_path}: {e}")

    # Determine default CSV output path if not specified
    if output_csv is None:
        output_csv = data_dir / "cv_evaluation.csv"

    write_csv_report(results, output_csv)

    report_text, warnings, suggestions = generate_console_report(results)
    if print_report:
        print(report_text)
        print(f"[Follica CV Eval] Detailed per-image CSV written to: {output_csv.resolve()}")

    return {
        "status": "success",
        "total_images": len(results),
        "results": results,
        "warnings": warnings,
        "suggestions": suggestions,
        "csv_path": str(output_csv.resolve()),
        "report_text": report_text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Follica ScalpCVPipeline Dataset Evaluation and Optical Diagnostics"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/eval",
        help="Path to evaluation dataset directory (default: data/eval)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to write the output CSV report (default: <data_dir>/cv_evaluation.csv)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    out_path = Path(args.output) if args.output else None

    run_evaluation(data_dir=data_path, output_csv=out_path, print_report=True)


if __name__ == "__main__":
    main()
