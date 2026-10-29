# Follica â€” Numeric Threshold Registry

> **IMPORTANT**: Every threshold listed in this document is a **provisional engineering
> heuristic**. None of these values have been fitted to a clinical dataset or validated
> against expert dermatological assessments. They were chosen for algorithmic plausibility
> only. Before any clinical or regulatory use, each threshold must be re-derived from
> appropriately collected and labelled data.

---

## Format

| File | Constant / Expression | Value | Controls | Status |
|------|-----------------------|-------|----------|--------|

---

## 1. Image Quality Gate Thresholds (`core/config.py`)

| File | Constant | Value | Controls | Status |
|------|----------|-------|----------|--------|
| `backend/app/core/config.py` | `QUALITY_MIN_BLUR_SCORE` | `60.0` | Minimum Laplacian variance. Images below this are rejected as too blurry. | **uncalibrated** |
| `backend/app/core/config.py` | `QUALITY_MIN_BRIGHTNESS` | `35.0` | Minimum mean luminance (Rec. 601). Images below this are rejected as underexposed. | **uncalibrated** |
| `backend/app/core/config.py` | `QUALITY_MAX_BRIGHTNESS` | `225.0` | Maximum mean luminance. Images above this are rejected as overexposed. | **uncalibrated** |
| `backend/app/core/config.py` | `QUALITY_MIN_DIMENSION` | `150` (px) | Minimum width and height. Images smaller than this lack sufficient resolution for morphological analysis. | **uncalibrated** |
| `backend/app/core/config.py` | `CONFIDENCE_GATE_THRESHOLD` | `0.65` | Deep learning predictions below this confidence are rerouted to Level 2 (Monitored Routine) instead of producing a definitive classification. | **uncalibrated** |
| `backend/app/core/config.py` | `MAX_IMAGE_SIZE_BYTES` | `10 Ã— 1024 Ã— 1024` (10 MB) | Maximum allowed upload size. Administrative / infrastructure limit, not derived from image quality requirements. | engineering limit |

---

## 2. Flash Glare Thresholds (`services/quality_gate.py`)

| File | Constant / Expression | Value | Controls | Status |
|------|-----------------------|-------|----------|--------|
| `backend/app/services/quality_gate.py` | `brightness > 240` (in `calculate_flash_glare_ratio`) | `240` | Max-channel brightness above which a pixel is considered a blown-out flash hotspot. | **uncalibrated** |
| `backend/app/services/quality_gate.py` | `color_diff < 20` (in `calculate_flash_glare_ratio`) | `20` | Achromatic colour-difference threshold for flash pixel identification. | **uncalibrated** |
| `backend/app/services/quality_gate.py` | `flash_glare_ratio <= 15.0` (in `evaluate`) | `15.0 %` | Maximum acceptable fraction of image area occupied by flash glare before the image is rejected. | **uncalibrated** |

---

## 3. Specular Glare Mask Thresholds (`services/cv_pipeline.py`)

| File | Constant / Expression | Value | Controls | Status |
|------|-----------------------|-------|----------|--------|
| `backend/app/services/cv_pipeline.py` | `brightness > 230` (in `build_specular_glare_mask`) | `230` | Brightness threshold separating specular highlights from diffuse skin pixels. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `color_diff < 25` (in `build_specular_glare_mask`) | `25` | Achromatic colour-difference threshold for glare pixel classification. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `min_channel > 190` (in `build_specular_glare_mask`) | `190` | All-channel brightness minimum ensuring the pixel is truly achromatic white, not a saturated colour. | **uncalibrated** |

---

## 4. Flake / Scaling Score Thresholds (`services/cv_pipeline.py`)

All cut-offs below are provisional heuristics. No clinical calibration has been performed.

### 4a. Top-Hat Morphology, Denoising, and Contour Filtering (OpenCV path)

| File | Constant / Expression | Value | Controls | Status |
|------|-----------------------|-------|----------|--------|
| `backend/app/services/cv_pipeline.py` | `cv2.GaussianBlur(img_bgr, (3, 3), sigmaX=1.2)` | `(3, 3), Ïƒ=1.2` | Pre-CLAHE smoothing to prevent sub-pixel Gaussian noise from being amplified into sharp intensity peaks. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `cv2.createCLAHE(clipLimit=1.5, ...)` | `1.5` | Contrast limit for adaptive histogram equalization, balanced to avoid over-amplifying flat skin texture. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `_build_hair_mask`: `gray < 80`, 11Ã—11 dilation | `80`, 11Ã—11 | Dark hair shaft threshold and dilation kernel to mask inter-hair skin gaps from false flake detection. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))` | `9 Ã— 9` | Top-Hat kernel size; larger than this represents skin texture rather than dandruff squames. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `cv2.threshold(tophat, 50, ...)` | `50` | Intensity threshold separating candidate flake pixels from background after denoising. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `area < 12 or area > 1500` | `12 â€“ 1500 pxÂ²` | Contour area filter; rejects single-pixel noise clusters (<12) and overly large structures (>1500). | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `8.0 <= edge_var <= 2500.0` | `8.0 â€“ 2500.0` | Laplacian variance window inside candidate contour; excludes smooth glare fringes (<8) and high-frequency noise spikes (>2500). | **uncalibrated** |

### 4b. Flake Coverage â†’ Grade â†’ Score Mapping (`_flake_ratio_to_score`)

| Grade | Coverage Cut-off | Score Range | Status |
|-------|-----------------|-------------|--------|
| Grade 0 (Absence) | < 0.1 % | 25.0 | **uncalibrated** |
| Grade 1 (Trace) | 0.1 % â€“ 0.5 % | 22.0 â€“ 24.9 | **uncalibrated** |
| Grade 2 (Mild) | 0.5 % â€“ 1.2 % | 18.0 â€“ 21.9 | **uncalibrated** |
| Grade 3 (Moderate) | 1.2 % â€“ 2.5 % | 13.0 â€“ 17.9 | **uncalibrated** |
| Grade 4 (Severe) | 2.5 % â€“ 4.0 % | 7.0 â€“ 12.9 | **uncalibrated** |
| Grade 5 (Very Severe) | > 4.0 % | 0.0 â€“ 6.9 | **uncalibrated** |

### 4c. NumPy Fallback Path (no OpenCV)

| File | Constant / Expression | Value | Controls | Status |
|------|-----------------------|-------|----------|--------|
| `backend/app/services/cv_pipeline.py` | `gray < 80` (hair mask) | `80` | Masks dark hair pixels in pure NumPy path. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `gray_glare_free > (local_mean + 55)` | `+55` | Brightness offset above local mean for classifying a pixel as a candidate flake. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `gray_glare_free > 185` | `185` | Absolute minimum brightness for a candidate flake pixel (ensures it is light-coloured). | **uncalibrated** |

---

## 5. Redness / Erythema Score Thresholds (`services/cv_pipeline.py`)

| File | Constant / Expression | Value | Controls | Status |
|------|-----------------------|-------|----------|--------|
| `backend/app/services/cv_pipeline.py` | `red_excess <= 34.0` | `34.0` | Red-excess (R âˆ’ mean(G,B)) below this is scored as optimal (25.0). Tolerates mild warm ambient lighting on neutral skin. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `score = 25.0 âˆ’ (excess_erythema Ã— 0.60)` | `0.60` | Linear penalty coefficient applied to red excess above 34.0 baseline. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `raw_red_excess > 50.0 and R/GB > 1.45` | `50.0, 1.45` | Global red cast threshold triggering low-confidence flag due to ambient lighting vs erythema ambiguity. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `_create_scalp_skin_mask`: `r > 60`, `g > 40` | `60`, `40` | Absolute channel thresholds defining the skin chromaticity region. Dark hair shafts are excluded. | **uncalibrated** |

---

## 6. Sebum / Lipid Barrier Score Thresholds (`services/cv_pipeline.py`)

| File | Constant / Expression | Value | Controls | Status |
|------|-----------------------|-------|----------|--------|
| `backend/app/services/cv_pipeline.py` | `brightness > 220` (shine mask) | `220` | Minimum brightness for a pixel to count as a shine/gloss candidate (proxy for luster). | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `color_diff < 30` (shine mask) | `30` | Achromatic colour-difference threshold for shine pixel identification (broader than specular mask). | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `shine_ratio <= 3.5` | `â‰¤ 3.5 %` | Optimal shine percentage range where full score (25.0) is awarded. Zero-shine matte scalps score 25.0 (no dryness penalty; dryness is measured in flake score). | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `excess_oil Ã— 2.5` penalty | `2.5` | Linear penalty coefficient for shine above 3.5 %. | **uncalibrated** |

---

## 7. Hair Coverage (indicative) Score Thresholds (`services/cv_pipeline.py`)

> API field name: `density_score`. This does NOT measure follicular density; it estimates visible dark-fiber area ratio on skin. Photo zoom, parting style, hair colour, and lighting all strongly affect the result.

| File | Constant / Expression | Value | Controls | Status |
|------|-----------------------|-------|----------|--------|
| `backend/app/services/cv_pipeline.py` | `L < 115` (LAB lightness) | `115` | Pixels darker than this in the LAB L-channel are classified as potential hair/dark-fiber pixels. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `p90 < 70` (dark skin detection) | `70` | 90th-percentile LAB L value; below this the image is too dark to separate hair from skin reliably (Fitzpatrick V-VI) -> returns neutral 12.5 + `density_confidence_low=True`. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `num_lines >= 20 and coverage_ratio < 0.005` (grey/blonde detection) | `20 lines, 0.5%` | HoughLinesP detects fiber-like structures; if many lines exist but dark coverage is very low, hair is light-coloured and unsegmentable -> returns neutral 12.5 + `density_confidence_low=True`. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `coverage_ratio < 0.005 and num_lines < 20` (strand-free) | `0.5%` | Below this combined threshold the image is classified as bare skin with no visible fibers -> score 0.0. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `50% fiber coverage -> score 25.0` (linear mapping) | `50%` | 50% of skin-region pixels classified as dark fibers maps linearly to full score 25.0. Coverage above 50% is capped at 25.0. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | HoughLinesP: `minLineLength=30, maxLineGap=10, threshold=40` | -- | Parameters controlling fiber-strand detection to distinguish structured hair from pixel noise. Noise produces ~5 detected lines; real grey hair produces ~200+. | **uncalibrated** |


---

## 8. EXIF Optical Quality Score Penalties (`services/quality_gate.py`)

| File | Condition | Penalty | Controls | Status |
|------|-----------|---------|----------|--------|
| `backend/app/services/quality_gate.py` | ISO > 1600 | âˆ’25 pts | Severe sensor chroma noise penalty. | **uncalibrated** |
| `backend/app/services/quality_gate.py` | ISO > 800 | âˆ’10 pts | Moderate grain penalty. | **uncalibrated** |
| `backend/app/services/quality_gate.py` | Exposure time > 0.04 s (> 1/25 s) | âˆ’15 pts | Elevated motion blur risk penalty. | **uncalibrated** |
| `backend/app/services/quality_gate.py` | Baseline optical score | `90.0` | Starting score when EXIF is present. | **uncalibrated** |
| `backend/app/services/quality_gate.py` | Default if no EXIF | `85.0` | Fallback when EXIF metadata is stripped or absent. | **uncalibrated** |

---

## 9. Gray-World White Balance Clamping (`services/cv_pipeline.py`)

| File | Constant / Expression | Value | Controls | Status |
|------|-----------------------|-------|----------|--------|
| `backend/app/services/cv_pipeline.py` | `np.clip(gray_target / mean_r, 0.85, 1.15)` | `0.85 â€“ 1.15` | Constrained channel scaling range; allows minor ambient color correction while preventing complete cancellation of true widespread vascular erythema. | **uncalibrated** |

---

## 10. Questionnaire Risk Indicators Count-Based Rules (`services/triage_engine.py`)

> Object: `risk_indicators` in `AnalysisResponse`. Transparent count-based evaluation of 11 self-reported lifestyle and symptom intake markers. Kept strictly separate from the image-derived `scalp_health_score` (self-reported, not clinically validated).

| File | Constant / Expression | Value | Controls | Status |
|------|-----------------------|-------|----------|--------|
| `backend/app/services/triage_engine.py` | `marker_count == 0` | `0` | No self-reported risk markers flagged -> overall level: `low`. | **uncalibrated** |
| `backend/app/services/triage_engine.py` | `1 <= marker_count <= 2 and not has_pain` | `1 – 2` | Mild self-reported risk markers -> overall level: `moderate`. | **uncalibrated** |
| `backend/app/services/triage_engine.py` | `marker_count >= 3 or has_pain` | `>= 3` / pain | Multiple risk markers or inflammatory scalp pain / trichodynia -> overall level: `elevated`. | **uncalibrated** |
| `backend/app/services/triage_engine.py` | `duration_months >= 3 or duration_weeks >= 12` | `3 mo / 12 wk` | Prolonged symptom timeline threshold identifying chronic vs acute involvement. | **uncalibrated** |

---

## 11. Heuristic Classification Fallback Rules (`services/classifier_service.py`)

> Evaluated when ONNX model weights are absent or in fallback evaluation mode. Note: This rule-based fallback is a stand-in until the CNN is retrained. The CNN class list (`CLASSES`, 5 classes) matches `training/train.py` and requires new labeled training data to natively output `CONDITION_ANDROGENETIC_ALOPECIA` or `CONDITION_SCALP_PSORIASIS`.

| File | Condition / Branch | Target Label | Confidence | Status |
|------|--------------------|--------------|------------|--------|
| `backend/app/services/classifier_service.py` | `flake_type == "plaque_crusty"` or `"plaque" in sensation` | `CONDITION_SCALP_PSORIASIS` | `0.82` | **uncalibrated** |
| `backend/app/services/classifier_service.py` | `primary_concern_zone == "patchy_spots"` | `CONDITION_ALOPECIA_AREATA` | `0.78` | **uncalibrated** |
| `backend/app/services/classifier_service.py` | `(family_history or gradual_chronic) and concern in ["crown_vertex", "hairline_temples", "central_parting", "diffuse_all_over"]` | `CONDITION_ANDROGENETIC_ALOPECIA` | `0.76` | **uncalibrated** |
| `backend/app/services/classifier_service.py` | `burning_or_pain and duration >= 1 mo` | `CONDITION_FOLLICULITIS` | `0.81` | **uncalibrated** |
| `backend/app/services/classifier_service.py` | `flake_type == "greasy_yellow_thick"` or `(itching and (hard_water or burning))` | `CONDITION_SEBORRHEIC_DERMATITIS` | `0.84` | **uncalibrated** |
| `backend/app/services/classifier_service.py` | `itching` or `flake_type == "dry_white_fine"` | `CONDITION_DRY_SCALP` | `0.77` | **uncalibrated** |

---

## Change Log

| Date | Change |
|------|--------|
| 2026-10-30 | Created. All thresholds collected from `cv_pipeline.py`, `quality_gate.py`, `config.py`. All marked **uncalibrated**. Removed all unsupported citations (Flament et al., PMC11975186, rÂ²=0.952, L'Oreal ScalpConsult Pro study, \$35B market). |
| 2026-10-30 (density fix) | Section 7 rewritten: `edge_energy >= 18.0` gradient threshold **removed**; replaced by hair-coverage segmentation thresholds (`coverage_ratio`, LAB L < 115, p90 < 70). See commit f2b128a. All new thresholds **uncalibrated**. |
| 2026-10-30 (risk indicators) | Section 10 added: questionnaire-derived risk indicators object added to `AnalysisResponse`, computed via transparent count rules (`low` / `moderate` / `elevated`) without blending into image-only `scalp_health_score`. All thresholds **uncalibrated**. |
| 2026-10-30 (questionnaire usage) | Removed unused `scan_angle`. Wired `biological_sex` + `age_group` into non-numeric correlation text notes (Norwood vs Ludwig, under-18 guardian advisory). Wired `washing_frequency`, `chemical_heat_styling`, `stress_level`, `sleep_quality` into tailored lifestyle text. Added frontend form selectors for `stress_level` and `sleep_quality`. All advisory rules are qualitative and **uncalibrated**. |
| 2026-10-30 (condition classification fixes) | Section 11 added: heuristic fallback rules updated. Family history + crown/vertex mapped to `CONDITION_ANDROGENETIC_ALOPECIA` (Level 2 triage). `patchy_spots` mapped to `CONDITION_ALOPECIA_AREATA` (Level 3). `plaque_crusty` mapped to `CONDITION_SCALP_PSORIASIS`. Added `plaque_crusty` option in frontend form. Documented that CNN `CLASSES` (5 classes) is constrained and requires model retraining. All fallback rules **uncalibrated**. |
