# Follica — Numeric Threshold Registry

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
| `backend/app/core/config.py` | `MAX_IMAGE_SIZE_BYTES` | `10 × 1024 × 1024` (10 MB) | Maximum allowed upload size. Administrative / infrastructure limit, not derived from image quality requirements. | engineering limit |

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
| `backend/app/services/cv_pipeline.py` | `cv2.GaussianBlur(img_bgr, (3, 3), sigmaX=1.2)` | `(3, 3), σ=1.2` | Pre-CLAHE smoothing to prevent sub-pixel Gaussian noise from being amplified into sharp intensity peaks. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `cv2.createCLAHE(clipLimit=1.5, ...)` | `1.5` | Contrast limit for adaptive histogram equalization, balanced to avoid over-amplifying flat skin texture. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `_build_hair_mask`: `gray < 80`, 11×11 dilation | `80`, 11×11 | Dark hair shaft threshold and dilation kernel to mask inter-hair skin gaps from false flake detection. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))` | `9 × 9` | Top-Hat kernel size; larger than this represents skin texture rather than dandruff squames. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `cv2.threshold(tophat, 50, ...)` | `50` | Intensity threshold separating candidate flake pixels from background after denoising. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `area < 12 or area > 1500` | `12 – 1500 px²` | Contour area filter; rejects single-pixel noise clusters (<12) and overly large structures (>1500). | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `8.0 <= edge_var <= 2500.0` | `8.0 – 2500.0` | Laplacian variance window inside candidate contour; excludes smooth glare fringes (<8) and high-frequency noise spikes (>2500). | **uncalibrated** |

### 4b. Flake Coverage → Grade → Score Mapping (`_flake_ratio_to_score`)

| Grade | Coverage Cut-off | Score Range | Status |
|-------|-----------------|-------------|--------|
| Grade 0 (Absence) | < 0.1 % | 25.0 | **uncalibrated** |
| Grade 1 (Trace) | 0.1 % – 0.5 % | 22.0 – 24.9 | **uncalibrated** |
| Grade 2 (Mild) | 0.5 % – 1.2 % | 18.0 – 21.9 | **uncalibrated** |
| Grade 3 (Moderate) | 1.2 % – 2.5 % | 13.0 – 17.9 | **uncalibrated** |
| Grade 4 (Severe) | 2.5 % – 4.0 % | 7.0 – 12.9 | **uncalibrated** |
| Grade 5 (Very Severe) | > 4.0 % | 0.0 – 6.9 | **uncalibrated** |

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
| `backend/app/services/cv_pipeline.py` | `red_excess <= 32.0` | `32.0` | Red-excess (R − mean(G,B)) below this is scored as optimal (25.0). Above this the score decays. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `score = 25.0 − (excess_erythema × 0.8)` | `0.8` | Linear penalty coefficient applied to red excess above the 32.0 baseline. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `_create_scalp_skin_mask`: `r > 60`, `g > 40` | `60`, `40` | Absolute channel thresholds defining the skin chromaticity region. Dark hair shafts are excluded. | **uncalibrated** |

---

## 6. Sebum / Lipid Barrier Score Thresholds (`services/cv_pipeline.py`)

| File | Constant / Expression | Value | Controls | Status |
|------|-----------------------|-------|----------|--------|
| `backend/app/services/cv_pipeline.py` | `brightness > 220` (shine mask) | `220` | Minimum brightness for a pixel to count as a shine/gloss candidate. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `color_diff < 30` (shine mask) | `30` | Achromatic colour-difference threshold for shine pixel identification (broader than specular mask to capture warm sebum gloss). | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `1.0 <= shine_ratio <= 3.5` | `1.0 % – 3.5 %` | Optimal sebum shine percentage range where full score (25.0) is awarded. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `excess_oil × 2.5` penalty | `2.5` | Linear penalty coefficient for shine above 3.5 %. | **uncalibrated** |
| `backend/app/services/cv_pipeline.py` | `dryness × 6.0` penalty | `6.0` | Linear penalty coefficient for shine below 1.0 % (dry scalp). | **uncalibrated** |

---

## 7. Hair Density / Follicle Coverage Score Thresholds (`services/cv_pipeline.py`)

| File | Constant / Expression | Value | Controls | Status |
|------|-----------------------|-------|----------|--------|
| `backend/app/services/cv_pipeline.py` | `edge_energy >= 18.0` | `18.0` | Mean absolute gradient energy threshold. Images above this score full density (25.0); below it the score scales linearly down to 0. | **uncalibrated** |

---

## 8. EXIF Optical Quality Score Penalties (`services/quality_gate.py`)

| File | Condition | Penalty | Controls | Status |
|------|-----------|---------|----------|--------|
| `backend/app/services/quality_gate.py` | ISO > 1600 | −25 pts | Severe sensor chroma noise penalty. | **uncalibrated** |
| `backend/app/services/quality_gate.py` | ISO > 800 | −10 pts | Moderate grain penalty. | **uncalibrated** |
| `backend/app/services/quality_gate.py` | Exposure time > 0.04 s (> 1/25 s) | −15 pts | Elevated motion blur risk penalty. | **uncalibrated** |
| `backend/app/services/quality_gate.py` | Baseline optical score | `90.0` | Starting score when EXIF is present. | **uncalibrated** |
| `backend/app/services/quality_gate.py` | Default if no EXIF | `85.0` | Fallback when EXIF metadata is stripped or absent. | **uncalibrated** |

---

## 9. Gray-World White Balance Clamping (`services/cv_pipeline.py`)

| File | Constant / Expression | Value | Controls | Status |
|------|-----------------------|-------|----------|--------|
| `backend/app/services/cv_pipeline.py` | `np.clip(gray_target / mean_r, 0.7, 1.4)` | `0.7 – 1.4` | Channel scaling range; prevents extreme colour correction that would produce unnatural results on very colour-cast images. | **uncalibrated** |

---

## Change Log

| Date | Change |
|------|--------|
| 2026-10-30 | Created. All thresholds collected from `cv_pipeline.py`, `quality_gate.py`, `config.py`. All marked **uncalibrated**. Removed all unsupported citations (Flament et al., PMC11975186, r²=0.952, L'Oreal ScalpConsult Pro study, \$35B market). |
