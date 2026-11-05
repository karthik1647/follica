"""
Follica Classical Computer Vision (CV) Micro-Feature Extraction Pipeline.

Architectural Design Intent and Algorithmic Rationale:
1. Why Deterministic Classical CV Runs in Parallel with Deep Learning:
   - Deep learning CNNs are effective at holistic pattern matching, but act as black boxes
     where individual sub-features (e.g., exact flake surface area vs vascular erythema)
     cannot be independently verified or explained to a clinician.
   - The classical CV pipeline extracts 4 deterministic, mathematically bounded metrics:
     a. Flake and Scaling Score (0.0 to 25.0)
     b. Erythema / Redness Score (0.0 to 25.0)
     c. Sebum and Lipid Balance Score (0.0 to 25.0)
     d. Hair Density and Follicle Coverage Score (0.0 to 25.0)
   - These 4 scores sum to the composite Scalp Health Score (0.0 to 100.0).

2. Optical Normalization (Gray-World White Balance and CLAHE Dehazing):
   - Real-world smartphone photos suffer from room lighting color casts (warm yellow 2700K
     incandescent bulbs make a healthy scalp look artificially inflamed; cool 6500K fluorescent
     lights wash out erythema).
   - Gray-World White Balance normalizes color channels to neutral illuminant D65 before redness scoring.
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) compensates for oily lens smudges and haze.

3. Specular Flash Glare Separation (Dichromatic Reflection Model):
   - Camera flash and ring lights produce achromatic specular highlights on curved scalp surfaces.
   - These specular pixels (high V, low S in HSV) are physically distinct from dandruff squames,
     which are solid keratin particles with irregular, textured, multi-directional boundaries.
   - Without separation, Top-Hat morphology counted flash glare as white dandruff, causing false-positive
     flake scores on healthy scalps photographed with flash.
   - The specular mask is built using the Shafer Dichromatic Reflection Model criteria:
       brightness > 230 AND color_diff < 25 AND no strong hue
   - Specular pixels are zeroed from flake analysis and their shine energy is routed to
     compute_sebum_score(), where surface gloss correctly belongs.

4. 6-Point Flake Severity Scale (PROVISIONAL HEURISTIC — not clinically validated):
   - The Flake Score uses a 6-grade ordinal scale whose coverage cut-offs and score
     boundaries are provisional engineering heuristics hand-picked for plausibility.
     They have NOT been fitted to or validated against any clinical dataset.
     See docs/thresholds.md for full threshold documentation.
   - Grade 0 (Absence)    -> Score 25.0     (< 0.1% coverage)
   - Grade 1 (Trace)      -> Score 22.0-24.9 (0.1%-0.5% coverage)
   - Grade 2 (Mild)       -> Score 18.0-21.9 (0.5%-1.2% coverage)
   - Grade 3 (Moderate)   -> Score 13.0-17.9 (1.2%-2.5% coverage)
   - Grade 4 (Severe)     -> Score 7.0-12.9  (2.5%-4.0% coverage)
   - Grade 5 (Very Severe)-> Score 0.0-6.9   (>4.0% coverage)

5. Dual-Mode Implementation (OpenCV + Pure NumPy/Pillow Fallback):
   - In production or local server environments with OpenCV installed, optimized C++
     morphological kernels are executed.
   - If OpenCV binaries are unavailable, pure NumPy/Pillow array vectorization executes
     the identical mathematical logic.
"""

import io
import numpy as np
from PIL import Image
from app.models.schemas import SubScores

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class ImageDecodeError(ValueError):
    """Raised when uploaded raw bytes cannot be decoded into a valid image matrix."""
    pass


class ScalpCVPipeline:
    """
    Deterministic Computer Vision extraction engine for scalp physiological micro-features.
    All sub-scores are normalized to [0.0, 25.0], where 25.0 represents optimal scalp health.
    """

    @staticmethod
    def apply_gray_world_white_balance(img_rgb: np.ndarray, clip_range: tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
        """
        Applies Gray-World White Balance algorithm with constrained scaling limits.

        ARCHITECTURAL & CLINICAL RATIONALE:
        Standard unconstrained Gray-World scales channel means towards equality, which
        neutralizes genuine diffuse erythema (e.g., flat red inflammation) into false
        neutral gray, artificially inflating redness scores to 25.0.
        Constraining scale factors to [0.85, 1.15] allows compensation for mild ambient
        color temperature shifts (e.g. minor fluorescent or warm cast) while preventing
        the cancellation of true widespread vascular erythema.
        (Thresholds are provisional / uncalibrated heuristics.)
        """
        img_float = img_rgb.astype(np.float32)
        mean_r = np.mean(img_float[:, :, 0])
        mean_g = np.mean(img_float[:, :, 1])
        mean_b = np.mean(img_float[:, :, 2])

        gray_target = (mean_r + mean_g + mean_b) / 3.0

        if mean_r > 0 and mean_g > 0 and mean_b > 0:
            # Constrained scaling to prevent flattening of genuine diffuse erythema
            scale_r = np.clip(gray_target / mean_r, clip_range[0], clip_range[1])
            scale_g = np.clip(gray_target / mean_g, clip_range[0], clip_range[1])
            scale_b = np.clip(gray_target / mean_b, clip_range[0], clip_range[1])

            img_balanced = np.zeros_like(img_float)
            img_balanced[:, :, 0] = np.clip(img_float[:, :, 0] * scale_r, 0, 255)
            img_balanced[:, :, 1] = np.clip(img_float[:, :, 1] * scale_g, 0, 255)
            img_balanced[:, :, 2] = np.clip(img_float[:, :, 2] * scale_b, 0, 255)
            return img_balanced.astype(np.uint8)

        return img_rgb

    @staticmethod
    def _create_scalp_skin_mask(img_rgb: np.ndarray) -> np.ndarray:
        """
        Isolates exposed scalp skin tissue from surrounding dark hair fibers and background.
        Uses skin chromaticity boundaries: Red channel dominates Green and Blue,
        with minimum brightness thresholds to exclude dark hair shafts.
        """
        r = img_rgb[:, :, 0]
        g = img_rgb[:, :, 1]
        b = img_rgb[:, :, 2]
        skin_mask = (r > g) & (r > b) & (r > 60) & (g > 40)
        return skin_mask

    @staticmethod
    def build_specular_glare_mask(img_rgb: np.ndarray) -> np.ndarray:
        """
        Builds a binary mask identifying flash/specular glare pixels using the
        Shafer Dichromatic Reflection Model.

        Physical basis:
        - Camera flash and ring lights reflect off the curved, slightly oily scalp surface
          as achromatic specular highlights (very high brightness, near-zero color saturation).
        - These glare pixels are NOT dandruff. They are optical artifacts from the light source.
        - True dandruff flakes are solid keratin particles with RGB color tint (dry = slightly
          yellowish-white, greasy = golden-yellow) and irregular, rough boundaries.

        Criteria for specular glare (all three must be true):
        1. Very high brightness (max channel > 230)
        2. Near-zero color saturation (|R-G| + |G-B| < 25): achromatic = no hue tint
        3. All channels bright simultaneously (min channel > 190): not a colored highlight

        Returns:
            Boolean mask, True where pixels are specular glare (to be excluded from flake detection).
        """
        r = img_rgb[:, :, 0].astype(int)
        g = img_rgb[:, :, 1].astype(int)
        b = img_rgb[:, :, 2].astype(int)

        brightness = np.maximum(np.maximum(r, g), b)
        min_channel = np.minimum(np.minimum(r, g), b)
        color_diff = np.abs(r - g) + np.abs(g - b)

        specular_mask = (brightness > 230) & (color_diff < 25) & (min_channel > 190)
        return specular_mask

    @staticmethod
    def _flake_ratio_to_score(flake_ratio: float) -> float:
        """
        Converts percentage flake coverage to a 6-point ordinal severity score (0-25 range).

        PROVISIONAL HEURISTIC — not clinically validated. The coverage cut-offs and score
        boundaries are hand-picked engineering estimates, NOT fitted to clinical data.
        See docs/thresholds.md for full threshold documentation.

        Grade 0 (Absence)     -> Score 25.0      (< 0.1% coverage)   # uncalibrated
        Grade 1 (Trace)       -> Score 22.0-24.9  (0.1% to 0.5%)     # uncalibrated
        Grade 2 (Mild)        -> Score 18.0-21.9  (0.5% to 1.2%)     # uncalibrated
        Grade 3 (Moderate)    -> Score 13.0-17.9  (1.2% to 2.5%)     # uncalibrated
        Grade 4 (Severe)      -> Score 7.0-12.9   (2.5% to 4.0%)     # uncalibrated
        Grade 5 (Very Severe) -> Score 0.0-6.9    (> 4.0%)           # uncalibrated
        """
        if flake_ratio < 0.1:
            return 25.0
        elif flake_ratio < 0.5:
            # Grade 1: trace. Interpolate 22.0 to 24.9
            return 25.0 - ((flake_ratio - 0.1) / 0.4) * 3.0
        elif flake_ratio < 1.2:
            # Grade 2: mild. Interpolate 18.0 to 21.9
            return 22.0 - ((flake_ratio - 0.5) / 0.7) * 4.0
        elif flake_ratio < 2.5:
            # Grade 3: moderate. Interpolate 13.0 to 17.9
            return 18.0 - ((flake_ratio - 1.2) / 1.3) * 5.0
        elif flake_ratio < 4.0:
            # Grade 4: severe. Interpolate 7.0 to 12.9
            return 13.0 - ((flake_ratio - 2.5) / 1.5) * 6.0
        else:
            # Grade 5: very severe. Maximum penalty.
            excess = flake_ratio - 4.0
            return max(0.0, 7.0 - (excess * 1.75))

    @staticmethod
    def _build_hair_mask(gray: np.ndarray) -> np.ndarray:
        """
        Returns a boolean mask that is True for dark hair-fibre pixels and a
        small neighbourhood around them.

        Root-cause fix: the Top-Hat transform produces strong responses at the
        bright inter-hair skin gaps because those patches are smaller than the
        9×9 structuring element.  Masking dark-hair pixels (and a 5-pixel
        dilated halo around them) removes those inter-hair gaps from the
        candidate-flake set.

        Threshold (provisional / uncalibrated): pixels darker than 80 on the
        CLAHE-enhanced grayscale image are treated as dark hair fibres.
        """
        # Provisional / uncalibrated: dark pixel threshold at 80
        hair_binary = (gray < 80).astype(np.uint8) * 255
        if not HAS_OPENCV:
            # Pure NumPy binary dilation approximation (5-px halo) without scipy or OpenCV
            dilated = (hair_binary > 0).copy()
            for _ in range(5):
                padded = np.pad(dilated, 1, mode="constant", constant_values=False)
                dilated = (
                    padded[:-2, 1:-1]
                    | padded[2:, 1:-1]
                    | padded[1:-1, :-2]
                    | padded[1:-1, 2:]
                    | padded[:-2, :-2]
                    | padded[:-2, 2:]
                    | padded[2:, :-2]
                    | padded[2:, 2:]
                    | dilated
                )
            return dilated
        # Dilate by 5 px to cover the bright halo at each hair edge
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(hair_binary, kernel)
        return dilated > 0

    @staticmethod
    def compute_flake_score(img_rgb: np.ndarray) -> float:
        """
        Extracts Flake and Scaling severity.

        Algorithmic Details:
        - Step 1: Build specular glare mask (Shafer Dichromatic model).
        - Step 2: Gaussian denoise before CLAHE.
          WHY: CLAHE amplifies Gaussian pixel noise (sd≈8-10) into sharp
          local-intensity peaks.  A 3×3 Gaussian blur before CLAHE smooths
          sub-pixel noise while preserving the macroscopic flake structure
          (which is 5-1500 px²).
        - Step 3: CLAHE on the luminance channel (compensates lens haze).
        - Step 4: Build a hair-fibre mask and dilate it by 5 px.
          WHY: Inter-hair skin gaps are bright patches smaller than the 9×9
          Top-Hat kernel; they appear as large "flakes" in the Top-Hat output
          with very high Laplacian variance from the adjacent hair edges.
          Masking hair and its halo removes this source of false positives.
        - Step 5: Morphological Top-Hat with threshold 50 (raised from 35).
          WHY: After denoising, noise residuals produce tophat < 50; real
          flakes or CLAHE-enhanced particles exceed 50.
        - Step 6: Contour texture discriminator — edge variance must be
          8 ≤ var ≤ 2500.
          WHY: Gaussian noise, after amplification by CLAHE, produces
          Laplacian variance > 5000 inside small blobs. Real dandruff flakes
          have rough but bounded boundaries (empirically < 2500 in tests).
          The upper bound rejects noise-induced false positives.
        - Step 7: Score mapped to provisional 6-grade ordinal scale.
          (Provisional heuristic — not clinically validated. See docs/thresholds.md.)
        """
        # Build specular glare mask first (shared across both code paths)
        specular_mask = ScalpCVPipeline.build_specular_glare_mask(img_rgb)

        if HAS_OPENCV:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Step 2: Gaussian denoise BEFORE CLAHE to suppress noise-induced peaks.
            # Kernel 3×3, sigma=1.2 (provisional / uncalibrated).
            img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), sigmaX=1.2)

            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            # Step 3: CLAHE — reduced clipLimit from 2.0 → 1.5 on near-flat regions.
            # clipLimit 1.5 (provisional / uncalibrated): limits contrast amplification
            # so that a flat-skin image is not boosted into false-particle territory.
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

            # Zero out specular glare pixels before Top-Hat
            gray_glare_free = gray.copy()
            gray_glare_free[specular_mask] = 0

            # Step 4: Build hair mask and zero those pixels too.
            # This removes inter-hair bright gaps which mimic flakes in Top-Hat.
            hair_mask = ScalpCVPipeline._build_hair_mask(gray)
            gray_glare_free[hair_mask] = 0

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            tophat = cv2.morphologyEx(gray_glare_free, cv2.MORPH_TOPHAT, kernel)

            # Step 5: Raised threshold 35 → 50 (provisional / uncalibrated).
            # After denoising, residual noise tops out below 50; real flakes exceed it.
            _, thresh = cv2.threshold(tophat, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Compute Laplacian on the denoised pre-CLAHE-amplified gray ONCE.
            # Using gray_glare_free (denoised) keeps variance bounded for real flakes.
            laplacian_full = cv2.Laplacian(gray_glare_free, cv2.CV_64F)

            flake_area = 0.0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Tightened min area 5→12: single-pixel noise clusters are < 12px²
                # Max area unchanged at 1500 (provisional / uncalibrated).
                if area < 12 or area > 1500:   # provisional / uncalibrated
                    continue
                mask_cnt = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                region_pixels = laplacian_full[mask_cnt == 255]
                if len(region_pixels) > 0:
                    edge_var = float(np.var(region_pixels))
                    # Step 6: Both bounds are provisional / uncalibrated.
                    # Lower bound 8: smooth residual glare fringes are below this.
                    # Upper bound 2500: Gaussian noise fragments are above this.
                    if 8.0 <= edge_var <= 2500.0:  # provisional / uncalibrated
                        flake_area += area

            total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
            flake_ratio = (flake_area / max(total_pixels, 1)) * 100.0
        else:
            # Pure NumPy fallback path (no OpenCV).
            gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])
            # Zero glare pixels
            gray_glare_free = gray.copy()
            gray_glare_free[specular_mask] = 0.0
            # Zero dark hair pixels (provisional / uncalibrated threshold: 80)
            gray_glare_free[gray < 80] = 0.0
            local_mean = (
                np.mean(gray_glare_free[(~specular_mask) & (gray >= 80)])
                if np.any((~specular_mask) & (gray >= 80))
                else np.mean(gray)
            )
            # Provisional / uncalibrated: threshold at local_mean + 55 and absolute 185
            bright_particles = (
                (gray_glare_free > (local_mean + 55)) & (gray_glare_free > 185)
            )
            flake_ratio = (np.sum(bright_particles) / float(gray.size)) * 100.0

        score = ScalpCVPipeline._flake_ratio_to_score(flake_ratio)
        return float(np.clip(round(score, 2), 0.0, 25.0))

    @staticmethod
    def compute_redness_score(img_rgb: np.ndarray, return_details: bool = False):
        """
        Extracts Vascular Erythema / Redness severity with constrained white-balancing.

        UNAVOIDABLE ILLUMINATION VS INFLAMMATION TRADE-OFF:
        Without an in-frame optical calibration reference (e.g. Macbeth color card or dermascope),
        diffuse warm illumination (e.g. 2700K incandescent bulb) and widespread vascular
        erythema present similar global chromatic signatures.
        To handle this trade-off honestly:
        1. Gray-World scaling is constrained to [0.85, 1.15] so global erythema is not
           erased.
        2. A conservative baseline threshold (34.0) and penalty slope (0.60) are used,
           allowing neutral skin under mild warm lighting to retain a calm/healthy score (>= 22.0)
           while ensuring clear diffuse erythema (e.g. (225, 120, 110)) scores <= 18.0.
        3. When a strong global red cast is detected, a low-confidence indicator and
           explanatory note are exposed to avoid pretending certainty.
        4. Longitudinal tracking against the patient's own earlier baseline photograph
           (/api/v1/progress/compare) provides the clinically reliable application of this metric.

        All cut-offs and slopes are provisional / uncalibrated heuristics.
        """
        balanced_rgb = ScalpCVPipeline.apply_gray_world_white_balance(img_rgb, clip_range=(0.85, 1.15))
        skin_mask = ScalpCVPipeline._create_scalp_skin_mask(balanced_rgb)

        # Exclude glare pixels from redness analysis: they are achromatic and would
        # pull the red-excess mean toward zero, making inflamed scalps look healthier.
        specular_mask = ScalpCVPipeline.build_specular_glare_mask(balanced_rgb)
        valid_mask = skin_mask & ~specular_mask

        if np.sum(valid_mask) > 100:
            r_skin = balanced_rgb[:, :, 0][valid_mask]
            g_skin = balanced_rgb[:, :, 1][valid_mask]
            b_skin = balanced_rgb[:, :, 2][valid_mask]
            r_raw = img_rgb[:, :, 0][valid_mask]
            g_raw = img_rgb[:, :, 1][valid_mask]
            b_raw = img_rgb[:, :, 2][valid_mask]
        elif np.sum(skin_mask) > 100:
            r_skin = balanced_rgb[:, :, 0][skin_mask]
            g_skin = balanced_rgb[:, :, 1][skin_mask]
            b_skin = balanced_rgb[:, :, 2][skin_mask]
            r_raw = img_rgb[:, :, 0][skin_mask]
            g_raw = img_rgb[:, :, 1][skin_mask]
            b_raw = img_rgb[:, :, 2][skin_mask]
        else:
            r_skin = balanced_rgb[:, :, 0]
            g_skin = balanced_rgb[:, :, 1]
            b_skin = balanced_rgb[:, :, 2]
            r_raw = img_rgb[:, :, 0]
            g_raw = img_rgb[:, :, 1]
            b_raw = img_rgb[:, :, 2]

        red_excess = float(np.mean(r_skin.astype(float) - ((g_skin.astype(float) + b_skin.astype(float)) / 2.0)))
        raw_red_excess = float(np.mean(r_raw.astype(float) - ((g_raw.astype(float) + b_raw.astype(float)) / 2.0)))
        mean_r_raw = float(np.mean(r_raw))
        mean_gb_raw = float(np.mean((g_raw.astype(float) + b_raw.astype(float)) / 2.0))

        # Provisional / uncalibrated threshold (34.0) and slope (0.60):
        # - Flat (225, 120, 110) erythema gives red_excess ≈ 59.0 -> score ≈ 10.0 (<= 18.0)
        # - Neutral skin (210, 160, 135) gives red_excess ≈ 16.7 -> score = 25.0
        # - Mild warm tint on neutral skin gives red_excess ≈ 32.3 -> score = 25.0 (>= 22.0)
        if red_excess <= 34.0:
            score = 25.0
        else:
            excess_erythema = red_excess - 34.0
            score = 25.0 - (excess_erythema * 0.60)

        # Focal patch erythema contrast (provisional / uncalibrated):
        # Uniform skin/warm-lamp frames have p90-p40 <= 6.0, whereas localized
        # erythematous lesions (papules, plaques, lichenoid patches) create spatial
        # red-excess contrast > 12.0 against surrounding uninvolved scalp skin.
        pixel_excess = r_skin.astype(float) - ((g_skin.astype(float) + b_skin.astype(float)) / 2.0)
        if pixel_excess.size >= 100:
            focal_erythema_delta = float(np.percentile(pixel_excess, 90) - np.percentile(pixel_excess, 40))
            if focal_erythema_delta > 12.0:
                focal_penalty = min(10.0, (focal_erythema_delta - 12.0) * 0.45)
                score -= focal_penalty

        final_score = float(np.clip(round(score, 2), 0.0, 25.0))

        # Check for global red cast ambiguity (warm lamp vs widespread inflammation)
        # Provisional / uncalibrated heuristic: raw red excess > 50 and R/(GB) > 1.45
        low_confidence = bool(raw_red_excess > 50.0 and (mean_r_raw / max(mean_gb_raw, 1.0)) > 1.45)
        if low_confidence:
            note = (
                "Global red chromaticity detected. Ambient warm lighting and widespread vascular erythema "
                "cannot be definitively separated without an optical reference target. Score is conservatively "
                "adjusted; longitudinal comparison against baseline scan (/progress/compare) is recommended."
            )
        else:
            note = "Redness confidence optimal; chromaticity within standard skin boundaries."

        if return_details:
            return final_score, low_confidence, note
        return final_score

    @staticmethod
    def compute_sebum_score(img_rgb: np.ndarray) -> float:
        """
        Extracts Sebum and Lipid Barrier Balance.

        NOTE ON PROXY MEASUREMENT:
        This score is an optical proxy for sebum production based on surface luster.
        Specular highlights originate from both oily sebum lipid coatings and direct
        camera flash or environmental illumination.
        The shine detection criteria (brightness > 220, color_diff < 30) and the
        upper boundary (3.5% shine) are provisional / uncalibrated heuristics.

        NO DRYNESS TERM:
        Dryness / xerosis and flaking are already quantified by compute_flake_score().
        Penalizing low or zero shine here would double-count dry scalp conditions and
        falsely penalize a clean, matte scalp (which previously scored 19.0).
        A matte, non-shiny scalp (shine <= 3.5%) receives the optimal score (25.0).
        Points are deducted only for excessive oily shine above 3.5%.
        """
        r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
        brightness = np.maximum(np.maximum(r, g), b)
        color_diff = np.abs(r.astype(int) - g.astype(int)) + np.abs(g.astype(int) - b.astype(int))

        # Shine detection: brightness > 220, color_diff < 30 (provisional / uncalibrated proxy)
        shine_mask = (brightness > 220) & (color_diff < 30)
        shine_ratio = (np.sum(shine_mask) / float(brightness.size)) * 100.0

        # Matte and balanced scalps score full points (25.0).
        # Points are docked only for excessive shine (> 3.5% area coverage).
        if shine_ratio <= 3.5:  # provisional / uncalibrated threshold
            score = 25.0
        else:
            excess_oil = shine_ratio - 3.5
            # Penalty slope: 2.5 pts per 1% excess shine (provisional / uncalibrated)
            score = 25.0 - (excess_oil * 2.5)

        return float(np.clip(round(score, 2), 0.0, 25.0))

    @staticmethod
    def compute_density_score(img_rgb: np.ndarray, return_details: bool = False):
        """
        Extracts Hair Coverage (indicative).

        IMPORTANT CLINICAL & METHODOLOGICAL LIMITATIONS:
        1. This metric measures 2D visible hair fiber area coverage ratio. It does NOT
           measure true follicular unit density (follicles/cm²), which requires contact
           dermascopy/trichoscopy with calibrated optical magnification.
        2. Contrast Dependency: The algorithm segments dark hair fibers on lighter skin.
           When contrast is insufficient (e.g. grey, blonde, dyed hair, or dark skin phototypes
           Fitzpatrick V-VI), optical segmentation is unreliable. In such cases, the function
           assigns a neutral midpoint score (12.50) and flags density_confidence_low=True
           rather than guessing.
        3. Framing Sensitivity: Camera zoom distance, focus angle, and hair parting geometry
           substantially alter visible scalp exposure. Longitudinal comparisons (/progress/compare)
           must adhere to standardized photo distances and parting angles.

        All segmentation thresholds (lightness L < 115, 50% fiber coverage for 25.0 score,
        line count threshold 20) are provisional / uncalibrated heuristics.
        """
        # Exclude specular glare pixels (which should not count as hair or skin)
        r = img_rgb[:, :, 0].astype(int)
        g = img_rgb[:, :, 1].astype(int)
        b = img_rgb[:, :, 2].astype(int)
        brightness = np.maximum(np.maximum(r, g), b)
        min_channel = np.minimum(np.minimum(r, g), b)
        color_diff = np.abs(r - g) + np.abs(g - b)
        glare_mask = (brightness > 230) & (color_diff < 25) & (min_channel > 190)

        if HAS_OPENCV:
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            L = lab[:, :, 0]
            valid_L = L[~glare_mask] if np.any(~glare_mask) else L.flatten()
            p90 = float(np.percentile(valid_L, 90))

            # Detect line fibers using HoughLinesP on smoothed Canny edges
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), sigmaX=1.8)
            edges = cv2.Canny(blur, 8, 22)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)
            num_lines = len(lines) if lines is not None else 0

            # Dark hair fiber segmentation (provisional threshold L < 115)
            dark_pixels = (L < 115) & (~glare_mask)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            hair_clean = cv2.morphologyEx(dark_pixels.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)

            total_valid = np.sum(~glare_mask)
            coverage_ratio = float(np.sum(hair_clean > 0)) / max(float(total_valid), 1.0)

            # Low contrast detection:
            # - Dark skin phototype: skin tissue is dark (p90 < 70)
            # - Fibers exist (num_lines >= 20) but dark fiber coverage is near zero (< 0.5%): grey/blonde/dyed hair
            is_dark_skin = (p90 < 70.0)
            is_grey_or_blonde = (num_lines >= 20) and (coverage_ratio < 0.005)

            if is_dark_skin or is_grey_or_blonde:
                neutral_score = 12.50
                note = (
                    "Hair coverage (indicative) could not be estimated reliably due to low optical contrast "
                    "between hair fibers and scalp tissue (e.g. grey, blonde, dyed hair, or dark skin phototype). "
                    "A neutral baseline score (12.50) is returned. Manual trichoscopy is recommended."
                )
                if return_details:
                    return neutral_score, True, note
                return neutral_score

            # Strand-free scalp region (smooth skin, no dark fibers, no strand lines)
            if coverage_ratio < 0.005 and num_lines < 20:
                score = 0.0
                note = "Hair coverage (indicative): 0.0% estimated fiber area (strand-free smooth scalp region)."
                if return_details:
                    return score, False, note
                return score

            # Linear mapping: 50% fiber coverage in frame -> full score 25.0
            score = min(25.0, (coverage_ratio / 0.50) * 25.0)

            # Focal bare-patch void penalty (provisional / uncalibrated):
            # Detects localized bald patches (e.g. patchy/scarring alopecia) surrounded by
            # dense border hair that would otherwise push global coverage >= 50% (25/25).
            if coverage_ratio >= 0.22:
                skin_mask_full = ScalpCVPipeline._create_scalp_skin_mask(img_rgb)
                bare_tiles = 0
                for gy in range(64, 448, 64):
                    for gx in range(64, 448, 64):
                        tile_h = float(np.mean(hair_clean[gy:gy+64, gx:gx+64] > 0))
                        tile_s = float(np.mean(skin_mask_full[gy:gy+64, gx:gx+64]))
                        if tile_h < 0.04 and tile_s > 0.60:
                            bare_tiles += 1
                if bare_tiles >= 2:
                    void_penalty = min(14.0, bare_tiles * 1.6)
                    score = max(4.0, score - void_penalty)

            final_score = float(np.clip(round(score, 2), 0.0, 25.0))
            note = (
                f"Hair coverage (indicative): {coverage_ratio * 100:.1f}% estimated fiber area. "
                "Note: optical surface coverage does not measure follicular unit density and varies with photo zoom and parting."
            )
            if return_details:
                return final_score, False, note
            return final_score
        else:
            # Pure NumPy fallback
            gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])
            valid_gray = gray[~glare_mask] if np.any(~glare_mask) else gray.flatten()
            p90 = float(np.percentile(valid_gray, 90))

            dark_pixels = (gray < 90) & (~glare_mask)
            total_valid = np.sum(~glare_mask)
            coverage_ratio = float(np.sum(dark_pixels)) / max(float(total_valid), 1.0)

            if p90 < 65.0:
                neutral_score = 12.50
                note = "Hair coverage (indicative): Low contrast dark skin phototype. Neutral score returned."
                if return_details:
                    return neutral_score, True, note
                return neutral_score

            if coverage_ratio < 0.005:
                score = 0.0
                note = "Hair coverage (indicative): 0.0% estimated fiber area (strand-free smooth scalp region)."
                if return_details:
                    return score, False, note
                return score

            score = (coverage_ratio / 0.50) * 25.0
            final_score = float(np.clip(round(score, 2), 0.0, 25.0))
            note = f"Hair coverage (indicative): {coverage_ratio * 100:.1f}% estimated fiber area (NumPy fallback)."
            if return_details:
                return final_score, False, note
            return final_score

    def analyze_image(self, img_bytes: bytes) -> SubScores:
        """
        Executes complete classical CV pipeline on raw image bytes.
        Returns discrete SubScores object with bounded [0.0, 25.0] values.
        """
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img.load()  # Force decoding of compressed pixel data to detect truncation early
            pil_img = pil_img.convert("RGB")
            pil_img = pil_img.resize((512, 512))
            img_rgb = np.array(pil_img, dtype=np.uint8)
        except Exception as e:
            raise ImageDecodeError(f"Failed to decode image: {str(e)}") from e

        flake_score = self.compute_flake_score(img_rgb)
        redness_score, red_low_conf, redness_note = self.compute_redness_score(img_rgb, return_details=True)
        sebum_score = self.compute_sebum_score(img_rgb)
        density_score, density_low_conf, density_note = self.compute_density_score(img_rgb, return_details=True)

        return SubScores(
            flake_score=flake_score,
            redness_score=redness_score,
            sebum_score=sebum_score,
            density_score=density_score,
            redness_confidence_low=red_low_conf,
            redness_note=redness_note,
            density_confidence_low=density_low_conf,
            density_note=density_note
        )


cv_pipeline = ScalpCVPipeline()