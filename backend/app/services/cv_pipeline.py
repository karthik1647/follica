"""
Follica Classical Computer Vision (CV) Micro-Feature Extraction Pipeline.

Extracts four discrete clinical scalp sub-scores in range [0.0, 25.0]:
1. Flake & Scaling Score (0=severe flaking, 25=completely clear)
2. Erythema / Redness Score (0=severe vascular erythema, 25=calm/non-inflamed)
3. Sebum & Lipid Balance Score (0=extreme oiliness/severe dehydration, 25=optimal)
4. Hair Density & Follicle Visibility (0=significant thinning, 25=dense coverage)

Supports both OpenCV (accelerated) and Pure NumPy/Pillow fallback for zero-dependency environments.
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


class ScalpCVPipeline:
    """Deterministic Computer Vision analysis for scalp micro-features."""

    @staticmethod
    def _create_scalp_skin_mask(img_rgb: np.ndarray) -> np.ndarray:
        """Creates a binary mask isolating exposed scalp skin."""
        # RGB skin thresholding: R > G, R > B, R > 60
        r = img_rgb[:, :, 0]
        g = img_rgb[:, :, 1]
        b = img_rgb[:, :, 2]
        skin_mask = (r > g) & (r > b) & (r > 60) & (g > 40)
        return skin_mask

    @staticmethod
    def compute_flake_score(img_rgb: np.ndarray) -> float:
        """
        Extracts flake & scaling severity.
        Detects bright, high-contrast white/yellowish epidermal particles.
        """
        if HAS_OPENCV:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            _, thresh = cv2.threshold(tophat, 35, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            flake_area = sum(cv2.contourArea(cnt) for cnt in contours if 5 < cv2.contourArea(cnt) < 1500)
            total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
            flake_ratio = (flake_area / max(total_pixels, 1)) * 100.0
        else:
            # Pure NumPy fallback
            gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])
            # High-pass filter via simple local contrast
            local_mean = np.mean(gray)
            bright_particles = (gray > (local_mean + 45)) & (gray > 180)
            flake_ratio = (np.sum(bright_particles) / float(gray.size)) * 100.0

        score = 25.0 - (flake_ratio * 5.0)
        return float(np.clip(round(score, 2), 0.0, 25.0))

    @staticmethod
    def compute_redness_score(img_rgb: np.ndarray) -> float:
        """
        Extracts vascular erythema / inflammation severity.
        Measures red-channel chrominance excess over green/blue skin baseline.
        """
        skin_mask = ScalpCVPipeline._create_scalp_skin_mask(img_rgb)
        
        if np.sum(skin_mask) > 100:
            r_skin = img_rgb[:, :, 0][skin_mask]
            g_skin = img_rgb[:, :, 1][skin_mask]
            b_skin = img_rgb[:, :, 2][skin_mask]
            # Red chrominance excess delta
            red_excess = np.mean(r_skin.astype(float) - ((g_skin.astype(float) + b_skin.astype(float)) / 2.0))
        else:
            red_excess = np.mean(img_rgb[:, :, 0].astype(float) - ((img_rgb[:, :, 1].astype(float) + img_rgb[:, :, 2].astype(float)) / 2.0))

        # Baseline uninflamed delta is typically ~20-35. Inflamed/erythematous reaches >55.
        if red_excess <= 32.0:
            score = 25.0
        else:
            excess_erythema = red_excess - 32.0
            score = 25.0 - (excess_erythema * 0.8)

        return float(np.clip(round(score, 2), 0.0, 25.0))

    @staticmethod
    def compute_sebum_score(img_rgb: np.ndarray) -> float:
        """
        Extracts sebum / oiliness balance.
        Analyzes specular reflection highlights.
        """
        r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
        # Specular shine: very high brightness across all channels with low color saturation
        brightness = np.maximum(np.maximum(r, g), b)
        color_diff = np.abs(r.astype(int) - g.astype(int)) + np.abs(g.astype(int) - b.astype(int))
        shine_mask = (brightness > 230) & (color_diff < 30)
        shine_ratio = (np.sum(shine_mask) / float(brightness.size)) * 100.0

        if 1.0 <= shine_ratio <= 3.5:
            score = 25.0
        elif shine_ratio > 3.5:
            excess_oil = shine_ratio - 3.5
            score = 25.0 - (excess_oil * 2.5)
        else:
            dryness = 1.0 - shine_ratio
            score = 25.0 - (dryness * 6.0)

        return float(np.clip(round(score, 2), 0.0, 25.0))

    @staticmethod
    def compute_density_score(img_rgb: np.ndarray) -> float:
        """
        Extracts hair density and coverage texture.
        Uses gradient magnitude / edge energy.
        """
        gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])
        # Discrete gradient along horizontal and vertical axes
        diff_y = np.abs(np.diff(gray, axis=0))
        diff_x = np.abs(np.diff(gray, axis=1))
        edge_energy = float(np.mean(diff_y) + np.mean(diff_x))

        if edge_energy >= 18.0:
            score = 25.0
        else:
            score = (edge_energy / 18.0) * 25.0

        return float(np.clip(round(score, 2), 0.0, 25.0))

    def analyze_image(self, img_bytes: bytes) -> SubScores:
        """Executes classical CV feature extraction on raw image bytes."""
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_img = pil_img.resize((512, 512))
            img_rgb = np.array(pil_img, dtype=np.uint8)
        except Exception:
            return SubScores(
                flake_score=15.0,
                redness_score=15.0,
                sebum_score=15.0,
                density_score=15.0
            )

        flake_score = self.compute_flake_score(img_rgb)
        redness_score = self.compute_redness_score(img_rgb)
        sebum_score = self.compute_sebum_score(img_rgb)
        density_score = self.compute_density_score(img_rgb)

        return SubScores(
            flake_score=flake_score,
            redness_score=redness_score,
            sebum_score=sebum_score,
            density_score=density_score
        )


cv_pipeline = ScalpCVPipeline()
