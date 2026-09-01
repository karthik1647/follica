"""
Follica Image Quality Pre-Inference Gate Service.

Architectural Design Intent & Algorithmic Rationale:
1. Why Pre-Inference Quality Validation is Essential:
   - Deep learning classification networks and classical CV micro-feature extractors
     (flake size detection, erythema redness thresholding, follicle counting) fail silently
     or produce erratic results when input photos suffer from severe optical blur or extreme
     exposure anomalies.
   - Intercepting poor-quality captures before execution saves cloud inference resources and
     prevents misleading or alarming health score outputs to the user.

2. Algorithmic Checks:
   a. Focus / Sharpness via Laplacian Variance:
      - The discrete 2D Laplacian operator computes the 2nd spatial derivative of pixel intensity.
      - Sharp, in-focus scalp images possess crisp follicular edges and flake boundaries,
        yielding high variance in Laplacian response.
      - Blurry images suffer from edge attenuation, causing the variance to drop below 60.0.
   b. Illumination & Exposure Distribution:
      - Trichoscopy features cannot be extracted in severely shadowed or washed-out images.
      - Mean luminance below 35.0 (out of 255) indicates underexposure where scalp skin
        chrominance cannot be differentiated from dark hair.
      - Mean luminance above 225.0 indicates severe overexposure or flashlight glare washing
        out epidermal texture.
   c. Dimensional Integrity:
      - Scalp photographs smaller than 150x150 pixels lack sufficient spatial resolution for
        morphological filtering and are rejected immediately.
   d. EXIF Metadata & Optical Exposure Evaluator:
      - Extracts camera hardware settings (ISO, Exposure Time, Flash, Aperture) if embedded.
      - High ISO (>800) indicates high sensor grain/chroma noise that can distort micro-features.
      - Slow shutter speed (<1/25s) flags motion blur risk.
      - Camera flash presence is tracked (flash provides optimal color rendering on hair partings).

3. Dual-Mode Implementation (OpenCV + Pure NumPy Fallback):
   - When OpenCV is available, hardware-accelerated cv2.Laplacian and cv2.cvtColor are used.
   - When running in minimal environments without C++ OpenCV binaries, pure NumPy discrete
     difference approximations and luminance matrix calculations provide equivalent gating.
"""

import io
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ExifTags

from app.core.config import settings

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class ImageQualityGate:
    """
    Validates scalp image quality and extracts optical EXIF metadata prior to running
    deep learning classification or classical computer vision micro-feature extraction.
    """

    def __init__(
        self,
        min_blur_score: float = settings.QUALITY_MIN_BLUR_SCORE,
        min_brightness: float = settings.QUALITY_MIN_BRIGHTNESS,
        max_brightness: float = settings.QUALITY_MAX_BRIGHTNESS,
        min_dimension: int = settings.QUALITY_MIN_DIMENSION,
    ):
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_dimension = min_dimension

    def _decode_image(self, image_bytes: bytes) -> Tuple[np.ndarray, Optional[Image.Image]]:
        """
        Decodes raw byte stream into an RGB NumPy array and preserves PIL Image for EXIF extraction.
        Handles RGBA, Grayscale, and standard RGB formats cleanly.
        """
        pil_img = Image.open(io.BytesIO(image_bytes))
        raw_pil = pil_img.copy()
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return np.array(pil_img), raw_pil

    def extract_exif_metadata(self, pil_img: Optional[Image.Image]) -> Dict[str, Any]:
        """
        Extracts photographic EXIF metadata (ISO, shutter speed, flash status, camera model).
        Returns a structured dictionary with optical quality metrics.
        """
        exif_data = {
            "has_exif": False,
            "iso": None,
            "exposure_time_sec": None,
            "flash_fired": None,
            "f_number": None,
            "camera_make": None,
            "camera_model": None,
            "optical_quality_score": 85.0  # Default baseline score if EXIF is stripped
        }

        if pil_img is None:
            return exif_data

        try:
            raw_exif = pil_img._getexif()
            if not raw_exif:
                return exif_data

            exif_data["has_exif"] = True
            tag_map = {ExifTags.TAGS[k]: v for k, v in raw_exif.items() if k in ExifTags.TAGS}

            # 1. ISO Speed Rating (Sensor Noise Indicator)
            iso = tag_map.get("ISOSpeedRatings") or tag_map.get("PhotographicSensitivity")
            if iso:
                exif_data["iso"] = int(iso) if isinstance(iso, (int, float)) else str(iso)

            # 2. Exposure Time (Motion Blur Indicator)
            exposure = tag_map.get("ExposureTime")
            if exposure:
                exif_data["exposure_time_sec"] = float(exposure)

            # 3. Flash Status
            flash = tag_map.get("Flash")
            if flash is not None:
                exif_data["flash_fired"] = bool(flash & 1) if isinstance(flash, int) else None

            # 4. Aperture & Camera Hardware
            f_num = tag_map.get("FNumber")
            if f_num:
                exif_data["f_number"] = float(f_num)

            exif_data["camera_make"] = tag_map.get("Make")
            exif_data["camera_model"] = tag_map.get("Model")

            # Calculate Optical Quality Score (0 to 100)
            score = 90.0
            if isinstance(exif_data["iso"], int):
                if exif_data["iso"] > 1600:
                    score -= 25.0  # Severe sensor chroma noise
                elif exif_data["iso"] > 800:
                    score -= 10.0  # Moderate grain

            if exif_data["exposure_time_sec"] and exif_data["exposure_time_sec"] > 0.04:  # > 1/25s
                score -= 15.0  # Elevated motion blur risk

            if exif_data["flash_fired"] is True:
                score = min(100.0, score + 5.0)  # Flash provides direct scalp illumination

            exif_data["optical_quality_score"] = float(round(np.clip(score, 20.0, 100.0), 1))

        except Exception:
            pass

        return exif_data

    def calculate_laplacian_variance(self, img_rgb: np.ndarray) -> float:
        """
        Calculates the focus sharpness score using the variance of the Laplacian.
        Higher variance indicates sharper structural edges.
        """
        if HAS_OPENCV:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            score = float(laplacian.var())
        else:
            # Pure NumPy discrete Laplacian approximation
            gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])
            padded = np.pad(gray, pad_width=1, mode="edge")
            laplacian = (
                padded[2:, 1:-1]
                + padded[:-2, 1:-1]
                + padded[1:-1, 2:]
                + padded[1:-1, :-2]
                - 4.0 * padded[1:-1, 1:-1]
            )
            score = float(np.var(laplacian))

        return round(score, 2)

    def calculate_brightness(self, img_rgb: np.ndarray) -> float:
        """
        Calculates the mean perceptual luminance across the entire photograph.
        Uses standard Rec. 601 coefficients: Y = 0.299*R + 0.587*G + 0.114*B.
        """
        gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])
        return float(round(np.mean(gray), 2))

    def evaluate(self, image_bytes: bytes) -> Tuple[bool, Dict[str, Any]]:
        """
        Performs full quality evaluation on raw image bytes.

        Returns:
            Tuple[bool, Dict[str, Any]]:
                - is_valid (bool): True if image satisfies all quality criteria.
                - report (dict): Metrics, pass/fail status, EXIF metadata, and actionable user guidance.
        """
        try:
            img_rgb, raw_pil = self._decode_image(image_bytes)
        except Exception as e:
            return False, {
                "is_valid": False,
                "error_code": "INVALID_IMAGE_DATA",
                "rejection_reason": "Failed to decode image file. File may be corrupted or in an unsupported format.",
                "user_guidance": "Please upload a valid JPEG, PNG, or WebP photograph of your scalp.",
                "metrics": {}
            }

        height, width = img_rgb.shape[:2]

        # 1. Dimension Check: Scalp trichoscopy requires minimum pixel resolution
        if width < self.min_dimension or height < self.min_dimension:
            return False, {
                "is_valid": False,
                "error_code": "IMAGE_RESOLUTION_TOO_LOW",
                "rejection_reason": f"Image dimensions ({width}x{height}px) are below the minimum required resolution ({self.min_dimension}x{self.min_dimension}px).",
                "user_guidance": "Please take a closer, higher-resolution photo of the affected scalp area.",
                "metrics": {
                    "width": width,
                    "height": height,
                    "min_dimension_required": self.min_dimension
                }
            }

        brightness_score = self.calculate_brightness(img_rgb)
        blur_score = self.calculate_laplacian_variance(img_rgb)
        exif_info = self.extract_exif_metadata(raw_pil)

        brightness_passed = self.min_brightness <= brightness_score <= self.max_brightness
        blur_passed = blur_score >= self.min_blur_score

        metrics = {
            "width": width,
            "height": height,
            "brightness_score": brightness_score,
            "min_brightness_threshold": self.min_brightness,
            "max_brightness_threshold": self.max_brightness,
            "brightness_passed": brightness_passed,
            "blur_score": blur_score,
            "min_blur_threshold": self.min_blur_score,
            "blur_passed": blur_passed,
            "exif": exif_info
        }

        # 2. Illumination / Exposure Check:
        if brightness_score < self.min_brightness:
            return False, {
                "is_valid": False,
                "error_code": "IMAGE_UNDEREXPOSED",
                "rejection_reason": f"Image brightness ({brightness_score}) indicates severe underexposure.",
                "user_guidance": "The photo is too dark. Please take the photo in a well-lit room or enable the camera flash.",
                "metrics": metrics
            }

        if brightness_score > self.max_brightness:
            return False, {
                "is_valid": False,
                "error_code": "IMAGE_OVEREXPOSED",
                "rejection_reason": f"Image brightness ({brightness_score}) indicates severe overexposure or intense glare.",
                "user_guidance": "The photo is washed out by bright light or harsh reflection. Please move away from direct overhead lighting and retake.",
                "metrics": metrics
            }

        # 3. Focus / Blur Check:
        if not blur_passed:
            return False, {
                "is_valid": False,
                "error_code": "IMAGE_TOO_BLURRY",
                "rejection_reason": f"Image focus sharpness score ({blur_score}) is below the minimum threshold ({self.min_blur_score}).",
                "user_guidance": "The photo is too blurry for trichoscopy analysis. Please hold the phone steady, tap the screen to focus on the scalp parting, and retake the photo.",
                "metrics": metrics
            }

        return True, {
            "is_valid": True,
            "error_code": None,
            "rejection_reason": None,
            "user_guidance": None,
            "metrics": metrics
        }


# Singleton service instance
quality_gate = ImageQualityGate()
