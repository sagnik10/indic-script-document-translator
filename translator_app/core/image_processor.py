"""Adaptive document cleanup with geometry-safe and mobile-photo modes."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from ..config.settings import Settings
from ..exceptions import CorruptedDocumentError
from .preprocessing_profiles import PROFILES, get_profile
from .quality_analysis import ImageQualityAnalyzer, ImageQualityMetrics, largest_page_contour
from .tesseract_runtime import configure_pytesseract, discover_tesseract_runtime

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessingOptions:
    grayscale: bool = True
    denoise: bool = True
    enhance_contrast: bool = True
    threshold: bool = True
    deskew: bool = True
    rotate_exif: bool = True
    resize_small_images: bool = True
    sharpen: bool = True
    border_cleanup: bool = True
    perspective_correction: bool = True
    shadow_correction: bool = True
    orientation_correction: bool = True
    local_region_enhancement: bool = True


@dataclass(slots=True)
class PreprocessedImage:
    image: Image.Image
    display_image: Image.Image
    original_size: tuple[int, int]
    applied_operations: list[str] = field(default_factory=list)
    profile: str = "clean_scan"
    quality: ImageQualityMetrics | None = None
    geometry_changed: bool = False
    candidate_images: dict[str, Image.Image] = field(default_factory=dict, repr=False)


def load_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CorruptedDocumentError("Image decoding failed") from exc


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _order_corners(points: np.ndarray) -> np.ndarray:
    rectangle = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    rectangle[0] = points[np.argmin(sums)]
    rectangle[2] = points[np.argmax(sums)]
    rectangle[1] = points[np.argmin(differences)]
    rectangle[3] = points[np.argmax(differences)]
    return rectangle


def _perspective_crop(color: np.ndarray) -> tuple[np.ndarray, bool]:
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    contour, confidence = largest_page_contour(gray)
    if contour is None or confidence < 0.42:
        return color, False
    source = _order_corners(contour)
    top_left, top_right, bottom_right, bottom_left = source
    width = int(
        max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
    )
    height = int(
        max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
    )
    if width < 300 or height < 300:
        return color, False
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    corrected = cv2.warpPerspective(
        color,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected, True


def _rotate_same_canvas(array: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.3 or abs(angle) > 12:
        return array
    height, width = array.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _correct_shadow(gray: np.ndarray) -> np.ndarray:
    kernel_size = max(15, (min(gray.shape) // 24) | 1)
    background = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    divided = cv2.divide(gray, np.maximum(background, 1), scale=255)
    return cv2.normalize(divided, None, 0, 255, cv2.NORM_MINMAX)


def _orientation_rotation(image: Image.Image, settings: Settings | None = None) -> int:
    """Use local Tesseract OSD when available; failure leaves orientation unchanged."""
    try:
        import pytesseract

        runtime = discover_tesseract_runtime(
            settings.tesseract_cmd if settings else None,
            settings.tessdata_directory if settings else None,
        )
        if runtime is None:
            return 0
        configure_pytesseract(pytesseract, runtime)
        probe = image.copy()
        probe.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        output = pytesseract.image_to_osd(
            probe,
            config="--psm 0",
        )
        match = re.search(r"Rotate:\s*(0|90|180|270)", output)
        confidence_match = re.search(r"Orientation confidence:\s*([0-9.]+)", output)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.0
        return int(match.group(1)) if match and confidence >= 1.5 else 0
    except Exception:
        return 0


def _cleanup_photocopy(gray: np.ndarray, strength: int) -> np.ndarray:
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if strength:
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return opened


class ImagePreprocessor:
    """Choose a measured profile, preserving clean pages and critical graphics."""

    def __init__(
        self,
        options: PreprocessingOptions | None = None,
        quality_analyzer: ImageQualityAnalyzer | None = None,
        default_upscale_factor: float = 2.0,
        settings: Settings | None = None,
    ) -> None:
        self.options = options or PreprocessingOptions()
        self.quality_analyzer = quality_analyzer or ImageQualityAnalyzer()
        self.default_upscale_factor = default_upscale_factor
        self.settings = settings

    def preprocess(
        self,
        image: Image.Image,
        *,
        profile: str = "auto",
        allow_geometry: bool = True,
        upscale_factor: float | None = None,
    ) -> PreprocessedImage:
        image = image.copy().convert("RGB")
        original_size = image.size
        operations: list[str] = []
        if self.options.rotate_exif:
            image = ImageOps.exif_transpose(image)
        if allow_geometry and self.options.orientation_correction:
            rotation = _orientation_rotation(image, self.settings)
            if rotation:
                image = image.rotate(-rotation, expand=True, fillcolor="white")
                operations.append(f"orientation correction {rotation} degrees")
        initial_quality = self.quality_analyzer.analyze(image)
        profile_name = (
            self.quality_analyzer.select_profile(initial_quality, allow_geometry)
            if profile == "auto"
            else profile
        )
        selected = get_profile(profile_name)
        color = np.asarray(image).copy()
        geometry_changed = False
        if (
            allow_geometry
            and self.options.perspective_correction
            and selected.geometry_correction
        ):
            color, geometry_changed = _perspective_crop(color)
            if geometry_changed:
                operations.append("document boundary crop and perspective correction")
        if self.options.deskew and abs(initial_quality.skew_angle) >= 0.3:
            color = _rotate_same_canvas(color, initial_quality.skew_angle)
            operations.append(f"deskew {initial_quality.skew_angle:.2f} degrees")
            geometry_changed = True
        display_image = Image.fromarray(color).convert("RGB")
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        variants: dict[str, np.ndarray] = {
            "rectified_original": color.copy(),
            "rectified_grayscale": gray.copy(),
        }
        illumination_candidate = _correct_shadow(gray)
        variants["illumination_corrected"] = illumination_candidate.copy()
        if (
            self.options.shadow_correction
            and selected.shadow_correction
            and initial_quality.shadow_score >= 0.12
        ):
            gray = illumination_candidate
            operations.append("shadow and illumination correction")
        mild_clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
        variants["clahe_grayscale"] = mild_clahe.copy()
        if self.options.enhance_contrast and (
            initial_quality.contrast < 0.72 or selected.clahe_clip_limit > 1.5
        ):
            clahe = cv2.createCLAHE(
                clipLimit=selected.clahe_clip_limit, tileGridSize=(8, 8)
            )
            gray = clahe.apply(gray)
            operations.append("CLAHE contrast enhancement")
            variants["clahe_grayscale"] = gray.copy()
        mild_denoise = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
        variants["mild_denoise"] = mild_denoise.copy()
        if self.options.denoise and (
            initial_quality.noise_score >= 0.18 or selected.denoise_strength >= 7
        ):
            gray = cv2.fastNlMeansDenoising(
                gray, None, selected.denoise_strength, 7, 21
            )
            operations.append("adaptive denoising")
            variants["denoised_grayscale"] = gray.copy()
        if selected.background_cleanup and initial_quality.background_noise >= 0.28:
            gray = _cleanup_photocopy(gray, selected.morphology_strength)
            operations.append("photocopy background cleanup")
        if self.options.threshold and selected.adaptive_threshold:
            thresholded = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                35,
                13,
            )
            ink_before = float(np.mean(gray < 210))
            ink_after = float(np.mean(thresholded < 128))
            if 0.35 * max(ink_before, 0.001) <= ink_after <= 2.8 * max(
                ink_before, 0.001
            ):
                variants["mild_adaptive_threshold"] = thresholded
                operations.append("generated mild adaptive-threshold OCR candidate")
        if self.options.sharpen and initial_quality.blur_score < 0.62:
            pil_gray = Image.fromarray(gray).filter(
                ImageFilter.UnsharpMask(
                    radius=1.1,
                    percent=int(90 * selected.sharpen_amount + 45),
                    threshold=2,
                )
            )
            gray = np.asarray(pil_gray)
            operations.append("edge-preserving sharpening")
            variants["sharpened_grayscale"] = gray.copy()
        variants["enhanced_grayscale"] = gray.copy()
        requested_scale = upscale_factor or self.default_upscale_factor
        if self.options.resize_small_images and (
            min(gray.shape) < 1500 or initial_quality.blur_score < 0.38
        ):
            scale = max(1.0, min(4.0, float(requested_scale)))
            if scale > 1.05:
                variants = {
                    name: cv2.resize(
                        candidate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                    )
                    for name, candidate in variants.items()
                }
                gray = variants["enhanced_grayscale"]
                operations.append(f"OCR upscale {scale:.1f}x")
        if self.options.border_cleanup and min(gray.shape) > 100:
            border = max(2, round(min(gray.shape) * 0.003))
            for candidate in variants.values():
                candidate[:border, :] = 255
                candidate[-border:, :] = 255
                candidate[:, :border] = 255
                candidate[:, -border:] = 255
        final_quality = self.quality_analyzer.analyze(display_image)
        LOGGER.info(
            "Preprocessing profile=%s operations=%d quality=%s",
            profile_name,
            len(operations),
            final_quality.to_dict(),
        )
        return PreprocessedImage(
            image=Image.fromarray(gray),
            display_image=display_image,
            original_size=original_size,
            applied_operations=operations,
            profile=profile_name,
            quality=final_quality,
            geometry_changed=geometry_changed,
            candidate_images={name: Image.fromarray(value) for name, value in variants.items()},
        )


__all__ = [
    "ImagePreprocessor",
    "PreprocessedImage",
    "PreprocessingOptions",
    "PROFILES",
    "image_to_png_bytes",
    "load_image",
]
