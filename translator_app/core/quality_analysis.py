"""Image-quality measurements and automatic preprocessing-profile selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class ImageQualityMetrics:
    width: int
    height: int
    brightness: float
    contrast: float
    blur_score: float
    noise_score: float
    shadow_score: float
    background_noise: float
    skew_angle: float
    border_confidence: float
    perspective_distortion: float
    handwriting_likelihood: float
    border_noise_fraction: float = 0.0
    rejected_border_artifacts: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def largest_page_contour(gray: np.ndarray) -> tuple[np.ndarray | None, float]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = float(gray.shape[0] * gray.shape[1])
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area_ratio = cv2.contourArea(contour) / max(1.0, page_area)
        if len(polygon) == 4 and area_ratio >= 0.35:
            return polygon.reshape(4, 2).astype(np.float32), min(1.0, area_ratio)
    return None, 0.0


def _skew_angle(gray: np.ndarray) -> float:
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=max(40, gray.shape[1] // 8),
        maxLineGap=12,
    )
    if lines is None:
        return 0.0
    angles = []
    for line in lines[:150]:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= 15:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def _component_irregularity(gray: np.ndarray) -> float:
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    heights = [
        stats[index, cv2.CC_STAT_HEIGHT]
        for index in range(1, count)
        if 8 <= stats[index, cv2.CC_STAT_AREA] <= 2500
    ]
    if len(heights) < 10:
        return 0.0
    mean = float(np.mean(heights))
    return min(1.0, float(np.std(heights)) / max(1.0, mean))


class ImageQualityAnalyzer:
    def analyze(self, image: Image.Image) -> ImageQualityMetrics:
        color = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        small = gray
        if max(gray.shape) > 1800:
            scale = 1800 / max(gray.shape)
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        from .visual_routing import suppress_border_noise

        border_result = suppress_border_noise(small)
        visual_content = border_result.image
        brightness = float(np.mean(small) / 255.0)
        contrast = min(1.0, float(np.std(small) / 80.0))
        laplacian = float(cv2.Laplacian(small, cv2.CV_64F).var())
        blur_score = min(1.0, laplacian / 350.0)
        residual = small.astype(np.float32) - cv2.GaussianBlur(small, (3, 3), 0).astype(np.float32)
        noise_score = min(1.0, float(np.mean(np.abs(residual))) / 18.0)
        illumination = cv2.GaussianBlur(
            small, (0, 0), sigmaX=max(7, min(small.shape) / 18)
        )
        shadow_score = min(1.0, float(np.std(illumination)) / 55.0)
        thresholded = cv2.threshold(
            visual_content, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        background_noise = min(1.0, float(np.mean(thresholded > 0)) / 0.35)
        contour, border_confidence = largest_page_contour(small)
        distortion = 0.0
        if contour is not None:
            ordered = contour[np.argsort(contour[:, 1])]
            top_pair = ordered[:2][np.argsort(ordered[:2, 0])]
            bottom_pair = ordered[2:][np.argsort(ordered[2:, 0])]
            top = np.linalg.norm(top_pair[0] - top_pair[1])
            bottom = np.linalg.norm(bottom_pair[0] - bottom_pair[1])
            left = np.linalg.norm(top_pair[0] - bottom_pair[0])
            right = np.linalg.norm(top_pair[1] - bottom_pair[1])
            distortion = min(
                1.0,
                abs(top - bottom) / max(1.0, max(top, bottom))
                + abs(left - right) / max(1.0, max(left, right)),
            )
        return ImageQualityMetrics(
            width=image.width,
            height=image.height,
            brightness=brightness,
            contrast=contrast,
            blur_score=blur_score,
            noise_score=noise_score,
            shadow_score=shadow_score,
            background_noise=background_noise,
            skew_angle=_skew_angle(visual_content),
            border_confidence=border_confidence,
            perspective_distortion=float(distortion),
            handwriting_likelihood=_component_irregularity(visual_content),
            border_noise_fraction=border_result.masked_fraction,
            rejected_border_artifacts=border_result.rejected_component_count,
        )

    @staticmethod
    def select_profile(metrics: ImageQualityMetrics, allow_geometry: bool = True) -> str:
        if allow_geometry and (
            metrics.perspective_distortion >= 0.08
            or 0.35 <= metrics.border_confidence < 0.92
            or metrics.shadow_score >= 0.28
        ):
            return "mobile_photo"
        if metrics.handwriting_likelihood >= 0.72:
            return "handwriting_heavy"
        if metrics.contrast < 0.42 and metrics.brightness > 0.58:
            return "faded_document"
        if metrics.noise_score >= 0.38 or metrics.background_noise >= 0.55:
            return "photocopy"
        return "clean_scan"
