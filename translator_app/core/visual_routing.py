"""Visual-first page/region script and handwriting routing.

This module deliberately consumes pixels before OCR. Its heuristic backend is
conservative and reports provenance/confidence; deployments may replace it with
an evaluated image classifier through :class:`VisualScriptClassifier`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from ..config.settings import Settings
from ..schemas import PageVisualType, ScriptType


@dataclass(frozen=True, slots=True)
class BorderNoiseResult:
    image: np.ndarray
    zones: tuple[tuple[int, int, int, int], ...]
    masked_fraction: float
    rejected_component_count: int


@dataclass(frozen=True, slots=True)
class VisualScriptEvidence:
    candidate: ScriptType
    confidence: float
    handwriting_probability: float
    page_type: PageVisualType
    reason: str
    provenance: str
    script_scores: dict[str, float] = field(default_factory=dict)
    features: dict[str, float | int] = field(default_factory=dict)
    line_boxes: tuple[tuple[int, int, int, int], ...] = ()
    noise_zones: tuple[tuple[int, int, int, int], ...] = ()
    rejected_noise_region_count: int = 0
    cleaned_image: Image.Image | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class RegionVisualEvidence:
    candidate: ScriptType
    confidence: float
    handwriting_probability: float
    reason: str
    is_textual: bool
    is_noise: bool
    features: dict[str, float | int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedScriptEvidence:
    script: ScriptType
    confidence: float
    reason: str


class VisualScriptClassifier(ABC):
    """Pluggable pixel-based script/handwriting classifier contract."""

    provider_id = "abstract_visual_script_classifier"

    @abstractmethod
    def classify_page(self, image: Image.Image) -> VisualScriptEvidence:
        raise NotImplementedError

    @abstractmethod
    def classify_region(
        self,
        image: Image.Image,
        page_evidence: VisualScriptEvidence,
    ) -> RegionVisualEvidence:
        raise NotImplementedError


def expected_script_prior(value: str) -> tuple[ScriptType, ...]:
    normalized = str(value or "auto").strip().casefold().replace("_", "+")
    if normalized in {"pa", "pan", "punjabi"}:
        return (ScriptType.GURMUKHI,)
    if normalized in {"hi", "hin", "hindi"}:
        return (ScriptType.DEVANAGARI,)
    if normalized in {"pa+hi", "hi+pa", "punjabi+hindi"}:
        return (ScriptType.GURMUKHI, ScriptType.DEVANAGARI)
    return ()


def language_for_script(script: ScriptType) -> str:
    return {
        ScriptType.GURMUKHI: "pa",
        ScriptType.DEVANAGARI: "hi",
        ScriptType.LATIN: "en",
    }.get(script, "und")


def resolve_script_evidence(
    visual: VisualScriptEvidence | RegionVisualEvidence,
    *,
    expected_language: str = "auto",
    ocr_script: ScriptType = ScriptType.UNKNOWN,
    ocr_confidence: float = 0.0,
    minimum_visual_confidence: float = 0.52,
) -> ResolvedScriptEvidence:
    """Resolve routing without allowing weak OCR to overwrite visual/user evidence."""
    priors = expected_script_prior(expected_language)
    if len(priors) == 1:
        return ResolvedScriptEvidence(
            priors[0],
            max(0.92, visual.confidence),
            f"user-selected {language_for_script(priors[0])} source-language routing prior",
        )
    if len(priors) == 2:
        if visual.candidate in priors and visual.confidence >= 0.38:
            return ResolvedScriptEvidence(
                visual.candidate,
                max(0.78, visual.confidence),
                "Punjabi + Hindi prior narrowed by visual script evidence",
            )
        if ocr_script in priors and ocr_confidence >= 0.78:
            return ResolvedScriptEvidence(
                ocr_script,
                ocr_confidence,
                "Punjabi + Hindi prior narrowed by validated Unicode evidence",
            )
        return ResolvedScriptEvidence(
            ScriptType.MIXED,
            0.78,
            "Punjabi + Hindi routing prior; visual evidence is ambiguous",
        )
    if visual.candidate in {
        ScriptType.GURMUKHI,
        ScriptType.DEVANAGARI,
        ScriptType.LATIN,
    } and visual.confidence >= minimum_visual_confidence:
        if (
            visual.candidate == ScriptType.LATIN
            and ocr_script in {ScriptType.GURMUKHI, ScriptType.DEVANAGARI}
            and ocr_confidence >= 0.82
        ):
            return ResolvedScriptEvidence(
                ocr_script,
                ocr_confidence,
                "validated Indic Unicode evidence corrected uncertain visual Latin evidence",
            )
        return ResolvedScriptEvidence(
            visual.candidate,
            visual.confidence,
            "visual-first script classifier",
        )
    if visual.candidate == ScriptType.MIXED and visual.confidence >= 0.38:
        return ResolvedScriptEvidence(
            ScriptType.MIXED,
            visual.confidence,
            "visual evidence indicates an Indic/mixed-script page",
        )
    if ocr_script != ScriptType.UNKNOWN and ocr_confidence >= 0.78:
        return ResolvedScriptEvidence(
            ocr_script,
            ocr_confidence,
            "visual evidence was inconclusive; meaningful validated Unicode evidence used",
        )
    return ResolvedScriptEvidence(
        ScriptType.UNKNOWN,
        max(visual.confidence, ocr_confidence * 0.25),
        "visual and meaningful Unicode evidence were inconclusive",
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.flatnonzero(mask)
    if not values.size:
        return []
    output: list[tuple[int, int]] = []
    start = previous = int(values[0])
    for value in values[1:]:
        current = int(value)
        if current > previous + 1:
            output.append((start, previous + 1))
            start = current
        previous = current
    output.append((start, previous + 1))
    return output


def suppress_border_noise(gray: np.ndarray, max_fraction: float = 0.18) -> BorderNoiseResult:
    """Mask dense edge bands/rules before visual statistics without cropping content."""
    if gray.ndim != 2:
        raise ValueError("Border suppression requires a grayscale image")
    cleaned = gray.copy()
    height, width = gray.shape
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    column_density = np.mean(binary > 0, axis=0)
    row_density = np.mean(binary > 0, axis=1)
    center_columns = column_density[int(width * 0.25) : max(int(width * 0.75), 1)]
    center_rows = row_density[int(height * 0.20) : max(int(height * 0.80), 1)]
    typical_column = float(np.median(center_columns)) if center_columns.size else 0.0
    typical_row = float(np.median(center_rows)) if center_rows.size else 0.0
    zones: list[tuple[int, int, int, int]] = []
    max_x = max(2, round(width * max_fraction))
    max_y = max(2, round(height * min(max_fraction, 0.10)))
    smooth_width = max(3, (width // 220) | 1)
    smooth_height = max(3, (height // 260) | 1)
    smooth_columns = np.convolve(
        column_density, np.ones(smooth_width) / smooth_width, mode="same"
    )
    smooth_rows = np.convolve(row_density, np.ones(smooth_height) / smooth_height, mode="same")
    column_threshold = max(0.22, typical_column * 2.6 + 0.02)
    row_threshold = max(0.28, typical_row * 2.8 + 0.02)
    for start, end in _runs(smooth_columns[:max_x] >= column_threshold):
        if start <= max(3, width // 100) and end - start >= 2:
            zones.append((start, 0, min(width, end + smooth_width), height))
    right_offset = width - max_x
    for start, end in _runs(smooth_columns[right_offset:] >= column_threshold):
        if end >= max_x - max(3, width // 100) and end - start >= 2:
            zones.append((max(0, right_offset + start - smooth_width), 0, width, height))
    for start, end in _runs(smooth_rows[:max_y] >= row_threshold):
        if start <= max(3, height // 120) and end - start >= 2:
            zones.append((0, start, width, min(height, end + smooth_height)))
    bottom_offset = height - max_y
    for start, end in _runs(smooth_rows[bottom_offset:] >= row_threshold):
        if end >= max_y - max(3, height // 120) and end - start >= 2:
            zones.append((0, max(0, bottom_offset + start - smooth_height), width, height))

    # Long binding/page rules near an edge are non-content even when not a solid band.
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 3))),
    )
    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rejected_components = 0
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_height < height * 0.32:
            continue
        if x <= width * max_fraction or x + box_width >= width * (1.0 - max_fraction):
            pad = max(2, box_width)
            zones.append((max(0, x - pad), max(0, y - 2), min(width, x + box_width + pad), min(height, y + box_height + 2)))
            rejected_components += 1
    mask = np.zeros_like(gray, dtype=np.uint8)
    for x0, y0, x1, y1 in zones:
        cleaned[y0:y1, x0:x1] = 255
        mask[y0:y1, x0:x1] = 1
    deduplicated = tuple(dict.fromkeys(zones))
    return BorderNoiseResult(
        cleaned,
        deduplicated,
        float(np.mean(mask > 0)),
        rejected_components + len(deduplicated),
    )


def detect_text_line_boxes(
    gray: np.ndarray,
    *,
    handwriting_heavy: bool,
    minimum_width: int = 28,
    minimum_height: int = 9,
    minimum_area: int = 180,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Group ink into logical line boxes and reject speckle/border artifacts."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        15,
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    page_area = gray.size
    component_mask = np.zeros_like(binary)
    accepted_labels = np.zeros(count, dtype=bool)
    rejected = 0
    heights: list[int] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        noise = (
            area < max(5, round(page_area * 0.000002))
            or area > page_area * 0.06
            or width < 2
            or height < 3
            or (width > gray.shape[1] * 0.72 and height <= max(5, gray.shape[0] * 0.004))
            or (height > gray.shape[0] * 0.45 and width <= max(5, gray.shape[1] * 0.006))
        )
        if noise:
            rejected += 1
            continue
        accepted_labels[index] = True
        heights.append(height)
    # Avoid one full-page boolean comparison per component.  Difficult
    # photocopies can contain thousands of speckles; label lookup keeps this
    # pass linear in the number of pixels instead of components × pixels.
    component_mask[accepted_labels[labels]] = 255
    median_height = float(np.median(heights)) if heights else max(8.0, gray.shape[0] / 80)
    join_width = max(9, round(median_height * (2.4 if handwriting_heavy else 1.7)))
    join_height = max(3, round(median_height * (0.38 if handwriting_heavy else 0.24)))
    joined = cv2.morphologyEx(
        component_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (join_width, join_height)),
    )
    joined = cv2.dilate(
        joined,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(3, join_width // 2), max(1, join_height // 2)),
        ),
        iterations=1,
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        crop_ink = float(np.mean(component_mask[y : y + height, x : x + width] > 0))
        if (
            width < minimum_width
            or height < minimum_height
            or area < minimum_area
            or crop_ink < 0.004
            or crop_ink > 0.72
        ):
            rejected += 1
            continue
        padding_x = max(3, round(median_height * 0.35))
        padding_y = max(2, round(median_height * 0.20))
        boxes.append(
            (
                max(0, x - padding_x),
                max(0, y - padding_y),
                min(gray.shape[1], x + width + padding_x),
                min(gray.shape[0], y + height + padding_y),
            )
        )
    boxes.sort(key=lambda item: (item[1], item[0]))
    # Merge contour fragments on the same baseline while preserving marginal notes.
    merged: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        previous = merged[-1]
        overlap = max(0, min(previous[3], box[3]) - max(previous[1], box[1]))
        minimum_box_height = max(1, min(previous[3] - previous[1], box[3] - box[1]))
        gap = box[0] - previous[2]
        if overlap / minimum_box_height >= 0.30 and gap <= median_height * 6.0:
            merged[-1] = (
                min(previous[0], box[0]),
                min(previous[1], box[1]),
                max(previous[2], box[2]),
                max(previous[3], box[3]),
            )
        else:
            merged.append(box)

    # A noisy photocopy can leave thousands of small, disconnected marks.  A
    # rectangular close then creates one contour spanning most of the page,
    # even though the underlying ink still has a clear horizontal line rhythm.
    # Use a projection-based result for handwriting-heavy pages.  Its threshold
    # is derived from the page's observed noise floor, rather than declaring a
    # row active after only a handful of pixels.
    if handwriting_heavy:
        projection_boxes = _projection_handwriting_line_boxes(
            gray,
            minimum_width=minimum_width,
            minimum_height=minimum_height,
            minimum_area=minimum_area,
        )
        if len(projection_boxes) >= 2:
            return projection_boxes, rejected
    return merged, rejected


def _projection_handwriting_line_boxes(
    gray: np.ndarray,
    *,
    minimum_width: int,
    minimum_height: int,
    minimum_area: int,
) -> list[tuple[int, int, int, int]]:
    """Recover line bands from a speckled handwriting-heavy grayscale page.

    This is deliberately a segmentation helper, not a recognizer.  It removes
    only obvious page-spanning rules, estimates the row-noise floor, and finds
    bands with substantially more coherent ink than the background.  It never
    emits character-sized boxes and returns no result when a line rhythm cannot
    be established safely.
    """
    height, width = gray.shape
    if height < max(40, minimum_height * 3) or width < minimum_width:
        return []

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        15,
    )

    # Remove only near-page-spanning rules.  Shorter Gurmukhi headlines and
    # underlines are retained as linguistic ink.
    horizontal_rules = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, round(width * 0.88)), 1)),
    )
    vertical_rules = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(60, round(height * 0.62)))),
    )
    binary = cv2.subtract(binary, cv2.bitwise_or(horizontal_rules, vertical_rules))

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    clean = np.zeros_like(binary)
    accepted_labels = np.zeros(count, dtype=bool)
    page_area = gray.size
    minimum_component_area = max(4, round(page_area * 0.0000015))
    component_heights: list[int] = []
    for index in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[index]
        )
        if (
            area < minimum_component_area
            or component_width < 2
            or component_height < 2
            or (
                component_width > width * 0.82
                and component_height <= max(5, height * 0.004)
            )
            or (
                component_height > height * 0.48
                and component_width <= max(5, width * 0.006)
            )
        ):
            continue
        accepted_labels[index] = True
        component_heights.append(component_height)
    if not component_heights:
        return []
    clean[accepted_labels[labels]] = 255

    glyph_scale = float(np.percentile(component_heights, 75))
    horizontal_join = max(5, min(31, round(max(4.0, glyph_scale) * 1.35)))
    grouped = cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_join, 1)),
    )
    row_counts = np.count_nonzero(grouped, axis=1).astype(np.float32)
    smoothing_window = max(3, min(11, round(max(5.0, glyph_scale) * 0.45) | 1))
    smoothed = np.convolve(
        row_counts,
        np.ones(smoothing_window, dtype=np.float32) / smoothing_window,
        mode="same",
    )
    positive = smoothed[smoothed > 0]
    if positive.size < minimum_height:
        return []

    noise_floor = float(np.median(positive))
    upper_density = float(np.percentile(positive, 90))
    # Uniform speckle has no pronounced line peaks and is not text evidence.
    if upper_density < max(width * 0.012, noise_floor * 1.30):
        return []
    normalized = np.clip(smoothed / max(1.0, float(smoothed.max())) * 255, 0, 255).astype(
        np.uint8
    )
    otsu_value, _unused = cv2.threshold(
        normalized.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    otsu_count = float(otsu_value) / 255.0 * float(smoothed.max())
    activation_threshold = max(
        3.0,
        width * 0.004,
        otsu_count * 0.82,
        noise_floor + 0.16 * max(0.0, upper_density - noise_floor),
    )
    active_rows = (smoothed >= activation_threshold).astype(np.uint8).reshape(-1, 1)

    # At 1600 px the previous 2.5%-of-page rule bridged 40 px and joined
    # neighboring lines.  Cap the bridge at 18 px and scale it conservatively.
    maximum_internal_gap = max(3, min(18, round(height * 0.012)))
    active_rows = cv2.morphologyEx(
        active_rows,
        cv2.MORPH_CLOSE,
        np.ones((maximum_internal_gap, 1), dtype=np.uint8),
    ).ravel() > 0

    line_runs = _runs(active_rows)
    boxes: list[tuple[int, int, int, int]] = []
    y_padding = max(2, min(8, round(height * 0.004)))
    for y0, y1 in line_runs:
        run_height = y1 - y0
        if run_height < max(5, round(minimum_height * 0.55)):
            continue
        # A near-page-height active band means noise still defeated the
        # segmentation.  Failing closed is safer than routing it as one line.
        if run_height > max(height * 0.18, minimum_height * 9):
            continue
        crop_y0 = max(0, y0 - y_padding)
        crop_y1 = min(height, y1 + y_padding)
        column_ink = np.count_nonzero(clean[crop_y0:crop_y1], axis=0).astype(np.float32)
        total_ink = float(column_ink.sum())
        if total_ink <= 0:
            continue
        # Trim sparse photocopy speckles at each horizontal extreme by ink
        # mass, not by a hard margin; genuine indented and marginal lines remain.
        cumulative = np.cumsum(column_ink)
        left = int(np.searchsorted(cumulative, total_ink * 0.025))
        right = int(np.searchsorted(cumulative, total_ink * 0.975)) + 1
        x_padding = max(3, min(8, round(run_height * 0.18)))
        x0, x1 = max(0, left - x_padding), min(width, right + x_padding)
        box_width, box_height = x1 - x0, crop_y1 - crop_y0
        if (
            box_width < minimum_width
            or box_height < minimum_height
            or box_width * box_height < minimum_area
        ):
            continue
        boxes.append((x0, crop_y0, x1, crop_y1))

    boxes.sort(key=lambda item: (item[1], item[0]))
    return boxes


def _component_features(gray: np.ndarray) -> dict[str, float | int]:
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    components = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if 5 <= area <= max(100, gray.size * 0.025) and width >= 2 and height >= 3:
            components.append((x, y, width, height, area))
    heights = np.array([item[3] for item in components], dtype=np.float32)
    widths = np.array([item[2] for item in components], dtype=np.float32)
    baselines = np.array([item[1] + item[3] for item in components], dtype=np.float32)
    ordered = sorted(components, key=lambda item: (item[1] + item[3] / 2, item[0]))
    gaps = np.array(
        [max(0, second[0] - (first[0] + first[2])) for first, second in zip(ordered, ordered[1:])],
        dtype=np.float32,
    )
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    strokes = distance[distance > 0]
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        holes = sum(1 for item in hierarchy[0] if item[3] >= 0)
    return {
        "ink_density": float(np.mean(binary > 0)),
        "component_count": len(components),
        "component_height_variation": float(np.std(heights) / max(1.0, np.mean(heights))) if heights.size else 0.0,
        "component_width_variation": float(np.std(widths) / max(1.0, np.mean(widths))) if widths.size else 0.0,
        "baseline_variation": float(np.std(baselines) / max(1.0, gray.shape[0])) if baselines.size else 0.0,
        "spacing_irregularity": float(np.std(gaps) / max(1.0, np.mean(gaps))) if gaps.size > 2 else 0.0,
        "stroke_width_variation": float(np.std(strokes) / max(0.5, np.mean(strokes))) if strokes.size else 0.0,
        "closed_loop_ratio": holes / max(1, len(contours)),
    }


def _headline_score(gray: np.ndarray, line_boxes: list[tuple[int, int, int, int]]) -> float:
    if not line_boxes:
        return 0.0
    values: list[float] = []
    for x0, y0, x1, y1 in line_boxes:
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        projection = np.mean(binary > 0, axis=1)
        peak_index = int(np.argmax(projection))
        peak = float(projection[peak_index])
        upper = peak_index / max(1, crop.shape[0]) <= 0.58
        values.append(min(1.0, peak / 0.28) if upper else min(0.35, peak))
    return float(np.mean(values)) if values else 0.0


def _line_curve_irregularity(
    gray: np.ndarray, line_boxes: list[tuple[int, int, int, int]]
) -> float:
    """Estimate baseline/stroke-path variation from pixels, independent of OCR."""
    values: list[float] = []
    for x0, y0, x1, y1 in line_boxes:
        crop = gray[y0:y1, x0:x1]
        if crop.shape[0] < 8 or crop.shape[1] < 24:
            continue
        binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        samples: list[tuple[float, float]] = []
        windows = min(28, max(8, crop.shape[1] // 18))
        for index in range(windows):
            left = round(index * crop.shape[1] / windows)
            right = max(left + 1, round((index + 1) * crop.shape[1] / windows))
            points = np.argwhere(binary[:, left:right] > 0)
            if points.size:
                samples.append(((left + right) / 2, float(np.percentile(points[:, 0], 78))))
        if len(samples) < 5:
            continue
        x_values = np.array([item[0] for item in samples], dtype=np.float32)
        y_values = np.array([item[1] for item in samples], dtype=np.float32)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        residual = y_values - (slope * x_values + intercept)
        values.append(min(1.0, float(np.std(residual)) / max(1.0, crop.shape[0] * 0.13)))
    return float(np.mean(values)) if values else 0.0


class HeuristicVisualScriptClassifier(VisualScriptClassifier):
    """Local fallback using grayscale morphology; confidence is intentionally capped."""

    provider_id = "local_visual_morphology_v1"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _analyze(self, image: Image.Image) -> VisualScriptEvidence:
        gray = np.asarray(image.convert("L"))
        noise = suppress_border_noise(gray, self.settings.border_noise_max_fraction)
        initial_boxes, initial_rejected = detect_text_line_boxes(
            noise.image,
            handwriting_heavy=False,
            minimum_width=self.settings.min_region_width,
            minimum_height=self.settings.min_region_height,
            minimum_area=self.settings.min_region_area,
        )
        page_features = _component_features(noise.image)
        line_count = len(initial_boxes)
        height_variation = min(1.0, float(page_features["component_height_variation"]) / 1.1)
        width_variation = min(1.0, float(page_features["component_width_variation"]) / 1.25)
        spacing = min(1.0, float(page_features["spacing_irregularity"]) / 1.8)
        baseline = min(1.0, float(page_features["baseline_variation"]) * 7.0)
        stroke = min(1.0, float(page_features["stroke_width_variation"]) / 1.0)
        curve = _line_curve_irregularity(noise.image, initial_boxes)
        connectedness = max(
            0.0,
            1.0
            - int(page_features["component_count"])
            / max(1.0, len(initial_boxes) * 4.0),
        )
        handwriting = max(
            0.0,
            min(
                1.0,
                0.10
                + height_variation * 0.18
                + width_variation * 0.12
                + spacing * 0.12
                + baseline * 0.12
                + stroke * 0.08
                + curve * 0.18
                + connectedness * 0.14
                + min(0.08, line_count * 0.012),
            ),
        )
        page_type = (
            PageVisualType.HANDWRITING_HEAVY
            if handwriting >= self.settings.handwriting_heavy_threshold
            else PageVisualType.PRINTED
            if handwriting <= 0.36 and line_count > 0
            else PageVisualType.MIXED
            if line_count > 0
            else PageVisualType.UNKNOWN
        )
        line_boxes, heavy_rejected = detect_text_line_boxes(
            noise.image,
            handwriting_heavy=page_type == PageVisualType.HANDWRITING_HEAVY,
            minimum_width=self.settings.min_region_width,
            minimum_height=self.settings.min_region_height,
            minimum_area=self.settings.min_region_area,
        )
        headline = _headline_score(noise.image, line_boxes)
        loop_ratio = min(1.0, float(page_features["closed_loop_ratio"]) / 0.16)
        indic_score = min(1.0, headline * 0.82 + loop_ratio * 0.18)
        latin_score = max(0.0, (1.0 - indic_score) * min(1.0, line_count / 3.0))
        gurmukhi_score = indic_score * (0.54 + loop_ratio * 0.24)
        devanagari_score = indic_score * (0.66 - loop_ratio * 0.12)
        scores = {
            ScriptType.GURMUKHI.value: gurmukhi_score,
            ScriptType.DEVANAGARI.value: devanagari_score,
            ScriptType.LATIN.value: latin_score,
        }
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_name, best_score = ordered[0]
        second_score = ordered[1][1]
        if line_count == 0 or float(page_features["ink_density"]) < 0.002:
            candidate, confidence = ScriptType.UNKNOWN, 0.0
            reason = "no reliable visual text lines survived noise filtering"
        elif indic_score >= self.settings.visual_indic_headline_threshold:
            if best_name in {"gurmukhi", "devanagari"} and best_score - second_score >= 0.10:
                candidate = ScriptType(best_name)
                confidence = min(0.68, 0.42 + indic_score * 0.20 + abs(best_score - second_score))
                reason = "Indic headline/stroke morphology; Gurmukhi versus Devanagari remains heuristic"
            else:
                candidate = ScriptType.MIXED
                confidence = min(0.64, 0.38 + indic_score * 0.22)
                reason = "visual morphology indicates Indic headline script but is ambiguous"
        elif latin_score >= 0.58 and handwriting < 0.58:
            candidate = ScriptType.LATIN
            confidence = min(0.72, 0.48 + latin_score * 0.22)
            reason = "visual line morphology is more consistent with Latin print"
        else:
            candidate = ScriptType.UNKNOWN
            confidence = min(0.45, max(indic_score, latin_score) * 0.45)
            reason = "visual script evidence is inconclusive"
        features = {
            **page_features,
            "headline_score": headline,
            "indic_visual_score": indic_score,
            "line_count": len(line_boxes),
            "line_curve_irregularity": curve,
            "component_connectedness": connectedness,
            "border_masked_fraction": noise.masked_fraction,
        }
        return VisualScriptEvidence(
            candidate=candidate,
            confidence=confidence,
            handwriting_probability=handwriting,
            page_type=page_type,
            reason=reason,
            provenance=self.provider_id,
            script_scores={key: round(value, 4) for key, value in scores.items()},
            features=features,
            line_boxes=tuple(line_boxes),
            noise_zones=noise.zones,
            rejected_noise_region_count=(
                # The initial and handwriting-aware passes inspect the same
                # pixels; summing them double-counted every rejected speckle.
                noise.rejected_component_count
                + max(initial_rejected, heavy_rejected)
            ),
            cleaned_image=Image.fromarray(noise.image),
        )

    def classify_page(self, image: Image.Image) -> VisualScriptEvidence:
        return self._analyze(image)

    def classify_region(
        self,
        image: Image.Image,
        page_evidence: VisualScriptEvidence,
    ) -> RegionVisualEvidence:
        if image.width < 8 or image.height < 8:
            return RegionVisualEvidence(
                ScriptType.UNKNOWN, 0.0, page_evidence.handwriting_probability, "region is too small", False, True
            )
        local = self._analyze(image)
        ink_density = float(local.features.get("ink_density", 0.0))
        component_count = int(local.features.get("component_count", 0))
        connected_handwriting_line = bool(
            page_evidence.page_type == PageVisualType.HANDWRITING_HEAVY
            and image.width / max(1, image.height) >= 2.0
            and 0.003 <= ink_density <= 0.55
        )
        is_noise = (
            (component_count == 0 and not connected_handwriting_line)
            or ink_density < 0.002
            or ink_density > 0.72
            or (image.width < self.settings.min_region_width and image.height < self.settings.min_region_height)
        )
        handwriting = min(
            1.0,
            page_evidence.handwriting_probability * 0.65
            + local.handwriting_probability * 0.35,
        )
        local_features = dict(local.features)
        local_features["local_handwriting_probability"] = local.handwriting_probability
        local_features["page_handwriting_probability"] = (
            page_evidence.handwriting_probability
        )
        candidate = local.candidate
        confidence = local.confidence
        reason = local.reason
        if candidate == ScriptType.UNKNOWN and page_evidence.candidate != ScriptType.UNKNOWN:
            candidate = page_evidence.candidate
            confidence = max(confidence, page_evidence.confidence * 0.72)
            reason = "region visual evidence was weak; inherited visual page candidate"
        return RegionVisualEvidence(
            candidate=candidate,
            confidence=confidence,
            handwriting_probability=handwriting,
            reason=reason,
            is_textual=not is_noise and (component_count >= 1 or connected_handwriting_line),
            is_noise=is_noise,
            features=local_features,
        )


__all__ = [
    "BorderNoiseResult",
    "HeuristicVisualScriptClassifier",
    "RegionVisualEvidence",
    "ResolvedScriptEvidence",
    "VisualScriptClassifier",
    "VisualScriptEvidence",
    "detect_text_line_boxes",
    "expected_script_prior",
    "language_for_script",
    "resolve_script_evidence",
    "suppress_border_noise",
]
