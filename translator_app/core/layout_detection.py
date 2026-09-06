"""Heuristic region segmentation for text, handwriting, forms, stamps, and signatures."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image

from ..config.settings import Settings
from ..schemas import BoundingBox, PageVisualType, Region, RegionType, ScriptType
from .region_merging import merge_text_regions
from .visual_routing import (
    RegionVisualEvidence,
    VisualScriptClassifier,
    VisualScriptEvidence,
    detect_text_line_boxes,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegionFeatures:
    ink_density: float
    component_count: int
    component_height_variation: float
    aspect_ratio: float
    line_density: float
    horizontal_line_density: float
    vertical_line_density: float
    color_saturation: float
    component_width_variation: float
    baseline_variation: float
    spacing_irregularity: float


def _features(gray: np.ndarray, color: np.ndarray | None = None) -> RegionFeatures:
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    component_stats = [
        (
            int(stats[index, cv2.CC_STAT_LEFT]),
            int(stats[index, cv2.CC_STAT_TOP]),
            int(stats[index, cv2.CC_STAT_WIDTH]),
            int(stats[index, cv2.CC_STAT_HEIGHT]),
        )
        for index in range(1, count)
        if 4 <= stats[index, cv2.CC_STAT_AREA] <= max(100, binary.size * 0.08)
    ]
    heights = [
        item[3]
        for item in component_stats
    ]
    widths = [
        item[2]
        for item in component_stats
    ]
    mean_height = float(np.mean(heights)) if heights else 0.0
    variation = float(np.std(heights) / max(1.0, mean_height)) if heights else 0.0
    mean_width = float(np.mean(widths)) if widths else 0.0
    width_variation = float(np.std(widths) / max(1.0, mean_width)) if widths else 0.0
    baselines = [top + height for _left, top, _width, height in component_stats]
    baseline_variation = float(np.std(baselines) / max(1.0, gray.shape[0])) if baselines else 0.0
    ordered_left = sorted(left for left, _top, _width, _height in component_stats)
    gaps = np.diff(ordered_left) if len(ordered_left) > 2 else np.array([])
    spacing_irregularity = float(np.std(gaps) / max(1.0, np.mean(gaps))) if gaps.size else 0.0
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, gray.shape[1] // 5), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, gray.shape[0] // 4))),
    )
    horizontal_density = float(np.mean(horizontal > 0))
    vertical_density = float(np.mean(vertical > 0))
    line_density = min(horizontal_density, vertical_density)
    saturation = 0.0
    if color is not None and color.size:
        saturation = float(np.mean(cv2.cvtColor(color, cv2.COLOR_RGB2HSV)[:, :, 1]) / 255.0)
    return RegionFeatures(
        ink_density=float(np.mean(binary > 0)),
        component_count=max(0, count - 1),
        component_height_variation=variation,
        aspect_ratio=gray.shape[1] / max(1, gray.shape[0]),
        line_density=line_density,
        horizontal_line_density=horizontal_density,
        vertical_line_density=vertical_density,
        color_saturation=saturation,
        component_width_variation=width_variation,
        baseline_variation=baseline_variation,
        spacing_irregularity=spacing_irregularity,
    )


def _component_line_boxes(gray: np.ndarray, settings: Settings) -> list[tuple[int, int, int, int]]:
    """Group connected ink components into line-sized boxes before region classification."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 15
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    components: list[tuple[int, int, int, int]] = []
    page_area = gray.shape[0] * gray.shape[1]
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < 6 or area > page_area * 0.08 or width < 2 or height < 3:
            continue
        if width > gray.shape[1] * 0.82 and height <= 5:
            continue
        components.append((x, y, width, height))
    components.sort(key=lambda item: (item[1] + item[3] / 2, item[0]))
    rows: list[list[tuple[int, int, int, int]]] = []
    for component in components:
        x, y, width, height = component
        center = y + height / 2
        match: list[tuple[int, int, int, int]] | None = None
        best_distance = float("inf")
        for row in rows:
            row_y0 = min(item[1] for item in row)
            row_y1 = max(item[1] + item[3] for item in row)
            overlap = max(0, min(y + height, row_y1) - max(y, row_y0))
            row_height = max(1, row_y1 - row_y0)
            row_center = (row_y0 + row_y1) / 2
            distance = abs(center - row_center)
            if (overlap / max(1, min(height, row_height)) >= 0.22 or distance <= 0.58 * max(height, row_height)) and distance < best_distance:
                match, best_distance = row, distance
        if match is None:
            match = []
            rows.append(match)
        match.append(component)

    boxes: list[tuple[int, int, int, int]] = []
    for row in rows:
        row.sort(key=lambda item: item[0])
        median_height = float(np.median([item[3] for item in row]))
        groups: list[list[tuple[int, int, int, int]]] = [[]]
        previous_right: int | None = None
        for component in row:
            gap = component[0] - previous_right if previous_right is not None else 0
            split_gap = max(gray.shape[1] * 0.09, median_height * 6.0)
            if groups[-1] and gap > split_gap:
                groups.append([])
            groups[-1].append(component)
            previous_right = component[0] + component[2]
        for group in groups:
            x0 = min(item[0] for item in group)
            y0 = min(item[1] for item in group)
            x1 = max(item[0] + item[2] for item in group)
            y1 = max(item[1] + item[3] for item in group)
            width, height = x1 - x0, y1 - y0
            if (
                width >= settings.min_region_width
                and height >= settings.min_region_height
                and width * height >= settings.min_region_area
            ):
                padding_x = max(2, round(median_height * 0.25))
                padding_y = max(2, round(median_height * 0.18))
                boxes.append(
                    (
                        max(0, x0 - padding_x),
                        max(0, y0 - padding_y),
                        min(gray.shape[1], x1 + padding_x),
                        min(gray.shape[0], y1 + padding_y),
                    )
                )
    return boxes


def _reading_order(regions: list[Region]) -> None:
    """Group overlapping baselines into rows, then preserve left-to-right side notes."""
    remaining = sorted(regions, key=lambda region: (region.bbox.y0, region.bbox.x0))
    ordered: list[Region] = []
    while remaining:
        anchor = remaining.pop(0)
        row = [anchor]
        row_top, row_bottom = anchor.bbox.y0, anchor.bbox.y1
        for candidate in remaining[:]:
            overlap = min(row_bottom, candidate.bbox.y1) - max(row_top, candidate.bbox.y0)
            if overlap > 0.25 * min(anchor.bbox.height, candidate.bbox.height):
                row.append(candidate)
                remaining.remove(candidate)
                row_top = min(row_top, candidate.bbox.y0)
                row_bottom = max(row_bottom, candidate.bbox.y1)
        ordered.extend(sorted(row, key=lambda region: region.bbox.x0))
    for index, region in enumerate(ordered):
        region.reading_order = index


class DocumentLayoutDetector:
    """Conservative CV segmentation; critical graphics are preserved even if classification is uncertain."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def _to_page_bbox(
        rectangle: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
        page_width: float,
        page_height: float,
    ) -> BoundingBox:
        x, y, width, height = rectangle
        return BoundingBox(
            x * page_width / image_width,
            y * page_height / image_height,
            (x + width) * page_width / image_width,
            (y + height) * page_height / image_height,
        )

    def _table_regions(
        self, gray: np.ndarray, page_width: float, page_height: float
    ) -> list[Region]:
        if not self.settings.enable_table_detection:
            return []
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
        )
        horizontal = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, gray.shape[1] // 12), 1)),
        )
        vertical = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, gray.shape[0] // 16))),
        )
        grid = cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        page_area = gray.size
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width * height < max(self.settings.min_region_area * 6, page_area * 0.015):
                continue
            if width < gray.shape[1] * 0.18 or height < gray.shape[0] * 0.035:
                continue
            horizontal_crop = horizontal[y : y + height, x : x + width]
            vertical_crop = vertical[y : y + height, x : x + width]
            # Gurmukhi/Devanagari headline strokes are horizontal but do not form a
            # table. Require sustained rules on both axes and actual intersections.
            if (
                np.count_nonzero(horizontal_crop) < width * 1.5
                or np.count_nonzero(vertical_crop) < height * 1.5
                or np.count_nonzero((horizontal_crop > 0) & (vertical_crop > 0)) < 4
            ):
                continue
            regions.append(
                Region(
                    page_number=1,
                    bbox=self._to_page_bbox(
                        (x, y, width, height), gray.shape[1], gray.shape[0], page_width, page_height
                    ),
                    region_type=RegionType.TABLE_FORM,
                    classification_confidence=0.88,
                    preserve_as_image=True,
                    overlaps_critical_graphic=True,
                    metadata={"pixel_bbox": [x, y, x + width, y + height], "table_rules": True},
                )
            )
        return regions

    def _stamp_regions(
        self,
        color: np.ndarray,
        page_width: float,
        page_height: float,
        *,
        handwriting_heavy: bool = False,
    ) -> list[Region]:
        """Return only compact, locally contrasting chromatic marks.

        Old photocopies and mobile photographs often have a yellow/brown cast,
        coloured shadows, or a saturated binding strip.  Saturation alone is
        therefore not stamp evidence.  A candidate must also be darker than its
        local background and have compact two-dimensional geometry.  The
        thresholds are deliberately stricter on handwriting-heavy pages because
        blue-ink words must not be reclassified as seals.
        """
        hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        background_saturation = float(np.percentile(saturation, 55))
        saturation_threshold = int(min(180, max(82, background_saturation + 34)))
        kernel_size = max(15, (min(color.shape[:2]) // 35) | 1)
        local_background = cv2.GaussianBlur(value, (kernel_size, kernel_size), 0)
        local_contrast = local_background.astype(np.int16) - value.astype(np.int16)
        saturation_mask = np.where(
            (saturation >= saturation_threshold)
            & (value >= 28)
            & ((local_contrast >= 10) | (value <= 145)),
            255,
            0,
        ).astype(np.uint8)
        saturation_mask = cv2.morphologyEx(
            saturation_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )
        contours, _ = cv2.findContours(
            saturation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions = []
        image_height, image_width = color.shape[:2]
        image_area = image_height * image_width
        minimum_dimension = max(10, round(min(image_height, image_width) * 0.012))
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if (
                area < self.settings.min_region_area * 2
                or area > image_area * 0.16
                or width < minimum_dimension
                or height < minimum_dimension
            ):
                continue
            aspect_ratio = width / max(1, height)
            touches_edge = (
                x <= image_width * 0.012
                or y <= image_height * 0.012
                or x + width >= image_width * 0.988
                or y + height >= image_height * 0.988
            )
            crop_mask = saturation_mask[y : y + height, x : x + width]
            chromatic_fill = float(np.mean(crop_mask > 0))
            contour_fill = float(cv2.contourArea(contour) / max(1, area))
            contour_perimeter = float(cv2.arcLength(contour, True))
            outline_circularity = (
                float(4.0 * np.pi * cv2.contourArea(contour) / (contour_perimeter**2))
                if contour_perimeter > 0.0
                else 0.0
            )
            mean_contrast = float(
                np.mean(local_contrast[y : y + height, x : x + width][crop_mask > 0])
            ) if np.any(crop_mask) else 0.0
            # Edge-reaching and elongated patches are normally binding bands,
            # shadows, page edges, or coloured handwriting strokes, not seals.
            if (
                aspect_ratio > 5.5
                or aspect_ratio < 1 / 5.5
                or (touches_edge and (aspect_ratio > 2.2 or aspect_ratio < 1 / 2.2))
                or chromatic_fill < 0.025
                or chromatic_fill > 0.68
                or (contour_fill < 0.035 and chromatic_fill < 0.075)
                or mean_contrast < 7.0
            ):
                continue
            if handwriting_heavy and (
                aspect_ratio > 3.0
                or aspect_ratio < 1 / 3.0
                or min(width, height) < min(image_height, image_width) * 0.018
                or (contour_fill < 0.07 and chromatic_fill < 0.10)
                # On a handwriting-heavy page, colour is commonly ordinary pen
                # ink.  Require a compact closed-outline shape before protecting
                # a chromatic mark as a seal; otherwise individual Gurmukhi
                # glyphs are removed from the HTR route as false stamps.
                or outline_circularity < 0.32
            ):
                continue
            regions.append(
                Region(
                    page_number=1,
                    bbox=self._to_page_bbox(
                        (x, y, width, height), color.shape[1], color.shape[0], page_width, page_height
                    ),
                    region_type=RegionType.STAMP_SEAL,
                    classification_confidence=min(
                        0.88,
                        0.62
                        + min(0.14, mean_contrast / 180.0)
                        + min(0.08, contour_fill * 0.18),
                    ),
                    preserve_as_image=True,
                    overlaps_critical_graphic=True,
                    metadata={
                        "pixel_bbox": [x, y, x + width, y + height],
                        "colored_mark": True,
                        "stamp_detection": "local_chromatic_contrast",
                        "chromatic_fill": chromatic_fill,
                        "local_contrast": mean_contrast,
                        "outline_circularity": outline_circularity,
                    },
                )
            )
        return regions

    def detect(
        self,
        ocr_image: Image.Image,
        display_image: Image.Image,
        *,
        page_number: int,
        page_width: float,
        page_height: float,
        quality_metrics: dict[str, float] | None = None,
        visual_evidence: VisualScriptEvidence | None = None,
        visual_classifier: VisualScriptClassifier | None = None,
    ) -> list[Region]:
        classification_image = (
            visual_evidence.cleaned_image
            if visual_evidence is not None and visual_evidence.cleaned_image is not None
            else ocr_image
        )
        gray = np.asarray(classification_image.convert("L"))
        color = np.asarray(display_image.convert("RGB"))
        page_features = _features(gray, color)
        handwriting_prior = (
            visual_evidence.handwriting_probability
            if visual_evidence is not None
            else float((quality_metrics or {}).get("handwriting_likelihood", 0.0))
        )
        if not handwriting_prior:
            handwriting_prior = min(
                1.0,
                page_features.component_height_variation * 0.34
                + page_features.component_width_variation * 0.22
                + page_features.spacing_irregularity * 0.18
                + page_features.baseline_variation * 2.5,
            )
        if visual_evidence is not None and visual_evidence.line_boxes:
            line_boxes = list(visual_evidence.line_boxes)
            rejected_noise_count = visual_evidence.rejected_noise_region_count
        else:
            line_boxes, rejected_noise_count = detect_text_line_boxes(
                gray,
                handwriting_heavy=(
                    visual_evidence is not None
                    and visual_evidence.page_type == PageVisualType.HANDWRITING_HEAVY
                ),
                minimum_width=self.settings.min_region_width,
                minimum_height=self.settings.min_region_height,
                minimum_area=self.settings.min_region_area,
            )
        regions: list[Region] = []
        table_regions = self._table_regions(gray, page_width, page_height)
        handwriting_heavy = bool(
            visual_evidence is not None
            and visual_evidence.page_type == PageVisualType.HANDWRITING_HEAVY
        )
        critical_regions = self._stamp_regions(
            color,
            page_width,
            page_height,
            handwriting_heavy=handwriting_heavy,
        )
        for x0, y0, x1, y1 in line_boxes:
            x, y, width, height = x0, y0, x1 - x0, y1 - y0
            area = width * height
            if area < self.settings.min_region_area or width < 12 or height < 6:
                continue
            if area > gray.shape[0] * gray.shape[1] * 0.82:
                continue
            crop_gray = gray[y : y + height, x : x + width]
            color_x0 = round(x * color.shape[1] / gray.shape[1])
            color_x1 = round((x + width) * color.shape[1] / gray.shape[1])
            color_y0 = round(y * color.shape[0] / gray.shape[0])
            color_y1 = round((y + height) * color.shape[0] / gray.shape[0])
            crop_color = color[color_y0:color_y1, color_x0:color_x1]
            features = _features(crop_gray, crop_color)
            region_visual = (
                visual_classifier.classify_region(
                    Image.fromarray(crop_gray), visual_evidence
                )
                if visual_classifier is not None and visual_evidence is not None
                else RegionVisualEvidence(
                    candidate=(
                        visual_evidence.candidate
                        if visual_evidence is not None
                        else ScriptType.UNKNOWN
                    ),
                    confidence=(
                        visual_evidence.confidence * 0.72
                        if visual_evidence is not None
                        else 0.0
                    ),
                    handwriting_probability=handwriting_prior,
                    reason="page-level visual fallback",
                    is_textual=features.component_count >= 1,
                    is_noise=False,
                )
            )
            if region_visual.is_noise or not region_visual.is_textual:
                rejected_noise_count += 1
                continue
            bbox = self._to_page_bbox(
                (x, y, width, height), gray.shape[1], gray.shape[0], page_width, page_height
            )
            critical_overlap = any(
                bbox.intersection_ratio(critical.bbox) >= 0.45
                for critical in critical_regions
            )
            if critical_overlap:
                # A verified seal remains a protected raster region.  Do not
                # create a second text/HTR route over the same pixels.
                rejected_noise_count += 1
                continue
            table_overlap = any(
                bbox.intersection_ratio(table.bbox) >= 0.55
                for table in table_regions
            )
            local_handwriting_probability = float(
                region_visual.features.get(
                    "local_handwriting_probability",
                    region_visual.handwriting_probability,
                )
            )
            local_headline_score = float(
                region_visual.features.get("headline_score", 1.0)
            )
            strongly_signature_like = bool(
                features.aspect_ratio >= 3.5
                and features.component_height_variation >= 0.65
                and features.ink_density < 0.28
                and y > gray.shape[0] * 0.45
                and (
                    not handwriting_heavy
                    or (
                        features.aspect_ratio >= 5.0
                        and features.component_count <= 6
                        and y > gray.shape[0] * 0.68
                        and width <= gray.shape[1] * 0.65
                        and local_headline_score < 0.16
                    )
                )
            )
            clearly_printed_latin = bool(
                region_visual.candidate == ScriptType.LATIN
                and region_visual.confidence >= 0.68
                and local_handwriting_probability <= 0.34
            )
            if strongly_signature_like:
                region_type, confidence = RegionType.SIGNATURE, 0.68
            elif (
                handwriting_heavy and not clearly_printed_latin
            ) or region_visual.handwriting_probability >= 0.68:
                region_type, confidence = (
                    RegionType.HANDWRITING,
                    max(
                        0.64,
                        handwriting_prior * 0.92,
                        region_visual.handwriting_probability,
                    ),
                )
            elif table_overlap:
                region_type, confidence = RegionType.TABLE_FORM, 0.86
            elif features.component_count >= 2:
                region_type, confidence = RegionType.PRINTED_TEXT, 0.72
            else:
                region_type, confidence = RegionType.UNKNOWN, 0.4
            preserve = region_type in {RegionType.SIGNATURE, RegionType.TABLE_FORM}
            region = Region(
                page_number=page_number,
                bbox=bbox,
                region_type=region_type,
                classification_confidence=confidence,
                preserve_as_image=preserve,
                overlaps_critical_graphic=region_type == RegionType.SIGNATURE,
                visual_script_candidate=region_visual.candidate,
                visual_script_confidence=region_visual.confidence,
                metadata={
                    "pixel_bbox": [x, y, x + width, y + height],
                    "features": asdict(features),
                    "page_handwriting_prior": handwriting_prior,
                    "region_handwriting_probability": region_visual.handwriting_probability,
                    "visual_script_reason": region_visual.reason,
                    "visual_features": region_visual.features,
                    "inside_detected_table": table_overlap,
                    "classification_reason": (
                        "ambiguous text-like line inherited handwriting-heavy page routing"
                        if region_type == RegionType.HANDWRITING and handwriting_heavy
                        else "region visual features"
                    ),
                },
            )
            if not any(
                region.bbox.intersection_ratio(existing.bbox) > 0.9
                and region.region_type == existing.region_type
                for existing in regions
            ):
                regions.append(region)
        for region in table_regions + critical_regions:
            region.page_number = page_number
            if region in critical_regions:
                region.metadata["pixel_bbox"] = [
                    round(region.bbox.x0 * gray.shape[1] / page_width),
                    round(region.bbox.y0 * gray.shape[0] / page_height),
                    round(region.bbox.x1 * gray.shape[1] / page_width),
                    round(region.bbox.y1 * gray.shape[0] / page_height),
                ]
            if not any(region.bbox.intersection_ratio(existing.bbox) > 0.85 for existing in regions):
                regions.append(region)
        fallback_handwriting = handwriting_heavy and not any(
            region.region_type == RegionType.HANDWRITING for region in regions
        )
        if not regions or fallback_handwriting:
            regions.append(
                Region(
                    page_number=page_number,
                    bbox=BoundingBox(0, 0, page_width, page_height),
                    region_type=(
                        RegionType.HANDWRITING if fallback_handwriting else RegionType.UNKNOWN
                    ),
                    classification_confidence=(
                        max(0.55, handwriting_prior) if fallback_handwriting else 0.2
                    ),
                    preserve_as_image=True,
                    visual_script_candidate=(
                        visual_evidence.candidate
                        if visual_evidence is not None
                        else ScriptType.UNKNOWN
                    ),
                    visual_script_confidence=(
                        visual_evidence.confidence if visual_evidence is not None else 0.0
                    ),
                    metadata={
                        "pixel_bbox": [0, 0, gray.shape[1], gray.shape[0]],
                        "routing_reason": (
                            "handwriting-heavy visual fallback; HTR will segment full-page lines"
                            if fallback_handwriting
                            else "no reliable line regions survived visual filtering"
                        ),
                        "rejected_noise_region_count": rejected_noise_count,
                        "critical_exclusion_region_ids": [
                            region.region_id for region in critical_regions
                        ],
                    },
                )
            )
        regions = merge_text_regions(
            regions,
            horizontal_gap_ratio=self.settings.region_merge_horizontal_gap_ratio,
            minimum_vertical_overlap=self.settings.region_merge_vertical_overlap,
        )
        for region in regions:
            region.metadata["page_rejected_noise_region_count"] = rejected_noise_count
        _reading_order(regions)
        return sorted(regions, key=lambda region: region.reading_order)
