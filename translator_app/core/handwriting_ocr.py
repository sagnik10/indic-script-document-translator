"""Line-level handwritten-text routing with safe Gurmukhi review fallback."""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from ..config.settings import Settings
from ..schemas import (
    BlockType,
    BoundingBox,
    ProcessingOptions,
    ProcessingStatus,
    ReconstructionType,
    Region,
    RegionType,
    ScriptType,
    TextBlock,
    TranslationStatus,
    UncertaintyState,
)
from .confidence_analysis import ConfidenceAnalyzer
from .htr_providers import (
    HTRProviderSpec,
    HandwritingRecognitionProvider,
    build_htr_providers,
    provider_status_records,
)
from .source_validation import normalize_language, script_ratio
from .visual_routing import detect_text_line_boxes

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class HandwritingOCRResult:
    blocks: list[TextBlock]
    warnings: list[str] = field(default_factory=list)


class HandwritingOCREngine(ABC):
    @abstractmethod
    def recognize_regions(
        self,
        image: Image.Image,
        regions: list[Region],
        language_hints: dict[str, str],
        *,
        page_number: int,
        options: ProcessingOptions,
        review_image: Image.Image | None = None,
    ) -> HandwritingOCRResult:
        raise NotImplementedError


def _line_boxes(crop: Image.Image) -> list[tuple[int, int, int, int]]:
    """Segment handwriting into full lines, never individual characters."""
    gray = np.asarray(crop.convert("L"))
    if gray.shape[0] < 60:
        return [(0, 0, crop.width, crop.height)]
    detected, _rejected = detect_text_line_boxes(
        gray,
        handwriting_heavy=True,
        minimum_width=max(18, min(28, crop.width // 4)),
        minimum_height=8,
        minimum_area=120,
    )
    if len(detected) >= 2:
        return detected
    if len(detected) == 1:
        _x0, y0, _x1, y1 = detected[0]
        if y1 - y0 <= crop.height * 0.45:
            return detected
    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        14,
    )
    projection = np.count_nonzero(binary, axis=1).astype(np.float32)
    positive = projection[projection > 0]
    if positive.size:
        noise_floor = float(np.median(positive))
        upper_density = float(np.percentile(positive, 90))
        threshold = max(
            3.0,
            gray.shape[1] * 0.004,
            noise_floor + 0.18 * max(0.0, upper_density - noise_floor),
        )
    else:
        threshold = float("inf")
    active = projection >= threshold
    runs: list[list[int]] = []
    for y, enabled in enumerate(active):
        if enabled and (not runs or y > runs[-1][-1] + 1):
            runs.append([y])
        elif enabled:
            runs[-1].append(y)
    merged_runs: list[list[int]] = []
    maximum_gap = max(3, min(18, round(gray.shape[0] * 0.012)))
    for run in runs:
        if merged_runs and run[0] - merged_runs[-1][-1] <= maximum_gap:
            merged_runs[-1].extend(run)
        else:
            merged_runs.append(run)
    boxes: list[tuple[int, int, int, int]] = []
    for run in merged_runs:
        y0, y1 = max(0, run[0] - 3), min(gray.shape[0], run[-1] + 4)
        if y1 - y0 < 8:
            continue
        columns = np.flatnonzero(np.any(binary[y0:y1] > 0, axis=0))
        if columns.size:
            x0, x1 = max(0, int(columns[0]) - 4), min(gray.shape[1], int(columns[-1]) + 5)
        else:
            x0, x1 = 0, gray.shape[1]
        if x1 - x0 >= 18:
            boxes.append((x0, y0, x1, y1))
    return boxes or [(0, 0, crop.width, crop.height)]


def _png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def _script_for_language(language: str, region: Region) -> ScriptType:
    canonical = normalize_language(language)
    if canonical == "pa":
        return ScriptType.GURMUKHI
    if canonical == "hi":
        return ScriptType.DEVANAGARI
    if canonical == "en":
        return ScriptType.LATIN
    if region.resolved_script != ScriptType.UNKNOWN:
        return region.resolved_script
    try:
        return ScriptType(str(region.metadata.get("dominant_script", "unknown")))
    except ValueError:
        return ScriptType.UNKNOWN


class TrOCRHandwritingEngine(HandwritingOCREngine):
    """Provider-orchestrating HTR engine retained under its legacy public name.

    Providers are selected only when their declared language and script
    capabilities match. No vision model is asked to translate an image.
    """

    def __init__(
        self,
        settings: Settings,
        device: str,
        providers: Iterable[HandwritingRecognitionProvider] | None = None,
        provider_specs: Iterable[HTRProviderSpec] | None = None,
    ) -> None:
        self.settings = settings
        self.device = device
        if providers is None:
            built, discovered = build_htr_providers(settings, device)
            self.providers = list(built)
            self.provider_specs = list(discovered)
        else:
            self.providers = list(providers)
            self.provider_specs = list(provider_specs or [])
        self._failed_providers: set[str] = set()

    def provider_status(self) -> list[dict[str, object]]:
        if self.provider_specs:
            return provider_status_records(self.provider_specs)
        return [
            {
                "provider_id": provider.capabilities.provider_id,
                "backend": provider.capabilities.backend,
                "model_id": provider.model_id,
                "supported_languages": sorted(provider.capabilities.supported_languages),
                "supported_scripts": sorted(
                    script.value for script in provider.capabilities.supported_scripts
                ),
                "confidence_capability": provider.capabilities.confidence_capability.value,
                "source_language_output_only": provider.capabilities.source_language_output_only,
                "handwriting_validated": provider.capabilities.handwriting_validated,
                "model_location": "injected",
            }
            for provider in self.providers
        ]

    def _provider_for(
        self, language: str, script: ScriptType
    ) -> HandwritingRecognitionProvider | None:
        return next(
            (
                provider
                for provider in self.providers
                if provider.capabilities.provider_id not in self._failed_providers
                and provider.capabilities.supports(language, script)
            ),
            None,
        )

    @staticmethod
    def _crop_region(
        image: Image.Image, region: Region
    ) -> tuple[Image.Image, tuple[int, int, int, int]]:
        pixel_bbox = region.metadata.get("pixel_bbox", [0, 0, image.width, image.height])
        x0, y0, x1, y1 = (int(value) for value in pixel_bbox)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.width, x1), min(image.height, y1)
        if x1 <= x0 or y1 <= y0:
            return image.copy(), (0, 0, image.width, image.height)
        return image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)

    @staticmethod
    def _crop_region_for_review(
        review_image: Image.Image,
        region: Region,
        recognition_size: tuple[int, int],
    ) -> Image.Image:
        pixel_bbox = region.metadata.get(
            "pixel_bbox", [0, 0, recognition_size[0], recognition_size[1]]
        )
        scale_x = review_image.width / max(1, recognition_size[0])
        scale_y = review_image.height / max(1, recognition_size[1])
        x0, y0, x1, y1 = (int(value) for value in pixel_bbox)
        review_box = (
            max(0, round(x0 * scale_x)),
            max(0, round(y0 * scale_y)),
            min(review_image.width, round(x1 * scale_x)),
            min(review_image.height, round(y1 * scale_y)),
        )
        if review_box[2] <= review_box[0] or review_box[3] <= review_box[1]:
            return review_image.copy()
        return review_image.crop(review_box)

    @staticmethod
    def _review_line_crop(
        review_crop: Image.Image,
        recognition_crop: Image.Image,
        line_box: tuple[int, int, int, int],
    ) -> Image.Image:
        scale_x = review_crop.width / max(1, recognition_crop.width)
        scale_y = review_crop.height / max(1, recognition_crop.height)
        x0, y0, x1, y1 = line_box
        return review_crop.crop(
            (
                max(0, round(x0 * scale_x)),
                max(0, round(y0 * scale_y)),
                min(review_crop.width, round(x1 * scale_x)),
                min(review_crop.height, round(y1 * scale_y)),
            )
        )

    @staticmethod
    def _line_bbox(
        region: Region, crop: Image.Image, line_box: tuple[int, int, int, int]
    ) -> BoundingBox:
        x0, y0, x1, y1 = line_box
        width_scale = region.bbox.width / max(1, crop.width)
        height_scale = region.bbox.height / max(1, crop.height)
        return BoundingBox(
            region.bbox.x0 + x0 * width_scale,
            region.bbox.y0 + y0 * height_scale,
            region.bbox.x0 + x1 * width_scale,
            region.bbox.y0 + y1 * height_scale,
        )

    def _unavailable_block(
        self,
        *,
        region: Region,
        page_number: int,
        language: str,
        script: ScriptType,
        line_index: int,
        line_image: Image.Image,
        bbox: BoundingBox,
        options: ProcessingOptions,
        reason: str,
    ) -> TextBlock:
        block = TextBlock(
            page_number=page_number,
            block_type=BlockType.LINE,
            source_bbox=bbox,
            source_text="[unreadable handwriting]",
            normalized_text="[unreadable handwriting]",
            detected_language=language,
            script=script,
            ocr_confidence=0.0,
            uncertainty_state=UncertaintyState.FLAGGED,
            translation_status=TranslationStatus.SKIPPED,
            is_ocr=True,
            region_type=RegionType.HANDWRITING,
            ocr_engine="htr_unavailable",
            is_handwritten=True,
            reconstruction_type=ReconstructionType.UNREADABLE,
            review_image_bytes=_png_bytes(line_image),
            provenance=[f"HTR_UNAVAILABLE: {reason}; source image preserved"],
            processing_statuses=[
                ProcessingStatus.HTR_UNAVAILABLE,
                ProcessingStatus.HANDWRITING_UNSUPPORTED,
                ProcessingStatus.UNREADABLE,
                ProcessingStatus.TRANSLATION_SKIPPED,
            ],
            validation_reason="handwriting recognition unavailable for this language/script",
            metadata={
                "region_id": region.region_id,
                "line_index": line_index,
                "source_origin": "htr_unavailable",
                "preserve_region_as_image": options.preserve_unreadable_handwriting_as_image,
                "handwriting_unsupported": True,
                "htr_unavailable": True,
                "htr_unavailable_reason": reason,
                "expected_script": script.value,
                "selected_recognition_engine": region.selected_recognition_engine,
            },
        )
        block.region_visual_script = region.visual_script_candidate
        block.visual_script_confidence = region.visual_script_confidence
        block.recognized_unicode_script = ScriptType.UNKNOWN
        block.resolved_script = script
        block.resolved_language = language
        block.script_resolution_reason = region.script_resolution_reason
        block.expected_language_prior = region.expected_language_prior
        return block

    def recognize_regions(
        self,
        image: Image.Image,
        regions: list[Region],
        language_hints: dict[str, str],
        *,
        page_number: int,
        options: ProcessingOptions,
        review_image: Image.Image | None = None,
    ) -> HandwritingOCRResult:
        warnings: list[str] = []
        blocks: list[TextBlock] = []
        analyzer = ConfidenceAnalyzer(
            options.ocr_low_confidence_threshold,
            options.handwriting_confidence_threshold,
        )
        unavailable_counts: dict[tuple[str, str], int] = {}
        for region in regions:
            if region.region_type != RegionType.HANDWRITING:
                continue
            language = normalize_language(language_hints.get(region.region_id, "und"))
            script = _script_for_language(language, region)
            provider = self._provider_for(language, script) if options.enable_handwriting_ocr else None
            region.resolved_language = language
            region.selected_recognition_engine = f"htr:{language}:{script.value}"
            region.metadata["htr_route_language"] = language
            region.metadata["htr_route_script"] = script.value
            crop, _pixel_bbox = self._crop_region(image, region)
            review_crop = self._crop_region_for_review(
                review_image or image, region, image.size
            )
            for line_index, line_box in enumerate(_line_boxes(crop)):
                line_image = crop.crop(line_box)
                source_line_image = self._review_line_crop(review_crop, crop, line_box)
                bbox = self._line_bbox(region, crop, line_box)
                if provider is None:
                    reason = (
                        "handwriting recognition was disabled"
                        if not options.enable_handwriting_ocr
                        else f"no configured provider declares {language}/{script.value} support"
                    )
                    block = self._unavailable_block(
                        region=region,
                        page_number=page_number,
                        language=language,
                        script=script,
                        line_index=line_index,
                        line_image=source_line_image,
                        bbox=bbox,
                        options=options,
                        reason=reason,
                    )
                    unavailable_counts[(language, reason)] = unavailable_counts.get((language, reason), 0) + 1
                    region.preserve_as_image = options.preserve_unreadable_handwriting_as_image
                    region.block_ids.append(block.block_id)
                    blocks.append(block)
                    continue
                try:
                    prediction = provider.recognize_line(line_image)
                except Exception:
                    LOGGER.warning(
                        "HTR inference failed page=%d provider=%s language=%s script=%s",
                        page_number,
                        provider.capabilities.provider_id,
                        language,
                        script.value,
                        exc_info=True,
                    )
                    self._failed_providers.add(provider.capabilities.provider_id)
                    reason = f"provider {provider.capabilities.provider_id!r} could not load or run"
                    block = self._unavailable_block(
                        region=region,
                        page_number=page_number,
                        language=language,
                        script=script,
                        line_index=line_index,
                        line_image=source_line_image,
                        bbox=bbox,
                        options=options,
                        reason=reason,
                    )
                    unavailable_counts[(language, reason)] = unavailable_counts.get((language, reason), 0) + 1
                    region.preserve_as_image = options.preserve_unreadable_handwriting_as_image
                    region.block_ids.append(block.block_id)
                    blocks.append(block)
                    continue
                confidence = prediction.confidence
                observed_ratio = script_ratio(prediction.text, script)
                script_consistent = bool(
                    prediction.text.strip()
                    and script != ScriptType.UNKNOWN
                    and observed_ratio >= self.settings.min_source_script_ratio
                )
                low_confidence = (
                    confidence is None
                    or confidence < options.handwriting_confidence_threshold
                    or not script_consistent
                )
                block = TextBlock(
                    page_number=page_number,
                    block_type=BlockType.LINE,
                    source_bbox=bbox,
                    source_text=prediction.text or "[unclear handwriting]",
                    normalized_text=prediction.text or "[unclear handwriting]",
                    detected_language=language,
                    script=script,
                    ocr_confidence=confidence,
                    is_ocr=True,
                    region_type=RegionType.HANDWRITING,
                    ocr_engine=f"htr:{prediction.provider_id}:{prediction.model_id}",
                    is_handwritten=True,
                    ocr_alternatives=prediction.alternatives,
                    review_image_bytes=_png_bytes(source_line_image),
                    metadata={
                        "region_id": region.region_id,
                        "line_index": line_index,
                        "source_origin": "htr",
                        "expected_script": script.value,
                        "htr_script_ratio": observed_ratio,
                        "htr_provider_id": prediction.provider_id,
                        "htr_model_id": prediction.model_id,
                        "htr_confidence_capability": provider.capabilities.confidence_capability.value,
                        "htr_source_language_output_only": provider.capabilities.source_language_output_only,
                        "htr_handwriting_validated": provider.capabilities.handwriting_validated,
                        "preserve_region_as_image": low_confidence,
                    },
                    provenance=[
                        f"HTR source transcription by {prediction.provider_id}; model={prediction.model_id}; "
                        f"confidence={confidence if confidence is not None else 'not_available'}; "
                        f"{script.value}_ratio={observed_ratio:.3f}"
                    ],
                )
                block.region_visual_script = region.visual_script_candidate
                block.visual_script_confidence = region.visual_script_confidence
                block.recognized_unicode_script = (
                    script if script_consistent else ScriptType.UNKNOWN
                )
                block.ocr_script_confidence = observed_ratio if script_consistent else 0.0
                block.resolved_script = script
                block.resolved_language = language
                block.script_resolution_reason = region.script_resolution_reason
                block.expected_language_prior = region.expected_language_prior
                block.linguistic_evidence_score = observed_ratio if script_consistent else 0.0
                analyzer.assess(block)
                if low_confidence:
                    if ProcessingStatus.HTR_LOW_CONFIDENCE not in block.processing_statuses:
                        block.processing_statuses.append(ProcessingStatus.HTR_LOW_CONFIDENCE)
                    block.uncertainty_state = UncertaintyState.LOW_OCR_CONFIDENCE
                    block.validation_reason = (
                        "HTR output script did not match the expected source script"
                        if not script_consistent
                        else "HTR confidence unavailable or below threshold"
                    )
                    region.preserve_as_image = options.preserve_unreadable_handwriting_as_image
                else:
                    if ProcessingStatus.HTR_RECOGNIZED not in block.processing_statuses:
                        block.processing_statuses.append(ProcessingStatus.HTR_RECOGNIZED)
                region.block_ids.append(block.block_id)
                blocks.append(block)
        for (language, reason), count in sorted(unavailable_counts.items()):
            label = language if language != "und" else "undetermined-language"
            warnings.append(
                f"Page {page_number}: {count} {label} handwritten line(s) were preserved for manual review: {reason}."
            )
        return HandwritingOCRResult(blocks, list(dict.fromkeys(warnings)))
