"""Region-aware printed OCR with low-confidence multi-pass comparison."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from PIL import Image

from ..config.settings import Settings
from ..schemas import BoundingBox, ProcessingOptions, Region, RegionType, ScriptType, TextBlock
from .confidence_analysis import ConfidenceAnalyzer
from .ocr_engine import OCREngine
from .ocr_ensemble import OCRCandidateComparator
from .script_detection import dominant_script

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PrintedOCRResult:
    blocks: list[TextBlock]
    warnings: list[str] = field(default_factory=list)


class PrintedTextOCR:
    """Run a primary Tesseract pass and alternate PSM passes only when evidence is weak."""

    def __init__(
        self, settings: Settings, engine: OCREngine | Callable[[], OCREngine]
    ) -> None:
        self.settings = settings
        self._engine_or_loader = engine
        self._engine: OCREngine | None = engine if isinstance(engine, OCREngine) else None
        self.comparator = OCRCandidateComparator()

    def _get_engine(self) -> OCREngine:
        if self._engine is None:
            loader = self._engine_or_loader
            if not callable(loader):
                raise RuntimeError("Printed OCR engine loader is invalid")
            self._engine = loader()
        return self._engine

    @staticmethod
    def _crop_for_region(image: Image.Image, region: Region) -> Image.Image:
        pixel_bbox = region.metadata.get("pixel_bbox")
        if pixel_bbox and len(pixel_bbox) == 4:
            x0, y0, x1, y1 = (int(value) for value in pixel_bbox)
        else:
            x0 = round(region.bbox.x0)
            y0 = round(region.bbox.y0)
            x1 = round(region.bbox.x1)
            y1 = round(region.bbox.y1)
        padding = 4
        return image.crop(
            (
                max(0, x0 - padding),
                max(0, y0 - padding),
                min(image.width, x1 + padding),
                min(image.height, y1 + padding),
            )
        )

    @staticmethod
    def _offset_blocks(blocks: list[TextBlock], region: Region) -> None:
        for block in blocks:
            bbox = block.source_bbox
            block.source_bbox = BoundingBox(
                bbox.x0 + region.bbox.x0,
                bbox.y0 + region.bbox.y0,
                bbox.x1 + region.bbox.x0,
                bbox.y1 + region.bbox.y0,
            )
            block.region_type = region.region_type
            block.metadata["region_id"] = region.region_id
            block.metadata["region_reading_order"] = region.reading_order
            block.metadata["region_classification_confidence"] = region.classification_confidence
            if region.region_type == RegionType.STAMP_SEAL:
                block.metadata["preserve_region_as_image"] = True
            elif region.region_type == RegionType.TABLE_FORM:
                # The renderer may replace validated cell text but must retain
                # rules and must never expand beyond the detected cell/region.
                block.metadata["preserve_table_rules"] = True

    def recognize_regions(
        self,
        image: Image.Image,
        regions: list[Region],
        *,
        page_number: int,
        options: ProcessingOptions,
        variants: dict[str, Image.Image] | None = None,
        probe_handwriting: bool = False,
        language_hints: dict[str, list[str]] | None = None,
    ) -> PrintedOCRResult:
        if not options.enable_printed_ocr:
            return PrintedOCRResult([], ["Printed-text OCR was disabled by the user."])
        warnings: list[str] = []
        blocks: list[TextBlock] = []
        analyzer = ConfidenceAnalyzer(
            options.ocr_low_confidence_threshold,
            options.handwriting_confidence_threshold,
        )
        psms = list(self.settings.printed_ocr_psm_candidates or [6, 11])
        engine: OCREngine | None = None
        candidate_variants = variants or {"enhanced_grayscale": image}
        variant_order = [
            name
            for name in (
                "enhanced_grayscale",
                "rectified_grayscale",
                "clahe_grayscale",
                "illumination_corrected",
                "denoised_grayscale",
                "mild_adaptive_threshold",
            )
            if name in candidate_variants
        ]
        if not variant_order:
            variant_order = list(candidate_variants)[:1]
        for region in regions:
            if region.region_type in {
                RegionType.SIGNATURE,
                RegionType.STAMP_SEAL,
                RegionType.GRAPHICAL_CONTENT,
            } or (region.region_type == RegionType.HANDWRITING and not probe_handwriting):
                continue
            crop = self._crop_for_region(candidate_variants[variant_order[0]], region)
            if crop.width < 8 or crop.height < 8:
                continue
            if engine is None:
                engine = self._get_engine()
            requested_languages = list(
                (language_hints or {}).get(region.region_id, options.ocr_languages)
            ) or list(options.ocr_languages)
            region.selected_recognition_engine = (
                "printed_ocr:" + "+".join(requested_languages)
            )
            candidate_blocks: list[TextBlock] = []
            primary_psm = 11 if region.region_type == RegionType.STAMP_SEAL else psms[0]
            result = engine.recognize(
                crop,
                page_number=page_number,
                page_width=region.bbox.width,
                page_height=region.bbox.height,
                requested_languages=requested_languages,
                low_confidence_threshold=options.ocr_low_confidence_threshold,
                psm=primary_psm,
            )
            warnings.extend(result.warnings)
            self._offset_blocks(result.blocks, region)
            for block in result.blocks:
                block.metadata["preprocessing_variant"] = variant_order[0]
                block.ocr_engine = f"{block.ocr_engine}:{variant_order[0]}"
            candidate_blocks.extend(result.blocks)
            aggregate = (
                sum(block.ocr_confidence or 0.0 for block in result.blocks) / len(result.blocks)
                if result.blocks
                else 0.0
            )
            if aggregate < options.ocr_low_confidence_threshold:
                # Compare several geometry-identical grayscale candidates. A harsh
                # threshold is only one candidate and can never erase the grayscale evidence.
                variant_limit = 2 if probe_handwriting else 4
                for variant_name in variant_order[1:variant_limit]:
                    alternate_crop = self._crop_for_region(candidate_variants[variant_name], region)
                    alternate = engine.recognize(
                        alternate_crop,
                        page_number=page_number,
                        page_width=region.bbox.width,
                        page_height=region.bbox.height,
                        requested_languages=requested_languages,
                        low_confidence_threshold=options.ocr_low_confidence_threshold,
                        psm=primary_psm,
                    )
                    warnings.extend(alternate.warnings)
                    self._offset_blocks(alternate.blocks, region)
                    for block in alternate.blocks:
                        block.metadata["preprocessing_variant"] = variant_name
                        block.ocr_engine = f"{block.ocr_engine}:{variant_name}"
                    candidate_blocks.extend(alternate.blocks)
                # A second segmentation mode is useful for a logical line/paragraph,
                # but never for character-sized contours.
                for psm in ([] if probe_handwriting else psms[1:2]):
                    if psm == primary_psm:
                        continue
                    alternate = engine.recognize(
                        crop,
                        page_number=page_number,
                        page_width=region.bbox.width,
                        page_height=region.bbox.height,
                        requested_languages=requested_languages,
                        low_confidence_threshold=options.ocr_low_confidence_threshold,
                        psm=psm,
                    )
                    self._offset_blocks(alternate.blocks, region)
                    candidate_blocks.extend(alternate.blocks)

            # Once Unicode provides a script clue, compare a script-specific OCR pass.
            joined = " ".join(block.source_text for block in candidate_blocks)
            script_name, script_confidence = dominant_script(joined)
            if region.resolved_script in {
                ScriptType.GURMUKHI,
                ScriptType.DEVANAGARI,
                ScriptType.LATIN,
            }:
                script_name = region.resolved_script.value
                script_confidence = max(script_confidence, region.visual_script_confidence)
            language_pack = {"gurmukhi": "pan", "devanagari": "hin", "latin": "eng"}.get(script_name)
            if language_pack and language_pack in requested_languages and script_confidence >= 0.35:
                routed = engine.recognize(
                    crop,
                    page_number=page_number,
                    page_width=region.bbox.width,
                    page_height=region.bbox.height,
                    requested_languages=[language_pack],
                    low_confidence_threshold=options.ocr_low_confidence_threshold,
                    psm=primary_psm,
                )
                warnings.extend(routed.warnings)
                self._offset_blocks(routed.blocks, region)
                for block in routed.blocks:
                    block.metadata["script_specific_ocr"] = script_name
                    block.ocr_engine = f"{block.ocr_engine}:{language_pack}_only"
                candidate_blocks.extend(routed.blocks)
            chosen = self.comparator.choose(
                candidate_blocks,
                expected_script=(
                    region.resolved_script
                    if region.resolved_script
                    in {ScriptType.GURMUKHI, ScriptType.DEVANAGARI, ScriptType.LATIN}
                    else None
                ),
            )
            if region.region_type == RegionType.STAMP_SEAL:
                # Seal graphics are never replaced; only clearly readable text enters the audit.
                chosen = [
                    block
                    for block in chosen
                    if (block.ocr_confidence or 0.0) >= max(0.75, options.ocr_low_confidence_threshold)
                ]
            for block in chosen:
                analyzer.assess(block)
                block.metadata["printed_ocr_probe"] = probe_handwriting
                block.region_visual_script = region.visual_script_candidate
                block.visual_script_confidence = region.visual_script_confidence
                block.resolved_script = region.resolved_script
                block.script_resolution_reason = region.script_resolution_reason
                block.expected_language_prior = region.expected_language_prior
                block.metadata["selected_recognition_engine"] = (
                    region.selected_recognition_engine
                )
                if not probe_handwriting:
                    region.block_ids.append(block.block_id)
            blocks.extend(chosen)
        blocks.sort(
            key=lambda block: (
                int(block.metadata.get("region_reading_order", 0)),
                block.source_bbox.y0,
                block.source_bbox.x0,
            )
        )
        return PrintedOCRResult(blocks, list(dict.fromkeys(warnings)))
