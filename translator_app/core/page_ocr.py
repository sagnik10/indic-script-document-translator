"""Visual-first per-page layout, OCR/HTR routing, and reading order."""

from __future__ import annotations

import io
from dataclasses import dataclass, field, replace

from PIL import Image, ImageDraw

from ..schemas import (
    PageVisualType,
    ProcessingOptions,
    Region,
    RegionType,
    ScriptType,
    TextBlock,
)
from .block_grouping import merge_text_blocks
from .handwriting_ocr import HandwritingOCREngine
from .language_detector import ScriptAwareLanguageDetector
from .layout_detection import DocumentLayoutDetector
from .printed_ocr import PrintedTextOCR
from .script_detection import meaningful_dominant_script, script_ratio
from .source_validation import build_page_context, normalize_language
from .visual_routing import (
    RegionVisualEvidence,
    VisualScriptClassifier,
    language_for_script,
    resolve_script_evidence,
)


@dataclass(slots=True)
class PageOCRResult:
    regions: list[Region]
    blocks: list[TextBlock]
    warnings: list[str] = field(default_factory=list)
    dominant_script: ScriptType = ScriptType.UNKNOWN
    dominant_script_confidence: float = 0.0
    visual_script_candidate: ScriptType = ScriptType.UNKNOWN
    visual_script_confidence: float = 0.0
    ocr_script_candidate: ScriptType = ScriptType.UNKNOWN
    ocr_script_confidence: float = 0.0
    resolved_script: ScriptType = ScriptType.UNKNOWN
    script_resolution_reason: str = "not_resolved"
    handwriting_probability: float = 0.0
    page_type: PageVisualType = PageVisualType.UNKNOWN
    detected_text_line_count: int = 0
    punjabi_htr_routes: int = 0
    hindi_htr_routes: int = 0
    printed_ocr_routes: int = 0
    rejected_noise_regions: int = 0


class PageOCRPipeline:
    """Choose recognition routes from pixels/user priors before inspecting OCR text."""

    def __init__(
        self,
        layout_detector: DocumentLayoutDetector,
        printed_ocr: PrintedTextOCR,
        handwriting_ocr: HandwritingOCREngine,
        language_detector: ScriptAwareLanguageDetector,
        visual_script_classifier: VisualScriptClassifier,
    ) -> None:
        self.layout_detector = layout_detector
        self.printed_ocr = printed_ocr
        self.handwriting_ocr = handwriting_ocr
        self.language_detector = language_detector
        self.visual_script_classifier = visual_script_classifier

    @staticmethod
    def _attach_review_crops(
        blocks: list[TextBlock],
        display_image: Image.Image,
        page_width: float,
        page_height: float,
    ) -> None:
        for block in blocks:
            if not block.is_ocr or block.review_image_bytes:
                continue
            x0 = max(0, round(block.source_bbox.x0 * display_image.width / max(1.0, page_width)))
            y0 = max(0, round(block.source_bbox.y0 * display_image.height / max(1.0, page_height)))
            x1 = min(
                display_image.width,
                round(block.source_bbox.x1 * display_image.width / max(1.0, page_width)),
            )
            y1 = min(
                display_image.height,
                round(block.source_bbox.y1 * display_image.height / max(1.0, page_height)),
            )
            if x1 <= x0 or y1 <= y0:
                continue
            stream = io.BytesIO()
            display_image.crop((x0, y0, x1, y1)).save(
                stream, format="PNG", optimize=True
            )
            block.review_image_bytes = stream.getvalue()

    @staticmethod
    def _visual_input(
        fallback: Image.Image,
        variants: dict[str, Image.Image] | None,
    ) -> tuple[str, Image.Image]:
        candidates = variants or {}
        for name in (
            "illumination_corrected",
            "clahe_grayscale",
            "mild_denoise",
            "rectified_grayscale",
            "enhanced_grayscale",
        ):
            if name in candidates:
                return name, candidates[name]
        return "ocr_fallback_grayscale", fallback.convert("L")

    @staticmethod
    def _mask_protected_regions_for_htr(
        image: Image.Image,
        regions: list[Region],
    ) -> Image.Image:
        """Exclude verified graphics from any overlapping handwriting fallback.

        Normally protected regions are never HTR routes.  This mask also covers
        the defensive whole-page handwriting fallback, whose rectangular crop
        can otherwise contain a seal or signature region.
        """
        protected = {
            RegionType.STAMP_SEAL,
            RegionType.SIGNATURE,
            RegionType.GRAPHICAL_CONTENT,
        }
        output = image.copy()
        draw = ImageDraw.Draw(output)
        for region in regions:
            if region.region_type not in protected:
                continue
            pixel_bbox = region.metadata.get("pixel_bbox")
            if not isinstance(pixel_bbox, (list, tuple)) or len(pixel_bbox) != 4:
                continue
            x0, y0, x1, y1 = (int(value) for value in pixel_bbox)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(output.width, x1), min(output.height, y1)
            if x1 > x0 and y1 > y0:
                draw.rectangle((x0, y0, x1, y1), fill="white")
        return output

    @staticmethod
    def _language_pack_routes(
        script: ScriptType,
        expected_language: str,
        installed_options: list[str],
    ) -> list[str]:
        expected = str(expected_language or "auto").casefold()
        if expected == "pa":
            requested = ["pan"]
        elif expected == "hi":
            requested = ["hin"]
        elif expected == "pa+hi":
            requested = ["pan", "hin"]
        else:
            requested = {
                ScriptType.GURMUKHI: ["pan"],
                ScriptType.DEVANAGARI: ["hin"],
                ScriptType.LATIN: ["eng"],
                ScriptType.MIXED: ["pan", "hin"],
            }.get(script, list(installed_options))
        # Explicit/visual routes stay script-specific even if a selected pack is
        # missing so the OCR layer can report the exact installation requirement.
        return requested or list(installed_options)

    @staticmethod
    def _htr_language_hint(
        region: Region,
        page_script: ScriptType,
        expected_language: str,
    ) -> str:
        expected = str(expected_language or "auto").casefold()
        if expected in {"pa", "hi"}:
            return expected
        if region.resolved_script in {ScriptType.GURMUKHI, ScriptType.DEVANAGARI}:
            return language_for_script(region.resolved_script)
        if page_script in {ScriptType.GURMUKHI, ScriptType.DEVANAGARI}:
            return language_for_script(page_script)
        if expected == "pa+hi" and region.visual_script_candidate in {
            ScriptType.GURMUKHI,
            ScriptType.DEVANAGARI,
        }:
            return language_for_script(region.visual_script_candidate)
        return "und"

    @staticmethod
    def _recognized_region_script(
        region: Region,
        blocks: list[TextBlock],
    ) -> tuple[ScriptType, float]:
        evidence = [
            block
            for block in blocks
            if str(block.metadata.get("region_id", "")) == region.region_id
            and not block.metadata.get("handwriting_unsupported")
        ]
        if not evidence:
            return ScriptType.UNKNOWN, 0.0
        joined = " ".join(block.normalized_text or block.source_text for block in evidence)
        return meaningful_dominant_script(joined)

    def process(
        self,
        ocr_image: Image.Image,
        display_image: Image.Image,
        *,
        page_number: int,
        page_width: float,
        page_height: float,
        options: ProcessingOptions,
        ocr_variants: dict[str, Image.Image] | None = None,
        quality_metrics: dict[str, float] | None = None,
    ) -> PageOCRResult:
        visual_variant, visual_image = self._visual_input(ocr_image, ocr_variants)
        visual = self.visual_script_classifier.classify_page(visual_image)
        independent_quality_handwriting = float(
            (quality_metrics or {}).get("handwriting_likelihood", 0.0)
        )
        fused_handwriting = max(
            visual.handwriting_probability,
            independent_quality_handwriting * 0.90,
        )
        if fused_handwriting > visual.handwriting_probability:
            visual = replace(
                visual,
                handwriting_probability=fused_handwriting,
                page_type=(
                    PageVisualType.HANDWRITING_HEAVY
                    if fused_handwriting
                    >= self.layout_detector.settings.handwriting_heavy_threshold
                    else visual.page_type
                ),
                reason=(
                    visual.reason
                    + "; page handwriting score fused from independent connected-component image analysis"
                ),
            )
        initial_page_resolution = resolve_script_evidence(
            visual,
            expected_language=options.expected_source_language,
            minimum_visual_confidence=self.layout_detector.settings.visual_script_min_confidence,
        )
        metrics = dict(quality_metrics or {})
        metrics.update(
            {
                "visual_handwriting_probability": visual.handwriting_probability,
                "visual_script_candidate": visual.candidate.value,
                "visual_script_confidence": visual.confidence,
            }
        )
        regions = self.layout_detector.detect(
            visual.cleaned_image or visual_image,
            display_image,
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            quality_metrics=metrics,
            visual_evidence=visual,
            visual_classifier=self.visual_script_classifier,
        )

        printed_language_hints: dict[str, list[str]] = {}
        for region in regions:
            region.expected_language_prior = options.expected_source_language
            if region.region_type in {
                RegionType.SIGNATURE,
                RegionType.STAMP_SEAL,
                RegionType.GRAPHICAL_CONTENT,
            }:
                region.resolved_script = ScriptType.UNKNOWN
                region.resolved_language = "und"
                region.selected_recognition_engine = "excluded:protected_graphic"
                region.script_resolution_reason = (
                    "protected graphic region excluded from OCR and HTR routing"
                )
                region.metadata.update(
                    {
                        "visual_script_candidate": region.visual_script_candidate.value,
                        "visual_script_confidence": region.visual_script_confidence,
                        "resolved_script": ScriptType.UNKNOWN.value,
                        "resolved_language": "und",
                        "script_resolution_reason": region.script_resolution_reason,
                        "expected_language_prior": options.expected_source_language,
                        "visual_classification_variant": visual_variant,
                        "recognition_excluded": True,
                        "selected_recognition_engine": region.selected_recognition_engine,
                    }
                )
                continue
            region_visual = RegionVisualEvidence(
                candidate=region.visual_script_candidate,
                confidence=region.visual_script_confidence,
                handwriting_probability=float(
                    region.metadata.get(
                        "region_handwriting_probability", visual.handwriting_probability
                    )
                ),
                reason=str(region.metadata.get("visual_script_reason", "visual region")),
                is_textual=True,
                is_noise=False,
            )
            region_resolution = resolve_script_evidence(
                region_visual,
                expected_language=options.expected_source_language,
                minimum_visual_confidence=self.layout_detector.settings.visual_script_min_confidence,
            )
            if region_resolution.script == ScriptType.UNKNOWN:
                region_resolution = initial_page_resolution
            region.resolved_script = region_resolution.script
            region.resolved_language = language_for_script(region_resolution.script)
            region.script_resolution_reason = region_resolution.reason
            region.metadata.update(
                {
                    "visual_script_candidate": region.visual_script_candidate.value,
                    "visual_script_confidence": region.visual_script_confidence,
                    "resolved_script": region.resolved_script.value,
                    "script_resolution_reason": region.script_resolution_reason,
                    "expected_language_prior": options.expected_source_language,
                    "visual_classification_variant": visual_variant,
                }
            )
            if region.region_type not in {
                RegionType.HANDWRITING,
                RegionType.SIGNATURE,
                RegionType.STAMP_SEAL,
                RegionType.GRAPHICAL_CONTENT,
            }:
                printed_language_hints[region.region_id] = self._language_pack_routes(
                    region.resolved_script,
                    options.expected_source_language,
                    options.ocr_languages,
                )

        # This is the dependency break: recognition routes already exist here,
        # before any OCR text has been produced.
        printed = self.printed_ocr.recognize_regions(
            ocr_image,
            regions,
            page_number=page_number,
            options=options,
            variants=ocr_variants,
            language_hints=printed_language_hints,
        )
        for block in printed.blocks:
            self.language_detector.annotate_block(block)

        handwriting_hints = {
            region.region_id: self._htr_language_hint(
                region,
                initial_page_resolution.script,
                options.expected_source_language,
            )
            for region in regions
            if region.region_type == RegionType.HANDWRITING
        }
        # The classifier's cleaned image has binding bands/page-edge noise
        # removed without aggressive thresholding.  Use it for recognition so a
        # dark photocopy border cannot become one page-sized handwriting line;
        # review crops still come from the untouched display image below.
        htr_base_image = visual.cleaned_image or visual_image
        htr_image = self._mask_protected_regions_for_htr(htr_base_image, regions)
        handwriting = self.handwriting_ocr.recognize_regions(
            htr_image,
            regions,
            handwriting_hints,
            page_number=page_number,
            options=options,
            review_image=display_image,
        )
        blocks = merge_text_blocks(
            printed.blocks + handwriting.blocks,
            horizontal_gap_ratio=self.layout_detector.settings.region_merge_horizontal_gap_ratio,
        )
        for block in blocks:
            self.language_detector.annotate_block(block)

        ocr_context = build_page_context(
            blocks,
            handwriting_likelihood=visual.handwriting_probability,
        )
        final_resolution = resolve_script_evidence(
            visual,
            expected_language=options.expected_source_language,
            ocr_script=ocr_context.ocr_script,
            ocr_confidence=ocr_context.ocr_confidence,
            minimum_visual_confidence=self.layout_detector.settings.visual_script_min_confidence,
        )
        final_context = build_page_context(
            blocks,
            handwriting_likelihood=visual.handwriting_probability,
            visual_script=visual.candidate,
            visual_confidence=visual.confidence,
            resolved_script=final_resolution.script,
            resolved_confidence=final_resolution.confidence,
            resolution_reason=final_resolution.reason,
            expected_language_prior=options.expected_source_language,
        )

        for region in regions:
            recognized_script, recognized_confidence = self._recognized_region_script(
                region, blocks
            )
            region.recognized_unicode_script = recognized_script
            region.recognized_unicode_script_confidence = recognized_confidence
            region.linguistic_evidence_score = recognized_confidence
            if (
                region.resolved_script in {ScriptType.UNKNOWN, ScriptType.MIXED}
                and recognized_script != ScriptType.UNKNOWN
                and recognized_confidence >= 0.78
            ):
                region.resolved_script = recognized_script
                region.resolved_language = language_for_script(recognized_script)
                region.script_resolution_reason = (
                    "meaningful validated Unicode evidence refined ambiguous visual routing"
                )
            region.metadata.update(
                {
                    "recognized_unicode_script": recognized_script.value,
                    "recognized_unicode_script_confidence": recognized_confidence,
                    "selected_recognition_engine": region.selected_recognition_engine,
                    "resolved_language": region.resolved_language,
                }
            )

        region_map = {region.region_id: region for region in regions}
        for block in blocks:
            region = region_map.get(str(block.metadata.get("region_id", "")))
            if region is None:
                continue
            block.region_visual_script = region.visual_script_candidate
            block.visual_script_confidence = region.visual_script_confidence
            block.expected_language_prior = options.expected_source_language
            block.recognized_unicode_script, block.ocr_script_confidence = (
                meaningful_dominant_script(block.normalized_text or block.source_text)
            )
            block.linguistic_evidence_score = block.ocr_script_confidence
            block.resolved_script = region.resolved_script
            block.resolved_language = normalize_language(block.detected_language)
            block.script_resolution_reason = region.script_resolution_reason
            block.metadata["unicode_script_ratio"] = (
                script_ratio(block.normalized_text or block.source_text, block.resolved_script)
                if block.resolved_script
                in {ScriptType.GURMUKHI, ScriptType.DEVANAGARI, ScriptType.LATIN}
                else 0.0
            )

        region_order = {region.region_id: region.reading_order for region in regions}
        blocks.sort(
            key=lambda block: (
                region_order.get(str(block.metadata.get("region_id", "")), 10_000),
                block.source_bbox.y0,
                block.source_bbox.x0,
            )
        )
        self._attach_review_crops(blocks, display_image, page_width, page_height)
        punjabi_htr = sum(
            region.metadata.get("htr_route_language") == "pa" for region in regions
        )
        hindi_htr = sum(
            region.metadata.get("htr_route_language") == "hi" for region in regions
        )
        printed_routes = sum(
            region.selected_recognition_engine.startswith("printed_ocr:")
            for region in regions
        )
        # Count the post-grouping logical blocks shown to the reviewer.  The HTR
        # engine can emit multiple horizontal fragments before baseline merging;
        # reporting that internal candidate count made a 28-line review appear
        # as 42 lines and obscured whether segmentation had actually improved.
        logical_line_count = sum(
            block.region_type
            in {
                RegionType.PRINTED_TEXT,
                RegionType.HANDWRITING,
                RegionType.MIXED,
                RegionType.TABLE_FORM,
            }
            for block in blocks
        )

        # Replace pre-merge HTR-unavailable counts with one concise warning based
        # on the exact logical blocks the user will review.
        handwriting_warnings = [
            warning
            for warning in handwriting.warnings
            if " handwritten line(s) were preserved for manual review:" not in warning
        ]
        unavailable_groups: dict[tuple[str, str], int] = {}
        for block in blocks:
            if not block.metadata.get("htr_unavailable"):
                continue
            key = (
                normalize_language(block.detected_language),
                str(
                    block.metadata.get("htr_unavailable_reason")
                    or "no validated source-language HTR model is configured"
                ),
            )
            unavailable_groups[key] = unavailable_groups.get(key, 0) + 1
        for (language, reason), count in sorted(unavailable_groups.items()):
            label = language if language != "und" else "undetermined-language"
            handwriting_warnings.append(
                f"Page {page_number}: {count} {label} handwritten line(s) were preserved "
                f"for manual review: {reason}."
            )
        return PageOCRResult(
            regions=regions,
            blocks=blocks,
            warnings=list(dict.fromkeys(printed.warnings + handwriting_warnings)),
            dominant_script=final_context.dominant_script,
            dominant_script_confidence=final_context.confidence,
            visual_script_candidate=visual.candidate,
            visual_script_confidence=visual.confidence,
            ocr_script_candidate=ocr_context.ocr_script,
            ocr_script_confidence=ocr_context.ocr_confidence,
            resolved_script=final_resolution.script,
            script_resolution_reason=final_resolution.reason,
            handwriting_probability=visual.handwriting_probability,
            page_type=visual.page_type,
            detected_text_line_count=logical_line_count,
            punjabi_htr_routes=punjabi_htr,
            hindi_htr_routes=hindi_htr,
            printed_ocr_routes=printed_routes,
            rejected_noise_regions=visual.rejected_noise_region_count,
        )
