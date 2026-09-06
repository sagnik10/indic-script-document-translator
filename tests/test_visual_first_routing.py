from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from translator_app.config.settings import Settings
from translator_app.core.handwriting_ocr import HandwritingOCRResult
from translator_app.core.language_detector import ScriptAwareLanguageDetector
from translator_app.core.layout_detection import DocumentLayoutDetector
from translator_app.core.page_ocr import PageOCRPipeline
from translator_app.core.ocr_engine import OCREngine, OCRResult
from translator_app.core.printed_ocr import PrintedTextOCR
from translator_app.core.printed_ocr import PrintedOCRResult
from translator_app.core.script_detection import linguistic_evidence_score
from translator_app.core.source_reconstruction import (
    SourceReconstructor,
    SourceSpanPredictionProvider,
    SpanPrediction,
)
from translator_app.core.visual_routing import (
    RegionVisualEvidence,
    VisualScriptClassifier,
    VisualScriptEvidence,
    detect_text_line_boxes,
    suppress_border_noise,
)
from translator_app.schemas import (
    BlockType,
    BoundingBox,
    PageVisualType,
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


class FixedVisualClassifier(VisualScriptClassifier):
    provider_id = "test_visual_gurmukhi"

    def __init__(self, script: ScriptType) -> None:
        self.script = script

    def classify_page(self, image: Image.Image) -> VisualScriptEvidence:
        return VisualScriptEvidence(
            candidate=self.script,
            confidence=0.88,
            handwriting_probability=0.91,
            page_type=PageVisualType.HANDWRITING_HEAVY,
            reason="test pixel classifier",
            provenance=self.provider_id,
            line_boxes=((20, 20, 280, 65),),
            cleaned_image=image.convert("L"),
        )

    def classify_region(
        self, image: Image.Image, page_evidence: VisualScriptEvidence
    ) -> RegionVisualEvidence:
        return RegionVisualEvidence(
            self.script,
            0.86,
            0.93,
            "test region pixels",
            True,
            False,
        )


class WeakRegionVisualClassifier(FixedVisualClassifier):
    """Page evidence is strong while individual line morphology is ambiguous."""

    def __init__(
        self,
        script: ScriptType,
        line_boxes: tuple[tuple[int, int, int, int], ...],
    ) -> None:
        super().__init__(script)
        self.line_boxes = line_boxes

    def classify_page(self, image: Image.Image) -> VisualScriptEvidence:
        evidence = super().classify_page(image)
        cleaned = suppress_border_noise(np.asarray(image.convert("L"))).image
        return VisualScriptEvidence(
            candidate=evidence.candidate,
            confidence=evidence.confidence,
            handwriting_probability=evidence.handwriting_probability,
            page_type=evidence.page_type,
            reason=evidence.reason,
            provenance=evidence.provenance,
            line_boxes=self.line_boxes,
            cleaned_image=Image.fromarray(cleaned),
        )

    def classify_region(
        self, image: Image.Image, page_evidence: VisualScriptEvidence
    ) -> RegionVisualEvidence:
        return RegionVisualEvidence(
            ScriptType.UNKNOWN,
            0.18,
            0.12,
            "ambiguous local pixels",
            True,
            False,
        )

class GarbagePrintedOCR:
    def recognize_regions(self, image, regions, **kwargs) -> PrintedOCRResult:
        blocks = []
        for index, text in enumerate(("|", "I", "£", "Qw", "eee", "?")):
            block = TextBlock(
                1,
                BlockType.LINE,
                BoundingBox(20 + index * 8, 20, 26 + index * 8, 36),
                text,
            )
            block.normalized_text = text
            block.is_ocr = True
            block.ocr_confidence = 0.12
            block.region_type = RegionType.PRINTED_TEXT
            block.metadata["region_id"] = regions[0].region_id
            blocks.append(block)
        return PrintedOCRResult(blocks)


class RecordingUnavailableHTR:
    def __init__(self) -> None:
        self.hints: dict[str, str] = {}
        self.recognition_image: Image.Image | None = None

    def recognize_regions(
        self,
        image,
        regions,
        language_hints,
        *,
        page_number,
        options,
        review_image=None,
    ) -> HandwritingOCRResult:
        self.hints = dict(language_hints)
        self.recognition_image = image.copy()
        blocks = []
        for region in regions:
            if region.region_type != RegionType.HANDWRITING:
                continue
            language = language_hints.get(region.region_id, "und")
            script = (
                ScriptType.GURMUKHI
                if language == "pa"
                else ScriptType.DEVANAGARI
                if language == "hi"
                else ScriptType.UNKNOWN
            )
            region.metadata["htr_route_language"] = language
            region.selected_recognition_engine = f"htr:{language}:{script.value}"
            block = TextBlock(
                page_number,
                BlockType.LINE,
                region.bbox,
                "[unreadable handwriting]",
            )
            block.normalized_text = block.source_text
            block.detected_language = language
            block.script = script
            block.resolved_script = script
            block.resolved_language = language
            block.is_ocr = True
            block.is_handwritten = True
            block.region_type = RegionType.HANDWRITING
            block.ocr_confidence = 0.0
            block.ocr_engine = "htr_unavailable"
            block.uncertainty_state = UncertaintyState.FLAGGED
            block.translation_status = TranslationStatus.SKIPPED
            block.reconstruction_type = ReconstructionType.UNREADABLE
            block.processing_statuses = [
                ProcessingStatus.HTR_UNAVAILABLE,
                ProcessingStatus.HANDWRITING_UNSUPPORTED,
                ProcessingStatus.UNREADABLE,
                ProcessingStatus.TRANSLATION_SKIPPED,
            ]
            block.metadata.update(
                {
                    "region_id": region.region_id,
                    "handwriting_unsupported": True,
                    "htr_unavailable": True,
                }
            )
            blocks.append(block)
        return HandwritingOCRResult(blocks)


def _pipeline(script: ScriptType) -> tuple[PageOCRPipeline, RecordingUnavailableHTR]:
    settings = Settings()
    htr = RecordingUnavailableHTR()
    return (
        PageOCRPipeline(
            DocumentLayoutDetector(settings),
            GarbagePrintedOCR(),
            htr,
            ScriptAwareLanguageDetector(),
            FixedVisualClassifier(script),
        ),
        htr,
    )


def _options(expected: str) -> ProcessingOptions:
    return ProcessingOptions(
        ocr_languages=["eng", "pan", "hin"],
        enable_printed_ocr=True,
        enable_handwriting_ocr=True,
        expected_source_language=expected,
    )


def test_gurmukhi_visual_page_with_ascii_garbage_never_resolves_to_latin() -> None:
    pipeline, htr = _pipeline(ScriptType.GURMUKHI)
    image = Image.new("L", (320, 100), "white")
    result = pipeline.process(
        image,
        image.convert("RGB"),
        page_number=1,
        page_width=320,
        page_height=100,
        options=_options("pa"),
        ocr_variants={"illumination_corrected": image},
    )
    assert result.visual_script_candidate == ScriptType.GURMUKHI
    assert result.ocr_script_candidate == ScriptType.UNKNOWN
    assert result.resolved_script == ScriptType.GURMUKHI
    assert result.page_type == PageVisualType.HANDWRITING_HEAVY
    assert any(region.region_type == RegionType.HANDWRITING for region in result.regions)
    assert set(htr.hints.values()) == {"pa"}
    assert result.punjabi_htr_routes == 1
    assert any(
        ProcessingStatus.HTR_UNAVAILABLE in block.processing_statuses
        for block in result.blocks
    )


def test_hindi_prior_routes_probable_handwriting_to_hindi_htr() -> None:
    pipeline, htr = _pipeline(ScriptType.UNKNOWN)
    image = Image.new("L", (320, 100), "white")
    result = pipeline.process(
        image,
        image.convert("RGB"),
        page_number=1,
        page_width=320,
        page_height=100,
        options=_options("hi"),
        ocr_variants={"clahe_grayscale": image},
    )
    assert result.resolved_script == ScriptType.DEVANAGARI
    assert set(htr.hints.values()) == {"hi"}
    assert result.hindi_htr_routes == 1


def test_explicit_script_routes_do_not_fall_back_to_english_ocr() -> None:
    assert PageOCRPipeline._language_pack_routes(
        ScriptType.GURMUKHI, "pa", ["eng"]
    ) == ["pan"]
    assert PageOCRPipeline._language_pack_routes(
        ScriptType.DEVANAGARI, "hi", ["eng"]
    ) == ["hin"]


def test_one_character_and_random_ascii_have_no_page_script_vote() -> None:
    for fragment in ("|", "I", "£", "?", "eee", "Qw", "ZXQV"):
        assert linguistic_evidence_score(fragment, ScriptType.LATIN) == 0.0
    assert linguistic_evidence_score("Government Hospital", ScriptType.LATIN) > 0.6


def _handwriting_like_page() -> Image.Image:
    image = np.full((900, 700), 240, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (45, 899), 25, -1)
    random = np.random.default_rng(3)
    for y in range(100, 760, 65):
        points = np.array(
            [
                (
                    60 + index * 25 + int(random.integers(-4, 5)),
                    y + int(random.integers(-8, 9)),
                )
                for index in range(22)
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [points], False, 25, 2)
        cv2.line(image, (65, y - 12), (610, y - 8), 30, 1)
    return Image.fromarray(image)


def test_visual_handwriting_and_line_detection_survive_failed_ocr() -> None:
    from translator_app.core.visual_routing import HeuristicVisualScriptClassifier

    settings = Settings()
    classifier = HeuristicVisualScriptClassifier(settings)
    page = _handwriting_like_page()
    evidence = classifier.classify_page(page)
    regions = DocumentLayoutDetector(settings).detect(
        evidence.cleaned_image or page,
        page.convert("RGB"),
        page_number=1,
        page_width=700,
        page_height=900,
        visual_evidence=evidence,
        visual_classifier=classifier,
    )
    handwriting = [region for region in regions if region.region_type == RegionType.HANDWRITING]
    assert evidence.page_type == PageVisualType.HANDWRITING_HEAVY
    assert evidence.handwriting_probability >= settings.handwriting_heavy_threshold
    assert 3 <= len(handwriting) <= 20


def test_dark_photocopy_border_is_masked_and_not_a_text_line() -> None:
    image = np.full((600, 500), 245, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (48, 599), 10, -1)
    cleaned = suppress_border_noise(image)
    boxes, _rejected = detect_text_line_boxes(
        cleaned.image,
        handwriting_heavy=True,
        minimum_width=28,
        minimum_height=9,
        minimum_area=180,
    )
    assert cleaned.zones
    assert cleaned.masked_fraction > 0.05
    assert boxes == []


def test_handwriting_heavy_lines_inherit_page_route_despite_coloured_artifacts() -> None:
    settings = Settings()
    image = Image.new("RGB", (360, 240), (236, 219, 168))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 34, 239), fill=(85, 54, 21))
    for y in (45, 95, 145):
        draw.line((55, y, 315, y + 4), fill=(25, 25, 25), width=3)
        for x in range(60, 300, 28):
            draw.arc((x, y - 10, x + 18, y + 13), 20, 320, fill=(25, 25, 25), width=2)
    classifier = WeakRegionVisualClassifier(
        ScriptType.GURMUKHI,
        ((50, 30, 325, 65), (50, 80, 325, 115), (50, 130, 325, 165)),
    )
    evidence = classifier.classify_page(image)
    regions = DocumentLayoutDetector(settings).detect(
        image.convert("L"),
        image,
        page_number=1,
        page_width=360,
        page_height=240,
        visual_evidence=evidence,
        visual_classifier=classifier,
    )
    handwriting = [region for region in regions if region.region_type == RegionType.HANDWRITING]
    assert len(handwriting) == 3
    assert not any(region.region_type == RegionType.STAMP_SEAL for region in regions)
    assert all(
        region.metadata["classification_reason"].startswith("ambiguous text-like")
        for region in handwriting
    )


def test_verified_stamp_cannot_suppress_handwriting_fallback() -> None:
    settings = Settings()
    image = Image.new("RGB", (360, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((245, 145, 325, 225), outline=(190, 15, 35), width=7)
    draw.ellipse((257, 157, 313, 213), outline=(190, 15, 35), width=3)
    classifier = WeakRegionVisualClassifier(ScriptType.GURMUKHI, ())
    evidence = classifier.classify_page(image)
    regions = DocumentLayoutDetector(settings).detect(
        image.convert("L"),
        image,
        page_number=1,
        page_width=360,
        page_height=240,
        visual_evidence=evidence,
        visual_classifier=classifier,
    )
    assert any(region.region_type == RegionType.STAMP_SEAL for region in regions)
    assert any(region.region_type == RegionType.HANDWRITING for region in regions)


def test_coloured_handwritten_glyphs_are_not_promoted_to_stamps() -> None:
    settings = Settings()
    image = Image.new("RGB", (900, 600), (238, 225, 196))
    draw = ImageDraw.Draw(image)
    ink = (82, 45, 30)
    # Irregular, locally saturated pen strokes with loops resemble the false
    # stamp candidates seen on Test1.jpeg, but they have no compact seal outline.
    for x, y in ((160, 220), (300, 210), (465, 225), (640, 215)):
        draw.arc((x, y, x + 35, y + 58), 30, 330, fill=ink, width=5)
        draw.line((x + 20, y + 4, x + 12, y + 62), fill=ink, width=4)
    detector = DocumentLayoutDetector(settings)
    stamps = detector._stamp_regions(
        np.asarray(image),
        page_width=900,
        page_height=600,
        handwriting_heavy=True,
    )
    assert stamps == []


class ForbiddenProtectedRegionOCR(OCREngine):
    def recognize(self, image: Image.Image, **kwargs) -> OCRResult:
        raise AssertionError("protected graphics must not be sent to printed OCR")


def test_stamp_signature_and_graphic_regions_are_excluded_from_printed_ocr() -> None:
    regions = [
        Region(1, BoundingBox(10, 10, 60, 60), region_type)
        for region_type in (
            RegionType.STAMP_SEAL,
            RegionType.SIGNATURE,
            RegionType.GRAPHICAL_CONTENT,
        )
    ]
    result = PrintedTextOCR(Settings(), ForbiddenProtectedRegionOCR()).recognize_regions(
        Image.new("L", (100, 100), "white"),
        regions,
        page_number=1,
        options=_options("pa"),
    )
    assert result.blocks == []


def test_htr_fallback_masks_protected_graphic_pixels() -> None:
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 60, 60), fill="red")
    stamp = Region(
        1,
        BoundingBox(20, 20, 60, 60),
        RegionType.STAMP_SEAL,
        metadata={"pixel_bbox": [20, 20, 60, 60]},
    )
    signature = Region(
        1,
        BoundingBox(65, 65, 90, 80),
        RegionType.SIGNATURE,
        metadata={"pixel_bbox": [65, 65, 90, 80]},
    )
    masked = PageOCRPipeline._mask_protected_regions_for_htr(
        image, [stamp, signature]
    )
    assert masked.getpixel((40, 40)) == (255, 255, 255)
    assert masked.getpixel((70, 70)) == (255, 255, 255)


def test_page_pipeline_routes_fallback_to_htr_and_never_ocr_stamp() -> None:
    settings = Settings()
    image = Image.new("RGB", (360, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((245, 145, 325, 225), outline=(190, 15, 35), width=7)
    draw.ellipse((257, 157, 313, 213), outline=(190, 15, 35), width=3)
    htr = RecordingUnavailableHTR()
    pipeline = PageOCRPipeline(
        DocumentLayoutDetector(settings),
        PrintedTextOCR(settings, ForbiddenProtectedRegionOCR()),
        htr,
        ScriptAwareLanguageDetector(),
        WeakRegionVisualClassifier(ScriptType.GURMUKHI, ()),
    )
    result = pipeline.process(
        image.convert("L"),
        image,
        page_number=1,
        page_width=360,
        page_height=240,
        options=_options("pa"),
        ocr_variants={"illumination_corrected": image.convert("L")},
    )
    assert result.punjabi_htr_routes == 1
    assert result.printed_ocr_routes == 0
    assert result.detected_text_line_count == 1
    assert set(htr.hints.values()) == {"pa"}
    stamp = next(
        region for region in result.regions if region.region_type == RegionType.STAMP_SEAL
    )
    assert stamp.selected_recognition_engine == "excluded:protected_graphic"
    assert stamp.resolved_script == ScriptType.UNKNOWN
    assert stamp.metadata["recognition_excluded"] is True


def test_page_pipeline_uses_border_cleaned_image_for_htr() -> None:
    settings = Settings()
    source = Image.new("L", (320, 120), "white")
    ImageDraw.Draw(source).rectangle((0, 0, 34, 119), fill="black")
    htr = RecordingUnavailableHTR()
    pipeline = PageOCRPipeline(
        DocumentLayoutDetector(settings),
        PrintedTextOCR(settings, ForbiddenProtectedRegionOCR()),
        htr,
        ScriptAwareLanguageDetector(),
        WeakRegionVisualClassifier(ScriptType.GURMUKHI, ()),
    )
    pipeline.process(
        source,
        source.convert("RGB"),
        page_number=1,
        page_width=320,
        page_height=120,
        options=_options("pa"),
        ocr_variants={"illumination_corrected": source},
    )
    assert htr.recognition_image is not None
    assert htr.recognition_image.getpixel((10, 60)) == 255


class ForbiddenReconstructionProvider(SourceSpanPredictionProvider):
    def __init__(self) -> None:
        self.called = False

    def predict_span(self, masked_source_text: str, **kwargs) -> SpanPrediction:
        self.called = True
        return SpanPrediction("ਸ਼ਬਦ", 0.99, "forbidden")


def test_missing_span_reconstruction_does_not_run_without_source_context() -> None:
    provider = ForbiddenReconstructionProvider()
    block = TextBlock(
        1,
        BlockType.LINE,
        BoundingBox(0, 0, 200, 30),
        "[missing]",
    )
    block.normalized_text = block.source_text
    block.detected_language = "pa"
    block.script = ScriptType.GURMUKHI
    block.is_ocr = True
    block.ocr_confidence = 0.0
    block.metadata["handwriting_unsupported"] = True
    SourceReconstructor(Settings(), provider).evaluate(block)
    assert not provider.called
    assert block.reconstruction_type == ReconstructionType.UNREADABLE
