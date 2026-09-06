from __future__ import annotations

from translator_app.config.settings import Settings
from translator_app.core.context_engine import ConservativeReconstructor
from translator_app.core.region_merging import merge_text_regions
from translator_app.core.layout_detection import DocumentLayoutDetector
from translator_app.core.source_validation import (
    PageLanguageContext,
    build_page_context,
    calculate_text_quality,
    normalize_language,
    resolve_language,
    script_ratio,
    validate_source_block,
)
from translator_app.core.translation_engine import TranslationProvider, TranslationService
from translator_app.schemas import (
    BlockType,
    BoundingBox,
    ReconstructionType,
    Region,
    RegionType,
    ScriptType,
    TextBlock,
    TranslationStatus,
)
from PIL import Image, ImageDraw


class RecordingProvider(TranslationProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def supports(self, source_language: str, target_language: str = "en") -> bool:
        return source_language in {"pa", "hi"} and target_language == "en"

    def translate_batch(
        self, texts: list[str], source_language: str, target_language: str = "en"
    ) -> list[str]:
        self.calls.append((texts, source_language))
        return ["generic fluent fallback" for _ in texts]


def _block(text: str, confidence: float = 0.8) -> TextBlock:
    block = TextBlock(1, BlockType.LINE, BoundingBox(0, 0, 120, 20), text)
    block.normalized_text = text
    block.ocr_confidence = confidence
    block.is_ocr = True
    return block


def test_language_aliases_are_canonicalized() -> None:
    for value in ("pa", "pan", "pan_Guru", "Punjabi"):
        assert normalize_language(value) == "pa"
    for value in ("hi", "hin", "hin_Deva", "Hindi"):
        assert normalize_language(value) == "hi"
    for value in (None, "", "und", "unknown", "mul"):
        assert normalize_language(value) == "und"


def test_gurmukhi_page_prior_rejects_short_ascii_ocr_fragments() -> None:
    context = PageLanguageContext(ScriptType.GURMUKHI, 0.91, 0.9)
    for garbage in ("eee", "Qw", "TT?", "Maa", "??"):
        assert resolve_language(garbage, None, context, 0.25).language == "und"


def test_gurmukhi_unicode_routes_to_punjabi_under_page_prior() -> None:
    text = "ਪੰਜਾਬ ਸਰਕਾਰ"
    resolution = resolve_language(
        text,
        ScriptType.GURMUKHI,
        PageLanguageContext(ScriptType.GURMUKHI, 0.94, 0.8),
        0.74,
    )
    assert resolution.language == "pa"
    assert script_ratio(text, ScriptType.GURMUKHI) > 0.9
    assert calculate_text_quality(text, ScriptType.GURMUKHI) >= 0.7


def test_und_and_punctuation_never_reach_translation_provider() -> None:
    provider = RecordingProvider()
    blocks = [_block("Qw", 0.2), _block("??", 0.1), _block("x", 0.9)]
    for block in blocks:
        block.detected_language = "und"
    warnings = TranslationService(provider, Settings(ocr_languages=["pan", "hin", "eng"])).translate_blocks(blocks)
    assert provider.calls == []
    assert all(block.translation_status == TranslationStatus.SKIPPED for block in blocks)
    assert len(warnings) == 1


def test_low_quality_source_does_not_invoke_generative_reconstruction() -> None:
    calls = 0

    def load_model():
        nonlocal calls
        calls += 1
        raise AssertionError("model loader must not be called")

    block = _block("??__??", 0.05)
    decision = ConservativeReconstructor(
        0.65,
        0.82,
        load_model,
        minimum_readable_ratio=0.72,
    ).evaluate(block)
    assert not decision.accepted
    assert block.reconstruction_type == ReconstructionType.UNREADABLE
    assert calls == 0


def test_adjacent_character_boxes_merge_into_one_line_region() -> None:
    regions = [
        Region(1, BoundingBox(10, 20, 20, 35), RegionType.HANDWRITING),
        Region(1, BoundingBox(23, 19, 35, 36), RegionType.HANDWRITING),
        Region(1, BoundingBox(39, 21, 52, 35), RegionType.HANDWRITING),
    ]
    merged = merge_text_regions(regions, horizontal_gap_ratio=2.5)
    assert len(merged) == 1
    assert merged[0].metadata["merged_component_count"] == 3


def test_random_ascii_on_gurmukhi_page_is_marked_unreadable() -> None:
    block = _block("eee TT?", 0.18)
    validation = validate_source_block(
        block,
        Settings(ocr_languages=["pan", "hin", "eng"]),
        PageLanguageContext(ScriptType.GURMUKHI, 0.92, 0.9),
    )
    assert not validation.valid
    assert block.detected_language == "und"
    assert block.translation_status == TranslationStatus.SKIPPED
    assert block.reconstruction_type == ReconstructionType.UNREADABLE


def test_visual_gurmukhi_prior_outweighs_marker_text_and_latin_header() -> None:
    marker = _block("[unreadable handwriting]", 0.0)
    marker.detected_language = "pa"
    marker.is_handwritten = True
    marker.metadata["handwriting_unsupported"] = True
    header = _block("GOVERNMENT", 0.85)
    context = build_page_context(
        [header, marker, marker],
        handwriting_likelihood=0.9,
        visual_script=ScriptType.GURMUKHI,
        visual_confidence=0.91,
        resolved_script=ScriptType.GURMUKHI,
        resolved_confidence=0.91,
        resolution_reason="visual-first Gurmukhi page evidence",
        expected_language_prior="pa",
    )
    assert context.dominant_script == ScriptType.GURMUKHI


def test_handwriting_heavy_page_cannot_silently_report_zero_handwriting_regions() -> None:
    image = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(image)
    for line_index, baseline in enumerate((90, 165, 240, 315)):
        x = 55
        for stroke_index in range(14):
            width = 8 + (stroke_index * 7 + line_index * 5) % 19
            height = 11 + (stroke_index * 11 + line_index * 3) % 24
            draw.line(
                [(x, baseline), (x + width // 2, baseline - height), (x + width, baseline + (stroke_index % 3) * 3)],
                fill="black",
                width=2 + stroke_index % 2,
            )
            x += width + 7 + (stroke_index % 4) * 3
    regions = DocumentLayoutDetector(Settings(ocr_languages=["pan", "eng"])).detect(
        image.convert("L"),
        image,
        page_number=1,
        page_width=600,
        page_height=450,
        quality_metrics={"handwriting_likelihood": 0.92},
    )
    assert any(region.region_type == RegionType.HANDWRITING for region in regions)
