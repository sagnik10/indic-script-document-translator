import io
import time

from PIL import Image, ImageDraw
import pytest

from translator_app.pipeline import DocumentTranslationPipeline
from translator_app.core.translation_engine import TranslationProvider
from translator_app.ui.app import (
    _has_unresolved_htr,
    _requires_review_pause,
    _unresolved_htr_block_ids,
)
from translator_app.schemas import (
    AnalysisResult,
    BlockType,
    BoundingBox,
    ContentKind,
    DocumentModel,
    FileFormat,
    PageModel,
    ProcessingOptions,
    ReconstructionType,
    ProcessingStatus,
    ReconstructionStatus,
    Region,
    RegionType,
    ScriptType,
    TextBlock,
    TranslationStatus,
    UncertaintyState,
)


class _RecordingPunjabiProvider(TranslationProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str, str]] = []

    def supports(self, source_language: str, target_language: str = "en") -> bool:
        return source_language == "pa" and target_language == "en"

    def translate_batch(
        self, texts: list[str], source_language: str, target_language: str = "en"
    ) -> list[str]:
        self.calls.append((texts, source_language, target_language))
        return ["This is Punjabi handwriting." for _text in texts]


class _TranslationModels:
    def __init__(self, provider: TranslationProvider) -> None:
        self.provider = provider

    def get_translation_provider(self) -> TranslationProvider:
        return self.provider


def test_human_review_changes_effective_text_but_preserves_raw_ocr() -> None:
    block = TextBlock(1, BlockType.LINE, BoundingBox(0, 0, 100, 20), "ਪਜਾਬ")
    block.normalized_text = "ਪਜਾਬ"
    block.uncertainty_state = UncertaintyState.FLAGGED
    document = DocumentModel(
        "review",
        "page.pdf",
        FileFormat.PDF,
        ContentKind.SCANNED,
        [PageModel(1, 100, 100, [block])],
        b"%PDF-",
    )
    analysis = AnalysisResult(document, ProcessingOptions(), time.perf_counter())
    pipeline = DocumentTranslationPipeline()
    pipeline.apply_review_edits(analysis, {block.block_id: "ਪੰਜਾਬ"})
    assert block.source_text == "ਪਜਾਬ"
    assert block.effective_source_text == "ਪੰਜਾਬ"
    assert block.reconstruction_type == ReconstructionType.MANUALLY_CONFIRMED
    assert block.reconstruction_status == ReconstructionStatus.MANUALLY_CONFIRMED
    assert block.reconstruction_confidence == 1.0
    assert ProcessingStatus.MANUALLY_CORRECTED in block.processing_statuses


def test_manual_gurmukhi_review_clears_htr_gate_but_preserves_provenance() -> None:
    block = TextBlock(1, BlockType.LINE, BoundingBox(0, 0, 180, 24), "[unreadable handwriting]")
    block.normalized_text = "[unreadable handwriting]"
    block.region_type = RegionType.HANDWRITING
    block.is_handwritten = True
    block.script = ScriptType.GURMUKHI
    block.detected_language = "pa"
    block.ocr_confidence = 0.0
    block.uncertainty_state = UncertaintyState.FLAGGED
    block.reconstruction_type = ReconstructionType.UNREADABLE
    block.metadata.update(
        {
            "handwriting_unsupported": True,
            "htr_unavailable": True,
            "preserve_region_as_image": True,
        }
    )
    block.processing_statuses.extend(
        [ProcessingStatus.HTR_UNAVAILABLE, ProcessingStatus.UNREADABLE]
    )
    page = PageModel(1, 200, 100, [block])
    page.metadata.update({"dominant_script": "gurmukhi", "dominant_script_confidence": 0.95})
    document = DocumentModel(
        "manual-htr",
        "scan.pdf",
        FileFormat.PDF,
        ContentKind.SCANNED,
        [page],
        b"%PDF-",
    )
    analysis = AnalysisResult(document, ProcessingOptions(), time.perf_counter())
    pipeline = DocumentTranslationPipeline()
    pipeline.apply_review_edits(analysis, {block.block_id: "ਇਹ ਪੰਜਾਬੀ ਹੱਥ ਲਿਖਤ ਹੈ"})
    assert block.source_text == "[unreadable handwriting]"
    assert block.effective_source_text == "ਇਹ ਪੰਜਾਬੀ ਹੱਥ ਲਿਖਤ ਹੈ"
    assert block.source_validated
    assert block.detected_language == "pa"
    assert block.uncertainty_state == UncertaintyState.CONFIRMED
    assert block.metadata["source_origin"] == "manual_correction"
    assert block.metadata["htr_unavailable"] is True
    assert "handwriting_unsupported" not in block.metadata
    assert "preserve_region_as_image" not in block.metadata
    assert ProcessingStatus.MANUALLY_CORRECTED in block.processing_statuses
    assert ProcessingStatus.HTR_UNAVAILABLE not in block.processing_statuses
    assert ProcessingStatus.UNREADABLE not in block.processing_statuses


def test_confirmed_htr_unavailable_gurmukhi_reaches_punjabi_translation() -> None:
    block = TextBlock(
        1,
        BlockType.LINE,
        BoundingBox(10, 10, 190, 40),
        "[unreadable handwriting]",
    )
    block.normalized_text = block.source_text
    block.region_type = RegionType.HANDWRITING
    block.is_handwritten = True
    block.is_ocr = True
    block.script = ScriptType.GURMUKHI
    block.detected_language = "pa"
    block.ocr_confidence = 0.0
    block.uncertainty_state = UncertaintyState.FLAGGED
    block.reconstruction_type = ReconstructionType.UNREADABLE
    block.translation_status = TranslationStatus.SKIPPED
    block.metadata.update(
        {
            "htr_unavailable": True,
            "handwriting_unsupported": True,
            "preserve_region_as_image": True,
            "automatic_translation_aborted": True,
        }
    )
    block.processing_statuses.extend(
        [
            ProcessingStatus.HTR_UNAVAILABLE,
            ProcessingStatus.HANDWRITING_UNSUPPORTED,
            ProcessingStatus.UNREADABLE,
            ProcessingStatus.TRANSLATION_SKIPPED,
        ]
    )
    page = PageModel(1, 200, 100, [block])
    page.resolved_page_script = ScriptType.GURMUKHI
    page.resolved_page_script_confidence = 0.95
    page.metadata.update(
        {"dominant_script": "gurmukhi", "dominant_script_confidence": 0.95}
    )
    document = DocumentModel(
        "manual-htr-translation",
        "scan.png",
        FileFormat.PNG,
        ContentKind.SCANNED,
        [page],
        b"source",
    )
    analysis = AnalysisResult(
        document,
        ProcessingOptions(review_before_render=False),
        time.perf_counter(),
    )
    provider = _RecordingPunjabiProvider()
    pipeline = DocumentTranslationPipeline()
    pipeline.models = _TranslationModels(provider)  # type: ignore[assignment]
    confirmed_source = (
        "\u0a87\u0a39 \u0a2a\u0a70\u0a1c\u0a3e\u0a2c\u0a40 "
        "\u0a39\u0a71\u0a25 \u0a32\u0a3f\u0a16\u0a24 \u0a39\u0a48"
    )

    assert _has_unresolved_htr(analysis)
    assert _unresolved_htr_block_ids(analysis) == {block.block_id}
    assert _requires_review_pause(analysis), "HTR_UNAVAILABLE must override an unchecked review option"
    pipeline.apply_review_edits(analysis, {block.block_id: confirmed_source})
    assert not _has_unresolved_htr(analysis)
    assert not _requires_review_pause(analysis)
    pipeline._translate(analysis, None)

    assert block.source_text == "[unreadable handwriting]"
    assert block.effective_source_text == confirmed_source
    assert block.source_validated
    assert block.detected_language == "pa"
    assert block.translation_status == TranslationStatus.TRANSLATED
    assert block.english_translation == "This is Punjabi handwriting."
    assert provider.calls == [([confirmed_source], "pa", "en")]
    assert "automatic_translation_aborted" not in block.metadata


def test_single_manual_gurmukhi_line_translates_and_renders_in_place() -> None:
    fitz = pytest.importorskip("pymupdf")
    source_image = Image.new("RGB", (360, 200), "white")
    ImageDraw.Draw(source_image).text((30, 72), "source handwriting line", fill="black")
    source_stream = io.BytesIO()
    source_image.save(source_stream, format="PNG")
    source_bytes = source_stream.getvalue()
    bbox = BoundingBox(20, 55, 335, 105)
    block = TextBlock(
        1,
        BlockType.LINE,
        bbox,
        "[unreadable handwriting]",
    )
    block.normalized_text = block.source_text
    block.region_type = RegionType.HANDWRITING
    block.is_handwritten = True
    block.is_ocr = True
    block.script = ScriptType.GURMUKHI
    block.detected_language = "pa"
    block.ocr_confidence = 0.0
    block.uncertainty_state = UncertaintyState.FLAGGED
    block.reconstruction_type = ReconstructionType.UNREADABLE
    block.translation_status = TranslationStatus.SKIPPED
    block.metadata.update(
        {
            "htr_unavailable": True,
            "handwriting_unsupported": True,
            "preserve_region_as_image": True,
            "automatic_translation_aborted": True,
        }
    )
    block.processing_statuses.extend(
        [
            ProcessingStatus.HTR_UNAVAILABLE,
            ProcessingStatus.HANDWRITING_UNSUPPORTED,
            ProcessingStatus.UNREADABLE,
            ProcessingStatus.TRANSLATION_SKIPPED,
        ]
    )
    region = Region(
        1,
        bbox,
        RegionType.HANDWRITING,
        preserve_as_image=True,
        block_ids=[block.block_id],
    )
    block.metadata["region_id"] = region.region_id
    page = PageModel(
        1,
        360,
        200,
        [block],
        [region],
        image_bytes=source_bytes,
        content_kind=ContentKind.SCANNED,
    )
    page.resolved_page_script = ScriptType.GURMUKHI
    page.resolved_page_script_confidence = 0.95
    page.metadata.update(
        {"dominant_script": "gurmukhi", "dominant_script_confidence": 0.95}
    )
    document = DocumentModel(
        "manual-htr-render",
        "scan.png",
        FileFormat.PNG,
        ContentKind.SCANNED,
        [page],
        source_bytes,
    )
    analysis = AnalysisResult(
        document,
        ProcessingOptions(review_before_render=True),
        time.perf_counter(),
    )
    provider = _RecordingPunjabiProvider()
    pipeline = DocumentTranslationPipeline()
    pipeline.models = _TranslationModels(provider)  # type: ignore[assignment]
    confirmed_source = (
        "\u0a87\u0a39 \u0a2a\u0a70\u0a1c\u0a3e\u0a2c\u0a40 "
        "\u0a39\u0a71\u0a25 \u0a32\u0a3f\u0a16\u0a24 \u0a39\u0a48"
    )

    pipeline.apply_review_edits(analysis, {block.block_id: confirmed_source})
    pipeline._translate(analysis, None)
    output_bytes, extension, _mime = pipeline.reconstruction_engine.rebuild(document)

    assert extension == "pdf"
    with fitz.open(stream=output_bytes, filetype="pdf") as output:
        assert output.page_count == 1
        assert "This is Punjabi handwriting." in output[0].get_text()
        assert output[0].rect == fitz.Rect(0, 0, 360, 200)
    assert block.translation_status == TranslationStatus.TRANSLATED
    assert block.metadata["replacement_applied"] is True
    assert block.metadata["primary_output_replacement_reason"] == (
        "validated_in_place_replacement"
    )


def test_manual_missing_span_confirmation_cannot_rewrite_surrounding_source() -> None:
    source = "ਮੈਂ ਅੱਜ [missing] ਨੂੰ ਜਾ ਰਿਹਾ ਹਾਂ"
    block = TextBlock(1, BlockType.LINE, BoundingBox(0, 0, 240, 24), source)
    block.normalized_text = source
    block.script = ScriptType.GURMUKHI
    block.detected_language = "pa"
    block.language_confidence = 0.95
    block.ocr_confidence = 0.82
    block.is_ocr = True
    block.missing_span_detected = True
    block.reconstruction_candidate = "ਸਕੂਲ"
    block.reconstruction_confidence = 0.78
    block.reconstruction_status = ReconstructionStatus.CANDIDATE_REVIEW
    block.reconstruction_type = ReconstructionType.MODEL_INFERRED
    start = source.index("[missing]")
    block.metadata.update(
        {
            "missing_span_start": start,
            "missing_span_end": start + len("[missing]"),
            "proposed_reconstructed_source_text": "ਮੈਂ ਅੱਜ ਸਕੂਲ ਨੂੰ ਜਾ ਰਿਹਾ ਹਾਂ",
        }
    )
    page = PageModel(1, 300, 100, [block])
    page.metadata.update({"dominant_script": "gurmukhi", "dominant_script_confidence": 0.95})
    document = DocumentModel(
        "missing-review",
        "scan.pdf",
        FileFormat.PDF,
        ContentKind.SCANNED,
        [page],
        b"%PDF-",
    )
    analysis = AnalysisResult(document, ProcessingOptions(), time.perf_counter())
    pipeline = DocumentTranslationPipeline()

    pipeline.apply_review_edits(
        analysis,
        {block.block_id: "ਅਸੀਂ ਅੱਜ ਸਕੂਲ ਨੂੰ ਜਾ ਰਿਹੇ ਹਾਂ"},
    )
    assert block.reconstruction_status == ReconstructionStatus.CANDIDATE_REVIEW
    assert any("outside the bounded span changed" in warning for warning in document.warnings)

    pipeline.apply_review_edits(
        analysis,
        {block.block_id: "ਮੈਂ ਅੱਜ ਸਕੂਲ ਨੂੰ ਜਾ ਰਿਹਾ ਹਾਂ"},
    )
    assert block.source_text == source
    assert block.reconstruction_candidate == "ਸਕੂਲ"
    assert block.reconstruction_status == ReconstructionStatus.MANUALLY_CONFIRMED
    assert block.reconstruction_type == ReconstructionType.MANUALLY_CONFIRMED
    assert block.source_validated
    assert block.detected_language == "pa"
