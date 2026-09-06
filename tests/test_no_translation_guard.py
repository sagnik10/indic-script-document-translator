from __future__ import annotations

import time

import pytest

from translator_app.exceptions import NoTranslationProducedError
from translator_app.pipeline import DocumentTranslationPipeline
from translator_app.schemas import (
    AnalysisResult,
    BlockType,
    BoundingBox,
    ContentKind,
    DocumentModel,
    FileFormat,
    PageModel,
    ProcessingOptions,
    ProcessingStatus,
    RegionType,
    ScriptType,
    TextBlock,
    TranslationStatus,
)


class _ForbiddenReconstructionEngine:
    def rebuild(self, document: DocumentModel):  # pragma: no cover - assertion path
        raise AssertionError("an unchanged document must not be rendered as translated")


class _PreservingReconstructionEngine:
    """Simulate a renderer that safely preserves every translated region."""

    def rebuild(self, document: DocumentModel):
        for block in document.blocks:
            block.metadata.update(
                {
                    "replacement_applied": False,
                    "preserved_original": True,
                    "primary_output_replacement_reason": "translation_did_not_fit",
                }
            )
        return b"%PDF-placeholder", "pdf", "application/pdf"


def test_translation_required_guard_stops_unchanged_htr_output_before_rendering() -> None:
    block = TextBlock(
        1,
        BlockType.LINE,
        BoundingBox(10, 10, 190, 35),
        "[unreadable handwriting]",
    )
    block.normalized_text = block.source_text
    block.is_ocr = True
    block.is_handwritten = True
    block.region_type = RegionType.HANDWRITING
    block.script = ScriptType.GURMUKHI
    block.detected_language = "pa"
    block.source_validated = False
    block.translation_status = TranslationStatus.SKIPPED
    block.metadata.update(
        {"htr_unavailable": True, "handwriting_unsupported": True}
    )
    block.processing_statuses.append(ProcessingStatus.HTR_UNAVAILABLE)
    document = DocumentModel(
        "no-translation",
        "scan.jpeg",
        FileFormat.JPEG,
        ContentKind.SCANNED,
        [PageModel(1, 200, 100, [block])],
        b"source",
    )
    analysis = AnalysisResult(
        document,
        ProcessingOptions(review_before_render=True),
        time.perf_counter(),
    )
    pipeline = DocumentTranslationPipeline()
    pipeline.reconstruction_engine = _ForbiddenReconstructionEngine()  # type: ignore[assignment]

    with pytest.raises(NoTranslationProducedError) as captured:
        pipeline.finalize(analysis, require_translation=True)

    assert "No Punjabi/Hindi text was translated" in captured.value.user_message
    assert block.english_translation is None


def test_translation_required_guard_rejects_output_when_renderer_applies_nothing() -> None:
    block = TextBlock(
        1,
        BlockType.LINE,
        BoundingBox(10, 10, 190, 35),
        "ਪੰਜਾਬੀ ਲਾਖ",
    )
    block.normalized_text = block.source_text
    block.detected_language = "pa"
    block.resolved_language = "pa"
    block.source_validated = True
    block.english_translation = "Punjabi text"
    block.translation_status = TranslationStatus.TRANSLATED
    document = DocumentModel(
        "translated-but-not-rendered",
        "scan.jpeg",
        FileFormat.JPEG,
        ContentKind.SCANNED,
        [PageModel(1, 200, 100, [block])],
        b"source",
    )
    analysis = AnalysisResult(
        document,
        ProcessingOptions(review_before_render=True),
        time.perf_counter(),
    )
    pipeline = DocumentTranslationPipeline()
    pipeline._translate = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    pipeline.reconstruction_engine = _PreservingReconstructionEngine()  # type: ignore[assignment]

    with pytest.raises(NoTranslationProducedError) as captured:
        pipeline.finalize(analysis, require_translation=True)

    assert "none of it could be safely placed" in captured.value.user_message
    assert block.metadata["replacement_applied"] is False
    assert document.metadata["applied_translation_replacement_count"] == 0
