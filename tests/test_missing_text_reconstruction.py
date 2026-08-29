from __future__ import annotations

from translator_app.config.settings import Settings
from translator_app.core.source_reconstruction import (
    SourceReconstructor,
    SourceSpanPredictionProvider,
    SpanPrediction,
)
from translator_app.core.source_validation import (
    PageLanguageContext,
    is_translatable_block,
    validate_source_block,
)
from translator_app.schemas import (
    BlockType,
    BoundingBox,
    ReconstructionStatus,
    ReconstructionType,
    ScriptType,
    TextBlock,
)


class RecordingSourceProvider(SourceSpanPredictionProvider):
    def __init__(self, candidate: str, confidence: float) -> None:
        self.candidate = candidate
        self.confidence = confidence
        self.calls: list[dict[str, object]] = []

    def predict_span(
        self,
        masked_source_text: str,
        *,
        language: str,
        expected_script: ScriptType,
        previous_text: str,
        next_text: str,
        ocr_alternatives: list[dict[str, object]],
    ) -> SpanPrediction:
        self.calls.append(
            {
                "text": masked_source_text,
                "language": language,
                "script": expected_script,
            }
        )
        return SpanPrediction(self.candidate, self.confidence, "test_source_model")


def _block(text: str, language: str, script: ScriptType) -> TextBlock:
    block = TextBlock(1, BlockType.LINE, BoundingBox(0, 0, 300, 30), text)
    block.normalized_text = text
    block.detected_language = language
    block.language_confidence = 0.95
    block.script = script
    block.ocr_confidence = 0.84
    block.is_ocr = True
    return block


def _context(script: ScriptType) -> PageLanguageContext:
    return PageLanguageContext(script, 0.96, 0.2)


def test_missing_punjabi_word_is_auto_accepted_only_as_source_text() -> None:
    settings = Settings()
    provider = RecordingSourceProvider("ਸਕੂਲ", 0.95)
    block = _block(
        "ਮੈਂ ਅੱਜ [missing] ਨੂੰ ਜਾ ਰਿਹਾ ਹਾਂ", "pa", ScriptType.GURMUKHI
    )
    decision = SourceReconstructor(settings, provider).evaluate(
        block, page_context=_context(ScriptType.GURMUKHI)
    )
    assert decision.accepted
    assert block.reconstruction_candidate == "ਸਕੂਲ"
    assert block.reconstructed_text == "ਮੈਂ ਅੱਜ ਸਕੂਲ ਨੂੰ ਜਾ ਰਿਹਾ ਹਾਂ"
    assert block.reconstruction_type == ReconstructionType.MODEL_INFERRED
    assert block.reconstruction_status == ReconstructionStatus.AUTO_ACCEPTED
    assert provider.calls[0]["language"] == "pa"
    assert "<MISSING_SPAN>" in str(provider.calls[0]["text"])
    assert validate_source_block(block, settings, _context(ScriptType.GURMUKHI)).valid
    assert is_translatable_block(block)


def test_missing_hindi_word_between_thresholds_requires_review() -> None:
    settings = Settings()
    provider = RecordingSourceProvider("विद्यालय", 0.78)
    block = _block(
        "मैं आज [missing] को जा रहा हूँ", "hi", ScriptType.DEVANAGARI
    )
    decision = SourceReconstructor(settings, provider).evaluate(
        block, page_context=_context(ScriptType.DEVANAGARI)
    )
    assert not decision.accepted
    assert block.reconstruction_candidate == "विद्यालय"
    assert block.reconstruction_status == ReconstructionStatus.CANDIDATE_REVIEW
    assert block.reconstructed_text is None
    assert not is_translatable_block(block)


def test_protected_number_gap_never_calls_source_model() -> None:
    provider = RecordingSourceProvider("09", 0.99)
    block = _block("ਮਿਤੀ 31.??.2016 ਦਰਜ ਹੈ", "pa", ScriptType.GURMUKHI)
    decision = SourceReconstructor(Settings(), provider).evaluate(
        block, page_context=_context(ScriptType.GURMUKHI)
    )
    assert not decision.accepted
    assert not provider.calls
    assert block.protected_entity_detected
    assert block.reconstruction_status == ReconstructionStatus.BLOCKED
    assert "[unclear]" in block.effective_source_text


def test_low_quality_line_does_not_invoke_reconstruction_provider() -> None:
    provider = RecordingSourceProvider("ਸ਼ਬਦ", 0.99)
    block = _block("?? __ ਪੰਜਾਬੀ", "pa", ScriptType.GURMUKHI)
    block.ocr_confidence = 0.08
    SourceReconstructor(Settings(), provider).evaluate(
        block, page_context=_context(ScriptType.GURMUKHI)
    )
    assert not provider.calls
    assert block.reconstruction_status == ReconstructionStatus.BLOCKED
    assert not is_translatable_block(block)


def test_mixed_script_line_is_not_reconstructed() -> None:
    provider = RecordingSourceProvider("ਦਸਤਾਵੇਜ਼", 0.99)
    block = _block(
        "ਇਹ [missing] document ਵਿੱਚ ਦਰਜ ਹੈ", "pa", ScriptType.GURMUKHI
    )
    SourceReconstructor(Settings(), provider).evaluate(
        block, page_context=_context(ScriptType.GURMUKHI)
    )
    assert not provider.calls
    assert block.reconstruction_status == ReconstructionStatus.BLOCKED


def test_person_name_context_remains_unresolved() -> None:
    provider = RecordingSourceProvider("ਗੁਰਪ੍ਰੀਤ", 0.99)
    block = _block("ਨਾਮ [missing] ਸਿੰਘ ਦਰਜ ਹੈ", "pa", ScriptType.GURMUKHI)
    SourceReconstructor(Settings(), provider).evaluate(
        block, page_context=_context(ScriptType.GURMUKHI)
    )
    assert not provider.calls
    assert block.protected_entity_detected
    assert block.reconstruction_type == ReconstructionType.UNREADABLE


def test_wrong_script_candidate_is_rejected_and_never_becomes_source() -> None:
    provider = RecordingSourceProvider("school", 0.99)
    block = _block(
        "ਮੈਂ ਅੱਜ [missing] ਨੂੰ ਜਾ ਰਿਹਾ ਹਾਂ", "pa", ScriptType.GURMUKHI
    )
    SourceReconstructor(Settings(), provider).evaluate(
        block, page_context=_context(ScriptType.GURMUKHI)
    )
    assert provider.calls
    assert block.reconstruction_status == ReconstructionStatus.REJECTED
    assert block.reconstruction_type == ReconstructionType.UNREADABLE
    assert "school" not in block.effective_source_text
    assert not is_translatable_block(block)


def test_candidate_below_review_threshold_preserves_unclear_marker() -> None:
    provider = RecordingSourceProvider("ਸਕੂਲ", 0.65)
    block = _block(
        "ਮੈਂ ਅੱਜ [missing] ਨੂੰ ਜਾ ਰਿਹਾ ਹਾਂ", "pa", ScriptType.GURMUKHI
    )
    SourceReconstructor(Settings(), provider).evaluate(
        block, page_context=_context(ScriptType.GURMUKHI)
    )
    assert block.reconstruction_status == ReconstructionStatus.REJECTED
    assert "[unclear]" in block.effective_source_text
    assert not is_translatable_block(block)
