from translator_app.core.context_engine import ConservativeReconstructor
from translator_app.schemas import BlockType, BoundingBox, TextBlock, UncertaintyState


def block(text: str, confidence: float) -> TextBlock:
    item = TextBlock(1, BlockType.LINE, BoundingBox(0, 0, 100, 20), text)
    item.ocr_confidence = confidence
    item.is_ocr = True
    return item


def test_safe_line_break_repair_is_accepted() -> None:
    item = block("transla-\ntion", 0.4)
    decision = ConservativeReconstructor(0.65, 0.82).evaluate(item)
    assert decision.accepted
    assert item.reconstructed_text == "translation"
    assert item.uncertainty_state == UncertaintyState.RECONSTRUCTED
    assert item.effective_source_text == "translation"


def test_low_confidence_without_candidate_stays_flagged() -> None:
    item = block("unclear", 0.2)
    decision = ConservativeReconstructor(0.65, 0.82).evaluate(item)
    assert not decision.accepted
    assert item.reconstructed_text is None
    assert item.uncertainty_state == UncertaintyState.FLAGGED
    assert item.effective_source_text == "unclear"


def test_candidate_below_acceptance_threshold_does_not_replace_source() -> None:
    item = block("transla-\ntion", 0.4)
    decision = ConservativeReconstructor(0.65, 0.99).evaluate(item)
    assert not decision.accepted
    assert item.reconstructed_text == "translation"
    assert item.effective_source_text == "transla-\ntion"

