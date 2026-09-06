from translator_app.core.context_engine import ConservativeReconstructor
from translator_app.schemas import BlockType, BoundingBox, ReconstructionType, TextBlock


def test_medical_missing_dosage_never_calls_inference_model() -> None:
    calls = []

    def forbidden_loader():
        calls.append(True)
        raise AssertionError("model must not be called")

    block = TextBlock(
        1,
        BlockType.TABLE_CELL,
        BoundingBox(0, 0, 100, 20),
        "Dose [missing] mg",
    )
    block.ocr_confidence = 0.2
    decision = ConservativeReconstructor(
        0.65, 0.82, forbidden_loader, document_domain="medical"
    ).evaluate(block)
    assert not decision.accepted
    assert not calls
    assert block.reconstruction_type == ReconstructionType.UNREADABLE
    assert "prohibited" in decision.reason

