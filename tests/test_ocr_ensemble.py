from translator_app.core.ocr_ensemble import OCRCandidateComparator
from translator_app.schemas import BlockType, BoundingBox, TextBlock


def candidate(text: str, confidence: float, engine: str) -> TextBlock:
    block = TextBlock(1, BlockType.LINE, BoundingBox(10, 10, 180, 30), text)
    block.ocr_confidence = confidence
    block.ocr_engine = engine
    block.is_ocr = True
    return block


def test_candidate_comparison_keeps_alternatives_and_source_engine() -> None:
    chosen = OCRCandidateComparator().choose(
        [
            candidate("ਪੰਜਾਬ ਸਰਕਾਰ", 0.79, "tesseract_psm_6"),
            candidate("ਪੰਜਾਬ ਸਰਕਾਰ", 0.71, "tesseract_psm_11"),
            candidate("ਪਜਾਬ ਸਰਕਾਰ", 0.58, "tesseract_threshold_variant"),
        ]
    )
    assert len(chosen) == 1
    assert chosen[0].source_text == "ਪੰਜਾਬ ਸਰਕਾਰ"
    assert chosen[0].ocr_engine == "tesseract_psm_6"
    assert len(chosen[0].ocr_alternatives) == 2
    assert chosen[0].metadata["ensemble_candidate_count"] == 3


def test_nonoverlapping_low_confidence_line_is_not_discarded() -> None:
    first = candidate("first", 0.2, "engine_a")
    second = candidate("unclear second", 0.1, "engine_a")
    second.source_bbox = BoundingBox(10, 60, 180, 80)
    selected = OCRCandidateComparator().choose([first, second])
    assert {block.source_text for block in selected} == {"first", "unclear second"}

