from translator_app.core.language_detector import ScriptAwareLanguageDetector
from translator_app.schemas import BlockType, BoundingBox, ScriptType, TextBlock


def test_script_detection_for_supported_languages() -> None:
    detector = ScriptAwareLanguageDetector()
    assert detector.detect("ਪੰਜਾਬੀ ਭਾਸ਼ਾ").language == "pa"
    assert detector.detect("தமிழ் மொழி").language == "ta"
    assert detector.detect("ગુજરાતી ભાષા").language == "gu"
    assert detector.detect("English text").language == "en"
    assert detector.detect("हिंदी भाषा").language in {"hi", "mr", "ne", "sa"}


def test_mixed_language_metadata_includes_spans() -> None:
    detector = ScriptAwareLanguageDetector()
    block = TextBlock(1, BlockType.LINE, BoundingBox(0, 0, 100, 20), "यह report है")
    detector.annotate_block(block)
    assert block.metadata["mixed_language"] is True
    languages = {span["language"] for span in block.metadata["language_spans"]}
    assert "en" in languages
    assert block.detected_language in {"hi", "mr", "ne", "sa"}
    assert block.script == ScriptType.MIXED


def test_priority_script_routing_distinguishes_punjabi_and_hindi() -> None:
    detector = ScriptAwareLanguageDetector()
    punjabi = TextBlock(1, BlockType.LINE, BoundingBox(0, 0, 100, 20), "ਪੰਜਾਬ ਸਰਕਾਰ")
    hindi = TextBlock(1, BlockType.LINE, BoundingBox(0, 20, 100, 40), "हिंदी पत्र")
    detector.annotate_block(punjabi)
    detector.annotate_block(hindi)
    assert (punjabi.detected_language, punjabi.script) == ("pa", ScriptType.GURMUKHI)
    assert hindi.detected_language in {"hi", "mr", "ne", "sa"}
    assert hindi.script == ScriptType.DEVANAGARI


def test_numbers_only_are_undetermined() -> None:
    assert ScriptAwareLanguageDetector().detect("2026-08-24").language == "und"
