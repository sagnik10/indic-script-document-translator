"""Script-aware block and token language detection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..schemas import DocumentModel, ScriptType, TextBlock
from .script_detection import SCRIPT_RANGES, char_script, script_counts
from .source_validation import LanguageResolution, PageLanguageContext, resolve_language

LOGGER = logging.getLogger(__name__)


SCRIPT_TO_LANGUAGE = {
    "bengali": "bn",
    "gurmukhi": "pa",
    "gujarati": "gu",
    "odia": "or",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "arabic": "ur",
    "latin": "en",
}


@dataclass(frozen=True, slots=True)
class LanguageDetection:
    language: str
    confidence: float
    scripts: tuple[str, ...]
    mixed: bool = False


class ScriptAwareLanguageDetector:
    """Use deterministic Unicode scripts and an optional classifier for ambiguity."""

    def __init__(self) -> None:
        self._classifier = None
        try:
            import langid

            self._classifier = langid
        except ImportError:
            LOGGER.info("langid is unavailable; ambiguous scripts use deterministic defaults")

    def _classify_ambiguous(self, text: str, candidates: set[str], fallback: str) -> str:
        if not self._classifier or len(text.strip()) < 4:
            return fallback
        try:
            language, _score = self._classifier.classify(text)
            return language if language in candidates else fallback
        except Exception:
            LOGGER.debug("Optional language classifier failed", exc_info=True)
            return fallback

    def detect(self, text: str) -> LanguageDetection:
        counts = script_counts(text)
        total = sum(counts.values())
        if not total:
            return LanguageDetection("und", 0.0, ())
        script, count = counts.most_common(1)[0]
        # In a mixed Indic/English block, route the Indic content through translation
        # and protect English spans. Otherwise a short Indic phrase beside a long
        # identifier or English label would be skipped entirely.
        non_latin = [(name, value) for name, value in counts.most_common() if name != "latin"]
        if script == "latin" and non_latin and non_latin[0][1] / total >= 0.2:
            script, count = non_latin[0]
        if script == "devanagari":
            language = self._classify_ambiguous(text, {"hi", "mr", "ne", "sa"}, "hi")
        elif script == "bengali":
            language = self._classify_ambiguous(text, {"bn", "as"}, "bn")
        else:
            language = SCRIPT_TO_LANGUAGE.get(script, "und")
        significant = tuple(
            name for name, value in counts.most_common() if value / total >= 0.12
        )
        mixed = len(significant) > 1
        confidence = min(0.99, count / total + (0.08 if count >= 4 else 0.0))
        return LanguageDetection(language, confidence, significant, mixed)

    def detect_spans(self, text: str) -> list[dict[str, str]]:
        """Return coarse language spans for mixed-language preservation."""
        spans: list[dict[str, str]] = []
        for part in re.findall(r"\s+|[^\s]+", text):
            if part.isspace():
                language = spans[-1]["language"] if spans else "und"
            else:
                language = self.detect(part).language
            if spans and spans[-1]["language"] == language:
                spans[-1]["text"] += part
            else:
                spans.append({"text": part, "language": language})
        return spans

    def annotate_block(self, block: TextBlock) -> None:
        if block.metadata.get("handwriting_unsupported"):
            language = block.detected_language
            block.script = {
                "pa": ScriptType.GURMUKHI,
                "hi": ScriptType.DEVANAGARI,
                "en": ScriptType.LATIN,
            }.get(language, ScriptType.UNKNOWN)
            block.metadata["scripts"] = (
                [block.script.value] if block.script != ScriptType.UNKNOWN else []
            )
            block.metadata["mixed_language"] = False
            block.metadata["language_spans"] = []
            return
        detection = self.detect(block.normalized_text or block.source_text)
        block.detected_language = detection.language
        block.language_confidence = detection.confidence
        block.metadata["scripts"] = list(detection.scripts)
        block.metadata["mixed_language"] = detection.mixed
        block.metadata["language_spans"] = self.detect_spans(
            block.normalized_text or block.source_text
        )
        script_map = {
            "gurmukhi": ScriptType.GURMUKHI,
            "devanagari": ScriptType.DEVANAGARI,
            "latin": ScriptType.LATIN,
        }
        if detection.mixed:
            block.script = ScriptType.MIXED
        elif detection.scripts:
            block.script = script_map.get(detection.scripts[0], ScriptType.UNKNOWN)
        elif any(character.isdigit() for character in block.source_text):
            block.script = ScriptType.DIGITS
        else:
            block.script = ScriptType.UNKNOWN

    def annotate_document(self, document: DocumentModel) -> None:
        for block in document.blocks:
            self.annotate_block(block)
