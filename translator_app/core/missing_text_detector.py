"""Detect bounded missing source-language spans without predicting their contents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..config.settings import Settings
from ..schemas import BoundingBox, ProcessingStatus, RegionType, ScriptType, TextBlock
from .script_detection import script_ratio
from .source_validation import calculate_text_quality, readable_character_ratio


EXPLICIT_MISSING_PATTERN = re.compile(
    r"(?:\[\s*(?:missing|illegible|unclear|unreadable)\s*\]|"
    r"\{\s*(?:missing|illegible|unclear)\s*\}|"
    r"<\s*(?:missing|illegible|unclear)\s*>|"
    r"_{2,}|\?{2,}|�+|\uFFFD+)",
    re.IGNORECASE,
)
REPEATED_PUNCTUATION_PATTERN = re.compile(r"([?_*#=~|.])\1{1,}")
MALFORMED_BOUNDARY_PATTERN = re.compile(r"(?<=\w)(?:\s{3,}|[-|]{2,})(?=\w)")
ISOLATED_PARTIAL_WORD_PATTERN = re.compile(r"(?:^|\s)(?:[-~]\S+|\S+[-~])(?:\s|$)")
ABRUPT_SENTENCE_GAP_PATTERN = re.compile(r"\w\s+(?:[?_*�…]+)\s+\w")
WORD_PATTERN = re.compile(r"[^\s\[\]{}<>?,;:!]+", re.UNICODE)

_PROTECTED_LABEL_PATTERN = re.compile(
    r"\b(?:name|father|mother|patient|signature|signed|address|phone|mobile|date|"
    r"case|file|ref(?:erence)?|dispatch|section|fir|dose|dosage|tablet|capsule|"
    r"medicine|drug|amount|rupees?|inr)\b|"
    r"(?:ਨਾਮ|ਪਿਤਾ|ਮਾਤਾ|ਮਰੀਜ਼|ਦਸਤਖਤ|ਪਤਾ|ਫੋਨ|ਮੋਬਾਈਲ|ਮਿਤੀ|ਤਾਰੀਖ|ਕੇਸ|ਫਾਈਲ|"
    r"ਹਵਾਲਾ|ਧਾਰਾ|ਦਵਾਈ|ਖੁਰਾਕ|ਰਕਮ|ਰੁਪਏ|ਸ੍ਰੀ|ਸ਼੍ਰੀ)|"
    r"(?:नाम|पिता|माता|मरीज|रोगी|हस्ताक्षर|पता|फोन|मोबाइल|दिनांक|तारीख|"
    r"केस|फाइल|संदर्भ|धारा|दवा|खुराक|राशि|रुपये|श्री)",
    re.IGNORECASE,
)
_PROTECTED_VALUE_PATTERN = re.compile(
    r"(?:₹|Rs\.?|INR|\$|€|£)|"
    r"\b\d{1,4}(?:[./:-]\d{1,4})+(?:[./:-]\d{1,4})?\b|"
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|kg|ml|mL|IU|units?|tablets?|capsules?)\b|"
    r"\b(?:case|file|ref|fir|section|no)\.?\s*[A-Za-z0-9./-]*\b|"
    r"\+?\d[\d\s().-]{6,}\d",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MissingSpanDetection:
    detected: bool
    start: int | None
    end: int | None
    span_text: str | None
    bbox: BoundingBox | None
    reasons: tuple[str, ...]
    reconstructable: bool
    readable_ratio: float
    context_quality: float
    validated_context_token_count: int
    protected_entity_detected: bool
    expected_script: ScriptType


def _expected_script(block: TextBlock, page_script: ScriptType) -> ScriptType:
    if block.detected_language == "pa":
        return ScriptType.GURMUKHI
    if block.detected_language == "hi":
        return ScriptType.DEVANAGARI
    if block.script in {ScriptType.GURMUKHI, ScriptType.DEVANAGARI}:
        return block.script
    return page_script


def _span_bbox(block: TextBlock, start: int, end: int, text_length: int) -> BoundingBox:
    width = block.source_bbox.width
    x0 = block.source_bbox.x0 + width * start / max(1, text_length)
    x1 = block.source_bbox.x0 + width * end / max(1, text_length)
    if x1 - x0 < 2.0:
        x1 = min(block.source_bbox.x1, x0 + 2.0)
    return BoundingBox(x0, block.source_bbox.y0, x1, block.source_bbox.y1)


def _script_token_count(text: str, script: ScriptType) -> int:
    return sum(
        1
        for token in WORD_PATTERN.findall(text)
        if len(token) >= 2
        and sum(1 for character in token if script_ratio(character, script) == 1.0) >= 2
        and script_ratio(token, script) >= 0.75
    )


def _protected_entity_near_span(
    text: str,
    start: int | None,
    end: int | None,
    block: TextBlock,
) -> bool:
    if block.region_type == RegionType.SIGNATURE:
        return True
    if start is None or end is None:
        window = text
    else:
        window = text[max(0, start - 80) : min(len(text), end + 80)]
    damaged_structured_value = False
    if start is not None and end is not None:
        left = text[max(0, start - 16) : start]
        right = text[end : min(len(text), end + 16)]
        damaged_structured_value = bool(
            (re.search(r"\d[\d./:-]*$", left) and re.match(r"[\d./:-]*\d", right))
            or re.search(r"(?:₹|Rs\.?|INR|\$|€|£)\s*\S*$", left, re.IGNORECASE)
        )
    return bool(
        _PROTECTED_LABEL_PATTERN.search(window)
        or _PROTECTED_VALUE_PATTERN.search(window)
        or damaged_structured_value
        or any(token and token in window for token in block.protected_tokens)
    )


class MissingTextDetector:
    """Find suspicious spans and calculate conservative reconstruction gates."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def detect(
        self,
        block: TextBlock,
        *,
        previous_text: str = "",
        next_text: str = "",
        page_script: ScriptType = ScriptType.UNKNOWN,
    ) -> MissingSpanDetection:
        text = block.normalized_text or block.source_text
        reasons: list[str] = []
        match = EXPLICIT_MISSING_PATTERN.search(text)
        reconstructable = bool(match)
        if match:
            reasons.append("explicit missing/unclear glyph marker")
        punctuation = REPEATED_PUNCTUATION_PATTERN.search(text)
        if punctuation:
            reasons.append("repeated punctuation or placeholder run")
            if match is None and punctuation.group(1) in {"?", "_"}:
                match = punctuation
                reconstructable = True
        malformed = MALFORMED_BOUNDARY_PATTERN.search(text)
        if malformed:
            reasons.append("malformed token boundary")
            if match is None:
                match = malformed
                reconstructable = False
        if ISOLATED_PARTIAL_WORD_PATTERN.search(text):
            reasons.append("isolated partial-word boundary")
        if ABRUPT_SENTENCE_GAP_PATTERN.search(text):
            reasons.append("abrupt sentence-level gap")
        if block.ocr_confidence is not None and block.ocr_confidence < self.settings.ocr_low_confidence_threshold:
            reasons.append("low OCR/HTR confidence")
        expected = _expected_script(block, page_script)
        if expected in {ScriptType.GURMUKHI, ScriptType.DEVANAGARI}:
            source_purity = script_ratio(EXPLICIT_MISSING_PATTERN.sub("", text), expected)
            if source_purity < self.settings.min_source_script_ratio:
                reasons.append("source line is inconsistent with the expected script")
        alternatives = [
            str(item.get("text", "")).strip()
            for item in block.ocr_alternatives
            if str(item.get("text", "")).strip()
        ]
        if alternatives:
            disagreement = min(
                SequenceMatcher(None, text, alternative).ratio()
                for alternative in alternatives
            )
            if disagreement < 0.65:
                reasons.append("OCR candidates disagree materially")
        start, end = (match.start(), match.end()) if match else (None, None)
        span_text = match.group(0) if match else None
        context_text = text
        if match:
            context_text = text[: match.start()] + " " + text[match.end() :]
        combined_context = " ".join(
            value.strip() for value in (previous_text, context_text, next_text) if value.strip()
        )
        readable_ratio = readable_character_ratio(context_text)
        context_quality = calculate_text_quality(combined_context or context_text, expected)
        context_token_count = _script_token_count(context_text, expected)
        if context_token_count < self.settings.min_validated_context_tokens:
            context_token_count += min(
                self.settings.min_validated_context_tokens - context_token_count,
                _script_token_count(" ".join((previous_text, next_text)), expected),
            )
        protected = _protected_entity_near_span(text, start, end, block)
        bbox = _span_bbox(block, start, end, len(text)) if start is not None and end is not None else None
        detection = MissingSpanDetection(
            detected=bool(reasons),
            start=start,
            end=end,
            span_text=span_text,
            bbox=bbox,
            reasons=tuple(dict.fromkeys(reasons)),
            reconstructable=reconstructable and start is not None and end is not None,
            readable_ratio=readable_ratio,
            context_quality=context_quality,
            validated_context_token_count=context_token_count,
            protected_entity_detected=protected,
            expected_script=expected,
        )
        self.annotate_block(block, detection)
        block.metadata["missing_span_previous_text"] = previous_text
        block.metadata["missing_span_next_text"] = next_text
        return detection

    @staticmethod
    def annotate_block(block: TextBlock, detection: MissingSpanDetection) -> None:
        block.missing_span_detected = detection.detected
        block.missing_span_bbox = detection.bbox
        block.readable_character_ratio = detection.readable_ratio
        block.validated_context_token_count = detection.validated_context_token_count
        block.protected_entity_detected = detection.protected_entity_detected
        block.metadata["missing_span_reasons"] = list(detection.reasons)
        block.metadata["missing_span_start"] = detection.start
        block.metadata["missing_span_end"] = detection.end
        block.metadata["missing_span_original"] = detection.span_text
        block.metadata["missing_span_context_quality"] = detection.context_quality
        block.metadata["missing_span_expected_script"] = detection.expected_script.value
        if detection.detected and ProcessingStatus.MISSING_SPAN_DETECTED not in block.processing_statuses:
            block.processing_statuses.append(ProcessingStatus.MISSING_SPAN_DETECTED)


__all__ = [
    "EXPLICIT_MISSING_PATTERN",
    "MissingSpanDetection",
    "MissingTextDetector",
]
