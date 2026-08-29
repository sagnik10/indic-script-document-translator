"""Deterministic Unicode-script primitives shared by OCR routing and validation."""

from __future__ import annotations

from collections import Counter
import re

from ..schemas import ScriptType


SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "bengali": ((0x0980, 0x09FF),),
    "gurmukhi": ((0x0A00, 0x0A7F),),
    "gujarati": ((0x0A80, 0x0AFF),),
    "odia": ((0x0B00, 0x0B7F),),
    "tamil": ((0x0B80, 0x0BFF),),
    "telugu": ((0x0C00, 0x0C7F),),
    "kannada": ((0x0C80, 0x0CFF),),
    "malayalam": ((0x0D00, 0x0D7F),),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "latin": ((0x0041, 0x005A), (0x0061, 0x007A)),
}


def char_script(character: str) -> str | None:
    codepoint = ord(character)
    for script, ranges in SCRIPT_RANGES.items():
        if any(start <= codepoint <= end for start, end in ranges):
            return script
    return None


def script_counts(text: str) -> Counter[str]:
    return Counter(
        script for character in text if (script := char_script(character)) is not None
    )


def script_ratio(text: str, script: str | ScriptType) -> float:
    """Return a script's share of recognized alphabetic Unicode characters."""
    name = script.value if isinstance(script, ScriptType) else str(script).casefold()
    counts = script_counts(text)
    total = sum(counts.values())
    return counts.get(name, 0) / total if total else 0.0


def dominant_script(text: str) -> tuple[str, float]:
    counts = script_counts(text)
    total = sum(counts.values())
    if not total:
        return "unknown", 0.0
    name, count = counts.most_common(1)[0]
    return name, count / total


_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_GARBAGE_LATIN = re.compile(r"^(?:[il|]+|e{1,4}|q+w?|t+t*|x+x*)$", re.IGNORECASE)


def linguistic_evidence_score(
    text: str,
    expected_script: str | ScriptType | None = None,
) -> float:
    """Score meaningful language evidence, not merely Unicode membership.

    Isolated glyphs, currency symbols, punctuation, and common OCR fragments
    intentionally contribute zero or negligible page-level script evidence.
    """
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return 0.0
    counts = script_counts(value)
    total_letters = sum(counts.values())
    if total_letters == 0:
        return 0.0
    name, purity = dominant_script(value)
    expected = (
        expected_script.value
        if isinstance(expected_script, ScriptType)
        else str(expected_script or name).casefold()
    )
    expected_count = counts.get(expected, 0)
    if expected_count == 0:
        return 0.0
    punctuation = sum(
        not character.isalnum() and not character.isspace() for character in value
    )
    visible = sum(not character.isspace() for character in value)
    punctuation_ratio = punctuation / max(1, visible)
    if expected == "latin":
        words = _LATIN_WORD.findall(value)
        letters = "".join(words)
        if len(letters) < 4 or not words:
            return 0.0
        if len(words) == 1 and _GARBAGE_LATIN.fullmatch(words[0]):
            return 0.0
        if words and all(_GARBAGE_LATIN.fullmatch(word) for word in words):
            return 0.0
        if len(words) == 1 and not re.search(r"[aeiouy]", words[0], re.IGNORECASE):
            return 0.0
        if len(set(letters.casefold())) <= 1:
            return 0.0
        word_score = min(1.0, len(letters) / 10.0)
        vocabulary_shape = min(1.0, sum(len(word) >= 3 for word in words) / 2.0)
        score = purity * (0.48 * word_score + 0.32 * vocabulary_shape + 0.20)
    else:
        if expected_count < 4:
            return 0.0
        score = purity * (0.55 * min(1.0, expected_count / 10.0) + 0.45)
    if punctuation_ratio > 0.45:
        score *= max(0.0, 1.0 - (punctuation_ratio - 0.35) * 1.6)
    return max(0.0, min(1.0, score))


def meaningful_dominant_script(text: str) -> tuple[ScriptType, float]:
    """Return a script only when its text contains meaningful linguistic evidence."""
    name, purity = dominant_script(text)
    script = {
        "gurmukhi": ScriptType.GURMUKHI,
        "devanagari": ScriptType.DEVANAGARI,
        "latin": ScriptType.LATIN,
    }.get(name, ScriptType.UNKNOWN)
    if script == ScriptType.UNKNOWN:
        return ScriptType.UNKNOWN, 0.0
    evidence = linguistic_evidence_score(text, script)
    if evidence < 0.20:
        return ScriptType.UNKNOWN, evidence
    return script, min(1.0, evidence * purity)
