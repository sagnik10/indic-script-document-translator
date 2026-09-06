"""Conservative OCR normalization that retains raw source and correction provenance."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..schemas import ReconstructionType, TextBlock
from ..utils.text_utils import normalize_text


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    text: str
    corrections: tuple[str, ...]


SAFE_CHARACTER_REPLACEMENTS = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "\u00a0": " ",
    "\u200b": "",
    "\ufeff": "",
}


class OCRNormalizer:
    """Apply only reversible Unicode/spacing/line-break corrections, never semantic guesses."""

    def normalize(self, raw_text: str) -> NormalizationResult:
        text = unicodedata.normalize("NFC", raw_text)
        corrections: list[str] = []
        for source, target in SAFE_CHARACTER_REPLACEMENTS.items():
            if source in text:
                text = text.replace(source, target)
                corrections.append(f"replaced Unicode presentation character U+{ord(source):04X}")
        cleaned = normalize_text(text)
        if cleaned != text:
            corrections.append("normalized OCR whitespace")
        # Join a word split by an explicit line-end hyphen; the characters themselves are unchanged.
        joined = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", cleaned)
        if joined != cleaned:
            corrections.append("joined line-end hyphenation")
        return NormalizationResult(joined, tuple(corrections))

    def apply(self, block: TextBlock) -> NormalizationResult:
        result = self.normalize(block.source_text)
        block.normalized_text = result.text
        if result.corrections:
            block.reconstruction_type = ReconstructionType.OCR_CORRECTED
            block.provenance.extend(f"OCR_CORRECTED: {correction}" for correction in result.corrections)
            block.metadata["normalization_corrections"] = list(result.corrections)
            block.metadata["correction_origin"] = "automatic_normalization"
        return result
