"""Unicode normalization, segmentation, and structured-token protection."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


STRUCTURED_TOKEN_PATTERN = re.compile(
    r"(?:https?://[^\s]+|www\.[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"(?:\+?\d[\d\s().-]{6,}\d)|(?:₹|Rs\.?|INR|\$|€|£)\s?\d[\d,]*(?:\.\d+)?|"
    r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b|"
    r"\b[A-Z]{2,}[A-Z0-9_-]*\d[A-Z0-9_-]*\b|\b\d+(?:[.,:]\d+)*%?\b)",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize Unicode while conservatively cleaning OCR whitespace."""
    normalized = unicodedata.normalize("NFC", text.replace("\x00", ""))
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    return normalized.strip()


def split_sentences(text: str) -> list[str]:
    """Split Latin and Indic sentence terminators without discarding delimiters."""
    return [part.strip() for part in re.split(r"(?<=[.!?।॥])\s+", text) if part.strip()]


def chunk_text(text: str, max_characters: int) -> list[str]:
    """Create sentence-aware chunks, falling back to word boundaries."""
    if len(text) <= max_characters:
        return [text] if text else []
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        pieces = [sentence]
        if len(sentence) > max_characters:
            words = sentence.split()
            pieces, piece = [], ""
            for word in words:
                candidate = f"{piece} {word}".strip()
                if piece and len(candidate) > max_characters:
                    pieces.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                pieces.append(piece)
        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > max_characters:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def structured_tokens(text: str) -> list[str]:
    return [match.group(0) for match in STRUCTURED_TOKEN_PATTERN.finditer(text)]


def protect_structured_tokens(text: str) -> tuple[str, dict[str, str]]:
    """Replace structured values with stable sentinels before translation."""
    mapping: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        key = f"__DTX_TOKEN_{len(mapping):04d}__"
        mapping[key] = match.group(0)
        return key

    return STRUCTURED_TOKEN_PATTERN.sub(replace, text), mapping


def restore_structured_tokens(text: str, mapping: dict[str, str]) -> str:
    restored = text
    for key, value in mapping.items():
        variants: Iterable[str] = (key, key.lower(), key.replace("_", " "))
        replaced = False
        for variant in variants:
            if variant in restored:
                restored = restored.replace(variant, value)
                replaced = True
        if not replaced:
            restored = f"{restored} {value}".strip()
    return restored
