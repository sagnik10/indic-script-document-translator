"""Lightweight layout classification and native/OCR block reconciliation."""

from __future__ import annotations

from difflib import SequenceMatcher

from ..schemas import BlockType, TextBlock


def classify_text_block(text: str, font_size: float, page_height: float) -> BlockType:
    stripped = text.strip()
    if not stripped:
        return BlockType.UNKNOWN
    word_count = len(stripped.split())
    if font_size >= 20 and word_count <= 14:
        return BlockType.TITLE
    if font_size >= 14 and word_count <= 24:
        return BlockType.HEADING
    if page_height and word_count <= 18:
        return BlockType.LINE
    return BlockType.PARAGRAPH


def remove_duplicate_ocr_blocks(
    native_blocks: list[TextBlock], ocr_blocks: list[TextBlock]
) -> list[TextBlock]:
    """Avoid duplicate hybrid extraction using geometry plus fuzzy text similarity."""
    kept: list[TextBlock] = []
    for ocr_block in ocr_blocks:
        duplicate = False
        for native_block in native_blocks:
            overlap = ocr_block.source_bbox.intersection_ratio(native_block.source_bbox)
            if overlap < 0.45:
                continue
            similarity = SequenceMatcher(
                None,
                ocr_block.source_text.casefold(),
                native_block.source_text.casefold(),
            ).ratio()
            if similarity >= 0.35 or overlap >= 0.8:
                duplicate = True
                break
        if not duplicate:
            kept.append(ocr_block)
    return kept

