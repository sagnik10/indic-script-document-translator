"""Text fitting, safe expansion, and collision detection for translated blocks."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from ..schemas import BoundingBox, LayoutStatus


@dataclass(frozen=True, slots=True)
class TextFitResult:
    text: str
    lines: tuple[str, ...]
    font_size: float
    line_spacing: float
    bbox: BoundingBox
    status: LayoutStatus
    overflow: bool
    font_variant: str = "source"
    source_line_count: int = 1


@dataclass(frozen=True, slots=True)
class LayoutFidelityResult:
    """Internal placement-quality record; never rendered into the primary document."""

    score: float
    bbox_preservation: float
    font_size_preservation: float
    line_count_preservation: float
    placement_preservation: float
    collision_free: bool
    within_page: bool
    replacement_applied: bool
    preserved_original: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "bbox_preservation": round(self.bbox_preservation, 4),
            "font_size_preservation": round(self.font_size_preservation, 4),
            "line_count_preservation": round(self.line_count_preservation, 4),
            "placement_preservation": round(self.placement_preservation, 4),
            "collision_free": self.collision_free,
            "within_page": self.within_page,
            "replacement_applied": self.replacement_applied,
            "preserved_original": self.preserved_original,
            "reason": self.reason,
        }


def _character_width(character: str, font_size: float, width_scale: float = 1.0) -> float:
    if character in " ilI.,'`|!:":
        return font_size * 0.28 * width_scale
    if character in "MW@#%&":
        return font_size * 0.85 * width_scale
    return font_size * 0.54 * width_scale


def _wrap_text(
    text: str,
    width: float,
    font_size: float,
    *,
    width_scale: float = 1.0,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        current_width = 0.0
        for word in words:
            word_width = sum(_character_width(char, font_size, width_scale) for char in word)
            space_width = _character_width(" ", font_size, width_scale) if current else 0.0
            if current and current_width + space_width + word_width > width:
                lines.append(current)
                current, current_width = "", 0.0
            if word_width > width:
                for character in word:
                    character_width = _character_width(character, font_size, width_scale)
                    if current and current_width + character_width > width:
                        lines.append(current)
                        current, current_width = "", 0.0
                    current += character
                    current_width += character_width
            else:
                if current:
                    current += " "
                    current_width += space_width
                current += word
                current_width += word_width
        if current:
            lines.append(current)
    return lines


def collides(candidate: BoundingBox, obstacles: list[BoundingBox], padding: float = 1.5) -> bool:
    return any(candidate.intersects(obstacle, padding=padding) for obstacle in obstacles)


def _safe_expansions(
    bbox: BoundingBox,
    page_width: float,
    page_height: float,
    maximum: float,
) -> list[BoundingBox]:
    steps = max(1, ceil(maximum / 6))
    candidates: list[BoundingBox] = []
    for step in range(1, steps + 1):
        amount = min(maximum, step * 6.0)
        candidates.append(BoundingBox(bbox.x0, bbox.y0, bbox.x1, min(page_height, bbox.y1 + amount)))
        candidates.append(BoundingBox(bbox.x0, bbox.y0, min(page_width, bbox.x1 + amount), bbox.y1))
        candidates.append(
            BoundingBox(
                bbox.x0,
                bbox.y0,
                min(page_width, bbox.x1 + amount / 2),
                min(page_height, bbox.y1 + amount),
            )
        )
    return candidates


def fit_text(
    text: str,
    bbox: BoundingBox,
    *,
    original_font_size: float,
    min_font_size: float,
    page_width: float,
    page_height: float,
    obstacles: list[BoundingBox] | None = None,
    allow_expansion: bool = True,
    maximum_expansion: float = 36.0,
    source_line_count: int | None = None,
    original_line_spacing: float = 1.15,
) -> TextFitResult:
    """Fit text using the fidelity-first fallback order required by the renderer.

    The order is source-sized text, a compact face at the source size, gradual font
    reduction with conservative line tightening, and finally collision-safe local
    expansion. An overflow result never authorizes source-region removal.
    """
    obstacles = obstacles or []
    original = max(min_font_size, original_font_size)
    expected_lines = max(1, source_line_count or len(text.splitlines()) or 1)
    source_spacing = max(0.92, min(1.30, original_line_spacing))

    # Trying a compact metric before shrinking preserves the visual text scale.
    for variant, width_scale in (("source", 1.0), ("compact", 0.88)):
        lines = _wrap_text(text, max(1.0, bbox.width), original, width_scale=width_scale)
        needed_height = len(lines) * original * source_spacing
        if needed_height <= bbox.height:
            return TextFitResult(
                text,
                tuple(lines),
                original,
                source_spacing,
                bbox,
                LayoutStatus.FIT,
                False,
                variant,
                expected_lines,
            )

    size = original - 0.5
    while size >= min_font_size - 0.01:
        line_spacing = max(0.90, min(source_spacing, source_spacing - (original - size) * 0.012))
        for variant, width_scale in (("source", 1.0), ("compact", 0.88)):
            lines = _wrap_text(text, max(1.0, bbox.width), size, width_scale=width_scale)
            needed_height = len(lines) * size * line_spacing
            if needed_height <= bbox.height:
                return TextFitResult(
                    text,
                    tuple(lines),
                    size,
                    line_spacing,
                    bbox,
                    LayoutStatus.SHRUNK,
                    False,
                    variant,
                    expected_lines,
                )
        size -= 0.5
    if allow_expansion:
        for expanded in _safe_expansions(bbox, page_width, page_height, maximum_expansion):
            if collides(expanded, obstacles):
                continue
            for variant, width_scale in (("source", 1.0), ("compact", 0.88)):
                lines = _wrap_text(
                    text,
                    max(1.0, expanded.width),
                    min_font_size,
                    width_scale=width_scale,
                )
                if len(lines) * min_font_size * 0.90 <= expanded.height:
                    return TextFitResult(
                        text,
                        tuple(lines),
                        min_font_size,
                        0.90,
                        expanded,
                        LayoutStatus.EXPANDED,
                        False,
                        variant,
                        expected_lines,
                    )
    lines = _wrap_text(text, max(1.0, bbox.width), min_font_size, width_scale=0.88)
    return TextFitResult(
        text,
        tuple(lines),
        min_font_size,
        0.90,
        bbox,
        LayoutStatus.OVERFLOW,
        True,
        "compact",
        expected_lines,
    )


def calculate_layout_fidelity(
    source_bbox: BoundingBox,
    output_bbox: BoundingBox,
    *,
    source_font_size: float,
    output_font_size: float,
    source_line_count: int,
    output_line_count: int,
    page_width: float,
    page_height: float,
    collision_free: bool,
    replacement_applied: bool,
    preserved_original: bool = False,
    reason: str = "rendered",
) -> LayoutFidelityResult:
    """Score geometric fidelity without exposing any marker in the final page."""
    page_box = BoundingBox(0.0, 0.0, page_width, page_height)
    within_page = (
        output_bbox.x0 >= page_box.x0
        and output_bbox.y0 >= page_box.y0
        and output_bbox.x1 <= page_box.x1
        and output_bbox.y1 <= page_box.y1
    )
    union_area = source_bbox.area + output_bbox.area
    intersection_width = max(
        0.0, min(source_bbox.x1, output_bbox.x1) - max(source_bbox.x0, output_bbox.x0)
    )
    intersection_height = max(
        0.0, min(source_bbox.y1, output_bbox.y1) - max(source_bbox.y0, output_bbox.y0)
    )
    intersection = intersection_width * intersection_height
    union = max(1.0, union_area - intersection)
    bbox_preservation = min(1.0, intersection / union)
    font_size_preservation = max(
        0.0,
        1.0 - abs(output_font_size - source_font_size) / max(1.0, source_font_size),
    )
    line_count_preservation = max(
        0.0,
        1.0 - abs(output_line_count - source_line_count) / max(1, source_line_count),
    )
    source_cx = (source_bbox.x0 + source_bbox.x1) / 2.0
    source_cy = (source_bbox.y0 + source_bbox.y1) / 2.0
    output_cx = (output_bbox.x0 + output_bbox.x1) / 2.0
    output_cy = (output_bbox.y0 + output_bbox.y1) / 2.0
    page_diagonal = max(1.0, (page_width**2 + page_height**2) ** 0.5)
    center_distance = ((source_cx - output_cx) ** 2 + (source_cy - output_cy) ** 2) ** 0.5
    placement_preservation = max(0.0, 1.0 - center_distance / page_diagonal)
    if preserved_original and not replacement_applied:
        score = 1.0
    else:
        score = (
            bbox_preservation * 0.30
            + font_size_preservation * 0.25
            + line_count_preservation * 0.20
            + placement_preservation * 0.15
            + (0.05 if collision_free else 0.0)
            + (0.05 if within_page else 0.0)
        )
    return LayoutFidelityResult(
        score=max(0.0, min(1.0, score)),
        bbox_preservation=bbox_preservation,
        font_size_preservation=font_size_preservation,
        line_count_preservation=line_count_preservation,
        placement_preservation=placement_preservation,
        collision_free=collision_free,
        within_page=within_page,
        replacement_applied=replacement_applied,
        preserved_original=preserved_original,
        reason=reason,
    )
