"""Reconstruct OCR words/fragments into logical source lines and paragraphs."""

from __future__ import annotations

from statistics import median

from ..schemas import BlockType, BoundingBox, ReconstructionStatus, TextBlock


def _vertical_overlap(first: BoundingBox, second: BoundingBox) -> float:
    overlap = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return overlap / max(1.0, min(first.height, second.height))


def _merge_group(group: list[TextBlock]) -> TextBlock:
    ordered = sorted(group, key=lambda item: item.source_bbox.x0)
    anchor = ordered[0]
    anchor.source_text = " ".join(item.source_text.strip() for item in ordered if item.source_text.strip())
    anchor.normalized_text = " ".join(
        (item.normalized_text or item.source_text).strip()
        for item in ordered
        if (item.normalized_text or item.source_text).strip()
    )
    anchor.source_bbox = BoundingBox(
        min(item.source_bbox.x0 for item in ordered),
        min(item.source_bbox.y0 for item in ordered),
        max(item.source_bbox.x1 for item in ordered),
        max(item.source_bbox.y1 for item in ordered),
    )
    weights = [max(1, len(item.source_text.strip())) for item in ordered]
    confidences = [item.ocr_confidence or 0.0 for item in ordered]
    anchor.ocr_confidence = sum(value * weight for value, weight in zip(confidences, weights, strict=True)) / sum(weights)
    anchor.child_block_ids = [item.block_id for item in ordered[1:]]
    anchor.ocr_alternatives = [alternative for item in ordered for alternative in item.ocr_alternatives]
    anchor.provenance.append(f"Merged {len(ordered)} adjacent OCR fragments into one source line")
    anchor.metadata["merged_block_ids"] = [item.block_id for item in ordered]
    anchor.block_type = BlockType.LINE
    return anchor


def merge_text_blocks(blocks: list[TextBlock], horizontal_gap_ratio: float = 2.5) -> list[TextBlock]:
    """Merge only baseline-compatible fragments; never join distinct vertical lines."""
    pending = sorted(blocks, key=lambda item: (item.source_bbox.y0, item.source_bbox.x0))
    output: list[TextBlock] = []
    while pending:
        anchor = pending.pop(0)
        group = [anchor]
        region_id = anchor.metadata.get("region_id")
        changed = True
        while changed:
            changed = False
            group_box = BoundingBox(
                min(item.source_bbox.x0 for item in group),
                min(item.source_bbox.y0 for item in group),
                max(item.source_bbox.x1 for item in group),
                max(item.source_bbox.y1 for item in group),
            )
            scale = median(max(1.0, item.source_bbox.height) for item in group)
            for candidate in pending[:]:
                if candidate.metadata.get("region_id") != region_id:
                    continue
                gap = max(
                    0.0,
                    candidate.source_bbox.x0 - group_box.x1,
                    group_box.x0 - candidate.source_bbox.x1,
                )
                if (
                    candidate.region_type == anchor.region_type
                    and _vertical_overlap(group_box, candidate.source_bbox) >= 0.32
                    and gap <= horizontal_gap_ratio * max(scale, candidate.source_bbox.height)
                ):
                    group.append(candidate)
                    pending.remove(candidate)
                    changed = True
        output.append(_merge_group(group) if len(group) > 1 else anchor)
    return sorted(output, key=lambda item: (item.source_bbox.y0, item.source_bbox.x0))


def group_validated_lines(
    blocks: list[TextBlock],
    *,
    maximum_line_gap_ratio: float = 1.65,
    maximum_lines: int = 4,
) -> list[TextBlock]:
    """Group adjacent validated OCR lines for contextual translation and one layout target."""
    ordered = sorted(blocks, key=lambda item: (item.source_bbox.y0, item.source_bbox.x0))
    groups: list[list[TextBlock]] = []
    for block in ordered:
        if (
            not block.is_ocr
            or not block.source_validated
            or block.detected_language not in {"pa", "hi"}
            or block.metadata.get("automatic_translation_aborted")
            or block.reconstruction_status != ReconstructionStatus.NOT_DETECTED
        ):
            groups.append([block])
            continue
        match: list[TextBlock] | None = None
        if groups:
            candidate_group = groups[-1]
            previous = candidate_group[-1]
            gap = block.source_bbox.y0 - previous.source_bbox.y1
            scale = median(item.source_bbox.height for item in candidate_group)
            horizontal_overlap = max(
                0.0,
                min(block.source_bbox.x1, previous.source_bbox.x1)
                - max(block.source_bbox.x0, previous.source_bbox.x0),
            )
            overlap_ratio = horizontal_overlap / max(
                1.0, min(block.source_bbox.width, previous.source_bbox.width)
            )
            left_alignment = abs(block.source_bbox.x0 - previous.source_bbox.x0) <= max(18.0, scale * 2.0)
            if (
                len(candidate_group) < maximum_lines
                and previous.is_ocr
                and previous.source_validated
                and previous.detected_language == block.detected_language
                and previous.reconstruction_status == ReconstructionStatus.NOT_DETECTED
                and previous.region_type == block.region_type
                and 0 <= gap <= maximum_line_gap_ratio * max(1.0, scale)
                and (overlap_ratio >= 0.20 or left_alignment)
            ):
                match = candidate_group
        if match is None:
            groups.append([block])
        else:
            match.append(block)

    output: list[TextBlock] = []
    for group in groups:
        if len(group) == 1:
            output.append(group[0])
            continue
        anchor = group[0]
        anchor.source_text = "\n".join(item.source_text for item in group)
        anchor.normalized_text = "\n".join(item.normalized_text or item.source_text for item in group)
        anchor.source_bbox = BoundingBox(
            min(item.source_bbox.x0 for item in group),
            min(item.source_bbox.y0 for item in group),
            max(item.source_bbox.x1 for item in group),
            max(item.source_bbox.y1 for item in group),
        )
        weights = [max(1, len(item.effective_source_text)) for item in group]
        anchor.ocr_confidence = sum(
            (item.ocr_confidence or 0.0) * weight
            for item, weight in zip(group, weights, strict=True)
        ) / sum(weights)
        anchor.text_quality = sum(item.text_quality * weight for item, weight in zip(group, weights, strict=True)) / sum(weights)
        anchor.child_block_ids = [item.block_id for item in group[1:]]
        anchor.metadata["grouped_source_line_ids"] = [item.block_id for item in group]
        anchor.metadata["grouped_source_line_count"] = len(group)
        anchor.block_type = BlockType.PARAGRAPH
        anchor.provenance.append(
            f"Grouped {len(group)} validated source lines into one contextual translation block"
        )
        output.append(anchor)
    return sorted(output, key=lambda item: (item.source_bbox.y0, item.source_bbox.x0))
