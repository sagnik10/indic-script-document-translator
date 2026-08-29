"""Spatial grouping utilities that keep OCR at logical line/paragraph granularity."""

from __future__ import annotations

from statistics import median

from ..schemas import BoundingBox, Region, RegionType, ScriptType


def _vertical_overlap(first: BoundingBox, second: BoundingBox) -> float:
    overlap = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return overlap / max(1.0, min(first.height, second.height))


def _union(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )


def merge_text_regions(
    regions: list[Region],
    *,
    horizontal_gap_ratio: float = 2.5,
    minimum_vertical_overlap: float = 0.30,
) -> list[Region]:
    """Merge adjacent text components sharing a baseline; critical graphics stay separate."""
    mergeable = {
        RegionType.PRINTED_TEXT,
        RegionType.HANDWRITING,
        RegionType.MIXED,
        RegionType.UNKNOWN,
    }
    pending = sorted(regions, key=lambda item: (item.bbox.y0, item.bbox.x0))
    output: list[Region] = []
    while pending:
        anchor = pending.pop(0)
        if anchor.region_type not in mergeable:
            output.append(anchor)
            continue
        group = [anchor]
        changed = True
        while changed:
            changed = False
            group_bbox = _union([item.bbox for item in group])
            line_height = median(max(1.0, item.bbox.height) for item in group)
            for candidate in pending[:]:
                if candidate.region_type not in mergeable:
                    continue
                gap = max(0.0, candidate.bbox.x0 - group_bbox.x1, group_bbox.x0 - candidate.bbox.x1)
                compatible_type = (
                    candidate.region_type == anchor.region_type
                    or RegionType.UNKNOWN in {candidate.region_type, anchor.region_type}
                    or RegionType.MIXED in {candidate.region_type, anchor.region_type}
                )
                if (
                    compatible_type
                    and _vertical_overlap(group_bbox, candidate.bbox) >= minimum_vertical_overlap
                    and gap <= horizontal_gap_ratio * max(line_height, candidate.bbox.height)
                ):
                    group.append(candidate)
                    pending.remove(candidate)
                    changed = True
        if len(group) == 1:
            output.append(anchor)
            continue
        kinds = {item.region_type for item in group if item.region_type != RegionType.UNKNOWN}
        merged_type = kinds.pop() if len(kinds) == 1 else RegionType.MIXED
        visual_scripts = {
            item.visual_script_candidate
            for item in group
            if item.visual_script_candidate != ScriptType.UNKNOWN
        }
        merged_visual_script = (
            visual_scripts.pop()
            if len(visual_scripts) == 1
            else ScriptType.MIXED
            if visual_scripts
            else ScriptType.UNKNOWN
        )
        handwriting_probability = sum(
            float(item.metadata.get("region_handwriting_probability", 0.0))
            for item in group
        ) / len(group)
        merged = Region(
            page_number=anchor.page_number,
            bbox=_union([item.bbox for item in group]),
            region_type=merged_type,
            classification_confidence=sum(item.classification_confidence for item in group) / len(group),
            preserve_as_image=any(item.preserve_as_image for item in group),
            overlaps_critical_graphic=any(item.overlaps_critical_graphic for item in group),
            visual_script_candidate=merged_visual_script,
            visual_script_confidence=sum(
                item.visual_script_confidence for item in group
            )
            / len(group),
            metadata={
                "merged_region_ids": [item.region_id for item in group],
                "merged_component_count": len(group),
                "region_handwriting_probability": handwriting_probability,
                "visual_script_reason": "merged baseline-compatible visual line components",
                "page_rejected_noise_region_count": max(
                    int(item.metadata.get("page_rejected_noise_region_count", 0))
                    for item in group
                ),
            },
        )
        pixel_boxes = [item.metadata.get("pixel_bbox") for item in group]
        valid = [box for box in pixel_boxes if isinstance(box, list) and len(box) == 4]
        if valid:
            merged.metadata["pixel_bbox"] = [
                min(box[0] for box in valid),
                min(box[1] for box in valid),
                max(box[2] for box in valid),
                max(box[3] for box in valid),
            ]
        output.append(merged)
    for order, region in enumerate(sorted(output, key=lambda item: (item.bbox.y0, item.bbox.x0))):
        region.reading_order = order
    return sorted(output, key=lambda item: item.reading_order)
