"""Fidelity-first selectable-text PDF reconstruction on the source page template."""

from __future__ import annotations

import io
import logging
from hashlib import sha256
from typing import Any

from PIL import Image

from ..config.settings import Settings
from ..core.layout_engine import calculate_layout_fidelity, fit_text
from ..exceptions import DependencyUnavailableError, RenderingError
from ..schemas import (
    ContentKind,
    DocumentModel,
    FileFormat,
    LayoutStatus,
    ProcessingStatus,
    ReconstructionType,
    RegionType,
    TranslationStatus,
)

LOGGER = logging.getLogger(__name__)

_PROTECTED_REGION_TYPES = {
    RegionType.STAMP_SEAL,
    RegionType.SIGNATURE,
    RegionType.GRAPHICAL_CONTENT,
}
_SOURCE_LANGUAGE_ALIASES = {
    "pa": "pa", "pan": "pa", "pan_guru": "pa", "punjabi": "pa",
    "hi": "hi", "hin": "hi", "hin_deva": "hi", "hindi": "hi",
}


def _rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    try:
        return tuple(int(value[index:index + 2], 16) / 255 for index in (0, 2, 4))  # type: ignore[return-value]
    except (ValueError, TypeError):
        return (0.0, 0.0, 0.0)


def _font_name(block: Any, *, compact: bool = False) -> str:
    if compact:
        return "tibi" if block.font.bold and block.font.italic else "tibo" if block.font.bold else "tiit" if block.font.italic else "tiro"
    return "hebi" if block.font.bold and block.font.italic else "hebo" if block.font.bold else "heit" if block.font.italic else "helv"


def _alignment_value(value: str) -> int:
    return {"left": 0, "center": 1, "right": 2, "justify": 3}.get(value, 0)


def _open_output_base(document: DocumentModel, fitz: Any) -> Any:
    if document.file_format == FileFormat.PDF:
        return fitz.open(stream=document.source_bytes, filetype="pdf")
    image_bytes = document.pages[0].image_bytes or document.source_bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    output = fitz.open()
    model = document.pages[0]
    page = output.new_page(width=model.width, height=model.height)
    page.insert_image(page.rect, stream=buffer.getvalue(), keep_proportion=False)
    return output


def _region_for_block(page_model: Any, block: Any) -> Any | None:
    region_id = str(block.metadata.get("region_id", ""))
    return next(
        (
            region for region in page_model.regions
            if region.region_id == region_id or block.block_id in region.block_ids
        ),
        None,
    )


def _is_critical_region(region: Any) -> bool:
    """Return whether a region's pixels must remain completely untouched."""
    if region.region_type in _PROTECTED_REGION_TYPES:
        return True
    # Handwriting fallback regions are marked preserve-as-image while their HTR
    # result is unavailable.  That state protects unreadable blocks, but it must
    # not permanently forbid a later validated/manual source transcription from
    # replacing its bounded line.  Table regions similarly protect their rules,
    # not every validated text glyph inside a cell.
    if region.region_type in {
        RegionType.PRINTED_TEXT,
        RegionType.HANDWRITING,
        RegionType.TABLE_FORM,
    }:
        return False
    return bool(region.preserve_as_image or region.overlaps_critical_graphic)


def _render_clip_png(page: Any, block: Any, fitz: Any) -> bytes:
    bbox = block.source_bbox
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(2.0, 2.0),
        clip=fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
        alpha=False,
        annots=False,
    )
    return pixmap.tobytes("png")


def _neutralize_text_patch(original_patch: bytes) -> bytes | None:
    """Inpaint text-like ink while retaining paper texture and long form rules.

    If a bounded text mask cannot be identified, returning ``None`` makes the
    renderer preserve the exact source crop. This is safer than a white rectangle.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    encoded = np.frombuffer(original_patch, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.shape[0] < 5 or image.shape[1] < 5:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    block_size = max(9, min(41, (min(height, width) // 3) | 1))
    ink = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        9,
    )
    # Long, thin form rules at crop edges are not part of the text-removal mask.
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, int(width * 0.62)), 1)),
    )
    vertical = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, int(height * 0.72)))),
    )
    candidate = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.bitwise_or(horizontal, vertical)))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate, 8)
    filtered = np.zeros_like(candidate)
    crop_area = float(height * width)
    for label in range(1, count):
        _x, _y, component_width, component_height, area = stats[label]
        if area < 2 or area / crop_area > 0.16:
            continue
        if component_width > width * 0.88 or component_height > height * 0.88:
            continue
        filtered[labels == label] = 255
    mask_ratio = float(np.count_nonzero(filtered)) / max(1.0, crop_area)
    # A text line normally occupies a modest fraction of its crop.  A larger
    # mask is likely a seal, photograph, border, or thresholding failure.
    if mask_ratio < 0.0015 or mask_ratio > 0.28:
        return None
    filtered = cv2.dilate(
        filtered,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    neutralized = cv2.inpaint(image, filtered, 3, cv2.INPAINT_TELEA)
    ok, result = cv2.imencode(".png", neutralized)
    return result.tobytes() if ok else None


def _positive_overlap(left: Any, right: Any, padding: float = 0.5) -> bool:
    return not (
        left.x1 + padding <= right.x0
        or right.x1 + padding <= left.x0
        or left.y1 + padding <= right.y0
        or right.y1 + padding <= left.y0
    )


def _preserve_original(block: Any, page_model: Any, reason: str, *, overflow: bool = False) -> None:
    block.output_bbox = None
    block.layout_status = LayoutStatus.OVERFLOW if overflow else LayoutStatus.SKIPPED
    block.metadata.update({
        "render_mode": "preserve_original",
        "replacement_applied": False,
        "preserved_original": True,
        "preservation_reason": reason,
        "primary_output_replacement_reason": reason,
    })
    if overflow and ProcessingStatus.LAYOUT_OVERFLOW not in block.processing_statuses:
        block.processing_statuses.append(ProcessingStatus.LAYOUT_OVERFLOW)
    line_count = max(1, len((block.effective_source_text or "").splitlines()))
    fidelity = calculate_layout_fidelity(
        block.source_bbox,
        block.source_bbox,
        source_font_size=max(1.0, block.font.size),
        output_font_size=max(1.0, block.font.size),
        source_line_count=line_count,
        output_line_count=line_count,
        page_width=page_model.width,
        page_height=page_model.height,
        collision_free=True,
        replacement_applied=False,
        preserved_original=True,
        reason=reason,
    )
    block.metadata["layout_fidelity"] = fidelity.to_dict()
    block.metadata["format_fidelity_score"] = fidelity.score


def _rejection_reason(page_model: Any, block: Any) -> str | None:
    if block.translation_status != TranslationStatus.TRANSLATED or not block.english_translation:
        return "translation_not_available"
    if not block.source_validated:
        return "source_not_validated"
    language = None
    for value in (block.resolved_language, block.detected_language):
        language = _SOURCE_LANGUAGE_ALIASES.get(str(value or "").casefold().replace("-", "_"))
        if language:
            break
    if language not in {"pa", "hi"}:
        return "source_language_not_punjabi_or_hindi"
    if block.reconstruction_type == ReconstructionType.UNREADABLE:
        return "source_unreadable"
    if any(status in block.processing_statuses for status in (
        ProcessingStatus.HTR_UNAVAILABLE,
        ProcessingStatus.HANDWRITING_UNSUPPORTED,
        ProcessingStatus.UNREADABLE,
    )):
        return "recognition_unavailable_or_unreadable"
    if block.region_type in _PROTECTED_REGION_TYPES:
        return f"protected_{block.region_type.value}"
    region = _region_for_block(page_model, block)
    if region and _is_critical_region(region):
        return f"protected_{region.region_type.value}"
    if block.metadata.get("preserve_region_as_image"):
        return "preserve_region_as_image"
    for candidate in page_model.regions:
        if _is_critical_region(candidate) and block.source_bbox.intersection_ratio(candidate.bbox) > 0.0:
            return f"overlaps_protected_{candidate.region_type.value}"
    return None


def _table_cell_bbox(block: Any) -> Any | None:
    value = block.metadata.get("table_cell_bbox")
    if not isinstance(value, dict):
        return None
    try:
        from ..schemas import BoundingBox

        return BoundingBox(
            float(value["x0"]),
            float(value["y0"]),
            float(value["x1"]),
            float(value["y1"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _clip_digest(page: Any, bbox: Any, fitz: Any) -> str:
    """Hash a protected visual clip for pre-export preservation checks."""
    clip = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1) & page.rect
    if clip.is_empty or clip.width < 0.5 or clip.height < 0.5:
        return ""
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(1.5, 1.5),
        clip=clip,
        alpha=False,
        annots=False,
    )
    return sha256(pixmap.samples).hexdigest()


def _verify_protected_pixels(
    output: Any,
    source_template: Any,
    document: DocumentModel,
    fitz: Any,
) -> None:
    """Fail export if a seal, signature, graphic, or unreadable crop changed."""
    for page_model in document.pages:
        output_page = output.load_page(page_model.page_number - 1)
        source_page = source_template.load_page(page_model.page_number - 1)
        protected: list[Any] = [
            region.bbox for region in page_model.regions if _is_critical_region(region)
        ]
        protected.extend(
            block.source_bbox
            for block in page_model.blocks
            if (
                block.reconstruction_type == ReconstructionType.UNREADABLE
                or any(
                    status in block.processing_statuses
                    for status in (
                        ProcessingStatus.HTR_UNAVAILABLE,
                        ProcessingStatus.HANDWRITING_UNSUPPORTED,
                        ProcessingStatus.UNREADABLE,
                    )
                )
            )
        )
        seen: set[tuple[float, float, float, float]] = set()
        for bbox in protected:
            key = tuple(round(value, 3) for value in (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
            if key in seen:
                continue
            seen.add(key)
            if _clip_digest(output_page, bbox, fitz) != _clip_digest(source_page, bbox, fitz):
                raise RenderingError(
                    f"Protected visual content changed on page {page_model.page_number}"
                )


def _verify_primary_structure(output: Any, document: DocumentModel) -> None:
    if output.page_count != len(document.pages):
        raise RenderingError("Primary output page count differs from source")
    for model in document.pages:
        page = output.load_page(model.page_number - 1)
        if (
            abs(page.rect.width - model.width) > 0.05
            or abs(page.rect.height - model.height) > 0.05
            or int(page.rotation) != int(model.rotation)
        ):
            raise RenderingError("Primary output page geometry differs from source")
        applied = [
            block for block in model.blocks if block.metadata.get("replacement_applied")
        ]
        for block in applied:
            bbox = block.output_bbox
            if bbox is None or not (
                0 <= bbox.x0 <= bbox.x1 <= model.width
                and 0 <= bbox.y0 <= bbox.y1 <= model.height
            ):
                raise RenderingError("Translated text was placed outside the source page")
        for index, left in enumerate(applied):
            for right in applied[index + 1 :]:
                if left.output_bbox and right.output_bbox and _positive_overlap(
                    left.output_bbox, right.output_bbox
                ):
                    raise RenderingError("Translated text regions collide in the primary output")


def _preflight(fitz: Any, page_model: Any, block: Any, fit: Any) -> bool:
    scratch = fitz.open()
    try:
        page = scratch.new_page(width=page_model.width, height=page_model.height)
        return page.insert_textbox(
            fitz.Rect(fit.bbox.x0, fit.bbox.y0, fit.bbox.x1, fit.bbox.y1),
            "\n".join(fit.lines),
            fontsize=fit.font_size,
            fontname=_font_name(block, compact=fit.font_variant == "compact"),
            color=_rgb(block.font.color),
            align=_alignment_value(block.alignment),
            lineheight=fit.line_spacing,
        ) >= 0
    finally:
        scratch.close()


def reconstruct_pdf(document: DocumentModel, settings: Settings) -> bytes:
    """Return only the source-shaped primary translated document."""
    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise DependencyUnavailableError("PyMuPDF is required for PDF rendering") from exc
    try:
        output = _open_output_base(document, fitz)
        source_template = _open_output_base(document, fitz)
        with output, source_template:
            for page_model in document.pages:
                page = output.load_page(page_model.page_number - 1)
                source_page = source_template.load_page(page_model.page_number - 1)
                occupied: list[Any] = []
                for block in page_model.blocks:
                    if block.translation_status != TranslationStatus.TRANSLATED:
                        _preserve_original(
                            block,
                            page_model,
                            "translation_not_available",
                        )
                        continue
                    reason = _rejection_reason(page_model, block)
                    if reason:
                        _preserve_original(block, page_model, reason)
                        continue
                    associated_region = _region_for_block(page_model, block)
                    obstacles = [
                        other.source_bbox
                        for other in page_model.blocks
                        if other.block_id != block.block_id
                    ]
                    obstacles.extend(
                        region.bbox
                        for region in page_model.regions
                        if region is not associated_region and _is_critical_region(region)
                    )
                    cell_bbox = _table_cell_bbox(block)
                    constrained_to_table = bool(
                        block.region_type == RegionType.TABLE_FORM
                        or (
                            associated_region is not None
                            and associated_region.region_type == RegionType.TABLE_FORM
                        )
                        or cell_bbox is not None
                    )
                    fit = fit_text(
                        block.english_translation or "",
                        block.source_bbox,
                        original_font_size=max(settings.min_output_font_size, block.font.size),
                        min_font_size=settings.min_output_font_size,
                        page_width=page_model.width,
                        page_height=page_model.height,
                        obstacles=obstacles + occupied,
                        allow_expansion=(
                            settings.enable_safe_block_expansion
                            and not constrained_to_table
                            and int(block.rotation) == 0
                        ),
                        maximum_expansion=settings.max_block_expansion_points,
                        source_line_count=max(1, len((block.effective_source_text or "").splitlines())),
                        original_line_spacing=block.font.line_spacing,
                    )
                    if fit.overflow or not _preflight(fitz, page_model, block, fit):
                        _preserve_original(block, page_model, "translation_did_not_fit", overflow=True)
                        continue
                    if any(_positive_overlap(fit.bbox, existing) for existing in occupied):
                        _preserve_original(block, page_model, "translated_region_collision", overflow=True)
                        continue
                    if fit.bbox != block.source_bbox and any(
                        _positive_overlap(fit.bbox, obstacle) for obstacle in obstacles
                    ):
                        _preserve_original(block, page_model, "unsafe_bbox_expansion", overflow=True)
                        continue
                    if cell_bbox is not None and not (
                        cell_bbox.x0 <= fit.bbox.x0 <= fit.bbox.x1 <= cell_bbox.x1
                        and cell_bbox.y0 <= fit.bbox.y0 <= fit.bbox.y1 <= cell_bbox.y1
                    ):
                        _preserve_original(block, page_model, "translation_exceeds_table_cell", overflow=True)
                        continue
                    rect = fitz.Rect(block.source_bbox.x0, block.source_bbox.y0, block.source_bbox.x1, block.source_bbox.y1)
                    original_patch = _render_clip_png(source_page, block, fitz)
                    neutral_patch = _neutralize_text_patch(original_patch)
                    if neutral_patch is None:
                        _preserve_original(block, page_model, "background_neutralization_not_safe")
                        continue
                    native_text = (
                        page_model.content_kind != ContentKind.SCANNED
                        and not block.is_ocr
                        and block.metadata.get("origin") == "native_pdf"
                    )
                    try:
                        if native_text:
                            page.add_redact_annot(rect, fill=None, cross_out=False)
                            page.apply_redactions(images=0, graphics=0, text=0)
                        page.insert_image(
                            rect,
                            stream=neutral_patch,
                            keep_proportion=False,
                            overlay=True,
                        )
                        inserted = page.insert_textbox(
                            fitz.Rect(fit.bbox.x0, fit.bbox.y0, fit.bbox.x1, fit.bbox.y1),
                            "\n".join(fit.lines),
                            fontsize=fit.font_size,
                            fontname=_font_name(block, compact=fit.font_variant == "compact"),
                            color=_rgb(block.font.color),
                            align=_alignment_value(block.alignment),
                            rotate=(
                                int(block.rotation)
                                if int(block.rotation) in {0, 90, 180, 270}
                                else 0
                            ),
                            lineheight=fit.line_spacing,
                            overlay=True,
                        )
                        if inserted < 0:
                            raise RenderingError("Text insertion disagreed with preflight")
                    except Exception:
                        # Transactional fail-safe: restore the exact original visual crop.
                        page.insert_image(
                            rect,
                            stream=original_patch,
                            keep_proportion=False,
                            overlay=True,
                        )
                        _preserve_original(
                            block,
                            page_model,
                            "render_failed_original_restored",
                            overflow=True,
                        )
                        LOGGER.warning(
                            "PDF region render failed; original crop restored (page=%s, block=%s)",
                            block.page_number,
                            block.block_id,
                            exc_info=True,
                        )
                        continue
                    block.output_bbox = fit.bbox
                    block.layout_status = fit.status
                    block.metadata.update({
                        "render_mode": "in_place_replacement",
                        "replacement_applied": True,
                        "preserved_original": False,
                        "replacement_method": (
                            "native_text_redaction_and_inpaint"
                            if native_text
                            else "raster_text_inpaint"
                        ),
                        "rendered_font_name": _font_name(block, compact=fit.font_variant == "compact"),
                        "rendered_font_size": fit.font_size,
                        "rendered_line_count": len(fit.lines),
                        "primary_output_replacement_reason": "validated_in_place_replacement",
                        "source_layout": {
                            "bbox": {
                                "x0": block.source_bbox.x0,
                                "y0": block.source_bbox.y0,
                                "x1": block.source_bbox.x1,
                                "y1": block.source_bbox.y1,
                            },
                            "baseline": block.metadata.get("baseline"),
                            "font_family": block.font.family,
                            "font_size": block.font.size,
                            "font_weight": "bold" if block.font.bold else "regular",
                            "font_style": "italic" if block.font.italic else "regular",
                            "text_color": block.font.color,
                            "alignment": block.alignment,
                            "line_count": max(
                                1,
                                int(
                                    block.metadata.get(
                                        "grouped_source_line_count",
                                        len((block.effective_source_text or "").splitlines()),
                                    )
                                ),
                            ),
                            "line_spacing": block.font.line_spacing,
                            "rotation": block.rotation,
                            "table_cell_bbox": block.metadata.get("table_cell_bbox"),
                        },
                    })
                    fidelity = calculate_layout_fidelity(
                        block.source_bbox,
                        fit.bbox,
                        source_font_size=max(1.0, block.font.size),
                        output_font_size=fit.font_size,
                        source_line_count=max(1, len((block.effective_source_text or "").splitlines())),
                        output_line_count=max(1, len(fit.lines)),
                        page_width=page_model.width,
                        page_height=page_model.height,
                        collision_free=True,
                        replacement_applied=True,
                    )
                    block.metadata["layout_fidelity"] = fidelity.to_dict()
                    block.metadata["format_fidelity_score"] = fidelity.score
                    occupied.append(fit.bbox)
                records = [block.metadata["layout_fidelity"] for block in page_model.blocks if "layout_fidelity" in block.metadata]
                page_model.metadata["format_fidelity_score"] = round(sum(record["score"] for record in records) / len(records), 4) if records else 1.0
                page_model.metadata["diagnostic_pages_appended"] = 0
                page_model.metadata["primary_output_has_debug_overlays"] = False
            _verify_primary_structure(output, document)
            _verify_protected_pixels(output, source_template, document, fitz)
            document.metadata["applied_translation_replacement_count"] = sum(
                block.translation_status == TranslationStatus.TRANSLATED
                and block.metadata.get("replacement_applied") is True
                for block in document.blocks
            )
            document.metadata.update({
                "primary_output_only": True,
                "diagnostic_pages_appended": 0,
                "primary_output_has_debug_overlays": False,
            })
            output.set_metadata({
                **output.metadata,
                "title": f"English translation - {document.source_filename}",
                "subject": "In-place validated Punjabi/Hindi to English translation",
                "producer": "Local Multilingual Document Translator MVP",
            })
            return output.tobytes(garbage=4, deflate=True)
    except Exception as exc:
        if isinstance(exc, (DependencyUnavailableError, RenderingError)):
            raise
        raise RenderingError("PDF reconstruction failed") from exc


def reconstruct_translation_report(document: DocumentModel) -> bytes:
    """Build an explicitly requested diagnostic transcript, never a primary output."""
    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise DependencyUnavailableError("PyMuPDF is required for PDF rendering") from exc
    output = fitz.open()
    try:
        for page_model in document.pages:
            page = output.new_page(width=595.0, height=842.0)
            lines = ["English translation diagnostic report", f"Source page: {page_model.page_number}", ""]
            lines.extend(
                block.english_translation or ""
                for block in page_model.blocks
                if block.translation_status == TranslationStatus.TRANSLATED and block.english_translation
            )
            page.insert_textbox(page.rect + (48, 48, -48, -48), "\n\n".join(lines), fontsize=9, fontname="helv")
        return output.tobytes(garbage=4, deflate=True)
    finally:
        output.close()
