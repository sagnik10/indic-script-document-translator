"""In-memory preview rendering for the Streamlit interface."""

from __future__ import annotations

import io
import logging

from PIL import Image, ImageDraw

from ..schemas import DocumentModel, FileFormat

LOGGER = logging.getLogger(__name__)


def render_pdf_pages(pdf_bytes: bytes, max_pages: int = 8, dpi: int = 120) -> list[bytes]:
    try:
        import pymupdf as fitz
    except ImportError:
        return []
    images: list[bytes] = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            scale = dpi / 72.0
            for page_number in range(min(pdf.page_count, max_pages)):
                pixmap = pdf.load_page(page_number).get_pixmap(
                    matrix=fitz.Matrix(scale, scale), alpha=False
                )
                images.append(pixmap.tobytes("png"))
    except Exception:
        LOGGER.warning("PDF preview rendering failed", exc_info=True)
    return images


def render_source_preview(document: DocumentModel, max_pages: int = 8) -> list[bytes]:
    if document.file_format == FileFormat.PDF:
        return render_pdf_pages(document.source_bytes, max_pages=max_pages)
    if document.file_format in {FileFormat.PNG, FileFormat.JPEG}:
        try:
            image = Image.open(io.BytesIO(document.source_bytes)).convert("RGB")
            image.thumbnail((1400, 1800), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return [buffer.getvalue()]
        except Exception:
            LOGGER.warning("Image preview rendering failed", exc_info=True)
    return []


def render_enhanced_preview(document: DocumentModel, max_pages: int = 8) -> list[bytes]:
    enhanced = [
        page.enhanced_image_bytes
        for page in document.pages[:max_pages]
        if page.enhanced_image_bytes
    ]
    return [value for value in enhanced if value]


def render_debug_preview(document: DocumentModel, max_pages: int = 8) -> list[bytes]:
    """Draw region classes and confidence provenance without changing final output."""
    bases = render_source_preview(document, max_pages=max_pages)
    images: list[bytes] = []
    region_colors = {
        "printed_text": "#2f80ed",
        "handwriting": "#9b51e0",
        "table_form": "#00a67e",
        "stamp_seal": "#d62728",
        "signature": "#7f0000",
        "graphical_content": "#5b6573",
        "unknown": "#f2994a",
    }
    state_colors = {
        "confirmed": "#1b9e3e",
        "low_ocr_confidence": "#ff9800",
        "reconstructed": "#8e44ad",
        "flagged": "#d32f2f",
    }
    for page_index, page in enumerate(document.pages[:max_pages]):
        base_bytes = page.enhanced_image_bytes or (bases[page_index] if page_index < len(bases) else None)
        if not base_bytes:
            continue
        try:
            image = Image.open(io.BytesIO(base_bytes)).convert("RGB")
            draw = ImageDraw.Draw(image)
            scale_x = image.width / max(1.0, page.width)
            scale_y = image.height / max(1.0, page.height)
            for region in page.regions:
                rectangle = (
                    region.bbox.x0 * scale_x,
                    region.bbox.y0 * scale_y,
                    region.bbox.x1 * scale_x,
                    region.bbox.y1 * scale_y,
                )
                color = region_colors.get(region.region_type.value, "#f2994a")
                draw.rectangle(rectangle, outline=color, width=3)
                draw.text((rectangle[0] + 2, rectangle[1] + 2), region.region_type.value, fill=color)
            for block in page.blocks:
                rectangle = (
                    block.source_bbox.x0 * scale_x,
                    block.source_bbox.y0 * scale_y,
                    block.source_bbox.x1 * scale_x,
                    block.source_bbox.y1 * scale_y,
                )
                color = state_colors.get(block.uncertainty_state.value, "#2f80ed")
                draw.rectangle(rectangle, outline=color, width=2)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            images.append(buffer.getvalue())
        except Exception:
            LOGGER.warning("Debug preview rendering failed for page %d", page.page_number, exc_info=True)
    return images


def render_output_preview(output_bytes: bytes, extension: str, max_pages: int = 8) -> list[bytes]:
    if extension == "pdf":
        return render_pdf_pages(output_bytes, max_pages=max_pages)
    return []
