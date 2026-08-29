"""Native, scanned, and hybrid PDF extraction using PyMuPDF."""

from __future__ import annotations

import io
import logging
from collections.abc import Callable

from PIL import Image

from ..config.settings import Settings
from ..exceptions import (
    CorruptedDocumentError,
    DependencyUnavailableError,
    PasswordProtectedPDFError,
)
from ..schemas import (
    BlockType,
    BoundingBox,
    ContentKind,
    DocumentModel,
    FileFormat,
    FontMetadata,
    PageModel,
    PageVisualType,
    ProcessingOptions,
    Region,
    RegionType,
    ScriptType,
    TextBlock,
)
from ..utils.text_utils import normalize_text
from ..utils.validation import ValidatedFile
from .image_processor import ImagePreprocessor, image_to_png_bytes
from .layout_analyzer import classify_text_block, remove_duplicate_ocr_blocks
from .page_ocr import PageOCRPipeline

LOGGER = logging.getLogger(__name__)


def _color_hex(color_value: int) -> str:
    return f"#{int(color_value) & 0xFFFFFF:06x}"


def _alignment(x0: float, x1: float, page_width: float) -> str:
    left_gap, right_gap = x0, page_width - x1
    if abs(left_gap - right_gap) <= max(8, page_width * 0.025):
        return "center"
    if right_gap < max(10, left_gap * 0.2):
        return "right"
    return "left"


def _native_blocks(page: object, page_number: int) -> list[TextBlock]:
    page_dict = page.get_text("dict", flags=11)
    blocks: list[TextBlock] = []
    page_width = float(page.rect.width)
    for raw_block in page_dict.get("blocks", []):
        if raw_block.get("type") != 0:
            continue
        for line_index, line in enumerate(raw_block.get("lines", [])):
            spans = line.get("spans", [])
            text = "".join(str(span.get("text", "")) for span in spans)
            text = normalize_text(text)
            if not text:
                continue
            bbox_values = line.get("bbox") or raw_block.get("bbox")
            if not bbox_values:
                continue
            bbox = BoundingBox(*(float(value) for value in bbox_values[:4]))
            representative = max(spans, key=lambda span: float(span.get("size", 0)), default={})
            flags = int(representative.get("flags", 0))
            font_size = float(representative.get("size", max(8.0, bbox.height * 0.75)))
            font_name = str(representative.get("font", "Helvetica"))
            block_type = classify_text_block(text, font_size, float(page.rect.height))
            if bbox.y1 >= float(page.rect.height) * 0.94:
                block_type = BlockType.FOOTER
            elif bbox.y0 <= float(page.rect.height) * 0.055 and block_type not in {
                BlockType.TITLE,
                BlockType.HEADING,
            }:
                block_type = BlockType.HEADER
            block = TextBlock(
                page_number=page_number,
                block_type=block_type,
                source_bbox=bbox,
                source_text=text,
                normalized_text=text,
                font=FontMetadata(
                    family=font_name,
                    size=font_size,
                    bold=bool(flags & 16) or "bold" in font_name.casefold(),
                    italic=bool(flags & 2) or "italic" in font_name.casefold(),
                    color=_color_hex(int(representative.get("color", 0))),
                ),
                alignment=_alignment(bbox.x0, bbox.x1, page_width),
                rotation=0.0,
                is_ocr=False,
                region_type=(
                    RegionType.TABLE_FORM
                    if block_type == BlockType.TABLE_CELL
                    else RegionType.PRINTED_TEXT
                ),
                ocr_engine="pymupdf_native",
                provenance=["Native selectable PDF text extracted by PyMuPDF"],
                source_reference=f"pdf:{page_number}:block:{raw_block.get('number', 0)}:line:{line_index}",
                metadata={
                    "origin": "native_pdf",
                    "span_count": len(spans),
                    "writing_direction": line.get("dir", (1.0, 0.0)),
                },
            )
            blocks.append(block)
    return blocks


def _regions_for_native_blocks(blocks: list[TextBlock]) -> list[Region]:
    regions: list[Region] = []
    for order, block in enumerate(blocks):
        region = Region(
            page_number=block.page_number,
            bbox=block.source_bbox,
            region_type=block.region_type,
            classification_confidence=1.0,
            reading_order=order,
            block_ids=[block.block_id],
            preserve_as_image=block.region_type == RegionType.TABLE_FORM,
            overlaps_critical_graphic=block.region_type == RegionType.TABLE_FORM,
            metadata={"origin": "native_pdf"},
        )
        block.metadata["region_id"] = region.region_id
        block.metadata["region_reading_order"] = order
        regions.append(region)
    return regions


def _annotate_pdf_tables(page: object, blocks: list[TextBlock]) -> None:
    """Attach available PyMuPDF table/cell relationships without rebuilding graphics."""
    try:
        found = page.find_tables()
        tables = getattr(found, "tables", [])
        for table_index, table in enumerate(tables):
            for cell_index, cell in enumerate(getattr(table, "cells", [])):
                if not cell:
                    continue
                cell_bbox = BoundingBox(*(float(value) for value in cell[:4]))
                for block in blocks:
                    if block.source_bbox.intersection_ratio(cell_bbox) >= 0.5:
                        block.block_type = BlockType.TABLE_CELL
                        block.metadata["table_index"] = table_index
                        block.metadata["table_cell_index"] = cell_index
                        block.metadata["table_cell_bbox"] = {
                            "x0": cell_bbox.x0,
                            "y0": cell_bbox.y0,
                            "x1": cell_bbox.x1,
                            "y1": cell_bbox.y1,
                        }
    except Exception:
        LOGGER.debug("PDF table detection was unavailable for a page", exc_info=True)


def _image_coverage(page: object) -> float:
    area = max(1.0, float(page.rect.width * page.rect.height))
    covered = 0.0
    seen: set[tuple[float, float, float, float]] = set()
    try:
        for image_info in page.get_images(full=True):
            for rectangle in page.get_image_rects(image_info[0]):
                key = tuple(round(value, 1) for value in rectangle)
                if key not in seen:
                    seen.add(key)
                    covered += max(0.0, rectangle.width * rectangle.height)
    except Exception:
        LOGGER.debug("Could not calculate image coverage", exc_info=True)
    return min(1.0, covered / area)


class PDFProcessor:
    def __init__(
        self,
        settings: Settings,
        page_ocr_loader: Callable[[], PageOCRPipeline],
        preprocessor: ImagePreprocessor,
    ) -> None:
        self.settings = settings
        self.page_ocr_loader = page_ocr_loader
        self.preprocessor = preprocessor

    def process(
        self,
        validated: ValidatedFile,
        options: ProcessingOptions,
        stage_callback: Callable[[str, float], None] | None = None,
    ) -> DocumentModel:
        try:
            import pymupdf as fitz
        except ImportError as exc:
            raise DependencyUnavailableError("PyMuPDF is required for PDF processing") from exc
        try:
            source = fitz.open(stream=validated.data, filetype="pdf")
        except Exception as exc:
            raise CorruptedDocumentError("PyMuPDF could not open the PDF") from exc
        with source:
            if source.needs_pass:
                raise PasswordProtectedPDFError("Encrypted PDF requires a password")
            if source.page_count == 0:
                raise CorruptedDocumentError("PDF contains no pages")
            pages: list[PageModel] = []
            warnings: list[str] = []
            page_kinds: list[ContentKind] = []
            for page_index in range(source.page_count):
                if stage_callback:
                    stage_callback("extracting", (page_index + 1) / source.page_count)
                page = source.load_page(page_index)
                native = _native_blocks(page, page_index + 1)
                _annotate_pdf_tables(page, native)
                native_regions = _regions_for_native_blocks(native)
                native_characters = sum(len(block.source_text) for block in native)
                coverage = _image_coverage(page)
                needs_ocr = options.force_ocr or native_characters < self.settings.native_text_min_characters
                if coverage > 0.55 and native_characters < 200:
                    needs_ocr = True
                ocr_blocks: list[TextBlock] = []
                ocr_regions: list[Region] = []
                processed = None
                page_kind = ContentKind.NATIVE
                if needs_ocr:
                    if stage_callback:
                        stage_callback("ocr", (page_index + 1) / source.page_count)
                    matrix = fitz.Matrix(self.settings.ocr_dpi / 72.0, self.settings.ocr_dpi / 72.0)
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                    processed = (
                        self.preprocessor.preprocess(
                            image,
                            profile=options.preprocessing_profile,
                            allow_geometry=False,
                            upscale_factor=options.ocr_upscale_factor,
                        )
                        if options.enable_preprocessing
                        else None
                    )
                    ocr_image = processed.image if processed else image
                    display_image = processed.display_image if processed else image
                    ocr = self.page_ocr_loader().process(
                        ocr_image,
                        display_image,
                        page_number=page_index + 1,
                        page_width=float(page.rect.width),
                        page_height=float(page.rect.height),
                        options=options,
                        ocr_variants=(
                            processed.candidate_images
                            if processed
                            else {"original": image.convert("L")}
                        ),
                        quality_metrics=(
                            processed.quality.to_dict()
                            if processed and processed.quality
                            else {}
                        ),
                    )
                    warnings.extend(ocr.warnings)
                    ocr_blocks = remove_duplicate_ocr_blocks(native, ocr.blocks)
                    ocr_regions = ocr.regions
                    if native and ocr_blocks:
                        page_kind = ContentKind.MIXED
                    elif ocr_blocks or not native:
                        page_kind = ContentKind.SCANNED
                page_kinds.append(page_kind)
                pages.append(
                    PageModel(
                        page_number=page_index + 1,
                        width=float(page.rect.width),
                        height=float(page.rect.height),
                        rotation=int(page.rotation),
                        content_kind=page_kind,
                        blocks=sorted(native + ocr_blocks, key=lambda item: (item.source_bbox.y0, item.source_bbox.x0)),
                        regions=native_regions + ocr_regions,
                        enhanced_image_bytes=(
                            image_to_png_bytes(ocr_image)
                            if needs_ocr and page_index < self.settings.max_preview_pages
                            else None
                        ),
                        visual_page_script=(
                            ocr.visual_script_candidate if needs_ocr else ScriptType.UNKNOWN
                        ),
                        visual_page_script_confidence=(
                            ocr.visual_script_confidence if needs_ocr else 0.0
                        ),
                        ocr_page_script=(
                            ocr.ocr_script_candidate if needs_ocr else ScriptType.UNKNOWN
                        ),
                        ocr_page_script_confidence=(
                            ocr.ocr_script_confidence if needs_ocr else 0.0
                        ),
                        resolved_page_script=(
                            ocr.resolved_script if needs_ocr else ScriptType.UNKNOWN
                        ),
                        resolved_page_script_confidence=(
                            ocr.dominant_script_confidence if needs_ocr else 0.0
                        ),
                        script_resolution_reason=(
                            ocr.script_resolution_reason if needs_ocr else "native text page"
                        ),
                        expected_language_prior=options.expected_source_language,
                        handwriting_probability=(
                            ocr.handwriting_probability if needs_ocr else 0.0
                        ),
                        page_visual_type=(
                            ocr.page_type if needs_ocr else PageVisualType.PRINTED
                        ),
                        metadata={
                            "native_character_count": native_characters,
                            "image_coverage": coverage,
                            "preprocessing": processed.applied_operations if needs_ocr and processed else [],
                            "preprocessing_profile": processed.profile if processed else "native",
                            "quality_metrics": (
                                processed.quality.to_dict()
                                if processed and processed.quality
                                else {}
                            ),
                            "dominant_script": (
                                ocr.dominant_script.value if needs_ocr else "unknown"
                            ),
                            "dominant_script_confidence": (
                                ocr.dominant_script_confidence if needs_ocr else 0.0
                            ),
                            "visual_page_script": (
                                ocr.visual_script_candidate.value if needs_ocr else "unknown"
                            ),
                            "visual_page_script_confidence": (
                                ocr.visual_script_confidence if needs_ocr else 0.0
                            ),
                            "ocr_page_script": (
                                ocr.ocr_script_candidate.value if needs_ocr else "unknown"
                            ),
                            "ocr_page_script_confidence": (
                                ocr.ocr_script_confidence if needs_ocr else 0.0
                            ),
                            "resolved_page_script": (
                                ocr.resolved_script.value if needs_ocr else "unknown"
                            ),
                            "script_resolution_reason": (
                                ocr.script_resolution_reason if needs_ocr else "native text page"
                            ),
                            "expected_language_prior": options.expected_source_language,
                            "page_handwriting_probability": (
                                ocr.handwriting_probability if needs_ocr else 0.0
                            ),
                            "page_visual_type": (
                                ocr.page_type.value if needs_ocr else "printed"
                            ),
                            "detected_text_line_count": (
                                ocr.detected_text_line_count if needs_ocr else len(native)
                            ),
                            "punjabi_htr_routes": ocr.punjabi_htr_routes if needs_ocr else 0,
                            "hindi_htr_routes": ocr.hindi_htr_routes if needs_ocr else 0,
                            "printed_ocr_routes": ocr.printed_ocr_routes if needs_ocr else 0,
                            "rejected_noise_regions": ocr.rejected_noise_regions if needs_ocr else 0,
                        },
                    )
                )
            unique_kinds = set(page_kinds)
            content_kind = unique_kinds.pop() if len(unique_kinds) == 1 else ContentKind.MIXED
            if not any(page.blocks for page in pages):
                warnings.append(
                    "No text was extracted. The output preserves the original visual content for manual review."
                )
            return DocumentModel(
                document_id=validated.sha256[:20],
                source_filename=validated.filename,
                file_format=FileFormat.PDF,
                content_kind=content_kind,
                pages=pages,
                source_bytes=validated.data,
                mime_type=validated.mime_type,
                warnings=list(dict.fromkeys(warnings)),
                metadata={"source_page_count": len(pages)},
            )
