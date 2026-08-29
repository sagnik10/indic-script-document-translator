from __future__ import annotations

import io

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

from translator_app.config.settings import Settings
from translator_app.core.output_validation import validate_pdf_output
from translator_app.reconstruction.pdf_reconstructor import reconstruct_pdf
from translator_app.schemas import (
    BlockType,
    BoundingBox,
    ContentKind,
    DocumentModel,
    FileFormat,
    FontMetadata,
    LayoutStatus,
    PageModel,
    ProcessingStatus,
    ReconstructionType,
    Region,
    RegionType,
    TextBlock,
    TranslationStatus,
)


def _validated_hindi_block(
    bbox: BoundingBox,
    source: str,
    translation: str,
    *,
    region_type: RegionType = RegionType.PRINTED_TEXT,
    is_ocr: bool = False,
) -> TextBlock:
    return TextBlock(
        page_number=1,
        block_type=BlockType.LINE,
        source_bbox=bbox,
        source_text=source,
        normalized_text=source,
        detected_language="hi",
        resolved_language="hi",
        english_translation=translation,
        translation_status=TranslationStatus.TRANSLATED,
        source_validated=True,
        font=FontMetadata(size=12),
        region_type=region_type,
        is_ocr=is_ocr,
    )


def _native_pdf_document(blocks: list[TextBlock]) -> DocumentModel:
    source = fitz.open()
    page = source.new_page(width=300, height=200)
    page.insert_text((40, 70), "SOURCE WORDS", fontsize=12)
    page.draw_circle((250, 55), 20, color=(0.8, 0.0, 0.0), width=3)
    source_bytes = source.tobytes()
    source.close()
    return DocumentModel(
        "pdf-integration",
        "source.pdf",
        FileFormat.PDF,
        ContentKind.NATIVE,
        [PageModel(1, 300, 200, blocks)],
        source_bytes,
    )


def _raster_source() -> tuple[bytes, BoundingBox, BoundingBox]:
    image = Image.new("RGB", (360, 240), (239, 235, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 351, 231), outline=(55, 55, 55), width=2)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    text_bbox = BoundingBox(28, 56, 226, 102)
    draw.text((38, 65), "SOURCE LINE", fill=(15, 15, 15), font=font)
    stamp_bbox = BoundingBox(264, 142, 338, 216)
    draw.ellipse(
        (stamp_bbox.x0, stamp_bbox.y0, stamp_bbox.x1, stamp_bbox.y1),
        outline=(180, 25, 25),
        width=5,
    )
    draw.line((16, 122, 344, 122), fill=(60, 60, 60), width=2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), text_bbox, stamp_bbox


def _render_pdf_page(data: bytes) -> np.ndarray:
    with fitz.open(stream=data, filetype="pdf") as pdf:
        pixmap = pdf[0].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )[:, :, :3]


def test_native_pdf_replaces_only_validated_text_with_searchable_english() -> None:
    block = _validated_hindi_block(
        BoundingBox(35, 52, 180, 76),
        "SOURCE WORDS",
        "TRANSLATED WORDS",
    )
    block.metadata["origin"] = "native_pdf"
    document = _native_pdf_document([block])

    output = reconstruct_pdf(document, Settings(ocr_languages=["eng"]))

    with fitz.open(stream=output, filetype="pdf") as rebuilt:
        assert rebuilt.page_count == 1
        assert rebuilt[0].rect.width == 300
        assert rebuilt[0].rect.height == 200
        text = " ".join(rebuilt[0].get_text().split())
        assert "TRANSLATED WORDS" in text
        assert "SOURCE WORDS" not in text
    assert block.metadata["render_mode"] == "in_place_replacement"
    assert block.metadata["format_fidelity_score"] > 0.8
    assert document.metadata["diagnostic_pages_appended"] == 0
    assert document.metadata["primary_output_has_debug_overlays"] is False
    assert validate_pdf_output(output, document) == []


def test_raster_page_keeps_background_and_stamp_while_inserting_selectable_text() -> None:
    source_bytes, text_bbox, stamp_bbox = _raster_source()
    block = _validated_hindi_block(
        text_bbox,
        "SOURCE LINE",
        "English line",
        is_ocr=True,
    )
    text_region = Region(1, text_bbox, RegionType.PRINTED_TEXT, block_ids=[block.block_id])
    block.metadata["region_id"] = text_region.region_id
    stamp = Region(
        1,
        stamp_bbox,
        RegionType.STAMP_SEAL,
        preserve_as_image=True,
        overlaps_critical_graphic=True,
    )
    document = DocumentModel(
        "raster-integration",
        "scan.png",
        FileFormat.PNG,
        ContentKind.SCANNED,
        [
            PageModel(
                1,
                360,
                240,
                [block],
                regions=[text_region, stamp],
                image_bytes=source_bytes,
                content_kind=ContentKind.SCANNED,
            )
        ],
        source_bytes,
    )

    output = reconstruct_pdf(document, Settings(ocr_languages=["eng"]))

    with fitz.open(stream=output, filetype="pdf") as rebuilt:
        assert rebuilt.page_count == 1
        assert "English line" in rebuilt[0].get_text()
        assert rebuilt[0].rect == fitz.Rect(0, 0, 360, 240)
    rendered = _render_pdf_page(output)
    original = np.asarray(Image.open(io.BytesIO(source_bytes)).convert("RGB"))
    outside = np.ones(original.shape[:2], dtype=bool)
    outside[int(text_bbox.y0) : int(text_bbox.y1), int(text_bbox.x0) : int(text_bbox.x1)] = False
    assert np.max(np.abs(rendered[outside].astype(int) - original[outside].astype(int))) == 0
    sx0, sy0, sx1, sy1 = map(
        int, (stamp_bbox.x0, stamp_bbox.y0, stamp_bbox.x1, stamp_bbox.y1)
    )
    assert np.array_equal(rendered[sy0:sy1, sx0:sx1], original[sy0:sy1, sx0:sx1])
    assert block.metadata["replacement_method"] == "raster_text_inpaint"
    assert validate_pdf_output(output, document) == []


def test_manually_validated_handwriting_can_replace_a_preserve_fallback_line() -> None:
    source_bytes, text_bbox, _stamp_bbox = _raster_source()
    block = _validated_hindi_block(
        text_bbox,
        "विश्वसनीय हस्तलिखित स्रोत",
        "Reviewed handwritten source",
        region_type=RegionType.HANDWRITING,
        is_ocr=True,
    )
    block.is_handwritten = True
    block.reconstruction_type = ReconstructionType.MANUALLY_CONFIRMED
    block.metadata["htr_unavailable"] = True  # retained as historical audit provenance
    region = Region(
        1,
        text_bbox,
        RegionType.HANDWRITING,
        preserve_as_image=True,
        block_ids=[block.block_id],
    )
    block.metadata["region_id"] = region.region_id
    document = DocumentModel(
        "reviewed-handwriting",
        "reviewed.png",
        FileFormat.PNG,
        ContentKind.SCANNED,
        [PageModel(1, 360, 240, [block], [region], image_bytes=source_bytes)],
        source_bytes,
    )

    output = reconstruct_pdf(document, Settings(ocr_languages=["eng"]))

    with fitz.open(stream=output, filetype="pdf") as rebuilt:
        assert "Reviewed handwritten source" in rebuilt[0].get_text()
    assert block.metadata["replacement_applied"] is True
    assert block.metadata["primary_output_replacement_reason"] == (
        "validated_in_place_replacement"
    )


def test_stamp_signature_or_graphic_region_is_never_translated_or_overlaid() -> None:
    bbox = BoundingBox(35, 43, 170, 65)
    block = _validated_hindi_block(
        bbox,
        "SOURCE WORDS",
        "English translation",
        region_type=RegionType.STAMP_SEAL,
    )
    stamp = Region(
        1,
        bbox,
        RegionType.STAMP_SEAL,
        preserve_as_image=True,
        overlaps_critical_graphic=True,
        block_ids=[block.block_id],
    )
    block.metadata["region_id"] = stamp.region_id
    document = _native_pdf_document([block])
    document.pages[0].regions = [stamp]
    original_pixels = _render_pdf_page(document.source_bytes)

    output = reconstruct_pdf(document, Settings(ocr_languages=["eng"]))

    with fitz.open(stream=output, filetype="pdf") as rebuilt:
        text = rebuilt[0].get_text()
        assert "SOURCE WORDS" in text
        assert "English translation" not in text
    assert np.array_equal(_render_pdf_page(output), original_pixels)
    assert block.metadata["render_mode"] == "preserve_original"
    assert block.metadata["format_fidelity_score"] == 1.0
    assert block.layout_status == LayoutStatus.SKIPPED
    assert document.metadata["applied_translation_replacement_count"] == 0


def test_unreadable_region_fails_safe_to_original_pixels() -> None:
    block = _validated_hindi_block(
        BoundingBox(35, 52, 65, 62),
        "SOURCE WORDS",
        "This English translation is far too long to fit in the source region safely",
    )
    block.reconstruction_type = ReconstructionType.UNREADABLE
    block.processing_statuses.append(ProcessingStatus.UNREADABLE)
    document = _native_pdf_document([block])
    before = _render_pdf_page(document.source_bytes)

    output = reconstruct_pdf(document, Settings(ocr_languages=["eng"]))

    assert np.array_equal(_render_pdf_page(output), before)
    assert block.layout_status == LayoutStatus.SKIPPED
    assert block.metadata["preserved_original"] is True
    assert validate_pdf_output(output, document) == []


def test_long_translation_collision_preserves_original_instead_of_overwriting() -> None:
    first = _validated_hindi_block(
        BoundingBox(35, 52, 70, 63),
        "SOURCE WORDS",
        "A very long English translation that cannot fit in this tiny rectangle",
    )
    second = TextBlock(
        1,
        BlockType.LINE,
        BoundingBox(70, 50, 195, 78),
        "NEIGHBOR",
    )
    document = _native_pdf_document([first, second])
    before = _render_pdf_page(document.source_bytes)

    output = reconstruct_pdf(
        document,
        Settings(
            ocr_languages=["eng"],
            min_output_font_size=9,
            enable_safe_block_expansion=True,
        ),
    )

    assert np.array_equal(_render_pdf_page(output), before)
    assert first.layout_status == LayoutStatus.OVERFLOW
    assert first.metadata["replacement_applied"] is False
    assert first.metadata["preservation_reason"] == "translation_did_not_fit"
