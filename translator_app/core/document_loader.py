"""Validated routing for PDF, DOCX, and image inputs."""

from __future__ import annotations

from collections.abc import Callable

from ..config.settings import Settings
from ..schemas import ContentKind, DocumentModel, FileFormat, PageModel, ProcessingOptions
from ..utils.validation import ValidatedFile
from .docx_processor import DOCXProcessor
from .image_processor import ImagePreprocessor, image_to_png_bytes, load_image
from .page_ocr import PageOCRPipeline
from .pdf_processor import PDFProcessor


class DocumentLoader:
    def __init__(
        self,
        settings: Settings,
        page_ocr_loader: Callable[[], PageOCRPipeline],
        preprocessor: ImagePreprocessor,
    ) -> None:
        self.settings = settings
        self.page_ocr_loader = page_ocr_loader
        self.preprocessor = preprocessor
        self.pdf_processor = PDFProcessor(settings, page_ocr_loader, preprocessor)
        self.docx_processor = DOCXProcessor(settings, page_ocr_loader, preprocessor)

    def _process_image(
        self,
        validated: ValidatedFile,
        options: ProcessingOptions,
        stage_callback: Callable[[str, float], None] | None,
    ) -> DocumentModel:
        image = load_image(validated.data)
        processed = (
            self.preprocessor.preprocess(
                image,
                profile=options.preprocessing_profile,
                allow_geometry=True,
                upscale_factor=options.ocr_upscale_factor,
            )
            if options.enable_preprocessing
            else None
        )
        ocr_image = processed.image if processed else image
        display_image = processed.display_image if processed else image
        width_points = display_image.width * 72.0 / 96.0
        height_points = display_image.height * 72.0 / 96.0
        if stage_callback:
            stage_callback("ocr", 0.5)
        ocr = self.page_ocr_loader().process(
            ocr_image,
            display_image,
            page_number=1,
            page_width=width_points,
            page_height=height_points,
            options=options,
            ocr_variants=processed.candidate_images if processed else {"original": image.convert("L")},
            quality_metrics=(processed.quality.to_dict() if processed and processed.quality else {}),
        )
        page = PageModel(
            page_number=1,
            width=width_points,
            height=height_points,
            blocks=ocr.blocks,
            regions=ocr.regions,
            content_kind=ContentKind.SCANNED,
            image_bytes=image_to_png_bytes(display_image),
            enhanced_image_bytes=image_to_png_bytes(ocr_image),
            visual_page_script=ocr.visual_script_candidate,
            visual_page_script_confidence=ocr.visual_script_confidence,
            ocr_page_script=ocr.ocr_script_candidate,
            ocr_page_script_confidence=ocr.ocr_script_confidence,
            resolved_page_script=ocr.resolved_script,
            resolved_page_script_confidence=ocr.dominant_script_confidence,
            script_resolution_reason=ocr.script_resolution_reason,
            expected_language_prior=options.expected_source_language,
            handwriting_probability=ocr.handwriting_probability,
            page_visual_type=ocr.page_type,
            metadata={
                "source_pixel_size": image.size,
                "processed_pixel_size": display_image.size,
                "preprocessing": processed.applied_operations if processed else [],
                "preprocessing_profile": processed.profile if processed else "none",
                "quality_metrics": processed.quality.to_dict() if processed and processed.quality else {},
                "geometry_changed": bool(processed and processed.geometry_changed),
                "dominant_script": ocr.dominant_script.value,
                "dominant_script_confidence": ocr.dominant_script_confidence,
                "visual_page_script": ocr.visual_script_candidate.value,
                "visual_page_script_confidence": ocr.visual_script_confidence,
                "ocr_page_script": ocr.ocr_script_candidate.value,
                "ocr_page_script_confidence": ocr.ocr_script_confidence,
                "resolved_page_script": ocr.resolved_script.value,
                "script_resolution_reason": ocr.script_resolution_reason,
                "expected_language_prior": options.expected_source_language,
                "page_handwriting_probability": ocr.handwriting_probability,
                "page_visual_type": ocr.page_type.value,
                "detected_text_line_count": ocr.detected_text_line_count,
                "punjabi_htr_routes": ocr.punjabi_htr_routes,
                "hindi_htr_routes": ocr.hindi_htr_routes,
                "printed_ocr_routes": ocr.printed_ocr_routes,
                "rejected_noise_regions": ocr.rejected_noise_regions,
            },
        )
        warnings = list(ocr.warnings)
        if not ocr.blocks:
            warnings.append(
                "No text was extracted from the image. The output preserves the source image for manual review."
            )
        return DocumentModel(
            document_id=validated.sha256[:20],
            source_filename=validated.filename,
            file_format=validated.file_format,
            content_kind=ContentKind.SCANNED,
            pages=[page],
            source_bytes=validated.data,
            mime_type=validated.mime_type,
            warnings=warnings,
        )

    def load(
        self,
        validated: ValidatedFile,
        options: ProcessingOptions,
        stage_callback: Callable[[str, float], None] | None = None,
    ) -> DocumentModel:
        if validated.file_format == FileFormat.PDF:
            return self.pdf_processor.process(validated, options, stage_callback)
        if validated.file_format == FileFormat.DOCX:
            return self.docx_processor.process(validated, options, stage_callback)
        if validated.file_format in {FileFormat.PNG, FileFormat.JPEG}:
            return self._process_image(validated, options, stage_callback)
        raise ValueError(f"No document processor for {validated.file_format}")
