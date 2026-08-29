"""Flat, review-oriented JSON audit generation."""

from __future__ import annotations

import json
from dataclasses import asdict

from .schemas import DocumentModel, LayoutStatus, ProcessingStatus, TranslationStatus


def _bbox(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    return {
        "x0": float(value.x0),
        "y0": float(value.y0),
        "x1": float(value.x1),
        "y1": float(value.y1),
    }


class AuditReportGenerator:
    def build(self, document: DocumentModel) -> dict[str, object]:
        blocks = document.blocks
        records = []
        for page in document.pages:
            region_map = {region.region_id: region for region in page.regions}
            for block in page.blocks:
                region = region_map.get(str(block.metadata.get("region_id", "")))
                records.append(
                    {
                        "page_number": block.page_number,
                        "block_id": block.block_id,
                        "region_id": region.region_id if region else block.metadata.get("region_id"),
                        "region_coordinates": _bbox(region.bbox) if region else _bbox(block.source_bbox),
                        "region_type": block.region_type.value,
                        "raw_ocr_text": block.source_text,
                        "normalized_text": block.normalized_text,
                        "script": block.script.value,
                        "region_visual_script": block.region_visual_script.value,
                        "visual_script_confidence": block.visual_script_confidence,
                        "recognized_unicode_script": block.recognized_unicode_script.value,
                        "ocr_script_confidence": block.ocr_script_confidence,
                        "resolved_script": block.resolved_script.value,
                        "resolved_language": block.resolved_language,
                        "script_resolution_reason": block.script_resolution_reason,
                        "expected_language_prior": block.expected_language_prior,
                        "linguistic_evidence_score": block.linguistic_evidence_score,
                        "detected_language": block.detected_language,
                        "language_confidence": block.language_confidence,
                        "text_quality": block.text_quality,
                        "source_validated": block.source_validated,
                        "validation_reason": block.validation_reason,
                        "ocr_engine": block.ocr_engine,
                        "ocr_confidence": block.ocr_confidence,
                        "ocr_alternatives": block.ocr_alternatives,
                        "htr_provider_id": block.metadata.get("htr_provider_id"),
                        "htr_model_id": block.metadata.get("htr_model_id"),
                        "htr_confidence_capability": block.metadata.get(
                            "htr_confidence_capability"
                        ),
                        "htr_source_language_output_only": block.metadata.get(
                            "htr_source_language_output_only"
                        ),
                        "htr_handwriting_validated": block.metadata.get(
                            "htr_handwriting_validated"
                        ),
                        "source_origin": block.metadata.get("source_origin", "ocr"),
                        "correction_origin": block.metadata.get("correction_origin"),
                        "manual_review_action": block.metadata.get("manual_review_action"),
                        "is_handwritten": block.is_handwritten,
                        "is_uncertain": block.is_uncertain,
                        "reconstruction_type": block.reconstruction_type.value,
                        "missing_span_detected": block.missing_span_detected,
                        "missing_span_bbox": _bbox(block.missing_span_bbox),
                        "reconstruction_candidate": block.reconstruction_candidate,
                        "was_reconstructed": block.reconstructed_text is not None,
                        "reconstructed_text": block.reconstructed_text,
                        "reconstruction_confidence": block.reconstruction_confidence,
                        "reconstruction_method": block.reconstruction_method,
                        "reconstruction_status": block.reconstruction_status.value,
                        "readable_character_ratio": block.readable_character_ratio,
                        "validated_context_token_count": block.validated_context_token_count,
                        "protected_entity_detected": block.protected_entity_detected,
                        "protected_tokens": block.protected_tokens,
                        "english_translation": block.english_translation,
                        "translation_status": block.translation_status.value,
                        "translation_confidence_or_status": block.translation_confidence_or_status,
                        "font_size": block.font_size,
                        "font_style": block.font_style,
                        "rotation": block.rotation,
                        "source_coordinates": _bbox(block.source_bbox),
                        "output_coordinates": _bbox(block.output_bbox),
                        "layout_status": (
                            "LAYOUT_OVERFLOW"
                            if block.layout_status.value == "overflow"
                            else block.layout_status.value
                        ),
                        "preserved_as_image": bool(
                            block.metadata.get("preserve_region_as_image")
                            or (region and region.preserve_as_image)
                        ),
                        "provenance": block.provenance,
                        "processing_statuses": [
                            status.value for status in block.processing_statuses
                        ],
                    }
                )
        return {
            "schema_version": "1.1",
            "document": {
                "document_id": document.document_id,
                "source_filename": document.source_filename,
                "file_format": document.file_format.value,
                "content_kind": document.content_kind.value,
                "page_count": len(document.pages),
                "warnings": document.warnings,
                "metadata": document.metadata,
            },
            "statistics": {
                "handwriting_regions": sum(
                    region.region_type.value == "handwriting"
                    for page in document.pages
                    for region in page.regions
                ),
                "htr_recognized_lines": sum(
                    ProcessingStatus.HTR_RECOGNIZED in block.processing_statuses
                    for block in blocks
                ),
                "manually_reviewed_lines": sum(
                    ProcessingStatus.MANUALLY_CORRECTED in block.processing_statuses
                    for block in blocks
                ),
                "source_validated_punjabi_lines": sum(
                    block.source_validated and block.detected_language == "pa"
                    for block in blocks
                ),
                "translated_lines": sum(
                    block.translation_status == TranslationStatus.TRANSLATED
                    for block in blocks
                ),
                "rejected_handwriting_lines": sum(
                    block.is_handwritten and not block.source_validated for block in blocks
                ),
                "layout_overflows": sum(
                    block.layout_status == LayoutStatus.OVERFLOW for block in blocks
                ),
                "missing_spans_detected": sum(block.missing_span_detected for block in blocks),
                "auto_reconstructed_spans": sum(
                    block.reconstruction_status.value == "AUTO_ACCEPTED" for block in blocks
                ),
                "reconstruction_review_candidates": sum(
                    block.reconstruction_status.value == "CANDIDATE_REVIEW" for block in blocks
                ),
                "manually_confirmed_sources": sum(
                    block.reconstruction_status.value == "MANUALLY_CONFIRMED" for block in blocks
                ),
                "unresolved_missing_spans": sum(
                    block.missing_span_detected
                    and block.reconstruction_status.value in {"BLOCKED", "REJECTED", "UNAVAILABLE"}
                    for block in blocks
                ),
                "punjabi_htr_routes": sum(
                    int(page.metadata.get("punjabi_htr_routes", 0))
                    for page in document.pages
                ),
                "hindi_htr_routes": sum(
                    int(page.metadata.get("hindi_htr_routes", 0))
                    for page in document.pages
                ),
                "printed_ocr_routes": sum(
                    int(page.metadata.get("printed_ocr_routes", 0))
                    for page in document.pages
                ),
                "rejected_noise_regions": sum(
                    int(page.metadata.get("rejected_noise_regions", 0))
                    for page in document.pages
                ),
            },
            "pages": [
                {
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "content_kind": page.content_kind.value,
                    "expected_language_prior": page.expected_language_prior,
                    "visual_page_script": page.visual_page_script.value,
                    "visual_page_script_confidence": page.visual_page_script_confidence,
                    "ocr_page_script": page.ocr_page_script.value,
                    "ocr_page_script_confidence": page.ocr_page_script_confidence,
                    "resolved_page_script": page.resolved_page_script.value,
                    "resolved_page_script_confidence": page.resolved_page_script_confidence,
                    "script_resolution_reason": page.script_resolution_reason,
                    "handwriting_probability": page.handwriting_probability,
                    "page_visual_type": page.page_visual_type.value,
                    "preprocessing_profile": page.metadata.get("preprocessing_profile"),
                    "quality_metrics": page.metadata.get("quality_metrics"),
                    "regions": [
                        {
                            **asdict(region),
                            "bbox": _bbox(region.bbox),
                            "region_type": region.region_type.value,
                            "visual_script_candidate": region.visual_script_candidate.value,
                            "visual_script_confidence": region.visual_script_confidence,
                            "recognized_unicode_script": region.recognized_unicode_script.value,
                            "recognized_unicode_script_confidence": region.recognized_unicode_script_confidence,
                            "resolved_script": region.resolved_script.value,
                            "resolved_language": region.resolved_language,
                            "script_resolution_reason": region.script_resolution_reason,
                            "selected_recognition_engine": region.selected_recognition_engine,
                            "linguistic_evidence_score": region.linguistic_evidence_score,
                            "rejected_as_noise": region.rejected_as_noise,
                        }
                        for region in page.regions
                    ],
                }
                for page in document.pages
            ],
            "text_blocks": records,
            "disclaimer": (
                "OCR, handwriting recognition, reconstruction, and machine translation are probabilistic. "
                "MODEL_INFERRED content is not confirmed source content; MANUALLY_CONFIRMED records "
                "an explicit source-language review; UNREADABLE content requires review."
            ),
        }

    def to_json_bytes(self, document: DocumentModel) -> bytes:
        return json.dumps(
            self.build(document), ensure_ascii=False, indent=2
        ).encode("utf-8")
