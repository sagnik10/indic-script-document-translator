import json

from translator_app.report_generator import AuditReportGenerator
from translator_app.schemas import (
    BlockType,
    BoundingBox,
    ContentKind,
    DocumentModel,
    FileFormat,
    PageModel,
    PageVisualType,
    Region,
    RegionType,
    ReconstructionStatus,
    ReconstructionType,
    ScriptType,
    TextBlock,
)


def test_audit_contains_required_per_block_provenance_fields() -> None:
    region = Region(1, BoundingBox(0, 0, 100, 30), RegionType.HANDWRITING)
    block = TextBlock(1, BlockType.LINE, region.bbox, "[unclear handwriting]")
    block.region_type = RegionType.HANDWRITING
    block.is_handwritten = True
    block.ocr_engine = "handwriting_unavailable"
    block.metadata["source_origin"] = "htr_unavailable"
    block.metadata["region_id"] = region.region_id
    block.missing_span_detected = True
    block.missing_span_bbox = BoundingBox(20, 0, 45, 30)
    block.reconstruction_candidate = "ਸ਼ਬਦ"
    block.reconstruction_confidence = 0.93
    block.reconstruction_method = "script_consistent_ocr_alternative"
    block.reconstruction_status = ReconstructionStatus.AUTO_ACCEPTED
    block.reconstruction_type = ReconstructionType.MODEL_INFERRED
    block.region_visual_script = ScriptType.GURMUKHI
    block.recognized_unicode_script = ScriptType.GURMUKHI
    block.resolved_script = ScriptType.GURMUKHI
    block.resolved_language = "pa"
    block.expected_language_prior = "pa"
    block.linguistic_evidence_score = 0.91
    region.visual_script_candidate = ScriptType.GURMUKHI
    region.resolved_script = ScriptType.GURMUKHI
    region.resolved_language = "pa"
    region.selected_recognition_engine = "htr:pa:gurmukhi"
    region.block_ids.append(block.block_id)
    document = DocumentModel(
        "audit",
        "scan.pdf",
        FileFormat.PDF,
        ContentKind.SCANNED,
        [
            PageModel(
                1,
                100,
                100,
                [block],
                regions=[region],
                visual_page_script=ScriptType.GURMUKHI,
                resolved_page_script=ScriptType.GURMUKHI,
                expected_language_prior="pa",
                handwriting_probability=0.9,
                page_visual_type=PageVisualType.HANDWRITING_HEAVY,
            )
        ],
        b"%PDF-",
    )
    payload = json.loads(AuditReportGenerator().to_json_bytes(document))
    record = payload["text_blocks"][0]
    assert {
        "handwriting_regions",
        "htr_recognized_lines",
        "manually_reviewed_lines",
        "source_validated_punjabi_lines",
        "translated_lines",
        "rejected_handwriting_lines",
        "layout_overflows",
        "missing_spans_detected",
        "auto_reconstructed_spans",
        "reconstruction_review_candidates",
        "manually_confirmed_sources",
        "unresolved_missing_spans",
    }.issubset(payload["statistics"])
    required = {
        "page_number",
        "region_coordinates",
        "raw_ocr_text",
        "normalized_text",
        "detected_language",
        "ocr_engine",
        "ocr_confidence",
        "is_handwritten",
        "was_reconstructed",
        "reconstruction_confidence",
        "english_translation",
        "translation_status",
        "layout_status",
        "source_origin",
        "htr_provider_id",
        "missing_span_detected",
        "missing_span_bbox",
        "reconstruction_candidate",
        "reconstruction_method",
        "reconstruction_status",
        "readable_character_ratio",
        "validated_context_token_count",
        "protected_entity_detected",
        "region_visual_script",
        "recognized_unicode_script",
        "resolved_script",
        "resolved_language",
        "script_resolution_reason",
        "expected_language_prior",
        "linguistic_evidence_score",
    }
    assert required.issubset(record)
    assert record["reconstruction_type"] == "MODEL_INFERRED"
    assert record["reconstruction_status"] == "AUTO_ACCEPTED"
    assert payload["statistics"]["missing_spans_detected"] == 1
