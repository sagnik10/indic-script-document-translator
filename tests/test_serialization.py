from translator_app.schemas import (
    BlockType,
    BoundingBox,
    ContentKind,
    DocumentModel,
    FileFormat,
    LayoutStatus,
    PageVisualType,
    PageModel,
    ReconstructionStatus,
    ReconstructionType,
    Region,
    RegionType,
    ScriptType,
    TextBlock,
)


def test_document_block_serialization_round_trip_excludes_source_bytes() -> None:
    block = TextBlock(1, BlockType.HEADING, BoundingBox(1, 2, 100, 30), "नमस्ते")
    block.output_bbox = BoundingBox(1, 2, 110, 36)
    block.layout_status = LayoutStatus.EXPANDED
    block.review_image_bytes = b"private-crop"
    block.missing_span_detected = True
    block.missing_span_bbox = BoundingBox(40, 2, 55, 30)
    block.reconstruction_candidate = "शब्द"
    block.reconstruction_confidence = 0.94
    block.reconstruction_method = "local_masked_language_model"
    block.reconstruction_status = ReconstructionStatus.AUTO_ACCEPTED
    block.reconstruction_type = ReconstructionType.MODEL_INFERRED
    block.region_visual_script = ScriptType.DEVANAGARI
    block.recognized_unicode_script = ScriptType.DEVANAGARI
    block.resolved_script = ScriptType.DEVANAGARI
    block.resolved_language = "hi"
    region = Region(1, block.source_bbox, RegionType.PRINTED_TEXT)
    region.visual_script_candidate = ScriptType.DEVANAGARI
    region.resolved_script = ScriptType.DEVANAGARI
    page = PageModel(
        1,
        595,
        842,
        [block],
        [region],
        content_kind=ContentKind.MIXED,
        visual_page_script=ScriptType.DEVANAGARI,
        ocr_page_script=ScriptType.DEVANAGARI,
        resolved_page_script=ScriptType.DEVANAGARI,
        page_visual_type=PageVisualType.PRINTED,
    )
    document = DocumentModel(
        "doc-id",
        "source.pdf",
        FileFormat.PDF,
        ContentKind.MIXED,
        [page],
        b"confidential",
    )
    serialized = document.to_dict()
    assert "source_bytes" not in serialized
    assert "image_bytes" not in serialized["pages"][0]
    assert "review_image_bytes" not in serialized["pages"][0]["blocks"][0]
    restored = DocumentModel.from_dict(serialized, source_bytes=b"restored")
    assert restored.source_bytes == b"restored"
    assert restored.pages[0].blocks[0].source_bbox == block.source_bbox
    assert restored.pages[0].blocks[0].layout_status == LayoutStatus.EXPANDED
    restored_block = restored.pages[0].blocks[0]
    assert restored_block.missing_span_bbox == block.missing_span_bbox
    assert restored_block.reconstruction_status == ReconstructionStatus.AUTO_ACCEPTED
    assert restored_block.reconstruction_candidate == "शब्द"
    assert restored_block.reconstruction_method == "local_masked_language_model"
    assert restored_block.region_visual_script == ScriptType.DEVANAGARI
    assert restored.pages[0].visual_page_script == ScriptType.DEVANAGARI
    assert restored.pages[0].resolved_page_script == ScriptType.DEVANAGARI
    assert restored.pages[0].page_visual_type == PageVisualType.PRINTED
    assert restored.pages[0].regions[0].resolved_script == ScriptType.DEVANAGARI
