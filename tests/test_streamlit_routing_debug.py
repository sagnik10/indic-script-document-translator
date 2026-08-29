from streamlit.testing.v1 import AppTest


ROUTING_APP = r'''
from translator_app.schemas import (
    BlockType, BoundingBox, ContentKind, DocumentModel, FileFormat, PageModel,
    PageVisualType, Region, RegionType, ScriptType, TextBlock,
)
from translator_app.ui.components import render_routing_debug

region = Region(1, BoundingBox(10, 20, 280, 55), RegionType.HANDWRITING)
region.visual_script_candidate = ScriptType.GURMUKHI
region.visual_script_confidence = 0.84
region.resolved_script = ScriptType.GURMUKHI
region.resolved_language = "pa"
region.script_resolution_reason = "user-selected pa source-language routing prior"
region.selected_recognition_engine = "htr:pa:gurmukhi"
block = TextBlock(1, BlockType.LINE, region.bbox, "[unreadable handwriting]")
block.normalized_text = block.source_text
block.region_type = RegionType.HANDWRITING
block.region_visual_script = ScriptType.GURMUKHI
block.resolved_script = ScriptType.GURMUKHI
block.detected_language = "pa"
block.validation_reason = "configured HTR provider does not support this handwriting language"
block.metadata["region_id"] = region.region_id
region.block_ids.append(block.block_id)
page = PageModel(
    1, 300, 100, [block], [region], content_kind=ContentKind.SCANNED,
    visual_page_script=ScriptType.GURMUKHI,
    visual_page_script_confidence=0.82,
    ocr_page_script=ScriptType.UNKNOWN,
    resolved_page_script=ScriptType.GURMUKHI,
    resolved_page_script_confidence=0.95,
    script_resolution_reason="user-selected pa source-language routing prior",
    expected_language_prior="pa",
    handwriting_probability=0.91,
    page_visual_type=PageVisualType.HANDWRITING_HEAVY,
)
page.metadata.update({
    "detected_text_line_count": 1,
    "punjabi_htr_routes": 1,
    "hindi_htr_routes": 0,
    "printed_ocr_routes": 0,
    "rejected_noise_regions": 4,
})
document = DocumentModel(
    "routing", "Test1.jpeg", FileFormat.JPEG, ContentKind.SCANNED,
    [page], b"source"
)
render_routing_debug(document)
'''


def test_routing_debug_separates_visual_ocr_and_resolved_evidence() -> None:
    app = AppTest.from_string(ROUTING_APP, default_timeout=20).run()
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Expected language"] == "pa"
    assert metrics["Visual script"] == "gurmukhi"
    assert metrics["OCR-derived script"] == "unknown"
    assert metrics["Resolved script"] == "gurmukhi"
    assert metrics["Page type"] == "handwriting_heavy"
    assert metrics["Punjabi HTR routes"] == "1"
