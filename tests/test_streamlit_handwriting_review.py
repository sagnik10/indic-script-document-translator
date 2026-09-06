from streamlit.testing.v1 import AppTest


REVIEW_APP = r'''
import io
import time
from PIL import Image
import streamlit as st

from translator_app.schemas import (
    AnalysisResult, BlockType, BoundingBox, ContentKind, DocumentModel,
    FileFormat, PageModel, ProcessingOptions, ProcessingStatus, ReconstructionStatus,
    ReconstructionType, RegionType, ScriptType, TextBlock, UncertaintyState,
)
from translator_app.ui.components import render_review_editor

stream = io.BytesIO()
Image.new("RGB", (240, 55), "white").save(stream, format="PNG")
block = TextBlock(1, BlockType.LINE, BoundingBox(0, 0, 200, 30), "[unreadable handwriting]")
block.block_id = "htr-review-line"
block.normalized_text = "[unreadable handwriting]"
block.is_handwritten = True
block.region_type = RegionType.HANDWRITING
block.script = ScriptType.GURMUKHI
block.detected_language = "pa"
block.uncertainty_state = UncertaintyState.FLAGGED
block.review_image_bytes = stream.getvalue()
block.processing_statuses = [ProcessingStatus.HTR_UNAVAILABLE, ProcessingStatus.UNREADABLE]
block.metadata.update({
    "htr_unavailable": True,
    "handwriting_unsupported": True,
    "preserve_region_as_image": True,
})
unconfirmed = TextBlock(1, BlockType.LINE, BoundingBox(0, 96, 200, 126), "[unreadable handwriting]")
unconfirmed.block_id = "htr-unconfirmed-line"
unconfirmed.normalized_text = "[unreadable handwriting]"
unconfirmed.is_handwritten = True
unconfirmed.region_type = RegionType.HANDWRITING
unconfirmed.script = ScriptType.GURMUKHI
unconfirmed.detected_language = "pa"
unconfirmed.uncertainty_state = UncertaintyState.FLAGGED
unconfirmed.review_image_bytes = stream.getvalue()
unconfirmed.processing_statuses = [ProcessingStatus.HTR_UNAVAILABLE, ProcessingStatus.UNREADABLE]
unconfirmed.metadata.update({
    "htr_unavailable": True,
    "handwriting_unsupported": True,
    "preserve_region_as_image": True,
})
missing = TextBlock(
    1, BlockType.LINE, BoundingBox(0, 35, 200, 65),
    "ਮੈਂ ਅੱਜ [missing] ਨੂੰ ਜਾ ਰਿਹਾ ਹਾਂ"
)
missing.block_id = "missing-review-line"
missing.normalized_text = missing.source_text
missing.is_ocr = True
missing.script = ScriptType.GURMUKHI
missing.detected_language = "pa"
missing.uncertainty_state = UncertaintyState.CANDIDATE
missing.reconstruction_status = ReconstructionStatus.CANDIDATE_REVIEW
missing.reconstruction_type = ReconstructionType.MODEL_INFERRED
missing.reconstruction_candidate = "ਸਕੂਲ"
missing.reconstruction_confidence = 0.78
missing.review_image_bytes = stream.getvalue()
missing.metadata.update({
    "proposed_reconstructed_source_text": "ਮੈਂ ਅੱਜ ਸਕੂਲ ਨੂੰ ਜਾ ਰਿਹਾ ਹਾਂ",
    "missing_span_previous_text": "ਪਿਛਲੀ ਲਾਈਨ",
    "missing_span_next_text": "ਅਗਲੀ ਲਾਈਨ",
})
automatic = TextBlock(
    1, BlockType.LINE, BoundingBox(0, 68, 200, 95),
    "ਉਹ [missing] ਗਿਆ"
)
automatic.block_id = "automatic-review-line"
automatic.normalized_text = automatic.source_text
automatic.is_ocr = True
automatic.script = ScriptType.GURMUKHI
automatic.detected_language = "pa"
automatic.uncertainty_state = UncertaintyState.RECONSTRUCTED
automatic.reconstruction_status = ReconstructionStatus.AUTO_ACCEPTED
automatic.reconstruction_type = ReconstructionType.MODEL_INFERRED
automatic.reconstruction_candidate = "ਘਰ"
automatic.reconstructed_text = "ਉਹ ਘਰ ਗਿਆ"
automatic.reconstruction_confidence = 0.94
automatic.review_image_bytes = stream.getvalue()
document = DocumentModel(
    "ui-review", "scan.jpg", FileFormat.JPEG, ContentKind.SCANNED,
    [PageModel(1, 200, 130, [block, unconfirmed, missing, automatic])], b"source"
)
analysis = AnalysisResult(document, ProcessingOptions(), time.perf_counter())
edits, submitted = render_review_editor(analysis)
if submitted:
    st.session_state["captured_review_edits"] = edits
'''


def test_handwriting_review_renders_crop_editor_and_confirmation_gate() -> None:
    app = AppTest.from_string(REVIEW_APP, default_timeout=20).run()
    assert not app.exception
    assert any(
        item.label == "Recognized Punjabi/Hindi source text" for item in app.text_area
    )
    assert any(item.label == "Confirmed source line" for item in app.text_area)
    assert any("MODEL_INFERRED" in item.value for item in app.success)
    assert any(
        "I confirm this source transcription" in item.label for item in app.checkbox
    )
    assert any(
        "Translate confirmed source lines" in item.label for item in app.button
    )
    assert any("Automatic Punjabi handwriting transcription is unavailable" in item.value for item in app.warning)
    assert any(
        "HTR unavailable" in item.value
        and "manual source transcription is required" in item.value
        for item in app.caption
    )
    assert not any("Allow an output with unconfirmed handwriting" in item.label for item in app.checkbox)
    assert any("At least one confirmed line" in item.value for item in app.caption)


def test_single_htr_unavailable_line_returns_only_explicitly_confirmed_gurmukhi() -> None:
    app = AppTest.from_string(REVIEW_APP, default_timeout=20).run()
    source_text = "\u0a87\u0a39 \u0a2a\u0a70\u0a1c\u0a3e\u0a2c\u0a40 \u0a39\u0a71\u0a25 \u0a32\u0a3f\u0a16\u0a24 \u0a39\u0a48"
    source_editor = next(
        item for item in app.text_area
        if item.label == "Recognized Punjabi/Hindi source text"
    )
    confirmation = next(
        item for item in app.checkbox
        if "I confirm this source transcription" in item.label
    )
    submit = next(
        item for item in app.button
        if "Translate confirmed source lines" in item.label
    )

    source_editor.set_value(source_text)
    confirmation.set_value(True)
    submit.click()
    app.run()

    assert not app.exception
    captured = app.session_state["captured_review_edits"]
    assert list(captured.values()) == [source_text]
    assert "htr-unconfirmed-line" not in captured
