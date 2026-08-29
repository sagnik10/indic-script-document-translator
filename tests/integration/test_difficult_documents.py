from pathlib import Path

import pytest

from translator_app.pipeline import DocumentTranslationPipeline
from translator_app.schemas import ProcessingOptions, RegionType


FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "difficult_documents"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("filename", "handwriting", "profile"),
    [
        ("punjabi_printed.pdf", False, "photocopy"),
        ("hindi_printed.pdf", False, "photocopy"),
        ("gurmukhi_handwriting.jpg", True, "handwriting_heavy"),
        ("devanagari_handwriting.jpg", True, "handwriting_heavy"),
        ("mixed_medical_form.jpg", True, "mobile_photo"),
        ("low_quality_photocopy.pdf", False, "photocopy"),
        ("mobile_photo.jpg", False, "mobile_photo"),
    ],
)
def test_private_difficult_document_scaffolding(
    filename: str, handwriting: bool, profile: str
) -> None:
    fixture = FIXTURE_DIRECTORY / filename
    if not fixture.exists():
        pytest.skip(f"Private integration fixture not installed: {filename}")
    options = ProcessingOptions(
        ocr_languages=["eng", "pan", "hin"],
        preprocessing_profile=profile,
        enable_handwriting_ocr=handwriting,
        review_before_render=True,
        debug_bounding_boxes=True,
    )
    analysis = DocumentTranslationPipeline().analyze(
        fixture.name, fixture.read_bytes(), options
    )
    assert analysis.document.pages
    assert all(page.width > 0 and page.height > 0 for page in analysis.document.pages)
    assert all(block.provenance for block in analysis.document.blocks)
    assert all(
        block.detected_language != "und" or not block.source_validated
        for block in analysis.document.blocks
    )
    assert all(
        not block.source_validated or block.text_quality >= 0
        for block in analysis.document.blocks
    )
    if handwriting:
        assert any(
            region.region_type == RegionType.HANDWRITING
            for page in analysis.document.pages
            for region in page.regions
        ), "handwriting fixture produced no handwriting regions"
    assert analysis.debug_preview_images or analysis.document.file_format.value == "docx"
