import io

from PIL import Image, ImageDraw

from translator_app.config.settings import Settings
from translator_app.pipeline import DocumentTranslationPipeline
from translator_app.schemas import ProcessingOptions


def test_scanned_image_can_be_preserved_when_all_ocr_is_disabled() -> None:
    image = Image.new("RGB", (500, 700), "white")
    ImageDraw.Draw(image).text((50, 80), "Source image remains visible", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    settings = Settings(
        ocr_languages=["eng"],
        handwriting_models={"en": None, "hi": None, "pa": None},
        preprocessing_profile="clean_scan",
    )
    options = ProcessingOptions(
        ocr_languages=["eng"],
        preprocessing_profile="clean_scan",
        enable_printed_ocr=False,
        enable_handwriting_ocr=False,
    )
    result = DocumentTranslationPipeline(settings).process(
        "preserve.png", buffer.getvalue(), options
    )
    assert result.output_filename == "preserve_translated_en.pdf"
    assert result.output_bytes.startswith(b"%PDF-")
    assert result.summary.text_block_count == 0
    assert any("No text was extracted" in warning for warning in result.summary.warnings)
