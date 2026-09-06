from translator_app.config.settings import Settings
from translator_app.core.document_loader import DocumentLoader
from translator_app.core.image_processor import ImagePreprocessor
from translator_app.schemas import ContentKind, DocumentModel, FileFormat, PageModel, ProcessingOptions
from translator_app.utils.validation import ValidatedFile


def test_pdf_is_routed_to_pdf_processor(monkeypatch) -> None:
    settings = Settings(ocr_languages=["eng"])
    loader = DocumentLoader(settings, lambda: object(), ImagePreprocessor())
    expected = DocumentModel(
        "id",
        "sample.pdf",
        FileFormat.PDF,
        ContentKind.NATIVE,
        [PageModel(1, 100, 100)],
        b"%PDF-1.7",
    )
    called = {}

    def fake_process(validated, options, callback):
        called["format"] = validated.file_format
        return expected

    monkeypatch.setattr(loader.pdf_processor, "process", fake_process)
    validated = ValidatedFile(
        "sample.pdf", b"%PDF-1.7", FileFormat.PDF, "application/pdf", "abc"
    )
    result = loader.load(validated, ProcessingOptions(ocr_languages=["eng"]))
    assert result is expected
    assert called["format"] == FileFormat.PDF

