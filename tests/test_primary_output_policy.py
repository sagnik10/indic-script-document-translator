from __future__ import annotations

from translator_app.config.settings import Settings
from translator_app.core import reconstruction_engine as module
from translator_app.core.reconstruction_engine import DocumentReconstructionEngine
from translator_app.schemas import ContentKind, DocumentModel, FileFormat, PageModel


def _image_document() -> DocumentModel:
    return DocumentModel(
        document_id="primary-output-policy",
        source_filename="source.jpeg",
        file_format=FileFormat.JPEG,
        content_kind=ContentKind.SCANNED,
        pages=[PageModel(page_number=1, width=100, height=120)],
        source_bytes=b"source-image",
        metadata={"reconstruction_mode": "translation_only_report"},
    )


def test_legacy_report_mode_cannot_replace_primary_document(monkeypatch) -> None:
    calls: list[str] = []

    def fake_reconstruct_pdf(document, settings):
        calls.append("format_preserving_pdf")
        return b"%PDF-primary"

    def forbidden_report(document):
        raise AssertionError("diagnostic transcript was used as primary output")

    monkeypatch.setattr(module, "reconstruct_pdf", fake_reconstruct_pdf)
    monkeypatch.setattr(module, "reconstruct_translation_report", forbidden_report)

    data, extension, mime = DocumentReconstructionEngine(Settings()).rebuild(
        _image_document()
    )

    assert data == b"%PDF-primary"
    assert extension == "pdf"
    assert mime == "application/pdf"
    assert calls == ["format_preserving_pdf"]


def test_translation_report_requires_explicit_diagnostic_call(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "reconstruct_translation_report",
        lambda document: b"%PDF-diagnostic",
    )

    data, extension, mime = (
        DocumentReconstructionEngine.build_diagnostic_translation_report(
            _image_document()
        )
    )

    assert data == b"%PDF-diagnostic"
    assert extension == "pdf"
    assert mime == "application/pdf"
