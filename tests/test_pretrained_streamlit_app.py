"""Smoke test for the pretrained OCR Streamlit entry point."""

from io import BytesIO
from pathlib import Path

import pymupdf
from docx import Document
from PIL import Image
from streamlit.testing.v1 import AppTest


def test_upload_app_starts_without_processing_models() -> None:
    app_path = (
        Path(__file__).resolve().parents[1]
        / "pretrained_page_ocr"
        / "streamlit_app.py"
    )

    app = AppTest.from_file(str(app_path)).run(timeout=20)

    assert not app.exception
    assert [title.value for title in app.title] == [
        "Hindi and Gurmukhi document translator"
    ]


def test_uploaded_page_exposes_fast_mode_and_full_page_action() -> None:
    app_path = (
        Path(__file__).resolve().parents[1]
        / "pretrained_page_ocr"
        / "streamlit_app.py"
    )
    stream = BytesIO()
    Image.new("RGB", (80, 100), "white").save(stream, format="PNG")

    app = AppTest.from_file(str(app_path)).run(timeout=20)
    app.file_uploader[0].upload(
        "page.png", stream.getvalue(), "image/png"
    ).run(timeout=20)

    assert not app.exception
    assert app.segmented_control[0].value == "fast_document"
    assert app.segmented_control[0].options == [
        "Fast document",
        "Hybrid handwriting accuracy",
    ]
    assert [button.label for button in app.button] == [
        "Transcribe and translate document"
    ]


def test_scanned_pdf_upload_is_validated_without_running_models() -> None:
    app_path = (
        Path(__file__).resolve().parents[1]
        / "pretrained_page_ocr"
        / "streamlit_app.py"
    )
    document = pymupdf.open()
    page = document.new_page(width=300, height=400)
    page.insert_text((30, 40), "Scanned page placeholder")
    pdf_bytes = document.tobytes()
    document.close()

    app = AppTest.from_file(str(app_path)).run(timeout=20)
    app.file_uploader[0].upload(
        "scan.pdf", pdf_bytes, "application/pdf"
    ).run(timeout=20)

    assert not app.exception
    assert app.segmented_control[0].value == "fast_document"
    assert any("PDF" in caption.value for caption in app.caption)


def test_docx_upload_is_validated_without_running_models() -> None:
    app_path = (
        Path(__file__).resolve().parents[1]
        / "pretrained_page_ocr"
        / "streamlit_app.py"
    )
    document = Document()
    document.add_paragraph("नमस्ते दुनिया")
    stream = BytesIO()
    document.save(stream)

    app = AppTest.from_file(str(app_path)).run(timeout=20)
    app.file_uploader[0].upload(
        "letter.docx",
        stream.getvalue(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ).run(timeout=20)

    assert not app.exception
    assert app.segmented_control[0].value == "fast_document"
    assert any("DOCX" in caption.value for caption in app.caption)
