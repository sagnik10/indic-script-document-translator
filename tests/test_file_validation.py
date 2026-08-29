from __future__ import annotations

import io
import zipfile

import pytest

from translator_app.exceptions import FileValidationError
from translator_app.schemas import FileFormat
from translator_app.utils.file_utils import sanitize_filename, translated_output_name
from translator_app.utils.validation import validate_upload


def make_minimal_docx_container() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", "<w:document />")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("name", "data", "expected"),
    [
        ("scan.pdf", b"%PDF-1.7\n", FileFormat.PDF),
        ("scan.png", b"\x89PNG\r\n\x1a\nrest", FileFormat.PNG),
        ("photo.jpeg", b"\xff\xd8\xff\xe0rest", FileFormat.JPEG),
        ("letter.docx", make_minimal_docx_container(), FileFormat.DOCX),
    ],
)
def test_supported_uploads(name: str, data: bytes, expected: FileFormat) -> None:
    validated = validate_upload(name, data, 10_000)
    assert validated.file_format == expected
    assert len(validated.sha256) == 64


def test_extension_signature_mismatch_is_rejected() -> None:
    with pytest.raises(FileValidationError):
        validate_upload("actually-not-a-pdf.pdf", b"hello", 1_000)


def test_empty_oversize_and_unsupported_are_rejected() -> None:
    with pytest.raises(FileValidationError):
        validate_upload("empty.pdf", b"", 1_000)
    with pytest.raises(FileValidationError):
        validate_upload("large.pdf", b"%PDF-" + b"x" * 100, 20)
    with pytest.raises(FileValidationError):
        validate_upload("malware.exe", b"MZ", 1_000)


def test_filename_sanitization_prevents_traversal_and_safe_output_name() -> None:
    assert sanitize_filename("../../My regional report.PDF") == "My_regional_report.pdf"
    assert translated_output_name("../../My regional report.PDF", "pdf") == (
        "My_regional_report_translated_en.pdf"
    )

