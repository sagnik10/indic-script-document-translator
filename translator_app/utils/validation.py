"""Untrusted upload validation using extension, signature, and container checks."""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..exceptions import FileValidationError
from ..schemas import FileFormat
from .file_utils import sanitize_filename


MIME_BY_FORMAT = {
    FileFormat.PDF: "application/pdf",
    FileFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    FileFormat.PNG: "image/png",
    FileFormat.JPEG: "image/jpeg",
}


@dataclass(frozen=True, slots=True)
class ValidatedFile:
    filename: str
    data: bytes
    file_format: FileFormat
    mime_type: str
    sha256: str


def _format_from_extension(filename: str) -> FileFormat:
    extension = Path(filename).suffix.lower()
    mapping = {
        ".pdf": FileFormat.PDF,
        ".docx": FileFormat.DOCX,
        ".png": FileFormat.PNG,
        ".jpg": FileFormat.JPEG,
        ".jpeg": FileFormat.JPEG,
    }
    if extension not in mapping:
        raise FileValidationError(f"Unsupported extension: {extension or '[none]'}")
    return mapping[extension]


def _validate_signature(data: bytes, file_format: FileFormat) -> None:
    if file_format == FileFormat.PDF and not data.startswith(b"%PDF-"):
        raise FileValidationError("PDF signature mismatch")
    if file_format == FileFormat.PNG and not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise FileValidationError("PNG signature mismatch")
    if file_format == FileFormat.JPEG and not data.startswith(b"\xff\xd8\xff"):
        raise FileValidationError("JPEG signature mismatch")
    if file_format == FileFormat.DOCX:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = set(archive.namelist())
                required = {"[Content_Types].xml", "word/document.xml"}
                if not required.issubset(names):
                    raise FileValidationError("ZIP container is not a DOCX document")
        except zipfile.BadZipFile as exc:
            raise FileValidationError("Invalid DOCX ZIP container") from exc


def validate_upload(filename: str, data: bytes, max_size: int) -> ValidatedFile:
    """Validate an upload without executing or writing it to a user-controlled path."""
    if not data:
        raise FileValidationError("The uploaded file is empty")
    if len(data) > max_size:
        raise FileValidationError(
            f"File size {len(data)} exceeds configured limit {max_size}"
        )
    safe_name = sanitize_filename(filename)
    file_format = _format_from_extension(safe_name)
    _validate_signature(data, file_format)
    return ValidatedFile(
        filename=safe_name,
        data=data,
        file_format=file_format,
        mime_type=MIME_BY_FORMAT[file_format],
        sha256=hashlib.sha256(data).hexdigest(),
    )
