"""Safe output naming, audit serialization, and optional local persistence."""

from __future__ import annotations

from pathlib import Path

from .schemas import DocumentModel
from .report_generator import AuditReportGenerator
from .utils.file_utils import safe_write_new, translated_output_name


def make_output_filename(source_filename: str, extension: str) -> str:
    return translated_output_name(source_filename, extension)


def make_audit_json(document: DocumentModel) -> bytes:
    return AuditReportGenerator().to_json_bytes(document)


def persist_output(directory: Path, filename: str, data: bytes) -> Path:
    """Persist only when explicitly requested, never overwriting an existing output."""
    return safe_write_new(directory / filename, data)
