"""Output backend selection for the typed intermediate document model."""

from __future__ import annotations

from ..config.settings import Settings
from ..schemas import DocumentModel, FileFormat
from ..reconstruction.docx_reconstructor import reconstruct_docx
from ..reconstruction.pdf_reconstructor import reconstruct_pdf, reconstruct_translation_report


class DocumentReconstructionEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def rebuild(self, document: DocumentModel) -> tuple[bytes, str, str]:
        """Build only the format-preserving primary document.

        ``translation_only_report`` was historically accepted as a reconstruction
        mode.  It is intentionally ignored here: a transcript must never replace
        the user's document as the primary export.  Call
        :meth:`build_diagnostic_translation_report` explicitly for an optional
        diagnostic artifact.
        """
        if document.file_format == FileFormat.DOCX:
            return reconstruct_docx(document), "docx", (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        return reconstruct_pdf(document, self.settings), "pdf", "application/pdf"

    @staticmethod
    def build_diagnostic_translation_report(document: DocumentModel) -> tuple[bytes, str, str]:
        """Create an explicitly requested diagnostic transcript, never primary output."""
        return reconstruct_translation_report(document), "pdf", "application/pdf"
