"""Domain-specific exceptions with safe messages for the UI."""

from __future__ import annotations


class DocumentTranslatorError(Exception):
    """Base error carrying a concise user-facing explanation."""

    user_message = "The document could not be processed. See the application log for details."

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        if user_message:
            self.user_message = user_message


class FileValidationError(DocumentTranslatorError):
    user_message = "The uploaded file is unsupported, empty, too large, or does not match its extension."


class CorruptedDocumentError(DocumentTranslatorError):
    user_message = "The document appears to be corrupted or unreadable."


class PasswordProtectedPDFError(DocumentTranslatorError):
    user_message = "Password-protected PDFs are not supported. Please upload an unlocked copy."


class DependencyUnavailableError(DocumentTranslatorError):
    user_message = "A required local component is unavailable. Check the setup instructions in README.md."


class OCRUnavailableError(DependencyUnavailableError):
    user_message = "OCR is unavailable or the required language pack is missing. Check the Tesseract setup instructions."


class TranslationUnavailableError(DependencyUnavailableError):
    user_message = "The local translation model is unavailable. Source text was retained where translation could not run."


class NoTranslationProducedError(DocumentTranslatorError):
    """Raised when a translated deliverable was requested but no block changed."""

    user_message = (
        "No Punjabi/Hindi text was translated, so an unchanged source copy was not exported "
        "as a translated document. Confirm at least one credible source-language line in the "
        "review step and try again."
    )


class RenderingError(DocumentTranslatorError):
    user_message = "The translated document could not be rendered. No source file was overwritten."
