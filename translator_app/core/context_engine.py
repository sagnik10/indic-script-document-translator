"""Backward-compatible facade for the bounded source reconstruction service."""

from __future__ import annotations

from typing import Any, Callable

from ..config.settings import Settings
from .missing_text_detector import EXPLICIT_MISSING_PATTERN as MISSING_PATTERN
from .source_reconstruction import (
    LocalMaskedLanguageModelProvider,
    ReconstructionDecision,
    SourceReconstructor,
)


class ConservativeReconstructor(SourceReconstructor):
    """Compatibility adapter; new code should use :class:`SourceReconstructor`."""

    def __init__(
        self,
        low_confidence_threshold: float,
        accept_threshold: float,
        model_loader: Callable[[], Any | None] | None = None,
        document_domain: str = "auto",
        minimum_readable_ratio: float = 0.72,
    ) -> None:
        review_threshold = min(0.70, accept_threshold)
        settings = Settings(
            ocr_low_confidence_threshold=low_confidence_threshold,
            reconstruction_accept_threshold=accept_threshold,
            auto_reconstruct_threshold=accept_threshold,
            review_reconstruct_threshold=review_threshold,
            min_context_quality=max(minimum_readable_ratio, 0.72),
            reconstruction_min_readable_ratio=minimum_readable_ratio,
        )
        provider = (
            LocalMaskedLanguageModelProvider(model_loader) if model_loader is not None else None
        )
        super().__init__(
            settings,
            provider,
            auto_threshold=accept_threshold,
            review_threshold=review_threshold,
            minimum_context_quality=max(minimum_readable_ratio, 0.72),
            document_domain=document_domain,
        )


__all__ = ["ConservativeReconstructor", "MISSING_PATTERN", "ReconstructionDecision"]
