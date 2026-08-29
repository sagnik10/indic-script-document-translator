"""Confidence and provenance state transitions shared by OCR and HTR engines."""

from __future__ import annotations

from ..schemas import ProcessingStatus, ReconstructionType, TextBlock, UncertaintyState


class ConfidenceAnalyzer:
    def __init__(self, printed_threshold: float, handwriting_threshold: float) -> None:
        self.printed_threshold = printed_threshold
        self.handwriting_threshold = handwriting_threshold

    def assess(self, block: TextBlock) -> TextBlock:
        threshold = self.handwriting_threshold if block.is_handwritten else self.printed_threshold
        if not block.source_text.strip() or block.source_text.startswith("[unclear"):
            block.uncertainty_state = UncertaintyState.FLAGGED
            block.reconstruction_type = ReconstructionType.UNREADABLE
            block.provenance.append("UNREADABLE: no reliable OCR/HTR output")
            if ProcessingStatus.UNREADABLE not in block.processing_statuses:
                block.processing_statuses.append(ProcessingStatus.UNREADABLE)
        elif block.ocr_confidence is None or block.ocr_confidence < threshold:
            block.uncertainty_state = UncertaintyState.LOW_OCR_CONFIDENCE
            block.reconstruction_type = ReconstructionType.OCR_EXTRACTED
            block.provenance.append(
                f"OCR_EXTRACTED below threshold {threshold:.2f}; retained for review"
            )
            if ProcessingStatus.OCR_LOW_CONFIDENCE not in block.processing_statuses:
                block.processing_statuses.append(ProcessingStatus.OCR_LOW_CONFIDENCE)
        else:
            block.uncertainty_state = UncertaintyState.CONFIRMED
            block.reconstruction_type = ReconstructionType.OCR_EXTRACTED
            block.provenance.append("OCR_EXTRACTED above configured threshold")
            if ProcessingStatus.OCR_CONFIRMED not in block.processing_statuses:
                block.processing_statuses.append(ProcessingStatus.OCR_CONFIRMED)
        return block
