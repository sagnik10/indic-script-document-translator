"""Constrained missing-span reconstruction in Punjabi/Hindi source text only."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from ..config.settings import Settings
from ..schemas import (
    ProcessingStatus,
    ReconstructionStatus,
    ReconstructionType,
    ScriptType,
    TextBlock,
    TranslationStatus,
    UncertaintyState,
)
from ..utils.text_utils import normalize_text
from .missing_text_detector import EXPLICIT_MISSING_PATTERN, MissingSpanDetection, MissingTextDetector
from .script_detection import script_counts, script_ratio
from .source_validation import (
    PageLanguageContext,
    calculate_text_quality,
    normalize_language,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SpanPrediction:
    candidate: str
    confidence: float
    method: str


@dataclass(frozen=True, slots=True)
class ReconstructionDecision:
    source_text: str
    candidate_text: str | None
    confidence: float | None
    accepted: bool
    reason: str
    status: ReconstructionStatus = ReconstructionStatus.NOT_DETECTED


class SourceSpanPredictionProvider(ABC):
    """Predict only the short missing source span, never a translation/full rewrite."""

    @abstractmethod
    def predict_span(
        self,
        masked_source_text: str,
        *,
        language: str,
        expected_script: ScriptType,
        previous_text: str,
        next_text: str,
        ocr_alternatives: list[dict[str, Any]],
    ) -> SpanPrediction | None:
        raise NotImplementedError


class LocalMaskedLanguageModelProvider(SourceSpanPredictionProvider):
    """Lazy adapter around a local Transformers fill-mask pipeline."""

    def __init__(self, model_loader: Callable[[], Any | None] | None) -> None:
        self.model_loader = model_loader

    def predict_span(
        self,
        masked_source_text: str,
        *,
        language: str,
        expected_script: ScriptType,
        previous_text: str,
        next_text: str,
        ocr_alternatives: list[dict[str, Any]],
    ) -> SpanPrediction | None:
        if self.model_loader is None:
            return None
        model = self.model_loader()
        if model is None:
            return None
        mask_token = getattr(getattr(model, "tokenizer", None), "mask_token", "[MASK]")
        model_input = masked_source_text.replace("<MISSING_SPAN>", mask_token, 1)
        context = " ".join(
            part.strip() for part in (previous_text, model_input, next_text) if part.strip()
        )
        predictions = model(context, top_k=5)
        if predictions and isinstance(predictions[0], list):
            predictions = predictions[0]
        for prediction in predictions or []:
            token = normalize_text(str(prediction.get("token_str", ""))).strip()
            if token:
                return SpanPrediction(
                    token,
                    max(0.0, min(1.0, float(prediction.get("score", 0.0)))),
                    "local_masked_language_model",
                )
        return None


def _add_status(block: TextBlock, status: ProcessingStatus) -> None:
    if status not in block.processing_statuses:
        block.processing_statuses.append(status)


class SourceReconstructor:
    """Detect, gate, predict, and validate a bounded source-language span."""

    def __init__(
        self,
        settings: Settings,
        provider: SourceSpanPredictionProvider | None = None,
        *,
        auto_threshold: float | None = None,
        review_threshold: float | None = None,
        minimum_context_quality: float | None = None,
        document_domain: str = "auto",
    ) -> None:
        self.settings = settings
        self.detector = MissingTextDetector(settings)
        self.provider = provider
        self.auto_threshold = (
            settings.auto_reconstruct_threshold if auto_threshold is None else auto_threshold
        )
        self.review_threshold = (
            settings.review_reconstruct_threshold if review_threshold is None else review_threshold
        )
        self.minimum_context_quality = (
            settings.min_context_quality
            if minimum_context_quality is None
            else minimum_context_quality
        )
        self.document_domain = document_domain.casefold()

    @staticmethod
    def _deterministic_candidate(text: str) -> SpanPrediction | None:
        normalized = normalize_text(text)
        repaired = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", normalized)
        repaired = re.sub(r"(?<=\w)\s+([,.;:!?।])", r"\1", repaired)
        if repaired != normalized:
            return SpanPrediction(repaired, 0.96, "deterministic_ocr_normalization")
        return None

    @staticmethod
    def _mask_source(source: str, detection: MissingSpanDetection) -> str:
        assert detection.start is not None and detection.end is not None
        return source[: detection.start] + "<MISSING_SPAN>" + source[detection.end :]

    @staticmethod
    def _replace_span(source: str, detection: MissingSpanDetection, candidate: str) -> str:
        assert detection.start is not None and detection.end is not None
        return normalize_text(source[: detection.start] + candidate + source[detection.end :])

    @staticmethod
    def _alternative_candidate(
        block: TextBlock,
        source: str,
        detection: MissingSpanDetection,
    ) -> SpanPrediction | None:
        if detection.start is None:
            return None
        source_tokens = list(re.finditer(r"\S+", source))
        missing_index = next(
            (
                index
                for index, match in enumerate(source_tokens)
                if match.start() <= detection.start < match.end()
            ),
            None,
        )
        if missing_index is None:
            return None
        candidates: list[SpanPrediction] = []
        for alternative in block.ocr_alternatives:
            alternative_text = normalize_text(str(alternative.get("text", "")))
            tokens = alternative_text.split()
            if len(tokens) != len(source_tokens) or missing_index >= len(tokens):
                continue
            candidate = tokens[missing_index].strip()
            if not candidate or EXPLICIT_MISSING_PATTERN.search(candidate):
                continue
            confidence = max(0.0, min(1.0, float(alternative.get("confidence") or 0.0)))
            candidates.append(
                SpanPrediction(candidate, confidence, "script_consistent_ocr_alternative")
            )
        return max(candidates, key=lambda item: item.confidence) if candidates else None

    def _candidate_is_safe(
        self,
        candidate: str,
        detection: MissingSpanDetection,
    ) -> tuple[bool, str]:
        value = normalize_text(candidate).strip()
        if not value:
            return False, "reconstruction provider returned an empty span"
        tokens = value.split()
        if len(tokens) > self.settings.max_reconstruction_span_tokens or len(value) > 64:
            return False, "provider attempted to rewrite more than the bounded missing span"
        if EXPLICIT_MISSING_PATTERN.search(value):
            return False, "candidate still contains missing-glyph markers"
        if any(character.isdigit() for character in value):
            return False, "numeric reconstruction is prohibited"
        expected = detection.expected_script
        if expected not in {ScriptType.GURMUKHI, ScriptType.DEVANAGARI}:
            return False, "source script is not eligible for reconstruction"
        purity = script_ratio(value, expected)
        if purity < max(0.80, self.settings.min_source_script_ratio):
            return False, f"candidate is not consistently {expected.value} source text"
        return True, "candidate is bounded and source-script consistent"

    def _mark_unresolved(
        self,
        block: TextBlock,
        source: str,
        detection: MissingSpanDetection,
        reason: str,
        status: ReconstructionStatus,
    ) -> ReconstructionDecision:
        if detection.start is not None and detection.end is not None:
            block.reconstructed_text = self._replace_span(source, detection, "[unclear]")
        block.reconstruction_status = status
        block.reconstruction_type = ReconstructionType.UNREADABLE
        block.uncertainty_state = UncertaintyState.FLAGGED
        block.translation_status = TranslationStatus.SKIPPED
        _add_status(block, ProcessingStatus.UNREADABLE)
        _add_status(block, ProcessingStatus.TRANSLATION_SKIPPED)
        block.provenance.append(f"UNREADABLE missing span: {reason}")
        return ReconstructionDecision(source, block.reconstruction_candidate, block.reconstruction_confidence, False, reason, status)

    def evaluate(
        self,
        block: TextBlock,
        previous_text: str = "",
        next_text: str = "",
        page_context: PageLanguageContext | None = None,
    ) -> ReconstructionDecision:
        source = block.normalized_text or normalize_text(block.source_text)
        block.normalized_text = source
        deterministic = self._deterministic_candidate(source)
        if deterministic is not None:
            block.reconstruction_candidate = deterministic.candidate
            block.reconstructed_text = deterministic.candidate
            block.reconstruction_confidence = deterministic.confidence
            block.reconstruction_method = deterministic.method
            accepted = deterministic.confidence >= self.auto_threshold
            if accepted:
                block.reconstructed_text = deterministic.candidate
                block.reconstruction_type = ReconstructionType.OCR_CORRECTED
                block.reconstruction_status = ReconstructionStatus.AUTO_ACCEPTED
                block.uncertainty_state = UncertaintyState.RECONSTRUCTED
                block.metadata["correction_origin"] = "automatic_normalization"
                _add_status(block, ProcessingStatus.RECONSTRUCTED_SOURCE)
            else:
                block.uncertainty_state = UncertaintyState.FLAGGED
            return ReconstructionDecision(
                source,
                deterministic.candidate,
                deterministic.confidence,
                accepted,
                deterministic.method,
                block.reconstruction_status,
            )

        context = page_context or PageLanguageContext(block.script, 0.0)
        detection = self.detector.detect(
            block,
            previous_text=previous_text,
            next_text=next_text,
            page_script=context.dominant_script,
        )
        if not detection.detected:
            if block.ocr_confidence is not None and block.ocr_confidence < self.settings.ocr_low_confidence_threshold:
                block.uncertainty_state = UncertaintyState.FLAGGED
            return ReconstructionDecision(source, None, None, False, "no missing span detected")
        block.reconstruction_status = ReconstructionStatus.DETECTED
        if not detection.reconstructable:
            return self._mark_unresolved(
                block,
                source,
                detection,
                "suspicion has no safe, bounded missing span",
                ReconstructionStatus.BLOCKED,
            )
        if detection.protected_entity_detected:
            return self._mark_unresolved(
                block,
                source,
                detection,
                "automatic inference prohibited for a protected name/date/number/medical/legal/address/signature context",
                ReconstructionStatus.BLOCKED,
            )
        language = normalize_language(block.detected_language)
        expected_language = "pa" if detection.expected_script == ScriptType.GURMUKHI else "hi"
        if language not in {"pa", "hi"} or language != expected_language:
            return self._mark_unresolved(
                block,
                source,
                detection,
                "source language/script is undetermined or inconsistent",
                ReconstructionStatus.BLOCKED,
            )
        if block.metadata.get("handwriting_unsupported") and not block.metadata.get("human_reviewed"):
            return self._mark_unresolved(
                block, source, detection, "unsupported handwriting cannot be inferred", ReconstructionStatus.BLOCKED
            )
        minimum_readable = max(
            self.settings.reconstruction_min_readable_ratio, self.minimum_context_quality
        )
        if detection.readable_ratio < minimum_readable:
            return self._mark_unresolved(
                block,
                source,
                detection,
                f"readable character ratio {detection.readable_ratio:.2f} is below {minimum_readable:.2f}",
                ReconstructionStatus.BLOCKED,
            )
        if detection.context_quality < self.minimum_context_quality:
            return self._mark_unresolved(
                block,
                source,
                detection,
                f"source context quality {detection.context_quality:.2f} is below {self.minimum_context_quality:.2f}",
                ReconstructionStatus.BLOCKED,
            )
        if detection.validated_context_token_count < self.settings.min_validated_context_tokens:
            return self._mark_unresolved(
                block,
                source,
                detection,
                "too few validated neighboring source-language tokens",
                ReconstructionStatus.BLOCKED,
            )
        cleaned_context = self._replace_span(source, detection, "")
        counts = script_counts(cleaned_context)
        total_script_characters = sum(counts.values())
        expected_count = counts.get(detection.expected_script.value, 0)
        competing_count = total_script_characters - expected_count
        if competing_count >= 3 and competing_count / max(1, total_script_characters) > 0.12:
            return self._mark_unresolved(
                block,
                source,
                detection,
                "mixed-script context is not trustworthy enough for reconstruction",
                ReconstructionStatus.BLOCKED,
            )
        if script_ratio(cleaned_context, detection.expected_script) < self.settings.min_source_script_ratio:
            return self._mark_unresolved(
                block,
                source,
                detection,
                "mixed-script context is not trustworthy enough for reconstruction",
                ReconstructionStatus.BLOCKED,
            )

        prediction = self._alternative_candidate(block, source, detection)
        if prediction is None and self.provider is not None:
            try:
                prediction = self.provider.predict_span(
                    self._mask_source(source, detection),
                    language=language,
                    expected_script=detection.expected_script,
                    previous_text=previous_text,
                    next_text=next_text,
                    ocr_alternatives=block.ocr_alternatives,
                )
            except Exception:
                LOGGER.warning("Source-span reconstruction provider failed", exc_info=True)
                prediction = None
        if prediction is None:
            return self._mark_unresolved(
                block,
                source,
                detection,
                "source reconstruction model/alternative is unavailable",
                ReconstructionStatus.UNAVAILABLE,
            )
        block.reconstruction_candidate = normalize_text(prediction.candidate).strip()
        block.reconstruction_confidence = max(0.0, min(1.0, prediction.confidence))
        block.reconstruction_method = prediction.method
        safe, safety_reason = self._candidate_is_safe(block.reconstruction_candidate, detection)
        if not safe:
            return self._mark_unresolved(
                block, source, detection, safety_reason, ReconstructionStatus.REJECTED
            )
        reconstructed = self._replace_span(source, detection, block.reconstruction_candidate)
        reconstructed_quality = calculate_text_quality(reconstructed, detection.expected_script)
        if reconstructed_quality < self.minimum_context_quality:
            return self._mark_unresolved(
                block,
                source,
                detection,
                "reconstructed source line failed final text-quality validation",
                ReconstructionStatus.REJECTED,
            )
        block.metadata["proposed_reconstructed_source_text"] = reconstructed
        block.provenance.append(
            f"MODEL_INFERRED missing-span candidate; method={prediction.method}; "
            f"confidence={block.reconstruction_confidence:.3f}; candidate not English"
        )
        if block.reconstruction_confidence >= self.auto_threshold:
            block.reconstructed_text = reconstructed
            block.reconstruction_type = ReconstructionType.MODEL_INFERRED
            block.reconstruction_status = ReconstructionStatus.AUTO_ACCEPTED
            block.uncertainty_state = UncertaintyState.RECONSTRUCTED
            block.metadata["correction_origin"] = "automatic_source_reconstruction"
            _add_status(block, ProcessingStatus.RECONSTRUCTED_SOURCE)
            _add_status(block, ProcessingStatus.RECONSTRUCTION_AUTO_ACCEPTED)
            return ReconstructionDecision(
                source,
                block.reconstruction_candidate,
                block.reconstruction_confidence,
                True,
                "source-language missing span auto-accepted",
                block.reconstruction_status,
            )
        if block.reconstruction_confidence >= self.review_threshold:
            block.reconstruction_status = ReconstructionStatus.CANDIDATE_REVIEW
            block.uncertainty_state = UncertaintyState.CANDIDATE
            block.translation_status = TranslationStatus.SKIPPED
            _add_status(block, ProcessingStatus.RECONSTRUCTION_REVIEW_REQUIRED)
            _add_status(block, ProcessingStatus.TRANSLATION_SKIPPED)
            return ReconstructionDecision(
                source,
                block.reconstruction_candidate,
                block.reconstruction_confidence,
                False,
                "source-language missing span requires manual confirmation",
                block.reconstruction_status,
            )
        return self._mark_unresolved(
            block,
            source,
            detection,
            "candidate confidence is below the review threshold",
            ReconstructionStatus.REJECTED,
        )

    def process_blocks(
        self,
        blocks: list[TextBlock],
        page_context: PageLanguageContext | None = None,
    ) -> list[ReconstructionDecision]:
        decisions: list[ReconstructionDecision] = []
        for index, block in enumerate(blocks):
            previous_text = blocks[index - 1].effective_source_text if index else ""
            next_text = blocks[index + 1].effective_source_text if index + 1 < len(blocks) else ""
            decisions.append(
                self.evaluate(block, previous_text, next_text, page_context=page_context)
            )
        return decisions


__all__ = [
    "LocalMaskedLanguageModelProvider",
    "ReconstructionDecision",
    "SourceReconstructor",
    "SourceSpanPredictionProvider",
    "SpanPrediction",
]
