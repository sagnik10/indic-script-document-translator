"""Strict source-language resolution and pre-translation quality gates."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..config.settings import Settings
from ..schemas import (
    ProcessingStatus,
    ReconstructionType,
    ScriptType,
    TextBlock,
    TranslationStatus,
    UncertaintyState,
)
from .script_detection import (
    dominant_script,
    linguistic_evidence_score,
    meaningful_dominant_script,
    script_counts,
    script_ratio,
)


LANGUAGE_ALIASES: dict[str, str] = {
    "pa": "pa",
    "pan": "pa",
    "pan_guru": "pa",
    "punjabi": "pa",
    "punjabi_gurmukhi": "pa",
    "hi": "hi",
    "hin": "hi",
    "hin_deva": "hi",
    "hindi": "hi",
    "en": "en",
    "eng": "en",
    "eng_latn": "en",
    "english": "en",
}
UNDETERMINED_ALIASES = {"", "und", "unknown", "none", "null", "mul"}
SUPPORTED_TRANSLATION_LANGUAGES = frozenset({"pa", "hi"})

_STRUCTURED_VALUE = re.compile(
    r"^(?:\+?[\d\s()./-]{3,}|[A-Z]{1,8}[-/:]\d[\w./-]*|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|https?://\S+|www\.\S+|"
    r"(?:Rs\.?|INR|₹)?\s*\d[\d,]*(?:\.\d+)?)$",
    re.IGNORECASE,
)
_GARBAGE_RUN = re.compile(r"([?_*#=~|])\1{1,}")


@dataclass(frozen=True, slots=True)
class PageLanguageContext:
    dominant_script: ScriptType = ScriptType.UNKNOWN
    confidence: float = 0.0
    handwriting_likelihood: float = 0.0
    visual_script: ScriptType = ScriptType.UNKNOWN
    visual_confidence: float = 0.0
    ocr_script: ScriptType = ScriptType.UNKNOWN
    ocr_confidence: float = 0.0
    resolution_reason: str = "not_resolved"
    expected_language_prior: str = "auto"


@dataclass(frozen=True, slots=True)
class LanguageResolution:
    language: str
    confidence: float
    script: ScriptType
    reason: str


@dataclass(frozen=True, slots=True)
class SourceValidation:
    valid: bool
    quality: float
    reason: str
    structured: bool = False


def normalize_language(language: object) -> str:
    if language is None:
        return "und"
    key = str(language).strip().casefold().replace("-", "_").replace(" ", "_")
    if key in UNDETERMINED_ALIASES:
        return "und"
    return LANGUAGE_ALIASES.get(key, key)


def is_structured_value(text: str) -> bool:
    value = " ".join(text.split()).strip()
    return bool(value and _STRUCTURED_VALUE.fullmatch(value))


def readable_character_ratio(text: str) -> float:
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    readable = sum(
        character.isalpha()
        or unicodedata.category(character).startswith("M")
        or character.isdigit()
        or character in ".,:;!?()[]{}'\"+-/%₹@#&"
        for character in visible
    )
    unreadable = sum(character in "�_" for character in visible)
    if "??" in text:
        unreadable += text.count("?")
    return max(0.0, min(1.0, (readable - unreadable) / len(visible)))


def calculate_text_quality(text: str, expected_script: str | ScriptType | None = None) -> float:
    value = " ".join(text.split()).strip()
    visible = [character for character in value if not character.isspace()]
    if not visible:
        return 0.0
    letters = [
        character
        for character in visible
        if character.isalpha() or unicodedata.category(character).startswith("M")
    ]
    letter_ratio = len(letters) / len(visible)
    length_score = min(1.0, len(letters) / 8.0)
    readable = readable_character_ratio(value)
    expected = None
    if expected_script is not None:
        expected = expected_script.value if isinstance(expected_script, ScriptType) else str(expected_script)
    purity = script_ratio(value, expected) if expected and expected not in {"unknown", "mixed", "digits"} else 1.0
    punctuation_ratio = sum(not character.isalnum() for character in visible) / len(visible)
    penalty = 0.0
    if _GARBAGE_RUN.search(value):
        penalty += 0.28
    if punctuation_ratio > 0.55:
        penalty += min(0.42, punctuation_ratio - 0.45)
    if len(letters) <= 1 and not is_structured_value(value):
        penalty += 0.35
    return max(
        0.0,
        min(1.0, 0.28 * letter_ratio + 0.22 * length_score + 0.22 * readable + 0.28 * purity - penalty),
    )


def _script_enum(name: str) -> ScriptType:
    return {
        "gurmukhi": ScriptType.GURMUKHI,
        "devanagari": ScriptType.DEVANAGARI,
        "latin": ScriptType.LATIN,
    }.get(name, ScriptType.UNKNOWN)


def resolve_language(
    text: str,
    script_hint: str | ScriptType | None = None,
    page_context: PageLanguageContext | None = None,
    ocr_confidence: float | None = None,
) -> LanguageResolution:
    """Resolve a canonical language; `und` remains a non-translatable outcome."""
    value = " ".join(text.split()).strip()
    if not value:
        return LanguageResolution("und", 0.0, ScriptType.UNKNOWN, "empty OCR output")
    if is_structured_value(value) and not any(character.isalpha() for character in value):
        return LanguageResolution("und", 1.0, ScriptType.DIGITS, "structured numeric token")
    counts = script_counts(value)
    letter_count = sum(counts.values())
    name, ratio = dominant_script(value)
    script = _script_enum(name)
    if letter_count == 0 or (letter_count <= 1 and not is_structured_value(value)):
        return LanguageResolution("und", 0.0, script, "insufficient linguistic characters")
    gurmukhi_ratio = counts.get("gurmukhi", 0) / letter_count
    devanagari_ratio = counts.get("devanagari", 0) / letter_count
    if gurmukhi_ratio >= 0.20 and counts.get("gurmukhi", 0) >= 4:
        return LanguageResolution(
            "pa",
            min(0.99, gurmukhi_ratio + min(counts["gurmukhi"], 8) * 0.02),
            ScriptType.GURMUKHI,
            "Gurmukhi Unicode source span",
        )
    if devanagari_ratio >= 0.20 and counts.get("devanagari", 0) >= 4:
        return LanguageResolution(
            "hi",
            min(0.99, devanagari_ratio + min(counts["devanagari"], 8) * 0.02),
            ScriptType.DEVANAGARI,
            "Devanagari Unicode source span",
        )
    if name == "latin" and ratio >= 0.60:
        prior = page_context or PageLanguageContext()
        quality = calculate_text_quality(value, ScriptType.LATIN)
        strong_indic_prior = (
            prior.dominant_script in {ScriptType.GURMUKHI, ScriptType.DEVANAGARI}
            and prior.confidence >= 0.60
        )
        genuine_english = (
            quality >= 0.68
            and (
                (letter_count >= 8 and (ocr_confidence is None or ocr_confidence >= 0.62))
                or (letter_count >= 4 and (ocr_confidence is None or ocr_confidence >= 0.78))
            )
        )
        if strong_indic_prior and not genuine_english:
            return LanguageResolution(
                "und",
                1.0 - prior.confidence,
                ScriptType.LATIN,
                "short Latin OCR conflicts with dominant Indic page script",
            )
        return LanguageResolution("en", min(0.98, ratio * quality + 0.10), script, "validated Latin text")
    hinted = script_hint.value if isinstance(script_hint, ScriptType) else str(script_hint or "").casefold()
    if hinted in {"gurmukhi", "devanagari"}:
        hinted_ratio = script_ratio(value, hinted)
        if hinted_ratio >= 0.25:
            language = "pa" if hinted == "gurmukhi" else "hi"
            return LanguageResolution(language, hinted_ratio, _script_enum(hinted), "script hint confirmed by Unicode")
    return LanguageResolution("und", 0.0, script, "script/language could not be resolved reliably")


def build_page_context(
    blocks: list[TextBlock],
    handwriting_likelihood: float = 0.0,
    *,
    visual_script: ScriptType = ScriptType.UNKNOWN,
    visual_confidence: float = 0.0,
    resolved_script: ScriptType = ScriptType.UNKNOWN,
    resolved_confidence: float = 0.0,
    resolution_reason: str = "recognized Unicode evidence",
    expected_language_prior: str = "auto",
) -> PageLanguageContext:
    weighted = {ScriptType.GURMUKHI: 0.0, ScriptType.DEVANAGARI: 0.0, ScriptType.LATIN: 0.0}
    for block in blocks:
        if block.metadata.get("handwriting_unsupported"):
            # A routing hint attached to unreadable pixels is not linguistic evidence.
            continue
        text = block.normalized_text or block.source_text
        confidence = max(0.1, block.ocr_confidence or (1.0 if not block.is_ocr else 0.0))
        script, evidence = meaningful_dominant_script(text)
        block.linguistic_evidence_score = evidence
        block.recognized_unicode_script = script
        block.ocr_script_confidence = evidence
        if script in weighted:
            multiplier = 1.0
            if block.metadata.get("printed_ocr_probe") and handwriting_likelihood >= 0.55:
                multiplier = 0.10 if script == ScriptType.LATIN else 1.25
            weighted[script] += max(1.0, len(text) ** 0.5) * confidence * evidence * multiplier
        for alternative in block.ocr_alternatives:
            alternative_text = str(alternative.get("text", ""))
            alternative_confidence = max(0.05, float(alternative.get("confidence") or 0.0))
            alternative_script, alternative_evidence = meaningful_dominant_script(
                alternative_text
            )
            if alternative_script in weighted:
                weighted[alternative_script] += (
                    max(1.0, len(alternative_text) ** 0.5)
                    * alternative_confidence
                    * alternative_evidence
                    * 0.30
                )
    total = sum(weighted.values())
    ocr_script, ocr_score = (
        max(weighted.items(), key=lambda item: item[1]) if total else (ScriptType.UNKNOWN, 0.0)
    )
    ocr_confidence = ocr_score / total if total else 0.0
    dominant = resolved_script if resolved_script != ScriptType.UNKNOWN else ocr_script
    confidence = resolved_confidence if resolved_script != ScriptType.UNKNOWN else ocr_confidence
    return PageLanguageContext(
        dominant,
        confidence,
        handwriting_likelihood,
        visual_script,
        visual_confidence,
        ocr_script,
        ocr_confidence,
        resolution_reason,
        expected_language_prior,
    )


def _add_status(block: TextBlock, status: ProcessingStatus) -> None:
    if status not in block.processing_statuses:
        block.processing_statuses.append(status)


def validate_source_block(
    block: TextBlock,
    settings: Settings,
    page_context: PageLanguageContext,
) -> SourceValidation:
    text = block.effective_source_text.strip()
    htr_unavailable = bool(
        (block.metadata.get("handwriting_unsupported") or block.metadata.get("htr_unavailable"))
        and not block.metadata.get("human_reviewed")
    )
    if htr_unavailable:
        preserved_language = normalize_language(block.detected_language)
        preserved_script = {
            "pa": ScriptType.GURMUKHI,
            "hi": ScriptType.DEVANAGARI,
            "en": ScriptType.LATIN,
        }.get(preserved_language, page_context.dominant_script)
        resolution = LanguageResolution(
            preserved_language if preserved_language in {"pa", "hi", "en"} else "und",
            page_context.confidence,
            preserved_script,
            "handwriting provider unavailable; language is a region-level hint only",
        )
    else:
        resolution = resolve_language(text, block.script, page_context, block.ocr_confidence)
    block.detected_language = normalize_language(resolution.language)
    block.resolved_language = block.detected_language
    block.language_confidence = resolution.confidence
    block.script = resolution.script
    block.resolved_script = resolution.script
    block.script_resolution_reason = resolution.reason
    recognized_script, recognized_confidence = (
        (ScriptType.UNKNOWN, 0.0)
        if htr_unavailable
        else meaningful_dominant_script(text)
    )
    block.recognized_unicode_script = recognized_script
    block.ocr_script_confidence = recognized_confidence
    block.linguistic_evidence_score = (
        0.0
        if htr_unavailable
        else linguistic_evidence_score(
            text,
            recognized_script if recognized_script != ScriptType.UNKNOWN else None,
        )
    )
    if (
        block.detected_language == "und"
        and page_context.dominant_script
        in {ScriptType.GURMUKHI, ScriptType.DEVANAGARI}
        and page_context.confidence >= 0.52
        and recognized_script in {ScriptType.UNKNOWN, ScriptType.LATIN}
    ):
        block.resolved_script = page_context.dominant_script
        block.script_resolution_reason = (
            resolution.reason
            + "; Indic visual/page routing prior preserved while OCR fragment was rejected"
        )
    structured = is_structured_value(text)
    expected = resolution.script if resolution.script != ScriptType.UNKNOWN else None
    quality = calculate_text_quality(text, expected)
    block.text_quality = quality
    letter_count = sum(character.isalpha() for character in text)
    confidence = block.ocr_confidence
    confidence_threshold = (
        settings.handwriting_confidence_threshold
        if block.is_handwritten
        else settings.ocr_low_confidence_threshold
    )
    trusted_reconstruction = bool(
        block.uncertainty_state == UncertaintyState.RECONSTRUCTED
        and block.reconstruction_confidence is not None
        and block.reconstruction_confidence
        >= (
            settings.auto_reconstruct_threshold
            if block.reconstruction_type == ReconstructionType.MODEL_INFERRED
            else settings.reconstruction_accept_threshold
        )
        and block.reconstruction_type
        in {
            ReconstructionType.OCR_CORRECTED,
            ReconstructionType.MODEL_INFERRED,
            ReconstructionType.HUMAN_REVIEWED,
            ReconstructionType.MANUALLY_CONFIRMED,
        }
    )

    if confidence is not None and confidence < confidence_threshold:
        _add_status(block, ProcessingStatus.OCR_LOW_CONFIDENCE)
    else:
        _add_status(block, ProcessingStatus.OCR_CONFIRMED)

    if structured:
        block.source_validated = True
        block.validation_reason = "structured token preserved without translation"
        block.translation_status = TranslationStatus.NOT_REQUIRED
        _add_status(block, ProcessingStatus.SOURCE_VALIDATED)
        _add_status(block, ProcessingStatus.TRANSLATION_SKIPPED)
        return SourceValidation(True, quality, block.validation_reason, True)

    if htr_unavailable:
        language_label = {
            "pa": "Punjabi/Gurmukhi",
            "hi": "Hindi/Devanagari",
        }.get(block.detected_language, "source-language")
        reason = (
            f"no validated {language_label} handwriting model is configured; "
            "manual source transcription is required"
        )
    elif block.detected_language == "und":
        reason = resolution.reason
    elif block.is_ocr and sum(character.isalnum() for character in text) < settings.min_ocr_character_count:
        reason = "OCR output below minimum character count"
    elif letter_count < settings.min_source_letter_count:
        reason = "too few source-language letters"
    elif quality < settings.min_text_quality_score:
        reason = "source text quality below threshold"
    elif block.detected_language in {"pa", "hi"} and script_ratio(text, block.script) < (
        0.20
        if (
            script_ratio(text, ScriptType.LATIN) >= 0.20
            and ("@" in text or any(character.isdigit() for character in text))
        )
        else settings.min_source_script_ratio
    ):
        reason = "source script purity below threshold"
    elif (
        block.is_ocr
        and confidence is not None
        and confidence < confidence_threshold
        and not trusted_reconstruction
    ):
        reason = "OCR confidence below translation threshold"
    else:
        block.source_validated = True
        block.validation_reason = resolution.reason
        _add_status(block, ProcessingStatus.SOURCE_VALIDATED)
        return SourceValidation(True, quality, block.validation_reason)

    block.source_validated = False
    block.validation_reason = reason
    block.translation_status = TranslationStatus.SKIPPED
    _add_status(block, ProcessingStatus.TRANSLATION_SKIPPED)
    if block.detected_language == "und":
        _add_status(block, ProcessingStatus.LANGUAGE_UNCERTAIN)
    if htr_unavailable:
        _add_status(block, ProcessingStatus.HANDWRITING_UNSUPPORTED)
    if (
        block.detected_language == "und"
        or quality < settings.min_text_quality_score
        or readable_character_ratio(text) < settings.reconstruction_min_readable_ratio
    ):
        if block.reconstruction_type not in {
            ReconstructionType.HUMAN_REVIEWED,
            ReconstructionType.MANUALLY_CONFIRMED,
        }:
            block.reconstruction_type = ReconstructionType.UNREADABLE
        block.uncertainty_state = UncertaintyState.FLAGGED
        _add_status(block, ProcessingStatus.UNREADABLE)
    return SourceValidation(False, quality, reason)


def is_translatable_block(block: TextBlock) -> bool:
    return bool(
        block.source_validated
        and normalize_language(block.detected_language) in SUPPORTED_TRANSLATION_LANGUAGES
        and block.reconstruction_type != ReconstructionType.UNREADABLE
        and not (
            (block.metadata.get("handwriting_unsupported") or block.metadata.get("htr_unavailable"))
            and not block.metadata.get("human_reviewed")
        )
    )
