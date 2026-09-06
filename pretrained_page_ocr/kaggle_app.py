"""Kaggle-ready, inference-only Hindi/Gurmukhi handwritten page OCR.

Change ``PAGE_IMAGE`` below, run this file on a Kaggle GPU, and collect the
artifacts written to ``OUTPUT_DIRECTORY``.  The program never trains or writes
model weights.  Hugging Face uses its normal read-through model cache.

Kaggle install cell (run once before this script)::

    !sudo apt-get update -qq && sudo apt-get install -y -qq tesseract-ocr tesseract-ocr-pan tesseract-ocr-hin
    %pip install -q "transformers>=5.12.1,<6" "surya-ocr==0.22.1" \
        "pillow>=10.2,<11" "opencv-python-headless==4.11.0.86" \
        "sentencepiece>=0.2,<1" "protobuf>=5,<7" "pytesseract>=0.3.13,<1" \
        "python-docx>=1.1,<2" "sacremoses>=0.1,<1"

The primary Hindi recognizer is the requested Devanagari TrOCR checkpoint.
No credible public *line-level handwritten* Gurmukhi checkpoint was found.
The Gurmukhi path therefore uses Surya OCR 2 as a clearly reported
multilingual fallback.  Its public documentation includes handwritten-note
examples and Punjabi in its multilingual benchmark.  Official Tesseract
``pan`` is a last-resort fallback if Surya cannot load; it is not claimed to be
a handwriting-specialized model.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import html
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# User configuration: this is normally the only value that needs changing.
# ---------------------------------------------------------------------------
PAGE_IMAGE: str | None = "/kaggle/input/.../test.jpg"

OUTPUT_DIRECTORY = Path("/kaggle/working/page_ocr_result")
HINDI_MODEL_ID = "paudelanil/trocr-devanagari-2"
GURMUKHI_MODEL_ID = "datalab-to/surya-ocr-2"
TRANSLATION_MODEL_ID = "facebook/nllb-200-distilled-600M"
INDICTRANS2_MODEL_ID = "ai4bharat/indictrans2-indic-en-dist-200M"
HELSINKI_MODEL_ID = "Helsinki-NLP/opus-mt-ine-en"
OPENAI_TRANSLATION_MODEL_ID = os.environ.get("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini")
TRANSLATION_PROVIDER_ORDER = ("openai", "indictrans2", "azure", "nllb")

OCR_BATCH_SIZE = 8
VLM_BATCH_SIZE = 4
TRANSLATION_BATCH_SIZE = 8
TRANSLATION_CONTEXT_MAX_LINES = 3
TRANSLATION_CONTEXT_MAX_CHARACTERS = 420
LINE_PADDING_PIXELS = 12
MIN_LINE_WIDTH = 28
MIN_LINE_HEIGHT = 8
MIN_LINE_AREA = 260
MIN_SOURCE_LETTERS = 2
MIN_AUTOMATIC_LINE_LETTERS = 4
MIN_SCRIPT_PURITY = 0.62
MIN_OCR_CONFIDENCE = 0.28
MIN_TEXT_QUALITY = 0.52
# A readable Punjabi/Hindi source can still yield a short English rendering
# for labels, form fields, and legal boilerplate.  Keep it visible as a
# low-confidence translation instead of discarding it completely.
MIN_LOW_CONFIDENCE_TRANSLATION_QUALITY = 0.30
MAX_IMAGE_SIDE = 2600

ScriptName = Literal["gurmukhi", "devanagari", "mixed", "unknown"]
BBox = tuple[int, int, int, int]
ProgressCallback = Callable[[str], None]
ProcessingMode = Literal["fast_cpu", "line_accurate"]

GURMUKHI_START, GURMUKHI_END = 0x0A00, 0x0A7F
DEVANAGARI_START, DEVANAGARI_END = 0x0900, 0x097F


def _load_indic_processor_class() -> Any:
    """Load the installed inference extension without importing training helpers.

    Toolkit 1.1.1's package initializer imports a training collator that uses a
    removed Transformers 4 symbol. The independently compiled processor has no
    dependency on that collator. Resolve only the installed extension using
    Python's extension suffix list; never modify site-packages or Transformers.
    """

    package = importlib.util.find_spec("IndicTransToolkit")
    if package is None or not package.submodule_search_locations:
        raise ImportError("Install indictranstoolkit in the app's Python environment")
    for directory in package.submodule_search_locations:
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            path = Path(directory) / ("processor" + suffix)
            if not path.is_file():
                continue
            spec = importlib.util.spec_from_file_location("processor", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.IndicProcessor
    # Future pure-Python toolkit distributions can use the standard import.
    return importlib.import_module("IndicTransToolkit.processor").IndicProcessor


def _from_pretrained_cached_first(factory: Any, model_id: str, **kwargs: Any) -> Any:
    """Use a complete local Hugging Face cache before making a Hub request."""

    try:
        return factory.from_pretrained(
            model_id,
            local_files_only=True,
            **kwargs,
        )
    except (OSError, RuntimeError):
        return factory.from_pretrained(model_id, **kwargs)


def _use_max_new_tokens_only(model: Any) -> None:
    """Prevent Transformers' max_length/max_new_tokens conflict warning."""

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.max_length = None


def _finish_pretrained_model_load(model: Any, device: Any) -> Any:
    """Resolve tied weights, reject meta tensors, and then move a real model.

    ``to_empty`` is intentionally never used here: it would materialize random
    parameters and make OCR appear operational with invalid weights.
    """

    tie_weights = getattr(model, "tie_weights", None)
    if callable(tie_weights):
        tie_weights()
    tensors: list[tuple[str, Any]] = []
    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        tensors.extend((f"parameter:{name}", value) for name, value in named_parameters())
    named_buffers = getattr(model, "named_buffers", None)
    if callable(named_buffers):
        tensors.extend((f"buffer:{name}", value) for name, value in named_buffers())
    meta_names = [name for name, value in tensors if getattr(value, "is_meta", False)]
    if meta_names:
        preview = ", ".join(meta_names[:4])
        raise RuntimeError(
            "The pretrained checkpoint left unmaterialized meta tensors "
            f"({preview}). The model was rejected instead of initializing fake weights."
        )
    target = str(device)
    needs_move = not tensors or any(str(value.device) != target for _name, value in tensors)
    if needs_move:
        model = model.to(device)
    remaining_meta = [
        name
        for name, value in (
            list(model.named_parameters()) + list(model.named_buffers())
        )
        if getattr(value, "is_meta", False)
    ] if callable(getattr(model, "named_parameters", None)) and callable(
        getattr(model, "named_buffers", None)
    ) else []
    if remaining_meta:
        raise RuntimeError("The pretrained model still contains meta tensors after loading")
    return model


def _find_tesseract_executable() -> Path | None:
    """Find Tesseract from PATH, an explicit override, or standard Windows paths."""

    candidates: list[str | Path | None] = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Tesseract-OCR"
        / "tesseract.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Tesseract-OCR"
        / "tesseract.exe",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
    return None


def _find_tessdata_directory(required_languages: Sequence[str]) -> Path | None:
    """Locate a tessdata directory containing every required language pack."""

    executable = _find_tesseract_executable()
    candidates: list[str | Path | None] = [
        os.environ.get("TESSDATA_PREFIX"),
        Path(__file__).resolve().parents[1] / ".runtime" / "tessdata",
        executable.parent / "tessdata" if executable else None,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        directory = Path(candidate).expanduser()
        if directory.name.casefold() != "tessdata" and (directory / "tessdata").is_dir():
            directory = directory / "tessdata"
        if directory.is_dir() and all(
            (directory / f"{language}.traineddata").is_file()
            for language in required_languages
        ):
            return directory.resolve()
    return None


def _prepare_tesseract(required_languages: Sequence[str]) -> tuple[Any, str]:
    """Configure pytesseract for local Windows installs and bundled language data."""

    import pytesseract

    executable = _find_tesseract_executable()
    if executable is not None:
        pytesseract.pytesseract.tesseract_cmd = str(executable)
    tessdata = _find_tessdata_directory(required_languages)
    if tessdata is not None:
        # Passing a quoted ``--tessdata-dir`` through pytesseract leaves literal
        # quote characters in the Windows path. TESSDATA_PREFIX avoids that
        # subprocess parsing problem and is also supported by native Tesseract.
        os.environ["TESSDATA_PREFIX"] = str(tessdata)
    return pytesseract, ""


@dataclass(slots=True)
class EnhancedPage:
    """Image variants kept separate so faint strokes are never discarded."""

    corrected_rgb: np.ndarray
    enhanced_rgb: np.ndarray
    enhanced_gray: np.ndarray
    threshold: np.ndarray
    operations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TextRegion:
    """A detected logical text line in corrected-page coordinates."""

    bbox: BBox
    detection_confidence: float | None
    detector: str


@dataclass(slots=True)
class RecognitionCandidate:
    """One model's source-language reading of a line crop."""

    expected_script: ScriptName
    text: str
    confidence: float | None
    model_id: str
    provider_kind: str
    error: str | None = None
    detected_script: ScriptName = "unknown"
    script_purity: float = 0.0
    text_quality: float = 0.0
    source_letter_count: int = 0
    score: float = 0.0
    selection_note: str | None = None


@dataclass(slots=True)
class LineResult:
    """Auditable recognition, routing, and translation result for one line."""

    line: int
    bbox: BBox
    crop_file: str
    detection_confidence: float | None
    detector: str
    script: ScriptName
    text: str
    script_purity: float
    confidence: float | None
    text_quality: float
    recognition_model: str
    provider_kind: str
    accepted: bool
    review_required: bool
    review_reasons: list[str]
    english: str = ""
    translation_status: str = "not_attempted"
    translation_error: str | None = None
    translation_confidence: float | None = None
    translation_quality: float = 0.0
    translation_review_reasons: list[str] = field(default_factory=list)
    translation_context_group: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    page_number: int = 1
    # Provider audit data is additive: existing consumers can continue to use
    # translation_status while audits can see exactly how a result was made.
    translation_provider: str | None = None
    translation_source_language: str | None = None
    translation_attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ProtectedTranslationSource:
    """A source string with exact values hidden from machine translation."""

    text: str
    replacements: dict[str, str]
    source_tokens: tuple[str, ...]


@dataclass(slots=True)
class TranslationUnit:
    """One contextual source unit mapped back to one or more physical lines."""

    lines: list[LineResult]
    script: ScriptName
    source: str
    protected: ProtectedTranslationSource
    group_id: str


@dataclass(frozen=True, slots=True)
class FinalizedTranslation:
    """The only representation permitted to reach ``LineResult.english``."""

    english: str
    quality: float
    used_conservative_fallback: bool
    review_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TranslationCoverage:
    """Evidence that a model translation did not silently omit a long line."""

    complete: bool
    score: float
    reasons: tuple[str, ...]


TRANSLATION_PROTECTED_PATTERN = re.compile(
    r"(?:https?://[^\s]+|www\.[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"(?:\+?\d(?:[\s.-]*\d){6,})(?=\s|$)|(?:₹|Rs\.?|INR|\$|€|£)\s?\d[\d,]*(?:\.\d+)?|"
    r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b|"
    # Treat a legal sequence as one span, never as a run of individual digits.
    r"\b\d+\s*\(\d+\)(?:\s*,\s*\d+\s*\(\d+\))+(?=\s|$)|"
    r"\b(?:Sec(?:tion)?\.?|FIR|Case|File|Ref(?:erence)?|Dispatch|MRD|OPD|IPD)\s*"
    r"(?:No\.?\s*)?[A-Za-z0-9_./()-]+|"
    r"\b[A-Za-z]{1,10}[-/]?\d+(?:[-/][A-Za-z0-9]+)*\b|"
    r"\b\d+(?:\s*\(\d+\))?(?:[.,:/-]\d+(?:\s*\(\d+\))?)*%?\b|"
    # Existing English names, institutions, abbreviations, and medical terms
    # are pass-through content on an otherwise Indic source line.
    r"\b[A-Za-z][A-Za-z0-9&.'()/+-]*(?:\s+[A-Za-z][A-Za-z0-9&.'()/+-]*)*\b)",
    re.IGNORECASE,
)

# These markers are implementation-only sentinels and must never leave the
# pipeline. Keep this broad enough to catch a tokenizer splitting one marker.
INTERNAL_PLACEHOLDER_PATTERN = re.compile(
    r"(?:ZX+Q?PROTECTED|PROTECTED[A-Z_-]*QXZ|ZXQATOMIC|QXZ)", re.IGNORECASE
)
UNRELATED_HALLUCINATION_PATTERN = re.compile(
    r"\b(?:countries?\s+of\s+the\s+european\s+union|european\s+union|"
    r"united\s+nations|capital\s+of\s+(?:france|india)|weather\s+forecast)\b",
    re.IGNORECASE,
)

# Deliberately small, high-certainty lexical maps for a conservative fallback.
# Everything not in this list remains exactly as OCR supplied it.
CONSERVATIVE_GURMUKHI_LEXICON = {
    "ਮਿਤੀ": "Date", "ਸਮਾਂ": "Time", "ਨਾਮ": "Name", "ਜ਼ਿਲ੍ਹਾ": "District",
    "ਜਿਲ੍ਹਾ": "District", "ਰਾਜ": "State", "ਥਾਣਾ": "Police station",
    "ਪੁਲਿਸ": "Police", "ਰਿਪੋਰਟ": "Report", "ਬਿਆਨ": "Statement",
    "ਨੰਬਰ": "Number", "ਨੂੰ": "to", "ਦੇ": "of", "ਅਤੇ": "and",
}
CONSERVATIVE_DEVANAGARI_LEXICON = {
    "दिनांक": "Date", "तारीख": "Date", "समय": "Time", "नाम": "Name",
    "जिला": "District", "राज्य": "State", "थाना": "Police station",
    "पुलिस": "Police", "रिपोर्ट": "Report", "बयान": "Statement",
    "नंबर": "Number", "और": "and", "का": "of", "को": "to",
}


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace without guessing source content."""

    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\u200b", "").replace("\ufeff", "")
    return " ".join(value.strip().split())


def normalize_translation_source(value: str) -> str:
    """Normalize OCR spacing that is unambiguously invalid inside Indic glyphs.

    This only joins combining signs and virama sequences to their base glyph. It
    deliberately does not guess whether two ordinary source letters form one
    word, so it cannot silently reconstruct missing source content.
    """

    normalized = normalize_text(value)
    output: list[str] = []
    pending_space = False
    viramas = {"\u094d", "\u0a4d"}
    for character in normalized:
        if character.isspace():
            pending_space = bool(output)
            continue
        category = unicodedata.category(character)
        attaches_to_previous = category.startswith("M")
        follows_virama = bool(output and output[-1] in viramas)
        if pending_space and not attaches_to_previous and not follows_virama:
            output.append(" ")
        output.append(character)
        pending_space = False
    result = "".join(output)
    result = re.sub(r"\s+([।॥,.;:!?])", r"\1", result)
    return normalize_text(result)


def _translation_placeholder(index: int) -> str:
    value = index
    letters = ""
    while True:
        letters = chr(ord("A") + value % 26) + letters
        value = value // 26 - 1
        if value < 0:
            break
    return f"ZXQPROTECTED{letters}QXZ"


def protect_translation_source(value: str) -> ProtectedTranslationSource:
    """Protect structured values and existing Latin text before NLLB."""

    replacements: dict[str, str] = {}
    tokens: list[str] = []

    def replace(match: re.Match[str]) -> str:
        placeholder = _translation_placeholder(len(replacements))
        replacements[placeholder] = match.group(0)
        tokens.append(match.group(0))
        return placeholder

    protected = TRANSLATION_PROTECTED_PATTERN.sub(replace, value)
    return ProtectedTranslationSource(
        text=protected,
        replacements=replacements,
        source_tokens=tuple(tokens),
    )


def _protected_span_equivalence_pattern(token: str) -> re.Pattern[str]:
    """Match a protected span after harmless punctuation/numeral normalization."""

    digit_sets = [0x0030, 0x0660, 0x06F0, 0x0966, 0x0A66]
    parts: list[str] = []
    for character in token:
        if character.isdecimal():
            value = unicodedata.digit(character)
            parts.append("[" + "".join(chr(start + value) for start in digit_sets) + "]")
        elif character.isalnum():
            parts.append(re.escape(character))
        elif character.isspace() or character in "_-./(),:;":
            # The original token is restored on a match, so only comparison is
            # punctuation-insensitive; output preserves the source form.
            parts.append(r"[\s_\-./(),:;]*")
        else:
            # Currency symbols and other meaningful punctuation are part of
            # the exact source value, not ignorable separator noise.
            parts.append(re.escape(character))
    return re.compile("".join(parts), re.IGNORECASE)


def restore_translation_source(
    value: str,
    protected: ProtectedTranslationSource,
) -> tuple[str, list[str]]:
    """Restore protected values and report any placeholder the model lost."""

    restored = value
    missing: list[str] = []
    for placeholder, source_token in protected.replacements.items():
        # Generative tokenizers sometimes add spaces or punctuation around an
        # otherwise unchanged Latin sentinel. Match those harmless variations.
        characters = [re.escape(character) for character in placeholder]
        flexible = re.compile(r"[\s_\-]*".join(characters), re.IGNORECASE)
        restored, count = flexible.subn(lambda _match, token=source_token: token, restored)
        if count == 0:
            equivalent = _protected_span_equivalence_pattern(source_token)
            restored, count = equivalent.subn(
                lambda _match, token=source_token: token, restored, count=1
            )
            if count == 0:
                missing.append(source_token)
    return normalize_text(restored), missing


def _collapse_accidental_repetition(value: str) -> str:
    """Remove only immediately duplicated words/phrases, never rewrite meaning."""

    output = normalize_text(value)
    output = re.sub(r"\b([A-Za-z][A-Za-z'-]{1,})(?:\s+\1\b)+", r"\1", output, flags=re.I)
    return output


def _conservative_source_rendering(source: str, script: ScriptName) -> str:
    """Translate only a small known vocabulary; preserve all uncertain OCR verbatim."""

    lexicon = (
        CONSERVATIVE_GURMUKHI_LEXICON
        if script == "gurmukhi"
        else CONSERVATIVE_DEVANAGARI_LEXICON
    )
    # Indic word runs are the only portions eligible for deterministic mapping.
    pattern = r"[\u0A00-\u0A7F]+" if script == "gurmukhi" else r"[\u0900-\u097F]+"
    return normalize_text(re.sub(pattern, lambda match: lexicon.get(match.group(0), match.group(0)), source))


def _candidate_validation(
    source: str,
    candidate: str,
    protected: ProtectedTranslationSource,
) -> tuple[str, float, list[str]]:
    """Run the ordered post-generation cleanup and hard-validation checks."""

    restored, missing_tokens = restore_translation_source(candidate, protected)
    cleaned = _collapse_accidental_repetition(restored)
    quality, reasons = translation_quality(source, cleaned, missing_tokens)
    if INTERNAL_PLACEHOLDER_PATTERN.search(cleaned):
        reasons.append("translation leaked an internal protected-token marker")
    if (
        UNRELATED_HALLUCINATION_PATTERN.search(cleaned)
        and not UNRELATED_HALLUCINATION_PATTERN.search(source)
    ):
        reasons.append("translation contains unrelated hallucinated content")
    source_letters = int(script_statistics(source)["letters"])
    english_words = re.findall(r"[A-Za-z][A-Za-z'-]{1,}", cleaned)
    if source_letters < 5 and len(english_words) > 5:
        reasons.append("translation adds unsupported semantic content to sparse OCR source")
    return cleaned, quality, list(dict.fromkeys(reasons))


def _final_output_safety_reasons(
    source: str,
    output: str,
    protected: ProtectedTranslationSource,
) -> tuple[str, list[str]]:
    """Hard checks shared by model output and deterministic fallbacks."""

    restored, missing_tokens = restore_translation_source(output, protected)
    cleaned = _collapse_accidental_repetition(restored)
    _quality, quality_reasons = translation_quality(source, cleaned, missing_tokens)
    reasons = [
        reason
        for reason in quality_reasons
        if reason
        in {
            "translation model returned empty text",
            "translation contains excessive repetition",
            "translation did not preserve every protected value",
        }
    ]
    if INTERNAL_PLACEHOLDER_PATTERN.search(cleaned):
        reasons.append("translation leaked an internal protected-token marker")
    if re.search(r"\[\s*source\s+values\s*:", cleaned, flags=re.IGNORECASE):
        reasons.append("translation exposed an internal source-values suffix")
    # Measure the final safety bound against all source content, not only
    # alphabetic glyphs: a faithful legal/numeric fallback can be long while
    # containing very few letters.
    source_content = len(re.findall(r"\w", source, flags=re.UNICODE))
    output_content = len(re.findall(r"\w", cleaned, flags=re.UNICODE))
    if output_content > max(24, source_content * 10):
        reasons.append("translation length is implausible for final source content")
    if (
        UNRELATED_HALLUCINATION_PATTERN.search(cleaned)
        and not UNRELATED_HALLUCINATION_PATTERN.search(source)
    ):
        reasons.append("translation contains unrelated hallucinated content")
    return cleaned, list(dict.fromkeys(reasons))


def finalize_translation(
    source_text: str,
    candidate: str,
    *,
    script: ScriptName | None = None,
    protected: ProtectedTranslationSource | None = None,
) -> FinalizedTranslation:
    """Return the only safe final text allowed in a line's ``english`` field.

    Model text is restored and checked first.  Any hard failure is replaced by
    a deterministic source walk; the unsafe candidate is never returned.
    """

    source = normalize_translation_source(source_text)
    if not source:
        return FinalizedTranslation("", 0.0, False, ())
    route = script if script in {"gurmukhi", "devanagari"} else detect_script(source)
    protected = protected or protect_translation_source(source)
    cleaned, quality, candidate_reasons = _candidate_validation(source, candidate, protected)
    safe_candidate, final_reasons = _final_output_safety_reasons(source, cleaned, protected)
    hard_reasons = list(dict.fromkeys([*candidate_reasons, *final_reasons]))
    if candidate and not _has_hard_translation_failure(hard_reasons) and not final_reasons:
        return FinalizedTranslation(
            safe_candidate,
            quality,
            False,
            tuple(reason for reason in hard_reasons if reason not in HARD_TRANSLATION_REASONS),
        )

    # No model output that failed hard validation is ever retained. Unknown
    # script defaults to the untouched source, while Indic lines use only the
    # small explicit mapping in _conservative_source_rendering.
    conservative = (
        _conservative_source_rendering(source, route)
        if route in {"gurmukhi", "devanagari"}
        else source
    )
    safe_conservative, conservative_failures = _final_output_safety_reasons(
        source, conservative, protected
    )
    if conservative_failures:
        # The source itself should already be safe. This defensive branch
        # removes any accidental marker-looking OCR sequence without creating
        # new language, then rechecks the mandatory invariant.
        safe_conservative = INTERNAL_PLACEHOLDER_PATTERN.sub("", source)
        safe_conservative = re.sub(r"\[\s*source\s+values\s*:[^\]]*\]", "", safe_conservative, flags=re.I)
        safe_conservative = _collapse_accidental_repetition(safe_conservative)
    if not normalize_text(safe_conservative):
        safe_conservative = "[unreadable source]"
    return FinalizedTranslation(
        normalize_text(safe_conservative),
        0.0,
        True,
        ("conservative_translation_fallback", "low_source_quality"),
    )


def _source_script_words(source: str, script: ScriptName) -> list[str]:
    """Return meaningful source-script runs without treating numbers as words."""

    pattern = r"[\u0A00-\u0A7F]{2,}" if script == "gurmukhi" else r"[\u0900-\u097F]{2,}"
    return re.findall(pattern, source)


def _source_clause_count(source: str) -> int:
    """Estimate semantic clauses using only visible boundaries in this line."""

    parts = [part for part in re.split(r"[\u0964\u0965.!?;:]+|\s*,\s*", source) if part.strip()]
    return max(1, len(parts))


def translation_completeness(
    source: str,
    english: str,
    script: ScriptName,
    protected: ProtectedTranslationSource,
) -> TranslationCoverage:
    """Detect fluent but materially truncated model output.

    This deliberately uses several weak signals together.  It does *not* use a
    fixed character ratio: Punjabi names and legal references often compress in
    English, while a long prose line still needs enough English lexical content
    to represent its clauses.
    """

    source = normalize_translation_source(source)
    english = normalize_text(english)
    source_words = _source_script_words(source, script)
    source_letters = sum(len(word) for word in source_words)
    target_words = re.findall(r"[A-Za-z][A-Za-z'-]{1,}", english)
    clauses = _source_clause_count(source)
    reasons: list[str] = []

    # Short labels, names, codes, and mostly structured fields are validly
    # represented by a conservative rendering.  Do not make them invent prose.
    if len(source_words) < 7 or source_letters < 24:
        return TranslationCoverage(True, 1.0, ())

    # A translated long Indic line normally has at least about two English
    # lexical words per visible clause and materially more than an opening
    # fragment.  The word threshold catches line 12's old six-word output
    # without penalising ordinary Punjabi-to-English compression.
    minimum_words = max(5, math.ceil(len(source_words) * 0.38), clauses * 2)
    if len(target_words) < minimum_words:
        reasons.append("translation appears incomplete for the source content")

    target_letters = sum(len(word) for word in target_words)
    if len(source_words) >= 14 and target_letters < max(20, source_letters * 0.22):
        reasons.append("translation coverage is too short for this long source line")

    # Protected values are already a hard safety requirement, but keeping this
    # explicit makes omission diagnostics auditable rather than mysterious.
    _restored, missing = restore_translation_source(english, protected)
    if missing:
        reasons.append("translation coverage omitted a protected source span")

    coverage = min(1.0, len(target_words) / max(1, minimum_words))
    return TranslationCoverage(not reasons, coverage, tuple(dict.fromkeys(reasons)))


def meaningful_residual_script_words(
    source: str,
    english: str,
    script: ScriptName,
    protected: ProtectedTranslationSource,
) -> list[str]:
    """Find substantial untranslated Indic lexical content in an English result.

    Protected spans are intentionally excluded: names, codes, legal terms and
    unavoidable verbatim fragments may correctly remain in their source form.
    A small number of remaining words is also tolerated for proper nouns.
    """

    route_words = _source_script_words(source, script)
    value = english
    for token in protected.source_tokens:
        value = value.replace(token, " ")
    residual = _source_script_words(value, script)
    # Ignore one-character OCR noise and only report source words that are
    # meaningful lexical runs, not a script artefact invented by the model.
    source_set = {word.casefold() for word in route_words}
    return [word for word in residual if word.casefold() in source_set and len(word) >= 2]


def translation_candidate_is_complete(
    source: str,
    finalized: FinalizedTranslation,
    script: ScriptName,
    protected: ProtectedTranslationSource,
) -> tuple[bool, list[str]]:
    """Apply completeness plus script-aware residual validation to one result."""

    if finalized.used_conservative_fallback:
        return False, ["translation hard validation required conservative fallback"]
    coverage = translation_completeness(source, finalized.english, script, protected)
    reasons = list(coverage.reasons)
    residual = meaningful_residual_script_words(source, finalized.english, script, protected)
    source_words = _source_script_words(source, script)
    allowed = max(2, math.ceil(len(source_words) * 0.20))
    if len(residual) > allowed:
        reasons.append("translation_incomplete: meaningful source-script content remains")
    return not reasons, list(dict.fromkeys(reasons))


def split_translation_clauses(
    source: str,
    script: ScriptName,
    *,
    max_source_words: int = 12,
    max_characters: int = 110,
) -> list[str]:
    """Split one line into ordered, self-contained retry chunks.

    First retain visible clause punctuation, then use a bounded word split only
    for unusually long unpunctuated OCR lines.  No text from another document
    line is ever included.
    """

    source = normalize_translation_source(source)
    if not source:
        return []
    clauses = [part.strip() for part in re.split(r"(?<=[\u0964\u0965.!?;:,])\s+", source) if part.strip()]
    chunks: list[str] = []
    for clause in clauses or [source]:
        words = clause.split()
        current: list[str] = []
        source_word_count = 0
        for word in words:
            next_count = source_word_count + len(_source_script_words(word, script))
            next_value = " ".join([*current, word])
            if current and (next_count > max_source_words or len(next_value) > max_characters):
                chunks.append(" ".join(current))
                current = [word]
                source_word_count = len(_source_script_words(word, script))
            else:
                current.append(word)
                source_word_count = next_count
        if current:
            chunks.append(" ".join(current))
    return chunks or [source]


def translation_quality(
    source: str,
    translated: str,
    missing_protected_tokens: Sequence[str] = (),
) -> tuple[float, list[str]]:
    """Validate that NLLB returned bounded English rather than corrupt output."""

    output = normalize_text(translated)
    reasons: list[str] = []
    if not output:
        return 0.0, ["translation model returned empty text"]
    stats = script_statistics(output)
    letters = int(stats["letters"])
    latin = int(stats["latin"])
    latin_ratio = latin / max(1, letters)
    meaningful_words = re.findall(r"[A-Za-z][A-Za-z'-]{1,}", output)
    if not meaningful_words:
        reasons.append("translation has no meaningful English words")
    if letters and latin_ratio < 0.65:
        reasons.append("translation output is not predominantly English")
    source_letters = max(1, int(script_statistics(source)["letters"]))
    length_ratio = len(output) / source_letters
    if length_ratio < 0.18 or length_ratio > 8.0:
        reasons.append("translation length is implausible for the validated source")
    words = re.findall(r"\w+", output.casefold())
    repeated_ratio = 0.0
    if words:
        repeated_ratio = 1.0 - len(set(words)) / len(words)
    most_common_ratio = max((words.count(word) for word in set(words)), default=0) / max(1, len(words))
    if len(words) >= 8 and (repeated_ratio > 0.62 or most_common_ratio > 0.30):
        reasons.append("translation contains excessive repetition")
    if missing_protected_tokens:
        reasons.append("translation did not preserve every protected value")

    length_score = max(0.0, 1.0 - abs(min(3.5, length_ratio) - 1.25) / 3.5)
    score = (
        0.45 * min(1.0, latin_ratio)
        + 0.25 * length_score
        + 0.20 * (1.0 - min(1.0, repeated_ratio))
        + 0.10 * (0.0 if missing_protected_tokens else 1.0)
    )
    if reasons:
        score *= 0.55
    return float(max(0.0, min(1.0, score))), reasons


def script_statistics(text: str) -> dict[str, float | int]:
    """Return Unicode-script counts and ratios over linguistic characters."""

    normalized = normalize_text(text)
    letters = [character for character in normalized if character.isalpha()]
    gurmukhi = sum(
        GURMUKHI_START <= ord(character) <= GURMUKHI_END for character in letters
    )
    devanagari = sum(
        DEVANAGARI_START <= ord(character) <= DEVANAGARI_END for character in letters
    )
    latin = sum("LATIN" in unicodedata.name(character, "") for character in letters)
    total = len(letters)
    return {
        "letters": total,
        "gurmukhi": gurmukhi,
        "devanagari": devanagari,
        "latin": latin,
        "gurmukhi_ratio": gurmukhi / max(1, total),
        "devanagari_ratio": devanagari / max(1, total),
        "latin_ratio": latin / max(1, total),
    }


def detect_script(text: str) -> ScriptName:
    """Classify recognized Unicode independently from the selected model."""

    stats = script_statistics(text)
    gurmukhi = int(stats["gurmukhi"])
    devanagari = int(stats["devanagari"])
    if gurmukhi and devanagari:
        return "mixed"
    if gurmukhi:
        return "gurmukhi"
    if devanagari:
        return "devanagari"
    return "unknown"


def script_purity(text: str, script: ScriptName) -> float:
    stats = script_statistics(text)
    if script == "gurmukhi":
        return float(stats["gurmukhi_ratio"])
    if script == "devanagari":
        return float(stats["devanagari_ratio"])
    if script == "mixed":
        return float(stats["gurmukhi_ratio"]) + float(stats["devanagari_ratio"])
    return 0.0


def calculate_text_quality(text: str, expected_script: ScriptName) -> float:
    """Conservatively score whether OCR output resembles source-language text."""

    normalized = normalize_text(text)
    if not normalized:
        return 0.0
    stats = script_statistics(normalized)
    letters = int(stats["letters"])
    purity = script_purity(normalized, expected_script)
    visible = [character for character in normalized if not character.isspace()]
    symbol_ratio = sum(
        not character.isalnum() and not character.isspace() for character in visible
    ) / max(1, len(visible))
    replacement_markers = {"?", "\ufffd", "□", "■", "|"}
    replacement_ratio = sum(character in replacement_markers for character in visible) / max(
        1, len(visible)
    )
    repeated = bool(re.search(r"(.)\1{3,}", normalized))
    length_score = min(1.0, letters / 8.0)
    quality = (
        0.55 * purity
        + 0.25 * length_score
        + 0.20 * max(0.0, 1.0 - symbol_ratio - replacement_ratio)
    )
    if repeated:
        quality *= 0.55
    if letters < MIN_SOURCE_LETTERS:
        quality *= 0.45
    return float(max(0.0, min(1.0, quality)))


def _candidate_score(candidate: RecognitionCandidate) -> float:
    confidence = candidate.confidence if candidate.confidence is not None else 0.35
    fallback_penalty = 0.04 if candidate.provider_kind == "tesseract_fallback" else 0.0
    return max(
        0.0,
        0.48 * candidate.script_purity
        + 0.30 * candidate.text_quality
        + 0.22 * confidence
        - fallback_penalty,
    )


DOCUMENT_FIELD_PATTERN = re.compile(
    r"\b(?:G\.?D\.?|FIR|P\.?S\.?|Police\s+Station|State|District|"
    r"Date|Subject|ASI|SHO|Inspector|Constable|Section|No\.?|Case)\b",
    re.IGNORECASE,
)


def _document_aware_candidate_bonus(
    candidate: RecognitionCandidate,
    nearby_text: Sequence[str] = (),
) -> float:
    """Small, evidence-only tie breaker for administrative-document OCR.

    It scores only words actually recovered by a candidate.  The bonus is too
    small to rescue wrong-script/symbol-soup OCR, but helps select a coherent
    header or structured field over a marginally higher noisy alternative.
    """

    text = candidate.text
    bonus = min(0.055, 0.012 * len(DOCUMENT_FIELD_PATTERN.findall(text)))
    if re.search(r"\b\d+\s*\(\d+\)(?:\s*,\s*\d+\s*\(\d+\))+", text):
        bonus += 0.018
    if re.search(r"\b(?:ASI|SHO|P\.?S\.?)\b", text, re.IGNORECASE):
        bonus += 0.012
    # Nearby lines supply only a document-style consistency hint; they never
    # contribute source words or alter the selected candidate's text.
    if nearby_text and candidate.detected_script in {"gurmukhi", "devanagari"}:
        nearby_scripts = [detect_script(value) for value in nearby_text if value]
        if candidate.detected_script in nearby_scripts:
            bonus += 0.008
    if candidate.provider_kind in {"surya_multilingual", "trocr_devanagari"}:
        bonus += 0.008
    return bonus


def enrich_candidate(candidate: RecognitionCandidate) -> RecognitionCandidate:
    candidate.text = normalize_text(candidate.text)
    candidate.detected_script = detect_script(candidate.text)
    candidate.script_purity = script_purity(candidate.text, candidate.expected_script)
    candidate.text_quality = calculate_text_quality(
        candidate.text, candidate.expected_script
    )
    stats = script_statistics(candidate.text)
    key = "gurmukhi" if candidate.expected_script == "gurmukhi" else "devanagari"
    candidate.source_letter_count = int(stats[key])
    candidate.score = _candidate_score(candidate)
    return candidate


def choose_best_recognition(
    candidates: Sequence[RecognitionCandidate],
    *,
    nearby_text: Sequence[str] = (),
) -> tuple[RecognitionCandidate, bool, list[str]]:
    """Choose script-consistent OCR and surface uncertainty rather than hiding it."""

    if not candidates:
        empty = RecognitionCandidate(
            expected_script="unknown",
            text="",
            confidence=None,
            model_id="none",
            provider_kind="unavailable",
            error="no recognition candidate was produced",
        )
        return empty, False, ["no recognition candidate"]

    enriched = [enrich_candidate(candidate) for candidate in candidates]
    base_ordered = sorted(enriched, key=lambda candidate: candidate.score, reverse=True)
    ordered = sorted(
        enriched,
        key=lambda candidate: candidate.score + _document_aware_candidate_bonus(
            candidate, nearby_text
        ),
        reverse=True,
    )
    winner = ordered[0]
    reasons: list[str] = []
    if winner is not base_ordered[0]:
        winner.selection_note = (
            "document-aware reranking selected this candidate over the raw-score winner"
        )
    if winner.error:
        reasons.append(winner.error)
    if not winner.text:
        reasons.append("empty OCR output")
    if winner.source_letter_count < MIN_SOURCE_LETTERS:
        reasons.append("too few source-script letters")
    elif winner.source_letter_count < MIN_AUTOMATIC_LINE_LETTERS:
        reasons.append("unusually short OCR output")
    if winner.script_purity < MIN_SCRIPT_PURITY:
        reasons.append("low Unicode script purity")
    if winner.detected_script not in {winner.expected_script, "mixed"}:
        reasons.append("model/script disagreement")
    if winner.detected_script == "mixed":
        reasons.append("mixed-script output requires review")
    if winner.text_quality < MIN_TEXT_QUALITY:
        reasons.append("low linguistic text quality")
    if winner.confidence is None:
        reasons.append("OCR confidence unavailable")
    elif winner.confidence < MIN_OCR_CONFIDENCE:
        reasons.append("low OCR confidence")
    if len(ordered) > 1:
        runner_up = ordered[1]
        both_plausible = (
            winner.script_purity >= MIN_SCRIPT_PURITY
            and runner_up.script_purity >= MIN_SCRIPT_PURITY
            and winner.text_quality >= MIN_TEXT_QUALITY
            and runner_up.text_quality >= MIN_TEXT_QUALITY
            and winner.expected_script != runner_up.expected_script
        )
        if both_plausible:
            reasons.append("Hindi and Gurmukhi candidates both look plausible")

    # Tesseract's line confidence is deliberately pessimistic on photographs
    # and low-contrast photocopies.  A printed Punjabi/Hindi reading with
    # strong Unicode-script evidence is still substantially safer than a
    # generic OCR fallback.  Accept those lines for translation while keeping
    # the hard protections against empty text, wrong scripts, and symbol soup.
    # This is intentionally limited to actual script-specific printed OCR;
    # handwritten/model outputs and Latin fragments still require review.
    printed_indic_rescue = (
        winner.provider_kind == "tesseract_fallback"
        and winner.expected_script in {"gurmukhi", "devanagari"}
        and winner.source_letter_count >= MIN_SOURCE_LETTERS
        and winner.script_purity >= 0.84
        and winner.text_quality >= 0.34
        # Some Tesseract builds omit word confidences for an otherwise clear
        # Indic line.  Unicode purity and the linguistic-quality gate remain
        # mandatory, so absence of that optional score is not a rejection.
        and (winner.confidence is None or winner.confidence >= 0.05)
    )
    if printed_indic_rescue:
        soft_failures = {
            "unusually short OCR output",
            "low linguistic text quality",
            "low OCR confidence",
            "OCR confidence unavailable",
        }
        reasons = [reason for reason in reasons if reason not in soft_failures]
    accepted = not reasons
    return winner, accepted, reasons


def _order_quad(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _perspective_rectify(rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 55, 160)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    page_area = rgb.shape[0] * rgb.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        if cv2.contourArea(contour) < 0.32 * page_area:
            break
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) != 4 or not cv2.isContourConvex(approximation):
            continue
        points = _order_quad(approximation.reshape(4, 2).astype(np.float32))
        top_width = np.linalg.norm(points[1] - points[0])
        bottom_width = np.linalg.norm(points[2] - points[3])
        left_height = np.linalg.norm(points[3] - points[0])
        right_height = np.linalg.norm(points[2] - points[1])
        width = int(max(top_width, bottom_width))
        height = int(max(left_height, right_height))
        if width < 300 or height < 300:
            continue
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(points, destination)
        return cv2.warpPerspective(rgb, transform, (width, height)), True
    return rgb, False


def _estimate_skew(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 55, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(40, gray.shape[1] // 12),
        minLineLength=max(80, gray.shape[1] // 8),
        maxLineGap=25,
    )
    if lines is None:
        return 0.0
    angles: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -12.0 <= angle <= 12.0:
            angles.append(angle)
    return float(statistics.median(angles)) if len(angles) >= 3 else 0.0


def _rotate_bound(rgb: np.ndarray, angle: float) -> np.ndarray:
    height, width = rgb.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2 - center[0]
    matrix[1, 2] += new_height / 2 - center[1]
    return cv2.warpAffine(
        rgb,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _optional_tesseract_orientation(rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    try:
        pytesseract, tessdata_option = _prepare_tesseract(("osd",))

        result = pytesseract.image_to_osd(
            Image.fromarray(rgb),
            output_type=pytesseract.Output.DICT,
            config=tessdata_option,
        )
        rotation = int(result.get("rotate", 0) or 0)
        confidence = float(result.get("orientation_conf", 0.0) or 0.0)
        if rotation in {90, 180, 270} and confidence >= 8.0:
            return np.rot90(rgb, k={90: 3, 180: 2, 270: 1}[rotation]).copy(), True
    except Exception:
        pass
    return rgb, False


def correct_document(image: Image.Image) -> tuple[np.ndarray, list[str]]:
    """Apply conservative orientation, boundary, perspective, and skew correction."""

    operations = ["exif_orientation"]
    image = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.asarray(image)
    if max(rgb.shape[:2]) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(rgb.shape[:2])
        rgb = cv2.resize(
            rgb,
            (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        operations.append("bounded_resize")
    rgb, oriented = _optional_tesseract_orientation(rgb)
    if oriented:
        operations.append("tesseract_orientation")
    rgb, rectified = _perspective_rectify(rgb)
    if rectified:
        operations.append("perspective_rectification")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    angle = _estimate_skew(gray)
    if 0.35 <= abs(angle) <= 10.0:
        rgb = _rotate_bound(rgb, angle)
        operations.append(f"deskew_{angle:.2f}_degrees")
    return rgb, operations


def enhance_page(corrected_rgb: np.ndarray, operations: list[str] | None = None) -> EnhancedPage:
    """Create mild recognition variants without replacing the preserved page."""

    applied = list(operations or [])
    gray = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2GRAY)
    background_kernel = max(31, (min(gray.shape[:2]) // 18) | 1)
    # Dark ink on light paper must be removed from the *background estimate*.
    # A large closing operation fills narrow dark strokes while retaining slow
    # illumination/shadow changes. MORPH_OPEN would instead preserve the ink in
    # the divisor and can flatten faint handwriting to almost zero contrast.
    background = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (background_kernel, background_kernel)
        ),
    )
    illumination = cv2.divide(gray, np.maximum(background, 1), scale=245)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(illumination)
    enhanced = cv2.bilateralFilter(enhanced, 5, 32, 32)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    enhanced = cv2.addWeighted(enhanced, 1.25, blurred, -0.25, 0)
    threshold = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        13,
    )
    applied.extend(
        ["illumination_correction", "clahe", "mild_bilateral_denoise", "mild_unsharp"]
    )
    return EnhancedPage(
        corrected_rgb=corrected_rgb,
        enhanced_rgb=cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB),
        enhanced_gray=enhanced,
        threshold=threshold,
        operations=applied,
    )


def _clip_bbox(bbox: Sequence[float | int], width: int, height: int) -> BBox:
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _valid_region(bbox: BBox, page_width: int, page_height: int) -> bool:
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    area = width * height
    if width < MIN_LINE_WIDTH or height < MIN_LINE_HEIGHT or area < MIN_LINE_AREA:
        return False
    border = max(3, round(min(page_width, page_height) * 0.006))
    touches_edge = x1 <= border or y1 <= border or x2 >= page_width - border or y2 >= page_height - border
    is_border_artifact = touches_edge and (
        height > 0.72 * page_height or width > 0.94 * page_width
    )
    return not is_border_artifact


def _vertical_overlap(first: BBox, second: BBox) -> float:
    overlap = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return overlap / max(1, min(first[3] - first[1], second[3] - second[1]))


def merge_text_regions(regions: Sequence[TextRegion]) -> list[TextRegion]:
    """Merge adjacent fragments sharing a handwriting baseline into logical lines."""

    pending = sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0]))
    merged: list[TextRegion] = []
    for region in pending:
        x1, y1, x2, y2 = region.bbox
        match_index: int | None = None
        best_gap = float("inf")
        for index, existing in enumerate(merged):
            ex1, ey1, ex2, ey2 = existing.bbox
            mean_height = ((y2 - y1) + (ey2 - ey1)) / 2.0
            baseline_delta = abs(y2 - ey2)
            gap = max(0, max(x1, ex1) - min(x2, ex2))
            same_line = (
                _vertical_overlap(region.bbox, existing.bbox) >= 0.45
                or baseline_delta <= 0.42 * mean_height
            )
            if same_line and gap <= max(28.0, 2.2 * mean_height) and gap < best_gap:
                match_index, best_gap = index, gap
        if match_index is None:
            merged.append(region)
            continue
        existing = merged[match_index]
        confidence_values = [
            value
            for value in (existing.detection_confidence, region.detection_confidence)
            if value is not None
        ]
        merged[match_index] = TextRegion(
            bbox=(
                min(existing.bbox[0], x1),
                min(existing.bbox[1], y1),
                max(existing.bbox[2], x2),
                max(existing.bbox[3], y2),
            ),
            detection_confidence=(
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else None
            ),
            detector=existing.detector,
        )
    return merged


def sort_reading_order(
    regions: Sequence[TextRegion], page_width: int | None = None
) -> list[TextRegion]:
    """Sort single-column pages by rows and detect obvious two-column layouts."""

    if not regions:
        return []
    heights = [region.bbox[3] - region.bbox[1] for region in regions]
    median_height = float(statistics.median(heights))
    if page_width and len(regions) >= 6:
        centers = sorted((region.bbox[0] + region.bbox[2]) / 2 for region in regions)
        gaps = [(centers[index + 1] - centers[index], index) for index in range(len(centers) - 1)]
        largest_gap, split_index = max(gaps, default=(0.0, 0))
        if largest_gap > 0.22 * page_width and split_index >= 1 and len(centers) - split_index >= 3:
            boundary = (centers[split_index] + centers[split_index + 1]) / 2
            left = [region for region in regions if (region.bbox[0] + region.bbox[2]) / 2 <= boundary]
            right = [region for region in regions if region not in left]
            return sorted(left, key=lambda region: (region.bbox[1], region.bbox[0])) + sorted(
                right, key=lambda region: (region.bbox[1], region.bbox[0])
            )

    rows: list[list[TextRegion]] = []
    for region in sorted(regions, key=lambda item: (item.bbox[1], item.bbox[0])):
        center_y = (region.bbox[1] + region.bbox[3]) / 2
        target: list[TextRegion] | None = None
        for row in rows:
            row_center = statistics.mean((item.bbox[1] + item.bbox[3]) / 2 for item in row)
            if abs(center_y - row_center) <= 0.55 * median_height:
                target = row
                break
        if target is None:
            target = []
            rows.append(target)
        target.append(region)
    ordered: list[TextRegion] = []
    for row in sorted(rows, key=lambda value: min(item.bbox[1] for item in value)):
        ordered.extend(sorted(row, key=lambda item: item.bbox[0]))
    return ordered


def _opencv_detect(page: EnhancedPage) -> list[TextRegion]:
    """Detect complete text lines without erasing Indic headline strokes.

    The older fallback removed every long horizontal component before grouping.
    That is unsafe for Gurmukhi: the connecting headline is a real part of the
    writing, not a form rule.  Keep ordinary horizontal ink and suppress only
    near-page-width, very thin rules/borders.  A projection pass is combined
    with contour grouping so faint or disconnected handwriting is not dropped.
    """

    ink = cv2.bitwise_not(page.threshold)
    height, width = ink.shape
    border_x, border_y = max(4, width // 120), max(4, height // 120)
    ink[:border_y, :] = 0
    ink[-border_y:, :] = 0
    ink[:, :border_x] = 0
    ink[:, -border_x:] = 0

    # Only remove unmistakable page/form rules.  Do not use a broad horizontal
    # opening here: it strips the Gurmukhi shirorekha and causes missing words.
    rule_candidates = cv2.bitwise_or(
        cv2.morphologyEx(
            ink,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(80, round(width * 0.86)), 1)),
        ),
        cv2.morphologyEx(
            ink,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(100, round(height * 0.78)))),
        ),
    )
    rules = np.zeros_like(ink)
    labels, _components, stats, _centroids = cv2.connectedComponentsWithStats(
        rule_candidates, connectivity=8
    )
    for label in range(1, labels):
        x, y, rule_width, rule_height, _area = stats[label]
        horizontal_rule = (
            rule_width >= round(width * 0.86)
            and rule_height <= max(3, round(height * 0.003))
        )
        vertical_rule = (
            rule_height >= round(height * 0.78)
            and rule_width <= max(3, round(width * 0.003))
        )
        if horizontal_rule or vertical_rule:
            cv2.rectangle(
                rules,
                (int(x), int(y)),
                (int(x + rule_width), int(y + rule_height)),
                255,
                -1,
            )
    text_ink = cv2.subtract(ink, rules)
    connect_width = max(18, width // 55)
    connected = cv2.morphologyEx(
        text_ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (connect_width, 3)),
        iterations=1,
    )
    connected = cv2.dilate(
        connected,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 100), 3)),
        iterations=1,
    )
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[TextRegion] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        bbox = _clip_bbox((x, y, x + box_width, y + box_height), width, height)
        if _valid_region(bbox, width, height):
            regions.append(TextRegion(bbox, None, "opencv_projection_fallback"))

    # Always add projection bands, not only when contour grouping is empty.
    # Broken pen strokes often yield tiny contours but still form a clear row.
    row_ink = np.count_nonzero(text_ink, axis=1).astype(np.float32)
    window = max(3, min(11, (height // 300) * 2 + 1))
    smoothed = cv2.blur(row_ink.reshape(-1, 1), (1, window)).reshape(-1)
    nonzero = smoothed[smoothed > 0]
    baseline = float(np.percentile(nonzero, 25)) if nonzero.size else 0.0
    active = smoothed >= max(2.0, baseline * 0.24)
    # Bridge only small within-line gaps; larger gaps remain separate lines.
    bridge = max(2, min(7, height // 450))
    active = np.convolve(active.astype(np.uint8), np.ones(bridge, dtype=np.uint8), mode="same") > 0
    start: int | None = None
    for index, is_active in enumerate([*active.tolist(), False]):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            band_height = index - start
            if MIN_LINE_HEIGHT <= band_height <= max(32, round(height * 0.18)):
                band = text_ink[start:index]
                columns = np.where(np.count_nonzero(band, axis=0) > 0)[0]
                if columns.size:
                    bbox = _clip_bbox(
                        (columns[0], start, columns[-1] + 1, index), width, height
                    )
                    if _valid_region(bbox, width, height):
                        regions.append(
                            TextRegion(bbox, None, "opencv_projection_fallback")
                        )
            start = None
    return sort_reading_order(merge_text_regions(regions), width)


class ModelRegistry:
    """Lazy, single-instance model registry optimized for Kaggle GPU memory."""

    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            major, _minor = torch.cuda.get_device_capability(0)
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
            self.gpu_name = torch.cuda.get_device_name(0)
        else:
            self.dtype = torch.float32
            self.gpu_name = "CPU"
        self.detector: Any | None = None
        self.hindi_processor: Any | None = None
        self.hindi_model: Any | None = None
        self.gurmukhi_processor: Any | None = None
        self.gurmukhi_model: Any | None = None
        self.translation_tokenizer: Any | None = None
        self.translation_model: Any | None = None
        self.helsinki_tokenizer: Any | None = None
        self.helsinki_model: Any | None = None
        self.indictrans2_tokenizer: Any | None = None
        self.indictrans2_model: Any | None = None
        self.indictrans2_processor: Any | None = None
        self.status: dict[str, dict[str, Any]] = {
            "detector": {"id": "surya.detection.DetectionPredictor", "loaded": False},
            "hindi": {
                "id": HINDI_MODEL_ID,
                "primary_id": HINDI_MODEL_ID,
                "loaded": False,
                "fallback": False,
            },
            "gurmukhi": {
                "id": GURMUKHI_MODEL_ID,
                "primary_id": GURMUKHI_MODEL_ID,
                "loaded": False,
                "fallback": True,
                "fallback_reason": "no verified dedicated line-level Gurmukhi handwriting checkpoint",
            },
            "translation": {"id": TRANSLATION_MODEL_ID, "loaded": False},
            "openai": {
                "id": OPENAI_TRANSLATION_MODEL_ID,
                "loaded": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
            },
            "indictrans2": {"id": INDICTRANS2_MODEL_ID, "loaded": False},
            "helsinki": {"id": HELSINKI_MODEL_ID, "loaded": False},
        }

    def autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.device.type != "cuda":
            return contextlib.nullcontext()
        return self.torch.autocast(device_type="cuda", dtype=self.dtype)

    def _release(self, *names: str) -> None:
        for name in names:
            setattr(self, name, None)
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def release_detector(self) -> None:
        self._release("detector")

    def release_recognizers(self) -> None:
        self._release(
            "hindi_processor", "hindi_model", "gurmukhi_processor", "gurmukhi_model"
        )

    def release_translation(self) -> None:
        self._release(
            "translation_tokenizer",
            "translation_model",
            "helsinki_tokenizer",
            "helsinki_model",
            "indictrans2_tokenizer",
            "indictrans2_model",
            "indictrans2_processor",
        )

    def load_detector(self) -> Any:
        if self.detector is None:
            # Surya is an optional Kaggle dependency. Loading by module name
            # keeps local editors usable when only the OpenCV fallback is
            # installed, while still raising a normal import error at runtime.
            detection_module = importlib.import_module("surya.detection")
            detection_predictor = getattr(detection_module, "DetectionPredictor")

            # Own the detector in this process so releasing this registry also
            # releases GPU memory; the default constructor starts a shared
            # server that can outlive a Kaggle notebook run.
            self.detector = detection_predictor.local(
                device=self.device,
                dtype=self.dtype,
            )
            self.status["detector"]["loaded"] = True
        return self.detector

    def load_hindi(self) -> tuple[Any, Any]:
        if self.hindi_model is None:
            from transformers import (
                RobertaTokenizer,
                TrOCRProcessor,
                ViTImageProcessorPil,
                VisionEncoderDecoderModel,
            )

            # This checkpoint predates the modern AutoImageProcessor metadata.
            # Construct its documented ViT + Roberta processor explicitly so it
            # remains loadable on Transformers 5 without changing the model.
            image_processor = _from_pretrained_cached_first(
                ViTImageProcessorPil, HINDI_MODEL_ID
            )
            tokenizer = _from_pretrained_cached_first(RobertaTokenizer, HINDI_MODEL_ID)
            self.hindi_processor = TrOCRProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
            )
            hindi_load_options: dict[str, Any] = {"low_cpu_mem_usage": False}
            if self.device.type == "cuda":
                hindi_load_options["dtype"] = self.dtype
            loaded_hindi = _from_pretrained_cached_first(
                VisionEncoderDecoderModel,
                HINDI_MODEL_ID,
                **hindi_load_options,
            )
            self.hindi_model = _finish_pretrained_model_load(
                loaded_hindi,
                self.device,
            )
            _use_max_new_tokens_only(self.hindi_model)
            self.hindi_model.eval()
            self.status["hindi"]["loaded"] = True
        return self.hindi_processor, self.hindi_model

    def load_gurmukhi(self) -> tuple[Any, Any]:
        if self.gurmukhi_model is None:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
            try:
                from transformers import AutoConfig, GenerationConfig
            except ImportError:  # Compatibility with minimal test/legacy installs.
                AutoConfig = None  # type: ignore[assignment,misc]
                GenerationConfig = None  # type: ignore[assignment,misc]

            self.gurmukhi_processor = _from_pretrained_cached_first(
                AutoProcessor, GURMUKHI_MODEL_ID
            )
            tokenizer = getattr(self.gurmukhi_processor, "tokenizer", None)
            if tokenizer is not None:
                tokenizer.padding_side = "left"
                if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
            # Surya's nested Qwen config includes an obsolete EOS id outside
            # its declared vocabulary.  It is replaced below by the loaded
            # tokenizer's valid EOS id; clear the stale value before model
            # construction so Transformers does not emit a misleading warning.
            gurmukhi_config = (
                _from_pretrained_cached_first(AutoConfig, GURMUKHI_MODEL_ID)
                if AutoConfig is not None
                else None
            )
            for config in (gurmukhi_config, getattr(gurmukhi_config, "text_config", None)):
                if config is not None:
                    vocab_size = int(getattr(config, "vocab_size", 0) or 0)
                    eos_token_id = getattr(config, "eos_token_id", None)
                    if isinstance(eos_token_id, int) and vocab_size and not 0 <= eos_token_id < vocab_size:
                        config.eos_token_id = None
            gurmukhi_load_options: dict[str, Any] = {
                "dtype": self.dtype,
                "low_cpu_mem_usage": False,
            }
            if gurmukhi_config is not None:
                gurmukhi_load_options["config"] = gurmukhi_config
            if GenerationConfig is not None:
                generation_config = _from_pretrained_cached_first(
                    GenerationConfig, GURMUKHI_MODEL_ID
                )
                if tokenizer is not None:
                    generation_config.eos_token_id = tokenizer.eos_token_id
                    generation_config.pad_token_id = tokenizer.pad_token_id
                gurmukhi_load_options["generation_config"] = generation_config
            loaded_gurmukhi = _from_pretrained_cached_first(
                AutoModelForMultimodalLM,
                GURMUKHI_MODEL_ID,
                # The checkpoint declares BF16.  Older AVX2-only CPUs emulate
                # BF16 very slowly, so explicitly load FP32 on CPU.
                **gurmukhi_load_options,
            )
            self.gurmukhi_model = _finish_pretrained_model_load(
                loaded_gurmukhi,
                self.device,
            )
            if tokenizer is not None:
                # The repository's nested text config contains a stale Qwen
                # token id, while its tokenizer/generation config use 2 and 0.
                # Set generation ids from the loaded tokenizer to guarantee
                # early stopping instead of decoding every token budget.
                self.gurmukhi_model.generation_config.eos_token_id = tokenizer.eos_token_id
                self.gurmukhi_model.generation_config.pad_token_id = tokenizer.pad_token_id
            _use_max_new_tokens_only(self.gurmukhi_model)
            self.gurmukhi_model.eval()
            self.status["gurmukhi"]["loaded"] = True
        return self.gurmukhi_processor, self.gurmukhi_model

    def load_translation(self) -> tuple[Any, Any]:
        if self.translation_model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.translation_tokenizer = _from_pretrained_cached_first(
                AutoTokenizer, TRANSLATION_MODEL_ID
            )
            translation_load_options: dict[str, Any] = {"low_cpu_mem_usage": False}
            if self.device.type == "cuda":
                translation_load_options["dtype"] = self.dtype
            loaded_translation = _from_pretrained_cached_first(
                AutoModelForSeq2SeqLM,
                TRANSLATION_MODEL_ID,
                **translation_load_options,
            )
            self.translation_model = _finish_pretrained_model_load(
                loaded_translation,
                self.device,
            )
            _use_max_new_tokens_only(self.translation_model)
            self.translation_model.eval()
            self.status["translation"]["loaded"] = True
        return self.translation_tokenizer, self.translation_model

    def load_helsinki(self) -> tuple[Any, Any]:
        """Load the public Marian Indo-European-to-English model locally."""

        if self.helsinki_model is None:
            from transformers import MarianMTModel, MarianTokenizer
            from huggingface_hub import hf_hub_download

            # Marian can raise TypeError instead of OSError when a partially
            # populated cache lacks source.spm. Resolve all tokenizer assets
            # explicitly; each file uses Hugging Face's normal shared cache.
            assets = {}
            for filename in ("source.spm", "target.spm", "vocab.json"):
                try:
                    assets[filename] = hf_hub_download(HELSINKI_MODEL_ID, filename, local_files_only=True)
                except OSError:
                    assets[filename] = hf_hub_download(HELSINKI_MODEL_ID, filename)
            self.helsinki_tokenizer = MarianTokenizer(
                source_spm=assets["source.spm"], target_spm=assets["target.spm"],
                vocab=assets["vocab.json"],
            )
            options: dict[str, Any] = {"low_cpu_mem_usage": False}
            if self.device.type == "cuda":
                options["dtype"] = self.dtype
            loaded = _from_pretrained_cached_first(MarianMTModel, HELSINKI_MODEL_ID, **options)
            self.helsinki_model = _finish_pretrained_model_load(loaded, self.device)
            self.helsinki_model.eval()
            _use_max_new_tokens_only(self.helsinki_model)
            self.status["helsinki"]["loaded"] = True
        return self.helsinki_tokenizer, self.helsinki_model

    def load_indictrans2(self) -> tuple[Any, Any, Any]:
        """Load the official IndicTrans2 Indic→English model once, lazily.

        The checkpoint is gated by its publisher.  A missing accepted HF access
        condition or optional toolkit is a normal provider-unavailable event;
        the translation cascade then continues without exposing credentials.
        """

        if self.indictrans2_model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            try:
                # Optional compiled dependency: resolve it lazily so NLLB and
                # Azure remain usable on installations without the toolkit.
                IndicProcessor = _load_indic_processor_class()
            except ImportError as exc:
                raise RuntimeError("IndicTransToolkit inference processor could not be loaded") from exc
            self.indictrans2_tokenizer = _from_pretrained_cached_first(
                AutoTokenizer, INDICTRANS2_MODEL_ID, trust_remote_code=True
            )
            options: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": False}
            if self.device.type == "cuda":
                options["dtype"] = self.dtype
            loaded = _from_pretrained_cached_first(
                AutoModelForSeq2SeqLM, INDICTRANS2_MODEL_ID, **options
            )
            self.indictrans2_model = _finish_pretrained_model_load(loaded, self.device)
            _use_max_new_tokens_only(self.indictrans2_model)
            self.indictrans2_model.eval()
            self.indictrans2_processor = IndicProcessor(inference=True)
            self.status["indictrans2"].update({"loaded": True})
        return self.indictrans2_tokenizer, self.indictrans2_model, self.indictrans2_processor


def load_models() -> ModelRegistry:
    """Create a lazy registry; downloads occur once when each model is first used."""

    return ModelRegistry()


def _object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def detect_text_regions(page: EnhancedPage, models: ModelRegistry) -> list[TextRegion]:
    """Use pretrained Surya line detection, with a robust OpenCV fallback."""

    height, width = page.enhanced_gray.shape
    detector: Any | None = None
    try:
        detector = models.load_detector()
        prediction = detector([Image.fromarray(page.enhanced_rgb)])[0]
        regions: list[TextRegion] = []
        for item in _object_value(prediction, "bboxes", []) or []:
            raw_bbox = _object_value(item, "bbox")
            if raw_bbox is None:
                polygon = _object_value(item, "polygon", [])
                if polygon:
                    points = np.asarray(polygon, dtype=np.float32)
                    raw_bbox = (
                        points[:, 0].min(),
                        points[:, 1].min(),
                        points[:, 0].max(),
                        points[:, 1].max(),
                    )
            if raw_bbox is None:
                continue
            bbox = _clip_bbox(raw_bbox, width, height)
            if not _valid_region(bbox, width, height):
                continue
            confidence_value = _object_value(item, "confidence")
            confidence = (
                float(confidence_value) if confidence_value is not None else None
            )
            regions.append(TextRegion(bbox, confidence, "surya_text_detector"))
        regions = sort_reading_order(merge_text_regions(regions), width)
        if regions:
            return regions
        print("WARNING: Surya detected no valid lines; using the OpenCV fallback.")
    except Exception as exc:
        models.status["detector"]["error"] = f"{type(exc).__name__}: {exc}"
        print(f"WARNING: Surya line detection unavailable ({exc}); using OpenCV.")
    finally:
        detector = None
        models.release_detector()
    return _opencv_detect(page)


def prepare_crop(page_rgb: np.ndarray, bbox: BBox, padding: int = LINE_PADDING_PIXELS) -> Image.Image:
    """Return a padded RGB line crop without modifying page coordinates."""

    height, width = page_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
    crop = Image.fromarray(page_rgb[y1:y2, x1:x2]).convert("RGB")
    return ImageOps.expand(crop, border=max(4, padding // 2), fill="white")


def _square_pad(image: Image.Image, padding: int = 8) -> Image.Image:
    """Center an OCR crop on a white square without stretching its aspect ratio."""

    source = image.convert("RGB")
    side = max(source.width, source.height) + 2 * max(0, padding)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(source, ((side - source.width) // 2, (side - source.height) // 2))
    return canvas


def _split_words(line: Image.Image) -> list[Image.Image]:
    """Split a line for the word-trained Hindi checkpoint when gaps are clear."""

    gray = np.asarray(line.convert("L"))
    threshold = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 13
    )
    projection = np.count_nonzero(threshold, axis=0)
    active = projection > max(1, int(gray.shape[0] * 0.025))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate([*active.tolist(), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if len(runs) < 2:
        return [line]
    gap_limit = max(7, round(gray.shape[0] * 0.16))
    words: list[tuple[int, int]] = []
    current_start, current_end = runs[0]
    for run_start, run_end in runs[1:]:
        if run_start - current_end >= gap_limit:
            words.append((current_start, current_end))
            current_start, current_end = run_start, run_end
        else:
            current_end = run_end
    words.append((current_start, current_end))
    if not 2 <= len(words) <= 18:
        return [line]
    result = [
        ImageOps.expand(
            line.crop((max(0, x1 - 4), 0, min(line.width, x2 + 4), line.height)),
            border=4,
            fill="white",
        )
        for x1, x2 in words
        if x2 - x1 >= 4
    ]
    return result or [line]


def _batch(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), max(1, size)):
        yield items[start : start + max(1, size)]


def _report_progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _generation_confidences(model: Any, generated: Any) -> list[float | None]:
    try:
        transition = model.compute_transition_scores(
            generated.sequences,
            generated.scores,
            getattr(generated, "beam_indices", None),
            normalize_logits=True,
        )
        probabilities = transition.float().exp()
        token_ids = generated.sequences[:, -transition.shape[1] :]
        pad_token_id = getattr(model.generation_config, "pad_token_id", None)
        valid_tokens = (
            token_ids.ne(pad_token_id) if pad_token_id is not None else token_ids.ge(0)
        )
        confidences: list[float | None] = []
        for row, valid in zip(probabilities, valid_tokens, strict=True):
            selected = row[valid]
            confidences.append(float(selected.mean().item()) if selected.numel() else None)
        return confidences
    except Exception:
        return [None] * len(generated.sequences)


def _tesseract_single_pass(
    crop: Image.Image,
    language: str,
    script: ScriptName,
    page_segmentation_mode: int,
) -> RecognitionCandidate:
    """Recognize one prepared line crop with one Tesseract segmentation mode."""

    try:
        pytesseract, tessdata_option = _prepare_tesseract((language,))
        recognition_crop = crop.convert("RGB")
        if recognition_crop.height < 54:
            scale = min(3.0, max(1.5, 54.0 / max(1, recognition_crop.height)))
            recognition_crop = recognition_crop.resize(
                (
                    max(1, round(recognition_crop.width * scale)),
                    max(1, round(recognition_crop.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        config = " ".join(
            part
            for part in (
                tessdata_option,
                f"--oem 1 --psm {page_segmentation_mode}",
            )
            if part
        )

        data = pytesseract.image_to_data(
            recognition_crop,
            lang=language,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        words: list[str] = []
        confidences: list[float] = []
        for text, confidence in zip(data.get("text", []), data.get("conf", []), strict=False):
            normalized = normalize_text(text)
            try:
                value = float(confidence)
            except (TypeError, ValueError):
                value = -1.0
            if normalized:
                words.append(normalized)
                if value >= 0:
                    confidences.append(value / 100.0)
        return RecognitionCandidate(
            expected_script=script,
            text=" ".join(words),
            confidence=(sum(confidences) / len(confidences) if confidences else None),
            model_id=f"tesseract:{language}:psm{page_segmentation_mode}",
            provider_kind="tesseract_fallback",
        )
    except Exception as exc:
        return RecognitionCandidate(
            expected_script=script,
            text="",
            confidence=None,
            model_id=f"tesseract:{language}:psm{page_segmentation_mode}",
            provider_kind="unavailable",
            error=f"Tesseract {language} unavailable: {exc}",
        )


def _tesseract_candidate(crop: Image.Image, language: str, script: ScriptName) -> RecognitionCandidate:
    """Use a small line-level Tesseract ensemble for incomplete OCR crops.

    PSM 7 is best for clean single lines.  PSM 6 is a useful rescue pass for
    broken handwriting, uneven baselines, and crops containing two close text
    rows.  An adaptive-threshold rescue is attempted only when the normal pass
    is weak, avoiding a fourfold CPU cost for already legible lines.
    """

    candidates = [_tesseract_single_pass(crop, language, script, 7)]
    first = enrich_candidate(candidates[0])
    if first.score < 0.76:
        candidates.append(_tesseract_single_pass(crop, language, script, 6))
        # Small mobile-photo lines often become readable only after a modest
        # scale-up.  It is a rescue candidate, not a replacement for the
        # original grayscale crop, preserving fine Gurmukhi headline strokes.
        enlarged = crop.resize(
            (min(2400, crop.width * 2), min(900, crop.height * 2)),
            Image.Resampling.LANCZOS,
        )
        candidates.append(_tesseract_single_pass(enlarged, language, script, 7))
        gray = np.asarray(crop.convert("L"))
        threshold = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        candidates.append(
            _tesseract_single_pass(
                Image.fromarray(threshold).convert("RGB"), language, script, 7
            )
        )
    return max(
        (enrich_candidate(candidate) for candidate in candidates),
        key=lambda item: item.score,
    )


def recognize_hindi(
    crops: Sequence[Image.Image],
    models: ModelRegistry,
    progress_callback: ProgressCallback | None = None,
) -> list[RecognitionCandidate]:
    """Recognize Devanagari with the required pretrained TrOCR checkpoint."""

    try:
        processor, model = models.load_hindi()
    except Exception as exc:
        models._release("hindi_processor", "hindi_model")
        models.status["hindi"].update(
            {
                "id": "tesseract:hin",
                "loaded": False,
                "fallback": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"WARNING: Hindi TrOCR unavailable ({exc}); using Tesseract hin.")
        return [_tesseract_candidate(crop, "hin", "devanagari") for crop in crops]

    units: list[Image.Image] = []
    mapping: list[list[int]] = []
    for crop in crops:
        indices: list[int] = []
        for word in _split_words(crop):
            indices.append(len(units))
            units.append(_square_pad(word))
        mapping.append(indices)
    texts: list[str] = []
    confidences: list[float | None] = []
    torch = models.torch
    try:
        total_batches = max(1, math.ceil(len(units) / OCR_BATCH_SIZE))
        for batch_index, current in enumerate(_batch(units, OCR_BATCH_SIZE), start=1):
            _report_progress(
                progress_callback,
                f"Hindi handwriting OCR batch {batch_index}/{total_batches}",
            )
            pixels = processor(images=list(current), return_tensors="pt").pixel_values.to(
                models.device
            )
            if models.device.type == "cuda":
                pixels = pixels.to(dtype=models.dtype)
            with torch.inference_mode(), models.autocast():
                generated = model.generate(
                    pixel_values=pixels,
                    num_beams=4,
                    max_new_tokens=32,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            texts.extend(
                normalize_text(value)
                for value in processor.batch_decode(
                    generated.sequences, skip_special_tokens=True
                )
            )
            confidences.extend(_generation_confidences(model, generated))
    except Exception as exc:
        models.status["hindi"].update(
            {
                "id": "tesseract:hin",
                "fallback": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"WARNING: Hindi TrOCR inference failed ({exc}); using Tesseract hin.")
        return [_tesseract_candidate(crop, "hin", "devanagari") for crop in crops]
    finally:
        processor = None
        model = None
        models._release("hindi_processor", "hindi_model")

    results: list[RecognitionCandidate] = []
    for indices in mapping:
        line_text = " ".join(texts[index] for index in indices if texts[index])
        values = [confidences[index] for index in indices if confidences[index] is not None]
        results.append(
            RecognitionCandidate(
                expected_script="devanagari",
                text=line_text,
                confidence=(sum(values) / len(values) if values else None),
                model_id=HINDI_MODEL_ID,
                provider_kind="dedicated_hindi_handwriting",
            )
        )
    return results


def _surya_prompts(processor: Any, count: int) -> list[str]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    # This is Surya 2's documented block-OCR training-time prompt.
                    # Unicode/source-quality gates below enforce the Gurmukhi route.
                    "text": "OCR this block image to HTML.",
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return [prompt] * count


def recognize_gurmukhi(
    crops: Sequence[Image.Image],
    models: ModelRegistry,
    progress_callback: ProgressCallback | None = None,
    *,
    fast: bool = False,
) -> list[RecognitionCandidate]:
    """Recognize Gurmukhi with the verified multilingual Surya fallback."""

    try:
        processor, model = models.load_gurmukhi()
    except Exception as exc:
        models._release("gurmukhi_processor", "gurmukhi_model")
        models.status["gurmukhi"].update(
            {
                "loaded": False,
                "id": "tesseract:pan",
                "fallback": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"WARNING: Surya multilingual OCR unavailable ({exc}); using Tesseract pan.")
        return [_tesseract_candidate(crop, "pan", "gurmukhi") for crop in crops]

    results: list[RecognitionCandidate] = []
    torch = models.torch
    try:
        batch_size = VLM_BATCH_SIZE if models.device.type == "cuda" else (2 if fast else 1)
        token_budget = 128 if models.device.type == "cuda" else (48 if fast else 64)
        total_batches = max(1, math.ceil(len(crops) / batch_size))
        for batch_index, current in enumerate(_batch(crops, batch_size), start=1):
            _report_progress(
                progress_callback,
                (
                    f"Punjabi/Gurmukhi OCR batch {batch_index}/{total_batches} "
                    f"on {models.gpu_name}"
                ),
            )
            prompts = _surya_prompts(processor, len(current))
            inputs = processor(
                text=prompts,
                images=list(current),
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(models.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.inference_mode(), models.autocast():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=token_budget,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            input_length = inputs["input_ids"].shape[1]
            new_tokens = generated.sequences[:, input_length:]
            decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)
            batch_confidences = _generation_confidences(model, generated)
            for text, confidence in zip(decoded, batch_confidences, strict=True):
                results.append(
                    RecognitionCandidate(
                        expected_script="gurmukhi",
                        text=normalize_text(text),
                        confidence=confidence,
                        model_id=GURMUKHI_MODEL_ID,
                        provider_kind="multilingual_handwriting_fallback",
                    )
                )
    except Exception as exc:
        models.status["gurmukhi"].update(
            {
                "id": "tesseract:pan",
                "fallback": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"WARNING: Surya Gurmukhi inference failed ({exc}); using Tesseract pan.")
        results = [_tesseract_candidate(crop, "pan", "gurmukhi") for crop in crops]
    processor = None
    model = None
    models._release("gurmukhi_processor", "gurmukhi_model")
    return results


def _candidate_for_audit(candidate: RecognitionCandidate) -> dict[str, Any]:
    return {
        "expected_script": candidate.expected_script,
        "detected_script": candidate.detected_script,
        "text": candidate.text,
        "confidence": candidate.confidence,
        "script_purity": candidate.script_purity,
        "text_quality": candidate.text_quality,
        "source_letter_count": candidate.source_letter_count,
        "score": candidate.score,
        "selection_note": candidate.selection_note,
        "model_id": candidate.model_id,
        "provider_kind": candidate.provider_kind,
        "error": candidate.error,
    }


def _language_code(script: ScriptName) -> str | None:
    return {"gurmukhi": "pan_Guru", "devanagari": "hin_Deva"}.get(script)


def _can_share_translation_context(
    previous: LineResult,
    current: LineResult,
    group_size: int,
    group_characters: int,
) -> bool:
    """Return whether two validated scan lines can safely form one source unit."""

    if group_size >= TRANSLATION_CONTEXT_MAX_LINES:
        return False
    if group_characters + len(current.text) + 1 > TRANSLATION_CONTEXT_MAX_CHARACTERS:
        return False
    if previous.page_number != current.page_number or previous.script != current.script:
        return False
    if current.line != previous.line + 1:
        return False
    # Native text already arrives as reliable structural units. Avoid merging
    # separate Word/PDF objects because those mappings must remain exact.
    native_kinds = {"docx_native", "pymupdf_native_text"}
    if previous.provider_kind in native_kinds or current.provider_kind in native_kinds:
        return False
    if re.search(r"[.!?।॥]\s*$", previous.text):
        return False
    previous_height = max(1, previous.bbox[3] - previous.bbox[1])
    current_height = max(1, current.bbox[3] - current.bbox[1])
    vertical_gap = current.bbox[1] - previous.bbox[3]
    return -0.40 * max(previous_height, current_height) <= vertical_gap <= 1.80 * max(
        previous_height,
        current_height,
    )


def build_translation_units(lines: Sequence[LineResult]) -> list[TranslationUnit]:
    """Create one independent translation unit for every routable source line.

    OCR layout lines are output records, so combining them for translation and
    trying to split generated English afterwards can lose words or leave a
    record blank. Context may be used by a future translator, but it must
    never make multiple source lines share one English result.
    """

    units: list[TranslationUnit] = []
    for line in lines:
        source = normalize_translation_source(line.text)
        script = _translation_script_for_line(line)
        if not source or script is None:
            continue
        units.append(
            TranslationUnit(
                lines=[line],
                script=script,
                source=source,
                protected=protect_translation_source(source),
                group_id=f"p{line.page_number}:l{line.line}",
            )
        )
    return units


def _translation_script_for_line(line: LineResult) -> ScriptName | None:
    """Choose a translator language without overwriting OCR audit metadata."""

    if _language_code(line.script) is not None:
        return line.script
    stats = script_statistics(line.text)
    gurmukhi = int(stats["gurmukhi"])
    devanagari = int(stats["devanagari"])
    if gurmukhi > devanagari and gurmukhi:
        return "gurmukhi"
    if devanagari > gurmukhi and devanagari:
        return "devanagari"
    return None


def _split_context_translation(
    translation: str,
    source_lines: Sequence[LineResult],
    protected_tokens: Sequence[str] = (),
) -> list[str] | None:
    """Split one contextual translation without dropping or duplicating words."""

    if len(source_lines) == 1:
        return [translation]
    atomic = translation
    atomic_values: dict[str, str] = {}
    for index, token in enumerate(sorted(set(protected_tokens), key=len, reverse=True)):
        if token not in atomic:
            continue
        placeholder = f"ZXQATOMIC{index}QXZ"
        atomic = atomic.replace(token, placeholder)
        atomic_values[placeholder] = token
    words = atomic.split()
    if len(words) < len(source_lines):
        return None
    weights = [
        max(1, int(script_statistics(line.text)["letters"])) for line in source_lines
    ]
    remaining_words = len(words)
    remaining_weight = sum(weights)
    values: list[str] = []
    start = 0
    for index, weight in enumerate(weights[:-1]):
        remaining_lines = len(weights) - index - 1
        size = round(remaining_words * weight / max(1, remaining_weight))
        size = max(1, min(size, remaining_words - remaining_lines))
        values.append(" ".join(words[start : start + size]))
        start += size
        remaining_words -= size
        remaining_weight -= weight
    values.append(" ".join(words[start:]))
    for index, value in enumerate(values):
        for placeholder, token in atomic_values.items():
            value = value.replace(placeholder, token)
        values[index] = value
    return values if all(values) else None


HARD_TRANSLATION_REASONS = {
    "translation model returned empty text",
    "translation has no meaningful English words",
    "translation output is not predominantly English",
    "translation length is implausible for the validated source",
    "translation contains excessive repetition",
    "translation did not preserve every protected value",
    "translation leaked an internal protected-token marker",
    "translation contains unrelated hallucinated content",
    "translation adds unsupported semantic content to sparse OCR source",
}


def _has_hard_translation_failure(reasons: Sequence[str]) -> bool:
    return any(reason in HARD_TRANSLATION_REASONS for reason in reasons)


def _preserve_source_with_warning(line: LineResult, reason: str) -> None:
    """Meet the output invariant without inventing a translation."""

    source = normalize_text(line.text)
    if not source:
        return
    finalized = finalize_translation(source, source, script=line.script)
    line.english = finalized.english
    line.translation_status = "preserved_source_with_warnings"
    # This is a completed conservative fallback, not an unhandled exception.
    line.translation_error = None
    line.translation_confidence = None
    line.translation_quality = 0.0
    line.translation_review_reasons = list(
        dict.fromkeys([*line.translation_review_reasons, reason, *finalized.review_reasons])
    )
    line.review_required = True


def _preserve_unroutable_content(line: LineResult) -> None:
    """Preserve already-English or structured source without inventing text."""

    source = normalize_text(line.text)
    if not source:
        return
    stats = script_statistics(source)
    letters = int(stats["letters"])
    latin_ratio = float(stats["latin_ratio"])
    if letters == 0 or latin_ratio >= 0.65:
        line.english = finalize_translation(source, source, script=line.script).english
        line.translation_status = "preserved_english" if letters else "preserved_structured"
        return
    _preserve_source_with_warning(
        line,
        "no Punjabi/Hindi script available for machine translation; source preserved unchanged",
    )


def _ensure_translation_invariant(lines: Sequence[LineResult]) -> None:
    """Guarantee an auditable nonempty output for every nonempty OCR string."""

    for line in lines:
        if normalize_text(line.text) and not normalize_text(line.english):
            _preserve_source_with_warning(
                line,
                "translation produced no usable English; OCR source retained unchanged",
            )


def _finalize_lines_for_export(lines: Sequence[LineResult]) -> None:
    """Mandatory final gate immediately before JSON, text, or Word export."""

    for line in lines:
        source = normalize_text(line.text)
        if not source:
            continue
        route = _translation_script_for_line(line)
        # This is a completeness preflight, not a replacement for the final
        # safety gate below.  It prevents an alternate caller or legacy result
        # object from exporting a safe-but-truncated model opening as though it
        # were a completed translation.
        if route is not None and line.translation_status.startswith("translated"):
            protected = protect_translation_source(normalize_translation_source(source))
            provisional = FinalizedTranslation(
                english=normalize_text(line.english),
                quality=line.translation_quality,
                used_conservative_fallback=False,
                review_reasons=(),
            )
            complete, completeness_reasons = translation_candidate_is_complete(
                source, provisional, route, protected
            )
            if not complete:
                repaired = finalize_translation(source, "", script=route)
                line.english = repaired.english
                line.translation_status = "preserved_source_with_warnings"
                line.translation_error = None
                line.translation_confidence = None
                line.translation_quality = 0.0
                line.translation_review_reasons = list(
                    dict.fromkeys(
                        [
                            *line.translation_review_reasons,
                            "translation_coverage_warning",
                            "conservative_translation_fallback",
                            *completeness_reasons,
                        ]
                    )
                )
                line.review_required = True
        finalized = finalize_translation(source, line.english, script=line.script)
        line.english = finalized.english
        if finalized.used_conservative_fallback:
            line.translation_status = "preserved_source_with_warnings"
            line.translation_confidence = None
            line.translation_quality = 0.0
            line.translation_review_reasons = list(
                dict.fromkeys(
                    [*line.translation_review_reasons, *finalized.review_reasons]
                )
            )
            line.review_required = True
        line.translation_error = None
        if not normalize_text(line.english):
            raise AssertionError("non-empty OCR source reached export without English output")
        checked, failures = _final_output_safety_reasons(
            source, line.english, protect_translation_source(normalize_translation_source(source))
        )
        if failures or checked != normalize_text(line.english):
            # A malformed legacy candidate must never abort an otherwise valid
            # document export. Force the deterministic path and assert only
            # after that last repair has been attempted.
            repaired = finalize_translation(source, "", script=line.script)
            line.english = repaired.english
            line.translation_status = "preserved_source_with_warnings"
            line.translation_error = None
            line.translation_confidence = None
            line.translation_quality = 0.0
            line.translation_review_reasons = list(
                dict.fromkeys([*line.translation_review_reasons, *repaired.review_reasons])
            )
            line.review_required = True
            checked, failures = _final_output_safety_reasons(
                source,
                line.english,
                protect_translation_source(normalize_translation_source(source)),
            )
            if failures or checked != normalize_text(line.english):
                raise AssertionError(
                    f"unsafe conservative translation for line {line.line}: {failures}"
                )


def _translate_with_nllb(
    lines: list[LineResult],
    models: ModelRegistry,
    progress_callback: ProgressCallback | None = None,
    num_beams: int = 4,
) -> float:
    """Translate every nonempty OCR line while preserving its review evidence."""

    started = time.perf_counter()
    # Unknown/Latin/identifier-only content has no safe Indic language route.
    # Keeping it verbatim is the correct English result for labels, numbers,
    # names, and already-English source; it also guarantees a nonempty record.
    for line in lines:
        if normalize_text(line.text) and _translation_script_for_line(line) is None:
            _preserve_unroutable_content(line)
    units = build_translation_units(lines)
    if not units:
        _ensure_translation_invariant(lines)
        return time.perf_counter() - started
    try:
        tokenizer, model = models.load_translation()
    except Exception as exc:
        models.release_translation()
        for unit in units:
            for line in unit.lines:
                _preserve_source_with_warning(
                    line, f"translation model unavailable: {type(exc).__name__}: {exc}"
                )
        models.status["translation"]["error"] = f"{type(exc).__name__}: {exc}"
        _ensure_translation_invariant(lines)
        return time.perf_counter() - started

    torch = models.torch
    target_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    requested_beams = max(4, int(num_beams))

    def translate_batch(
        current: Sequence[TranslationUnit],
        beams: int,
        *,
        strict: bool,
    ) -> list[tuple[str, float | None]]:
        encoded = tokenizer(
            # The retry intentionally uses the original line, not internal
            # sentinel strings.  It is still validated against protected spans
            # before any result is accepted.
            [unit.protected.text if strict else unit.source for unit in current],
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(models.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.inference_mode(), models.autocast():
            output = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=384,
                num_beams=beams,
                length_penalty=1.0 if strict else 0.85,
                repetition_penalty=1.05 if strict else 1.0,
                no_repeat_ngram_size=3 if strict else 0,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        decoded = [
            normalize_text(value)
            for value in tokenizer.batch_decode(
                output.sequences,
                skip_special_tokens=True,
            )
        ]
        confidences = _generation_confidences(model, output)
        return list(zip(decoded, confidences, strict=True))

    def apply_strict_translation(
        unit: TranslationUnit,
        raw_translation: str,
        confidence: float | None,
    ) -> list[str]:
        finalized = finalize_translation(
            unit.source, raw_translation, script=unit.script, protected=unit.protected
        )
        # A strict candidate that needs the source-preserving fallback gets one
        # retry on the unprotected original line before any final storage.
        if finalized.used_conservative_fallback:
            return ["strict translation hard validation failed"]
        coverage = translation_completeness(
            unit.source, finalized.english, unit.script, unit.protected
        )
        if not coverage.complete:
            return ["translation_incomplete_retry", *coverage.reasons]
        for line in unit.lines:
            line.english = finalized.english
            line.translation_confidence = confidence
            line.translation_quality = finalized.quality
            line.translation_review_reasons = list(finalized.review_reasons)
            line.translation_context_group = unit.group_id
            line.translation_status = (
                "translated_with_warnings" if finalized.review_reasons else "translated"
            )
            line.translation_error = None
            # Retain OCR uncertainty: translating an unaccepted source does not
            # make that source accepted.
            line.review_required = bool(
                line.review_required or line.review_reasons or finalized.review_reasons
            )
        return []

    def retain_fallback_translation(
        unit: TranslationUnit,
        raw_translation: str,
        confidence: float | None,
        strict_reasons: Sequence[str],
    ) -> bool:
        finalized = finalize_translation(
            unit.source, raw_translation, script=unit.script, protected=unit.protected
        )
        if not finalized.used_conservative_fallback:
            coverage = translation_completeness(
                unit.source, finalized.english, unit.script, unit.protected
            )
            if not coverage.complete:
                return False
        for line in unit.lines:
            if not finalized.used_conservative_fallback:
                line.english = finalized.english
                line.translation_status = "translated_with_warnings"
                # Strict-path failures are obsolete once this independent retry
                # has passed all hard checks; do not leak stale errors.
                line.translation_error = None
                line.translation_confidence = confidence
                line.translation_quality = finalized.quality
                line.translation_review_reasons = list(
                    dict.fromkeys(
                        [
                            "approximate translation after strict validation retry",
                            *( ["low_source_quality"] if not line.accepted else [] ),
                            *finalized.review_reasons,
                        ]
                    )
                )
                line.translation_context_group = unit.group_id
                line.review_required = True
            else:
                line.english = finalized.english
                line.translation_status = "preserved_source_with_warnings"
                line.translation_error = None
                line.translation_confidence = None
                line.translation_quality = 0.0
                line.translation_review_reasons = list(
                    dict.fromkeys(
                        [
                            "conservative source-preserving fallback used",
                            "low_source_quality" if not line.accepted else "approximate translation unavailable",
                            *finalized.review_reasons,
                        ]
                    )
                )
                line.translation_context_group = unit.group_id
                line.review_required = True
        return not finalized.used_conservative_fallback

    def retain_chunked_translation(
        unit: TranslationUnit,
        raw_translation: str,
        confidence: float | None,
        prior_reasons: Sequence[str],
    ) -> bool:
        """Store an ordered chunk retry only after whole-line safety checks."""

        finalized = finalize_translation(
            unit.source, raw_translation, script=unit.script, protected=unit.protected
        )
        coverage = translation_completeness(
            unit.source, finalized.english, unit.script, unit.protected
        )
        if finalized.used_conservative_fallback or not coverage.complete:
            return False
        for line in unit.lines:
            line.english = finalized.english
            line.translation_status = "translated_with_warnings"
            line.translation_error = None
            line.translation_confidence = confidence
            line.translation_quality = finalized.quality
            line.translation_review_reasons = list(
                dict.fromkeys(
                    [
                        "translation_incomplete_retry",
                        "translation_coverage_warning",
                        *prior_reasons,
                        *finalized.review_reasons,
                    ]
                )
            )
            line.translation_context_group = unit.group_id
            line.review_required = bool(line.review_required or line.review_reasons)
        return True

    def retry_in_chunks(unit: TranslationUnit, prior_reasons: Sequence[str]) -> bool:
        """Retry only this source line in safe semantic-size chunks."""

        chunks = split_translation_clauses(unit.source, unit.script)
        if len(chunks) < 2:
            return False
        chunk_units = [
            TranslationUnit(
                lines=[],
                script=unit.script,
                source=chunk,
                protected=protect_translation_source(chunk),
                group_id=f"{unit.group_id}:chunk{index}",
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        try:
            translated = translate_batch(chunk_units, max(2, requested_beams - 1), strict=False)
        except Exception:
            return False
        rendered: list[str] = []
        confidences: list[float] = []
        for chunk, (raw, confidence) in zip(chunk_units, translated, strict=True):
            finalized = finalize_translation(
                chunk.source, raw, script=chunk.script, protected=chunk.protected
            )
            coverage = translation_completeness(
                chunk.source, finalized.english, chunk.script, chunk.protected
            )
            if finalized.used_conservative_fallback or not coverage.complete:
                return False
            rendered.append(finalized.english)
            if confidence is not None:
                confidences.append(confidence)
        confidence = sum(confidences) / len(confidences) if confidences else None
        return retain_chunked_translation(
            unit, " ".join(rendered), confidence, prior_reasons
        )

    def retry_unit(unit: TranslationUnit, strict_reasons: Sequence[str]) -> None:
        try:
            raw, confidence = translate_batch([unit], max(2, requested_beams - 2), strict=False)[0]
            if retain_fallback_translation(unit, raw, confidence, strict_reasons):
                return
            if retry_in_chunks(unit, strict_reasons):
                return
            for line in unit.lines:
                _preserve_source_with_warning(
                    line,
                    "translation_coverage_warning; complete model translation unavailable",
                )
        except Exception as exc:
            for line in unit.lines:
                _preserve_source_with_warning(
                    line,
                    "; ".join(
                        [
                            *strict_reasons,
                            f"fallback translation failed: {type(exc).__name__}: {exc}",
                        ]
                    ),
                )

    for script in ("gurmukhi", "devanagari"):
        group = [unit for unit in units if unit.script == script]
        source_language = _language_code(script)  # type: ignore[arg-type]
        if not group or source_language is None:
            continue
        tokenizer.src_lang = source_language
        total_batches = max(1, math.ceil(len(group) / TRANSLATION_BATCH_SIZE))
        for batch_index, current in enumerate(
            _batch(group, TRANSLATION_BATCH_SIZE), start=1
        ):
            _report_progress(
                progress_callback,
                f"English translation batch {batch_index}/{total_batches} ({source_language})",
            )
            try:
                translated = translate_batch(current, requested_beams, strict=True)
                for unit, (raw, confidence) in zip(current, translated, strict=True):
                    reasons = apply_strict_translation(unit, raw, confidence)
                    if reasons:
                        retry_unit(unit, reasons)
            except Exception as batch_exc:
                # One unusually shaped unit must not discard the full batch.
                for unit in current:
                    retry_unit(
                        unit,
                        [
                            "strict translation batch failed: "
                            f"{type(batch_exc).__name__}: {batch_exc}"
                        ],
                    )
                models.status["translation"]["last_batch_error"] = (
                    f"{type(batch_exc).__name__}: {batch_exc}"
                )
    models.status["translation"].update(
        {
            "decoding": "quality_beam_search",
            "num_beams": requested_beams,
            "context_max_lines": TRANSLATION_CONTEXT_MAX_LINES,
            "protected_tokens": True,
            "output_validation": True,
        }
    )
    tokenizer = None
    model = None
    models.release_translation()
    _ensure_translation_invariant(lines)
    return time.perf_counter() - started


@dataclass(slots=True)
class TranslationAttempt:
    """Sanitized provider audit record; it never contains request headers or keys."""

    provider: str
    outcome: str
    validation: str
    reason: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "provider": self.provider,
            "outcome": self.outcome,
            "validation": self.validation,
            "reason": self.reason,
        }


class TranslationProvider:
    """Small provider contract used by the ordered, per-line translation cascade."""

    name = "provider"

    def __init__(self) -> None:
        self.cache: dict[tuple[str, str, str], str] = {}
        self.unavailable_reason: str | None = None

    def translate(self, source: str, source_language: str) -> str:
        key = (normalize_translation_source(source), source_language, "eng_Latn")
        if key not in self.cache:
            self.cache[key] = self._translate_uncached(source, source_language)
        return self.cache[key]

    def _translate_uncached(self, source: str, source_language: str) -> str:
        raise NotImplementedError


class OpenAITranslationProvider(TranslationProvider):
    """Optional high-quality line translator configured by OPENAI_API_KEY."""

    name = "openai"
    _language_labels = {
        "pan_Guru": "Punjabi written in Gurmukhi script",
        "hin_Deva": "Hindi written in Devanagari script",
    }

    def __init__(self) -> None:
        super().__init__()
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.model = os.environ.get("OPENAI_TRANSLATION_MODEL", OPENAI_TRANSLATION_MODEL_ID).strip()
        if not self.api_key:
            self.unavailable_reason = "OpenAI API key is not configured"

    def _translate_uncached(self, source: str, source_language: str) -> str:
        if self.unavailable_reason:
            raise RuntimeError(self.unavailable_reason)
        language = self._language_labels.get(source_language)
        if language is None:
            raise ValueError("Unsupported OpenAI source language")
        prompt = (
            "Translate exactly one OCR line into English.\n"
            f"Source language: {language}.\n"
            "Rules:\n"
            "- Return only the English rendering, no notes.\n"
            "- Translate all readable Punjabi/Hindi words; do not stop after the first phrase.\n"
            "- Preserve names, dates, times, phone numbers, FIR/GD/case numbers, legal sections, "
            "IDs, abbreviations, punctuation-sensitive values, and existing English exactly.\n"
            "- If OCR is noisy, translate only what is visible and preserve uncertain fragments verbatim.\n"
            "- Do not invent missing facts or autocorrect uncertain OCR.\n\n"
            f"OCR line:\n{source}"
        )
        body = json.dumps(
            {
                "model": self.model,
                "input": prompt,
                "max_output_tokens": 700,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        for attempt, delay in enumerate((0.0, 0.8), start=1):
            if delay:
                time.sleep(delay)
            request = urllib.request.Request(
                "https://api.openai.com/v1/responses",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=35) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                text = normalize_text(str(payload.get("output_text") or ""))
                if text:
                    return text
                fragments: list[str] = []
                for item in payload.get("output", []) or []:
                    for content in item.get("content", []) or []:
                        value = content.get("text")
                        if isinstance(value, str):
                            fragments.append(value)
                return normalize_text(" ".join(fragments))
            except urllib.error.HTTPError as exc:
                # Do not retry auth/config/model errors; avoid exposing the key.
                if exc.code in {400, 401, 403, 404} or attempt == 2:
                    raise RuntimeError(f"OpenAI Responses API HTTP {exc.code}") from None
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                if attempt == 2:
                    raise RuntimeError("OpenAI Responses API timed out or was unavailable") from exc
        raise RuntimeError("OpenAI Responses API request failed")


class HelsinkiProvider(TranslationProvider):
    """Public Marian model: supports pan_Guru and hin, English-only target.

    This many-to-one model uses the source script directly, without NLLB/FLORES
    prefix tokens or a target-language tag. The public audit still uses FLORES.
    """

    name = "helsinki"

    def __init__(self, models: ModelRegistry) -> None:
        super().__init__()
        self.models = models

    def _translate_uncached(self, source: str, source_language: str) -> str:
        if source_language not in {"pan_Guru", "hin_Deva"}:
            raise ValueError("Unsupported Helsinki source language")
        if self.unavailable_reason:
            raise RuntimeError(self.unavailable_reason)
        try:
            tokenizer, model = self.models.load_helsinki()
        except Exception as exc:
            self.unavailable_reason = f"Helsinki model could not load ({type(exc).__name__})"
            raise RuntimeError(self.unavailable_reason) from exc
        encoded = tokenizer(source, return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[-1] > 480:
            raise ValueError("Source exceeds model window; clause retry required")
        encoded = {key: value.to(self.models.device) for key, value in encoded.items()}
        with self.models.torch.inference_mode(), self.models.autocast():
            generated = model.generate(
                **encoded, max_new_tokens=384, num_beams=4,
                no_repeat_ngram_size=3, repetition_penalty=1.05,
            )
        return normalize_text(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])


class IndicTrans2Provider(TranslationProvider):
    """Official local AI4Bharat IndicTrans2 Indic→English provider."""

    name = "indictrans2"

    def __init__(self, models: ModelRegistry) -> None:
        super().__init__()
        self.models = models

    def _translate_uncached(self, source: str, source_language: str) -> str:
        tokenizer, model, processor = self.models.load_indictrans2()
        try:
            prepared = processor.preprocess_batch(
                [source], src_lang=source_language, tgt_lang="eng_Latn"
            )
        except TypeError:
            prepared = processor.preprocess_batch([source], source_language, "eng_Latn")
        encoded = tokenizer(
            prepared, return_tensors="pt", padding=True, truncation=True, max_length=384
        )
        encoded = {
            key: value.to(self.models.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with self.models.torch.inference_mode(), self.models.autocast():
            output = model.generate(
                **encoded,
                max_new_tokens=384,
                num_beams=4,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
            )
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        try:
            translated = processor.postprocess_batch(decoded, lang="eng_Latn")
        except TypeError:
            translated = processor.postprocess_batch(decoded, "eng_Latn")
        return normalize_text(translated[0] if translated else "")


class AzureTranslatorProvider(TranslationProvider):
    """Optional Azure Translator REST fallback, configured only by environment."""

    name = "azure"
    _azure_languages = {"pan_Guru": "pa", "hin_Deva": "hi"}

    def __init__(self) -> None:
        super().__init__()
        self.key = os.environ.get("AZURE_TRANSLATOR_KEY", "").strip()
        self.region = os.environ.get("AZURE_TRANSLATOR_REGION", "").strip()
        self.endpoint = os.environ.get("AZURE_TRANSLATOR_ENDPOINT", "").strip()
        if not self.key or not self.endpoint:
            self.unavailable_reason = "Azure Translator is not configured"

    def _translate_uncached(self, source: str, source_language: str) -> str:
        if self.unavailable_reason:
            raise RuntimeError(self.unavailable_reason)
        source_code = self._azure_languages.get(source_language)
        if source_code is None:
            raise RuntimeError("Azure language route is unavailable")
        query = urllib.parse.urlencode({"api-version": "3.0", "from": source_code, "to": "en"})
        url = f"{self.endpoint.rstrip('/')}/translate?{query}"
        body = json.dumps([{"text": source}], ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Ocp-Apim-Subscription-Key": self.key,
        }
        if self.region:
            headers["Ocp-Apim-Subscription-Region"] = self.region
        # Bounded backoff only for transient errors. Authentication and bad
        # configuration never retry and secrets are never included in errors.
        for attempt, delay in enumerate((0.0, 0.4, 0.9), start=1):
            if delay:
                time.sleep(delay)
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=12) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return normalize_text(payload[0]["translations"][0]["text"])
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 403, 400} or attempt == 3 or exc.code not in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"Azure Translator HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == 3:
                    raise RuntimeError("Azure Translator request timed out or was unavailable") from exc
        raise RuntimeError("Azure Translator request failed")


def _append_translation_attempt(line: LineResult, attempt: TranslationAttempt) -> None:
    line.translation_attempts.append(attempt.as_dict())


def _store_provider_translation(
    line: LineResult,
    finalized: FinalizedTranslation,
    provider: str,
    source_language: str,
    *,
    chunked: bool = False,
) -> None:
    line.english = finalized.english
    line.translation_provider = provider
    line.translation_source_language = source_language
    line.translation_confidence = None
    line.translation_quality = finalized.quality
    line.translation_error = None
    line.translation_status = "translated_with_warnings" if (chunked or finalized.review_reasons or not line.accepted) else "translated"
    line.translation_review_reasons = list(
        dict.fromkeys(
            [
                *( ["translation_incomplete_retry"] if chunked else [] ),
                *( ["low_source_quality"] if not line.accepted else [] ),
                *finalized.review_reasons,
            ]
        )
    )
    line.review_required = bool(line.review_required or line.translation_review_reasons)


def _provider_candidate(
    provider: TranslationProvider,
    line: LineResult,
    source: str,
    source_language: str,
    script: ScriptName,
) -> tuple[FinalizedTranslation | None, list[str]]:
    """Translate one line then apply all existing hard and completeness gates."""

    try:
        raw = provider.translate(source, source_language)
    except Exception as exc:
        reason = provider.unavailable_reason or f"{type(exc).__name__}: {exc}"
        if provider.name == "openai":
            failure_text = str(exc)
            if "HTTP 400" in failure_text:
                provider.unavailable_reason = (
                    "OpenAI translation request was rejected; check OPENAI_TRANSLATION_MODEL"
                )
            elif "HTTP 401" in failure_text or "HTTP 403" in failure_text:
                provider.unavailable_reason = (
                    "OpenAI API key was rejected; check OPENAI_API_KEY"
                )
            elif "HTTP 404" in failure_text:
                provider.unavailable_reason = (
                    "OpenAI translation model was not found for this API key"
                )
            elif "HTTP 429" in failure_text:
                provider.unavailable_reason = (
                    "OpenAI rate or quota limit was reached; falling back for this run"
                )
            elif "timed out" in failure_text.lower() or "unavailable" in failure_text.lower():
                provider.unavailable_reason = (
                    "OpenAI translation was temporarily unavailable; falling back for this run"
                )
            if provider.unavailable_reason:
                reason = provider.unavailable_reason
        if provider.name == "indictrans2":
            # A missing gated-model approval/toolkit will not become available
            # for later lines in this run; avoid repeated downloads/errors.
            failure_type = type(exc).__name__
            if failure_type == "GatedRepoError" or "gated repo" in str(exc).lower():
                provider.unavailable_reason = (
                    "IndicTrans2 model access denied: accept the AI4Bharat model terms "
                    "on Hugging Face and sign in with an authorized account"
                )
            else:
                provider.unavailable_reason = (
                    f"IndicTrans2 could not load or run ({failure_type}); check its dependencies and model access"
                )
            reason = provider.unavailable_reason
            provider.models.status.setdefault("indictrans2", {}).update(
                {"loaded": False, "error": reason}
            )
            print(f"WARNING: {reason}")
        _append_translation_attempt(line, TranslationAttempt(provider.name, "unavailable", "not_run", reason))
        return None, [reason]
    protected = protect_translation_source(source)
    finalized = finalize_translation(source, raw, script=script, protected=protected)
    valid, reasons = translation_candidate_is_complete(source, finalized, script, protected)
    if valid:
        _append_translation_attempt(line, TranslationAttempt(provider.name, "accepted", "passed"))
        return finalized, []
    _append_translation_attempt(
        line,
        TranslationAttempt(provider.name, "rejected", "failed", "; ".join(reasons[:2]) or "hard validation failed"),
    )
    return None, reasons


def _provider_chunk_retry(
    provider: TranslationProvider,
    line: LineResult,
    source: str,
    source_language: str,
    script: ScriptName,
) -> FinalizedTranslation | None:
    chunks = split_translation_clauses(source, script)
    if len(chunks) < 2:
        return None
    rendered: list[str] = []
    for chunk in chunks:
        candidate, _reasons = _provider_candidate(
            provider, line, chunk, source_language, script
        )
        if candidate is None:
            return None
        rendered.append(candidate.english)
    protected = protect_translation_source(source)
    finalized = finalize_translation(source, " ".join(rendered), script=script, protected=protected)
    valid, reasons = translation_candidate_is_complete(source, finalized, script, protected)
    if not valid:
        _append_translation_attempt(
            line, TranslationAttempt(provider.name, "rejected", "chunk_validation_failed", "; ".join(reasons[:2]))
        )
        return None
    _append_translation_attempt(line, TranslationAttempt(provider.name, "accepted", "chunked_passed"))
    return finalized


def translate_text(
    lines: list[LineResult],
    models: ModelRegistry,
    progress_callback: ProgressCallback | None = None,
    num_beams: int = 4,
) -> float:
    """Translate with IndicTrans2 → optional Azure → NLLB → preservation."""

    started = time.perf_counter()
    providers: list[TranslationProvider] = [OpenAITranslationProvider()]
    if os.environ.get("OCR_ENABLE_INDICTRANS2_PROVIDER", "").strip() == "1":
        providers.append(IndicTrans2Provider(models))
    providers.append(AzureTranslatorProvider())
    if os.environ.get("OCR_ENABLE_HELSINKI_PROVIDER", "").strip() == "1":
        # The Helsinki OPUS checkpoint is public but produced question-mark
        # output for Gurmukhi/Devanagari on this Windows stack. Keep it opt-in
        # only so it cannot lower the default Word export quality.
        providers.append(HelsinkiProvider(models))
    pending: list[LineResult] = []
    for line in lines:
        source = normalize_translation_source(line.text)
        script = _translation_script_for_line(line)
        language = _language_code(script) if script is not None else None
        if not source or script is None or language is None:
            pending.append(line)
            continue
        line.translation_source_language = language
        translated = False
        for provider in providers:
            if provider.unavailable_reason:
                _append_translation_attempt(line, TranslationAttempt(provider.name, "unavailable", "not_run", provider.unavailable_reason))
                continue
            _report_progress(progress_callback, f"{provider.name} translation (line {line.line})")
            finalized, _reasons = _provider_candidate(provider, line, source, language, script)
            chunked = False
            if finalized is None and not provider.unavailable_reason:
                finalized = _provider_chunk_retry(provider, line, source, language, script)
                chunked = finalized is not None
            if finalized is not None:
                _store_provider_translation(
                    line, finalized, provider.name, language, chunked=chunked
                )
                translated = True
                break
        if not translated:
            pending.append(line)

    # NLLB remains the later local provider. Its established line/chunk logic
    # and finalizer are retained; only lines not accepted above reach it.
    if hasattr(models, "_release"):
        models._release("helsinki_tokenizer", "helsinki_model")
    if pending:
        _translate_with_nllb(pending, models, progress_callback, num_beams=num_beams)
        for line in pending:
            script = _translation_script_for_line(line)
            language = _language_code(script) if script is not None else None
            if language is None or not normalize_text(line.text):
                continue
            line.translation_source_language = language
            if line.translation_status.startswith("translated"):
                protected = protect_translation_source(normalize_translation_source(line.text))
                finalized = finalize_translation(
                    line.text, line.english, script=script, protected=protected
                )
                complete, reasons = translation_candidate_is_complete(
                    line.text, finalized, script, protected
                )
                if complete:
                    line.translation_provider = "nllb"
                    _append_translation_attempt(line, TranslationAttempt("nllb", "accepted", "passed"))
                else:
                    _preserve_source_with_warning(
                        line,
                        "all eligible translation providers failed completeness validation; source preserved",
                    )
                    line.translation_provider = "conservative_fallback"
                    _append_translation_attempt(
                        line,
                        TranslationAttempt("nllb", "rejected", "failed", "; ".join(reasons[:2])),
                    )
            else:
                line.translation_provider = "conservative_fallback"
                _append_translation_attempt(line, TranslationAttempt("nllb", "rejected", "failed", "NLLB did not produce a valid complete translation"))
                line.translation_review_reasons = list(dict.fromkeys([
                    *line.translation_review_reasons,
                    "all eligible translation providers failed; source preserved",
                ]))
    models.release_translation()
    _ensure_translation_invariant(lines)
    return time.perf_counter() - started


def _draw_detected_lines(page_rgb: np.ndarray, lines: Sequence[LineResult]) -> np.ndarray:
    output = cv2.cvtColor(page_rgb.copy(), cv2.COLOR_RGB2BGR)
    for line in lines:
        x1, y1, x2, y2 = line.bbox
        color = (52, 180, 66) if line.accepted and not line.review_required else (25, 120, 245)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = str(line.line)
        cv2.rectangle(output, (x1, max(0, y1 - 22)), (x1 + 30, y1), color, -1)
        cv2.putText(
            output,
            label,
            (x1 + 4, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def _line_transcription(line: LineResult) -> str:
    if line.accepted:
        return line.text
    return f"[REVIEW REQUIRED — line {line.line}] {line.text}".rstrip()


def _line_translation(line: LineResult) -> str:
    if line.translation_status in {
        "translated",
        "translated_low_confidence",
        "translated_with_warnings",
        "manual_translation",
        "preserved_english",
        "preserved_structured",
        "preserved_source_with_warnings",
    }:
        return line.english
    if line.review_required:
        return f"[REVIEW REQUIRED — line {line.line}]"
    return ""


def _word_document_line(line: LineResult) -> str:
    """Return one nonempty, provenance-safe Word paragraph per detected line."""

    english = normalize_text(line.english)
    source = normalize_text(line.text)
    if line.translation_status in {
        "translated",
        "translated_low_confidence",
        "translated_with_warnings",
        "manual_translation",
        "preserved_english",
        "preserved_structured",
        "preserved_source_with_warnings",
    } and english:
        return english
    if line.accepted and source:
        return (
            f"[Translation unavailable for line {line.line}; source retained] "
            f"{source}"
        )
    if source:
        return f"[Review required for line {line.line}; uncertain source retained] {source}"
    return f"[Review required; no reliable transcription for detected line {line.line}]"


def build_translated_docx(page: EnhancedPage, lines: Sequence[LineResult]) -> bytes:
    """Build a clean Word document containing every detected line in reading order."""

    from docx import Document
    from docx.enum.text import WD_LINE_SPACING
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    document = Document()
    section = document.sections[0]
    source_height, source_width = page.corrected_rgb.shape[:2]
    page_width_inches = 11.69 if source_width >= source_height else 8.27
    page_height_inches = page_width_inches * source_height / max(1, source_width)
    page_height_inches = max(5.0, min(22.0, page_height_inches))
    section.page_width = Inches(page_width_inches)
    section.page_height = Inches(page_height_inches)
    section.top_margin = Inches(0.45)
    section.bottom_margin = Inches(0.45)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)

    normal = document.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Nirmala UI")

    usable_height_points = max(1.0, (page_height_inches - 0.9) * 72.0)
    previous_bottom = 0
    previous_page = lines[0].page_number if lines else 1
    expected_paragraphs: list[str] = []
    for line in lines:
        if line.page_number != previous_page:
            document.add_page_break()
            previous_bottom = 0
            previous_page = line.page_number
        value = _word_document_line(line)
        expected_paragraphs.append(value)
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.keep_together = True
        paragraph.paragraph_format.keep_with_next = False
        paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        paragraph.paragraph_format.space_after = Pt(0)
        y_gap = max(0, line.bbox[1] - previous_bottom)
        paragraph.paragraph_format.space_before = Pt(
            min(18.0, y_gap / max(1, source_height) * usable_height_points)
        )
        previous_bottom = max(previous_bottom, line.bbox[3])
        run = paragraph.add_run(value)
        run.font.name = (
            "Aptos"
            if line.translation_status in {"translated", "translated_with_warnings"}
            else "Nirmala UI"
        )
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Nirmala UI")
        source_line_height = max(1, line.bbox[3] - line.bbox[1])
        run.font.size = Pt(max(8.0, min(13.0, source_line_height * 0.32)))

    stream = BytesIO()
    document.save(stream)
    content = stream.getvalue()
    reopened = Document(BytesIO(content))
    actual_paragraphs = [
        paragraph.text for paragraph in reopened.paragraphs if paragraph.text
    ]
    if actual_paragraphs != expected_paragraphs:
        raise OSError("Generated Word document failed its ordered-line validation")
    return content


def build_translated_docx_from_image_bytes(
    corrected_page: bytes, lines: Sequence[LineResult]
) -> bytes:
    """Rebuild the spatial Word output after a reviewer corrects source text.

    The review UI keeps artifacts as bytes, so this small public wrapper avoids
    re-running OCR merely to update the downloadable Word document.
    """

    with Image.open(BytesIO(corrected_page)) as image:
        rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return build_translated_docx(
        EnhancedPage(
            corrected_rgb=rgb,
            enhanced_rgb=rgb,
            enhanced_gray=gray,
            threshold=gray,
            operations=["reviewed source reconstruction"],
        ),
        lines,
    )


def save_results(
    *,
    input_image: Path,
    page: EnhancedPage,
    lines: list[LineResult],
    models: ModelRegistry,
    timings: dict[str, float],
    output_directory: Path = OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    """Write primary Word output plus image, text, JSON, and CSV artifacts."""

    # Export is the final safety net for callers that bypassed translate_text.
    _ensure_translation_invariant(lines)
    _finalize_lines_for_export(lines)
    output_directory.mkdir(parents=True, exist_ok=True)
    corrected_bgr = cv2.cvtColor(page.corrected_rgb, cv2.COLOR_RGB2BGR)
    corrected_path = output_directory / "corrected_page.jpg"
    detected_path = output_directory / "detected_lines.jpg"
    corrected_written = cv2.imwrite(str(corrected_path), corrected_bgr)
    detected_written = cv2.imwrite(
        str(detected_path),
        _draw_detected_lines(page.corrected_rgb, lines),
    )
    if not corrected_written or not detected_written:
        failed = [
            str(path)
            for path, written in (
                (corrected_path, corrected_written),
                (detected_path, detected_written),
            )
            if not written
        ]
        raise OSError(f"Could not write output image(s): {', '.join(failed)}")

    transcription = "\n".join(_line_transcription(line) for line in lines)
    english_translation = "\n".join(
        value for value in (_line_translation(line) for line in lines) if value
    )
    (output_directory / "transcription.txt").write_text(
        transcription + "\n", encoding="utf-8"
    )
    (output_directory / "translation_en.txt").write_text(
        english_translation + "\n", encoding="utf-8"
    )
    translated_docx_path = output_directory / "translated_en.docx"
    translated_docx_path.write_bytes(build_translated_docx(page, lines))

    payload = {
        "schema_version": "1.0",
        "input_image": str(input_image),
        "output_directory": str(output_directory),
        "inference_only": True,
        "models": models.status,
        "preprocessing_operations": page.operations,
        "lines": [asdict(line) for line in lines],
        "transcription": transcription,
        "english_translation": english_translation,
        "primary_output": translated_docx_path.name,
        "timings_seconds": timings,
        "accuracy_warning": (
            "Handwriting OCR is probabilistic. Review every uncertain line; this output "
            "is not a certified legal or medical transcription or translation."
        ),
    }
    (output_directory / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_fields = [
        "page_number",
        "line",
        "bbox",
        "script",
        "text",
        "script_purity",
        "confidence",
        "text_quality",
        "recognition_model",
        "provider_kind",
        "accepted",
        "review_required",
        "review_reasons",
        "english",
        "translation_confidence",
        "translation_quality",
        "translation_review_reasons",
        "translation_context_group",
        "translation_status",
        "translation_error",
        "crop_file",
    ]
    with (output_directory / "result.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for line in lines:
            row = asdict(line)
            row["bbox"] = json.dumps(row["bbox"])
            row["review_reasons"] = "; ".join(line.review_reasons)
            row["translation_review_reasons"] = "; ".join(
                line.translation_review_reasons
            )
            writer.writerow({key: row.get(key) for key in csv_fields})
    return payload


def _resolve_input(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        if "..." not in explicit:
            raise FileNotFoundError(f"PAGE_IMAGE does not exist: {candidate}")
        print("PAGE_IMAGE still contains '...'; auto-detecting an image under /kaggle/input.")
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        raise FileNotFoundError(
            "Set PAGE_IMAGE to an existing JPG/JPEG/PNG file; /kaggle/input is unavailable."
        )
    candidates = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
    )
    if not candidates:
        raise FileNotFoundError("No JPG/JPEG/PNG image was found under /kaggle/input")
    return candidates[0]


def process_page(
    image_path: Path,
    output_directory: Path = OUTPUT_DIRECTORY,
    progress_callback: ProgressCallback | None = None,
    processing_mode: ProcessingMode = "fast_cpu",
) -> dict[str, Any]:
    """Run preprocessing, OCR validation, translation, and Word generation."""

    if processing_mode not in {"fast_cpu", "line_accurate"}:
        raise ValueError(f"Unsupported processing mode: {processing_mode}")

    total_started = time.perf_counter()
    models = load_models()
    _report_progress(progress_callback, "Correcting and enhancing the uploaded page")
    with Image.open(image_path) as source:
        corrected_rgb, operations = correct_document(source)
    page = enhance_page(corrected_rgb, operations)

    detection_started = time.perf_counter()
    _report_progress(progress_callback, "Detecting logical handwritten text lines")
    regions = detect_text_regions(page, models)
    detection_time = time.perf_counter() - detection_started
    if not regions:
        raise RuntimeError("No credible text-line regions were detected")

    crops = [prepare_crop(page.enhanced_rgb, region.bbox) for region in regions]
    output_directory.mkdir(parents=True, exist_ok=True)
    crop_directory = output_directory / "line_crops"
    crop_directory.mkdir(parents=True, exist_ok=True)
    for stale in crop_directory.glob("line_*.jpg"):
        stale.unlink()
    crop_names: list[str] = []
    for index, crop in enumerate(crops, start=1):
        relative = Path("line_crops") / f"line_{index:03d}.jpg"
        crop.save(output_directory / relative, quality=95)
        crop_names.append(relative.as_posix())

    ocr_started = time.perf_counter()
    fast_mode = processing_mode == "fast_cpu"
    _report_progress(
        progress_callback,
        (
            f"Recognizing all {len(crops)} detected lines"
            + (" in faster CPU batches" if fast_mode else " with accuracy routing")
        ),
    )
    # Punjabi is the language and Gurmukhi is the Unicode script used to
    # write it.  Most official Punjabi documents are printed, so begin with
    # the installed Punjabi/Hindi OCR packs.  Loading a large handwriting VLM
    # before this check was both slow and often less accurate on clean forms.
    _report_progress(progress_callback, "Running Punjabi and Hindi printed-OCR candidates")
    tesseract_gurmukhi = [
        _tesseract_candidate(crop, "pan", "gurmukhi") for crop in crops
    ]
    tesseract_hindi = [
        _tesseract_candidate(crop, "hin", "devanagari") for crop in crops
    ]
    unresolved_indices = [
        index
        for index, candidates in enumerate(zip(tesseract_gurmukhi, tesseract_hindi))
        if not choose_best_recognition(list(candidates))[1]
    ]
    gurmukhi_candidates = [
        RecognitionCandidate(
            expected_script="gurmukhi",
            text="",
            confidence=None,
            model_id=GURMUKHI_MODEL_ID,
            provider_kind="not_routed",
            error="printed Punjabi OCR passed source-validation gates",
        )
        for _crop in crops
    ]
    hindi_candidates = [
        RecognitionCandidate(
            expected_script="devanagari",
            text="",
            confidence=None,
            model_id=HINDI_MODEL_ID,
            provider_kind="not_routed",
            error="Hindi OCR not needed after validated Gurmukhi recognition",
        )
        for _crop in crops
    ]
    if unresolved_indices and fast_mode:
        # Fast mode intentionally remains local and deterministic.  The
        # low-confidence lines remain editable in the review table rather
        # than being replaced by a hallucinated reading.
        models.status["hindi"].update(
            {
                "id": "tesseract:hin",
                "fallback": True,
                "skip_reason": "fast mode uses installed printed-OCR candidates",
            }
        )
    elif not fast_mode:
        _report_progress(
            progress_callback,
            (
                f"Running handwriting OCR only for {len(unresolved_indices)} "
                "weak printed-OCR lines"
            ),
        )
        if unresolved_indices:
            unresolved_crops = [crops[index] for index in unresolved_indices]
            htr_gurmukhi = recognize_gurmukhi(
                unresolved_crops, models, progress_callback, fast=False
            )
            htr_hindi = recognize_hindi(unresolved_crops, models, progress_callback)
            for index, gurmukhi, hindi in zip(
                unresolved_indices, htr_gurmukhi, htr_hindi, strict=True
            ):
                gurmukhi_candidates[index] = gurmukhi
                hindi_candidates[index] = hindi
    else:
        models.status["hindi"]["skipped"] = True
        models.status["hindi"]["skip_reason"] = (
            "all lines passed validated Gurmukhi source gates"
        )
    models.release_recognizers()
    lines: list[LineResult] = []
    for index, (region, hindi, gurmukhi) in enumerate(
        zip(regions, hindi_candidates, gurmukhi_candidates, strict=True), start=1
    ):
        candidates = [hindi, gurmukhi]
        if tesseract_gurmukhi:
            candidates.append(tesseract_gurmukhi[index - 1])
        if tesseract_hindi:
            candidates.append(tesseract_hindi[index - 1])
        winner, accepted, reasons = choose_best_recognition(
            candidates,
            nearby_text=[line.text for line in lines[-2:] if line.text],
        )
        script = winner.detected_script
        lines.append(
            LineResult(
                line=index,
                bbox=region.bbox,
                crop_file=crop_names[index - 1],
                detection_confidence=region.detection_confidence,
                detector=region.detector,
                script=script,
                text=winner.text,
                script_purity=winner.script_purity,
                confidence=winner.confidence,
                text_quality=winner.text_quality,
                recognition_model=winner.model_id,
                provider_kind=winner.provider_kind,
                accepted=accepted,
                review_required=not accepted,
                review_reasons=reasons,
                candidates=[
                    _candidate_for_audit(enrich_candidate(candidate))
                    for candidate in candidates
                ],
            )
        )
    ocr_time = time.perf_counter() - ocr_started
    _report_progress(progress_callback, "Translating validated source lines to English")
    translation_time = translate_text(
        lines,
        models,
        progress_callback,
        num_beams=4 if fast_mode else 6,
    )
    total_time = time.perf_counter() - total_started
    timings = {
        "detection": detection_time,
        "ocr": ocr_time,
        "translation": translation_time,
        "total": total_time,
    }
    models.status["processing"] = {"mode": processing_mode}
    payload = save_results(
        input_image=image_path,
        page=page,
        lines=lines,
        models=models,
        timings=timings,
        output_directory=output_directory,
    )
    accepted_count = sum(line.accepted for line in lines)
    review_count = sum(line.review_required for line in lines)
    print("\nINFERENCE SUMMARY")
    print(f"GPU              : {models.gpu_name}")
    print(f"Hindi model      : {models.status['hindi']['id']}")
    gurmukhi_id = str(models.status["gurmukhi"]["id"])
    if gurmukhi_id == GURMUKHI_MODEL_ID:
        gurmukhi_note = "multilingual handwriting fallback; not a dedicated Gurmukhi HTR checkpoint"
    else:
        gurmukhi_note = "last-resort printed OCR fallback; not handwriting-specialized"
    print(f"Gurmukhi model   : {gurmukhi_id} ({gurmukhi_note})")
    print(
        "Translation      : "
        f"OpenAI ({OPENAI_TRANSLATION_MODEL_ID}) → Azure (if configured) → "
        f"{models.status['translation']['id']}"
    )
    print(f"Detected regions : {len(lines)}")
    print(f"Accepted lines   : {accepted_count}")
    print(f"Review required  : {review_count}")
    provider_counts = Counter(line.translation_provider for line in lines if line.translation_provider)
    for language, script in (("Punjabi", "gurmukhi"), ("Hindi", "devanagari")):
        routed = [line for line in lines if line.script == script]
        print(
            f"{language} providers : OpenAI={sum(line.translation_provider == 'openai' for line in routed)}, "
            f"Helsinki={sum(line.translation_provider == 'helsinki' for line in routed)}, "
            f"IndicTrans2={sum(line.translation_provider == 'indictrans2' for line in routed)}, "
            f"Azure={sum(line.translation_provider == 'azure' for line in routed)}, "
            f"NLLB={sum(line.translation_provider == 'nllb' for line in routed)}, "
            f"conservative={sum(line.translation_provider == 'conservative_fallback' for line in routed)}"
        )
    print(f"Provider total    : {dict(provider_counts)}")
    print(f"OCR time         : {ocr_time:.2f}s")
    print(f"Translation time : {translation_time:.2f}s")
    print(f"Total time       : {total_time:.2f}s")
    print("\nPUNJABI/HINDI TRANSCRIPTION")
    print(payload["transcription"])
    print("\nENGLISH TRANSLATION")
    print(payload["english_translation"])
    print(f"\nResults written to: {output_directory}")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference-only handwritten Hindi/Gurmukhi page OCR"
    )
    parser.add_argument("--image", type=str, help="Override PAGE_IMAGE")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument(
        "--mode",
        choices=("fast_cpu", "line_accurate"),
        default="fast_cpu",
        help="Use faster CPU batching or the slower dual-neural-model accuracy path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        image_path = _resolve_input(args.image if args.image is not None else PAGE_IMAGE)
        process_page(image_path, args.output, processing_mode=args.mode)
        return 0
    except KeyboardInterrupt:
        print("Cancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
