"""Streamlit upload interface for the pretrained handwritten-page pipeline."""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import json
import mimetypes
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import streamlit as st
import numpy as np
import pandas as pd
from packaging.version import Version
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _suppress_benign_windows_disconnect() -> None:
    """Avoid a noisy traceback when a browser/HF socket closes on Windows.

    WinError 10054 is a peer-reset notification from the Proactor transport;
    it does not corrupt model state or the current OCR request.  Other asyncio
    errors still use the original exception handler and remain visible.
    """

    if sys.platform != "win32":
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Streamlit may execute the script before its server loop is actively
        # running. Attach to that thread's default loop as well.
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            return
    if getattr(loop, "_indic_ocr_disconnect_handler_installed", False):
        return
    previous_handler = loop.get_exception_handler()

    def handle(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        error = context.get("exception")
        if isinstance(error, ConnectionResetError) and getattr(error, "winerror", None) == 10054:
            return
        if previous_handler is not None:
            previous_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handle)
    setattr(loop, "_indic_ocr_disconnect_handler_installed", True)


_suppress_benign_windows_disconnect()

from pretrained_page_ocr.kaggle_app import (  # noqa: E402
    LineResult,
    ModelRegistry,
    build_translated_docx_from_image_bytes,
    calculate_text_quality,
    detect_script,
    meaningful_residual_script_words,
    normalize_text,
    protect_translation_source,
    process_page,
    script_purity,
    finalize_translation,
    translate_text,
)

try:  # The document router is kept separate from the Streamlit view.
    from pretrained_page_ocr.document_processor import process_document  # noqa: E402
except ImportError:  # Allows image-only installations to keep starting cleanly.
    process_document = None


MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".docx"}
DOWNLOAD_MIME_TYPES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json",
    ".csv": "text/csv; charset=utf-8",
}


@dataclass(frozen=True, slots=True)
class UploadInfo:
    """Validated, display-safe metadata for one uploaded document."""

    kind: str
    description: str
    preview: bytes | None = None


def _safe_input_name(original_name: str, extension: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original_name).stem).strip("._")
    return f"{stem or 'uploaded_document'}{extension}"


def _validate_image_upload(name: str, content: bytes) -> tuple[str, UploadInfo]:
    """Validate an image by decoded content rather than its extension alone."""

    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            image_format = str(image.format or "").upper()
            dimensions = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The upload is not a valid JPG, JPEG, or PNG image.") from exc
    if image_format not in ALLOWED_FORMATS:
        raise ValueError(f"Unsupported image format: {image_format or 'unknown'}.")
    if dimensions[0] <= 0 or dimensions[1] <= 0:
        raise ValueError("The uploaded image has invalid dimensions.")
    if dimensions[0] * dimensions[1] > MAX_IMAGE_PIXELS:
        raise ValueError("The image is too large to process safely (maximum 50 megapixels).")
    return (
        _safe_input_name(name, ALLOWED_FORMATS[image_format]),
        UploadInfo(
            kind="image",
            description=f"{dimensions[0]} × {dimensions[1]} pixels",
            preview=content,
        ),
    )


def _validate_pdf_upload(name: str, content: bytes) -> tuple[str, UploadInfo]:
    """Validate PDF structure and render a lightweight first-page preview."""

    if not content.startswith(b"%PDF-"):
        raise ValueError("The upload does not contain a valid PDF signature.")
    try:
        import pymupdf

        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("Password-protected PDFs are not supported.")
            if document.page_count < 1:
                raise ValueError("The PDF has no pages.")
            if document.page_count > 100:
                raise ValueError("The fast document interface supports up to 100 PDF pages per run.")
            page_count = document.page_count
            page = document.load_page(0)
            scale = min(
                1.25,
                1100.0 / max(1.0, page.rect.width),
                1500.0 / max(1.0, page.rect.height),
            )
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            preview = pixmap.tobytes("png")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The PDF is corrupted or cannot be opened.") from exc
    noun = "page" if page_count == 1 else "pages"
    return (
        _safe_input_name(name, ".pdf"),
        UploadInfo(kind="pdf", description=f"{page_count} {noun}", preview=preview),
    )


def _validate_docx_upload(name: str, content: bytes) -> tuple[str, UploadInfo]:
    """Validate that the ZIP payload is a genuine, readable Word document."""

    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            names = set(archive.namelist())
            required = {"[Content_Types].xml", "word/document.xml"}
            if not required.issubset(names):
                raise ValueError("The upload is not a valid DOCX document.")
            media_count = sum(name.startswith("word/media/") for name in names)

        from docx import Document

        document = Document(BytesIO(content))
        paragraph_count = sum(bool(paragraph.text.strip()) for paragraph in document.paragraphs)
        table_count = len(document.tables)
        if not paragraph_count and not table_count and not media_count:
            raise ValueError("The Word document is empty.")
    except ValueError:
        raise
    except (zipfile.BadZipFile, OSError, KeyError) as exc:
        raise ValueError("The DOCX file is corrupted or cannot be opened.") from exc
    description = (
        f"{paragraph_count} text paragraphs · {table_count} tables · "
        f"{media_count} embedded images"
    )
    return (
        _safe_input_name(name, ".docx"),
        UploadInfo(kind="docx", description=description),
    )


def _validate_upload(name: str, content: bytes) -> tuple[str, UploadInfo]:
    if not content:
        raise ValueError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("The document exceeds the 100 MB upload limit.")
    extension = Path(name).suffix.casefold()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Supported formats are PDF, DOCX, JPG, JPEG, and PNG.")
    if extension == ".pdf":
        return _validate_pdf_upload(name, content)
    if extension == ".docx":
        return _validate_docx_upload(name, content)
    return _validate_image_upload(name, content)


def _process_upload(
    content: bytes,
    safe_name: str,
    content_digest: str,
    processing_mode: str,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Process one immutable upload and return serializable output artifacts."""

    del content_digest  # Included in the cache key for explicit provenance.
    with tempfile.TemporaryDirectory(prefix="indic_htr_ui_") as temp_name:
        temp_directory = Path(temp_name)
        input_path = temp_directory / safe_name
        output_directory = temp_directory / "page_ocr_result"
        input_path.write_bytes(content)
        if processing_mode == "openai_direct":
            try:
                return _process_upload_with_openai(
                    content,
                    safe_name,
                    progress_callback=progress_callback,
                )
            except RuntimeError as exc:
                if progress_callback is not None:
                    progress_callback(
                        f"{exc} Falling back to the local document pipeline."
                    )
        if process_document is not None:
            payload = process_document(
                input_path,
                output_directory,
                progress_callback=progress_callback,
                processing_mode=processing_mode,
            )
        elif input_path.suffix.casefold() in {".jpg", ".jpeg", ".png"}:
            payload = process_page(
                input_path,
                output_directory,
                progress_callback=progress_callback,
                processing_mode=(
                    "line_accurate" if processing_mode == "line_accurate" else "fast_cpu"
                ),
            )
        else:
            raise RuntimeError(
                "The fast PDF/DOCX document processor is not available in this installation."
            )
        artifacts = {
            path.relative_to(output_directory).as_posix(): path.read_bytes()
            for path in output_directory.rglob("*")
            if path.is_file()
        }
    return payload, artifacts


def _openai_output_text(payload: dict[str, Any]) -> str:
    text = str(payload.get("output_text") or "").strip()
    if text:
        return text
    fragments: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                fragments.append(content["text"])
    return "\n".join(fragments).strip()


def _extract_json_object(value: str) -> dict[str, Any]:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("OpenAI extraction did not return a JSON object.")
    return parsed


def _fallback_bbox(index: int, total: int, width: int, height: int) -> tuple[int, int, int, int]:
    top_margin = max(8, int(height * 0.05))
    bottom_margin = max(top_margin + 1, int(height * 0.95))
    usable = max(1, bottom_margin - top_margin)
    line_height = max(12, usable // max(1, total))
    y1 = min(height - 2, top_margin + index * line_height)
    y2 = min(height - 1, y1 + max(10, int(line_height * 0.82)))
    return (max(0, int(width * 0.04)), y1, min(width - 1, int(width * 0.96)), y2)


def _preview_image_for_upload(content: bytes, safe_name: str) -> bytes:
    suffix = Path(safe_name).suffix.casefold()
    if suffix in {".jpg", ".jpeg", ".png"}:
        with Image.open(BytesIO(content)) as image:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=88)
            return buffer.getvalue()
    if suffix == ".pdf":
        import fitz

        document = fitz.open(stream=content, filetype="pdf")
        try:
            if len(document) == 0:
                raise ValueError("The PDF has no pages.")
            pixmap = document[0].get_pixmap(dpi=130, alpha=False, colorspace="rgb")
            return pixmap.tobytes("jpg")
        finally:
            document.close()
    image = Image.new("RGB", (1200, 1600), "white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def _direct_openai_content_parts(content: bytes, safe_name: str, prompt: str) -> list[dict[str, str]]:
    suffix = Path(safe_name).suffix.casefold()
    mime = mimetypes.types_map.get(suffix, "application/octet-stream")
    encoded = base64.b64encode(content).decode("ascii")
    parts: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
    if suffix in {".jpg", ".jpeg", ".png"}:
        parts.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime};base64,{encoded}",
            }
        )
    else:
        parts.append(
            {
                "type": "input_file",
                "filename": safe_name,
                "file_data": f"data:{mime};base64,{encoded}",
            }
        )
    return parts


def _openai_http_error_reason(exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8", errors="replace"))
    except Exception:
        return f"OpenAI extraction failed with HTTP {exc.code}."
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return f"OpenAI extraction failed with HTTP {exc.code}."
    code = str(error.get("code") or "").strip()
    message = str(error.get("message") or "").strip()
    if exc.code == 429:
        if code == "insufficient_quota":
            return "OpenAI quota or credits are not available for this API key."
        return "OpenAI rate limit was reached for this request."
    if message:
        return f"OpenAI extraction failed with HTTP {exc.code}: {message}"
    return f"OpenAI extraction failed with HTTP {exc.code}."


def _process_upload_with_openai(
    content: bytes,
    safe_name: str,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    model = os.environ.get("OPENAI_EXTRACTION_MODEL", os.environ.get("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini")).strip()
    if progress_callback is not None:
        progress_callback("OpenAI is extracting lines and translating the document")

    prompt = (
        "Extract every readable line from this Punjabi/Gurmukhi, Hindi/Devanagari, or mixed "
        "document and translate each line into complete English.\n"
        "Return strict JSON only, with this shape: "
        "{\"lines\":[{\"line\":1,\"page_number\":1,\"bbox\":[x1,y1,x2,y2],"
        "\"script\":\"gurmukhi|devanagari|mixed|unknown\",\"text\":\"source line\","
        "\"english\":\"complete English translation\"}],\"notes\":[]}\n"
        "Rules: preserve line order; do not merge separate source lines; preserve names, dates, "
        "times, phone numbers, FIR/GD/case numbers, legal sections, IDs, ranks, abbreviations, "
        "and existing English exactly; translate all readable Punjabi/Hindi content; if OCR is "
        "uncertain, keep uncertain fragments verbatim instead of inventing facts."
    )
    body = json.dumps(
        {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": _direct_openai_content_parts(content, safe_name, prompt),
                }
            ],
            "max_output_tokens": 6000,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    started = time.perf_counter()
    response_payload: dict[str, Any] | None = None
    last_error = "OpenAI extraction request failed."
    for attempt, delay in enumerate((0.0, 6.0), start=1):
        if delay:
            if progress_callback is not None:
                progress_callback("OpenAI rate limit hit; retrying once")
            time.sleep(delay)
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_error = _openai_http_error_reason(exc)
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise RuntimeError(last_error) from None
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last_error = "OpenAI extraction request timed out or was unavailable."
            if attempt == 2:
                raise RuntimeError(last_error) from exc
    if response_payload is None:
        raise RuntimeError(last_error)

    parsed = _extract_json_object(_openai_output_text(response_payload))
    source_image = _preview_image_for_upload(content, safe_name)
    with Image.open(BytesIO(source_image)) as image:
        width, height = image.size
    raw_lines = parsed.get("lines", [])
    if not isinstance(raw_lines, list) or not raw_lines:
        raise RuntimeError("OpenAI did not return any document lines.")

    lines: list[LineResult] = []
    total = len(raw_lines)
    for index, item in enumerate(raw_lines):
        if not isinstance(item, dict):
            continue
        source = normalize_text(str(item.get("text") or ""))
        if not source:
            continue
        script = str(item.get("script") or detect_script(source)).strip().lower()
        if script not in {"gurmukhi", "devanagari", "mixed", "unknown"}:
            script = detect_script(source)
        bbox_value = item.get("bbox")
        if (
            isinstance(bbox_value, list)
            and len(bbox_value) == 4
            and all(isinstance(value, (int, float)) for value in bbox_value)
        ):
            x1, y1, x2, y2 = [int(value) for value in bbox_value]
            bbox = (
                max(0, min(width - 1, x1)),
                max(0, min(height - 1, y1)),
                max(1, min(width, x2)),
                max(1, min(height, y2)),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                bbox = _fallback_bbox(index, total, width, height)
        else:
            bbox = _fallback_bbox(index, total, width, height)
        english_candidate = normalize_text(str(item.get("english") or ""))
        finalized = finalize_translation(source, english_candidate, script=script)  # type: ignore[arg-type]
        line_number = int(item.get("line") or index + 1)
        line = LineResult(
            line=line_number,
            bbox=bbox,
            crop_file="",
            detection_confidence=0.92,
            detector="openai_responses_direct",
            script=script,  # type: ignore[arg-type]
            text=source,
            script_purity=script_purity(source, script),  # type: ignore[arg-type]
            confidence=0.90,
            text_quality=0.90,
            recognition_model=model,
            provider_kind="openai_responses_direct",
            accepted=True,
            review_required=bool(finalized.used_conservative_fallback),
            review_reasons=[],
            english=finalized.english,
            translation_status=(
                "translated_with_warnings"
                if finalized.used_conservative_fallback or finalized.review_reasons
                else "translated"
            ),
            translation_confidence=None,
            translation_quality=finalized.quality,
            translation_review_reasons=list(finalized.review_reasons),
            translation_provider="openai_direct",
            translation_source_language=None,
            translation_attempts=[
                {
                    "provider": "openai_direct",
                    "outcome": "accepted",
                    "validation": "finalized",
                    "reason": None,
                }
            ],
            page_number=int(item.get("page_number") or 1),
        )
        lines.append(line)
    if not lines:
        raise RuntimeError("OpenAI returned line objects, but none contained source text.")

    from pretrained_page_ocr.kaggle_app import _draw_detected_lines, _finalize_lines_for_export

    _finalize_lines_for_export(lines)
    transcription = "\n".join(line.text for line in lines)
    english_translation = "\n".join(line.english for line in lines if line.english)
    detected = _draw_detected_lines(np.asarray(Image.open(BytesIO(source_image)).convert("RGB")), lines)
    detected_buffer = BytesIO()
    Image.fromarray(detected[:, :, ::-1]).save(detected_buffer, format="JPEG", quality=88)
    docx = build_translated_docx_from_image_bytes(source_image, lines)
    text_rows = []
    for line in lines:
        text_rows.append(
            {
                "line": line.line,
                "page_number": line.page_number,
                "script": line.script,
                "text": line.text,
                "english": line.english,
                "translation_status": line.translation_status,
                "translation_provider": line.translation_provider,
            }
        )
    import io

    csv_value = io.StringIO()
    writer = csv.DictWriter(csv_value, fieldnames=list(text_rows[0].keys()))
    writer.writeheader()
    writer.writerows(text_rows)
    payload = {
        "schema_version": "1.2",
        "input_document": safe_name,
        "input_image": safe_name,
        "output_directory": "streamlit_memory",
        "inference_only": True,
        "models": {
            "openai_direct": {
                "id": model,
                "loaded": True,
                "purpose": "line extraction, OCR, and English translation",
            }
        },
        "pages": [{"page_number": 1, "route": "openai_direct", "line_count": len(lines), "width": width, "height": height}],
        "lines": [asdict(line) for line in lines],
        "transcription": transcription,
        "english_translation": english_translation,
        "primary_output": "translated_en.docx",
        "timings_seconds": {"openai_direct": time.perf_counter() - started},
        "accuracy_warning": "OpenAI extraction is automatic and should be reviewed for legal or official use.",
    }
    result_json = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    artifacts = {
        "corrected_page.jpg": source_image,
        "detected_lines.jpg": detected_buffer.getvalue(),
        "transcription.txt": (transcription + "\n").encode("utf-8"),
        "translation_en.txt": (english_translation + "\n").encode("utf-8"),
        "result.json": result_json,
        "result.csv": csv_value.getvalue().encode("utf-8-sig"),
        "translated_en.docx": docx,
    }
    return payload, artifacts


def _download_name(upload_name: str, artifact_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(upload_name).stem).strip("._")
    artifact = Path(artifact_name)
    return f"{stem or 'document'}_{artifact.stem}{artifact.suffix}"


def _result_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in payload.get("lines", []):
        rows.append(
            {
                "Page": line.get("page_number", 1),
                "Line": line.get("line"),
                "Language/script": _script_label(line.get("script")),
                "Recognized text": line.get("text"),
                "OCR confidence": line.get("confidence"),
                "Script purity": line.get("script_purity"),
                "Accepted": line.get("accepted"),
                "English": line.get("english"),
                "Translation confidence": line.get("translation_confidence"),
                "Translation quality": line.get("translation_quality"),
                "Context group": line.get("translation_context_group"),
                "Translation status": line.get("translation_status"),
                "Translation provider": line.get("translation_provider"),
                "Review reason": "; ".join(
                    [
                        *line.get("review_reasons", []),
                        *line.get("translation_review_reasons", []),
                    ]
                ),
            }
        )
    return rows


def _script_label(script: str | None) -> str:
    """Use language-first wording without losing the Unicode-script detail."""

    return {
        "gurmukhi": "Punjabi (Gurmukhi)",
        "devanagari": "Hindi (Devanagari)",
        "mixed": "Mixed Punjabi/Hindi",
        "unknown": "Unknown",
    }.get(str(script), "Unknown")


def _line_from_payload(item: dict[str, Any]) -> LineResult:
    """Restore one auditable line without trusting browser-provided metadata."""

    bbox = tuple(int(value) for value in item.get("bbox", (0, 0, 1, 1)))
    if len(bbox) != 4:
        bbox = (0, 0, 1, 1)
    return LineResult(
        line=int(item.get("line", 0)),
        bbox=bbox,
        crop_file=str(item.get("crop_file", "")),
        detection_confidence=item.get("detection_confidence"),
        detector=str(item.get("detector", "review")),
        script=str(item.get("script", "unknown")),  # validated after editing
        text=str(item.get("text", "")),
        script_purity=float(item.get("script_purity") or 0.0),
        confidence=item.get("confidence"),
        text_quality=float(item.get("text_quality") or 0.0),
        recognition_model=str(item.get("recognition_model", "review")),
        provider_kind=str(item.get("provider_kind", "review")),
        accepted=bool(item.get("accepted")),
        review_required=bool(item.get("review_required")),
        review_reasons=list(item.get("review_reasons") or []),
        english=str(item.get("english") or ""),
        translation_status=str(item.get("translation_status") or "not_attempted"),
        translation_error=item.get("translation_error"),
        translation_confidence=item.get("translation_confidence"),
        translation_quality=float(item.get("translation_quality") or 0.0),
        translation_review_reasons=list(item.get("translation_review_reasons") or []),
        translation_context_group=item.get("translation_context_group"),
        translation_provider=item.get("translation_provider"),
        translation_source_language=item.get("translation_source_language"),
        translation_attempts=list(item.get("translation_attempts") or []),
        candidates=list(item.get("candidates") or []),
        page_number=int(item.get("page_number", 1)),
    )


def _review_editor_rows(lines: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Key": f"{item.get('page_number', 1)}:{item.get('line', 0)}",
                "Page": item.get("page_number", 1),
                "Line": item.get("line"),
                "Language": _script_label(item.get("script")),
                "Punjabi/Hindi source (editable)": item.get("text", ""),
                "Confirm source and translate": bool(item.get("accepted")),
                "English override (optional)": item.get("english", ""),
                "OCR confidence": item.get("confidence"),
                "Reason": "; ".join(item.get("review_reasons") or []),
            }
            for item in lines
        ]
    )


def _apply_review_edits(
    payload: dict[str, Any], edited: pd.DataFrame, corrected_page: bytes
) -> tuple[dict[str, Any], bytes, int]:
    """Apply explicit source corrections and translate only confirmed Indic text."""

    original = {_key: item for _key, item in (
        (f"{line.get('page_number', 1)}:{line.get('line', 0)}", line)
        for line in payload.get("lines", [])
    )}
    restored: list[LineResult] = []
    pending: list[LineResult] = []
    for row in edited.to_dict("records"):
        item = original.get(str(row.get("Key")))
        if item is None:
            continue
        line = _line_from_payload(item)
        source = normalize_text(str(row.get("Punjabi/Hindi source (editable)") or ""))
        confirmed = bool(row.get("Confirm source and translate"))
        english_override = normalize_text(str(row.get("English override (optional)") or ""))
        if confirmed:
            script = detect_script(source)
            if script not in {"gurmukhi", "devanagari"}:
                line.accepted = False
                line.review_required = True
                line.review_reasons = [
                    "confirmed source must contain Punjabi (Gurmukhi) or Hindi (Devanagari) text"
                ]
                line.translation_status = "not_attempted"
                line.english = ""
            else:
                line.text = source
                line.script = script
                line.script_purity = script_purity(source, script)
                line.text_quality = calculate_text_quality(source, script)
                line.confidence = 1.0
                line.recognition_model = "manual_source_confirmation"
                line.provider_kind = "manual_review"
                line.accepted = True
                line.review_required = False
                line.review_reasons = []
                if english_override:
                    line.english = english_override
                    line.translation_status = "manual_translation"
                    line.translation_confidence = 1.0
                    line.translation_quality = 1.0
                    line.translation_review_reasons = []
                elif line.translation_status not in {
                    "translated",
                    "translated_low_confidence",
                    "translated_with_warnings",
                    "manual_translation",
                } or source != normalize_text(str(item.get("text", ""))):
                    line.english = ""
                    line.translation_status = "not_attempted"
                    line.translation_error = None
                    line.translation_review_reasons = []
                    pending.append(line)
        restored.append(line)

    if pending:
        translate_text(pending, ModelRegistry(), num_beams=4)
    updated = dict(payload)
    updated["lines"] = [
        {
            **asdict(item),
            "bbox": list(item.bbox),
        }
        for item in restored
    ]
    updated["transcription"] = "\n".join(
        line.text if line.accepted else f"[REVIEW REQUIRED - line {line.line}] {line.text}".rstrip()
        for line in restored
    )
    updated["english_translation"] = "\n".join(
        line.english
        for line in restored
        if line.translation_status in {
            "translated",
            "translated_low_confidence",
            "translated_with_warnings",
            "manual_translation",
            "preserved_source_with_warnings",
        } and line.english
    )
    updated["reviewed_in_browser"] = True
    return updated, build_translated_docx_from_image_bytes(corrected_page, restored), len(pending)


def _runtime_dependency_error() -> str | None:
    try:
        version("PyMuPDF")
    except PackageNotFoundError:
        return "PyMuPDF is not installed; PDF documents cannot be opened."
    try:
        version("python-docx")
    except PackageNotFoundError:
        return "python-docx is not installed; the translated Word document cannot be created."
    try:
        Version(version("torchvision"))
    except (PackageNotFoundError, ValueError):
        return "Torchvision is not installed; the pretrained OCR vision models cannot load."
    try:
        installed = Version(version("transformers"))
    except (PackageNotFoundError, ValueError):
        return "Transformers is not installed."
    required = Version("5.12.1")
    if installed < required:
        return f"Transformers {installed} is installed, but this app requires {required} or newer."
    return None


st.set_page_config(
    page_title="Hindi and Gurmukhi document translator",
    page_icon="📝",
    layout="wide",
)

st.session_state.setdefault("pretrained_ocr_result", None)
st.session_state.setdefault("pretrained_ocr_digest", None)

try:
    secret_key = str(st.secrets.get("OPENAI_API_KEY", "") or "").strip()
    secret_model = str(st.secrets.get("OPENAI_TRANSLATION_MODEL", "") or "").strip()
    if secret_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = secret_key
    if secret_model and not os.environ.get("OPENAI_TRANSLATION_MODEL"):
        os.environ["OPENAI_TRANSLATION_MODEL"] = secret_model
except Exception:
    pass

st.title("Hindi and Gurmukhi document translator")
st.caption(
    "Upload a PDF, Word document, scan, or photograph. Native PDF/DOCX text is "
    "extracted directly; OCR runs only where a page is actually scanned."
)
st.caption(
    "Punjabi and Hindi translation now prioritizes OpenAI when `OPENAI_API_KEY` is set. "
    "The OpenAI fast mode extracts text and translates in one pass; Azure and NLLB remain fallbacks for local OCR mode."
)
try:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        st.warning(
            "OpenAI translation is not active. Set `OPENAI_API_KEY` in PowerShell or "
            "Streamlit secrets before running the app.",
            icon=":material/key:",
        )
except Exception:
    pass
st.info(
    "Fast document mode avoids handwriting models when native text is available. "
    "Uncertain text is preserved for review rather than guessed.",
    icon=":material/info:",
)

cuda_available = False
try:
    import torch

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        st.warning(
            "CUDA is not available. Native PDF/DOCX extraction is fast on CPU. "
            "Choose handwriting recognition only for genuinely handwritten sources.",
            icon=":material/speed:",
        )
except ImportError:
    pass

dependency_error = _runtime_dependency_error()
if dependency_error:
    st.error(dependency_error, icon=":material/package_2:")
    st.code(
        r".\.venv\Scripts\python.exe -m pip install -r "
        r".\pretrained_page_ocr\requirements-ui-windows.txt",
        language="powershell",
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "jpg", "jpeg", "png"],
    max_upload_size=100,
    help="Supported formats: PDF, DOCX, JPG, JPEG, and PNG. Maximum size: 100 MB.",
    key="pretrained_page_upload",
)

if uploaded_file is None:
    st.session_state.pretrained_ocr_result = None
    st.session_state.pretrained_ocr_digest = None
    st.stop()

upload_bytes = uploaded_file.getvalue()
try:
    safe_filename, upload_info = _validate_upload(uploaded_file.name, upload_bytes)
except ValueError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

digest = hashlib.sha256(b"openai-translation-v1:" + upload_bytes).hexdigest()

with st.container(border=True):
    st.markdown(f"**{uploaded_file.name}**")
    st.caption(f"{upload_info.kind.upper()} · {upload_info.description}")
    if upload_info.preview is not None:
        st.image(upload_info.preview, caption="Source preview")

with st.form("recognize_page", border=False):
    processing_mode = st.segmented_control(
        "Processing mode",
        options=["openai_direct", "fast_document", "line_accurate"],
        default=(
            "openai_direct"
            if os.environ.get("OPENAI_API_KEY", "").strip()
            else "fast_document"
        ),
        format_func=lambda value: {
            "openai_direct": "OpenAI fast extraction",
            "fast_document": "Printed Punjabi/Hindi document",
            "line_accurate": "Hybrid handwriting (optional)",
        }[value],
        help=(
            "OpenAI fast extraction sends the uploaded document to OpenAI for line OCR "
            "and translation in one request. Printed Punjabi/Hindi document is the local "
            "mode for forms, police "
            "reports, and scanned typed pages. It automatically selects Punjabi `pan` "
            "or Hindi `hin` OCR from page-script evidence and leaves English headings "
            "unchanged. Hybrid handwriting is slower and only for handwritten pages."
        ),
        width="stretch",
    )
    submitted = st.form_submit_button(
        "Transcribe and translate document",
        type="primary",
        icon=":material/translate:",
        width="stretch",
    )

request_identity = f"{digest}:{processing_mode}"
if st.session_state.pretrained_ocr_digest != request_identity:
    st.session_state.pretrained_ocr_result = None
    st.session_state.pretrained_ocr_digest = request_identity

if submitted:
    status = st.status("Preparing the page and loading pretrained models…", expanded=True)
    try:
        def report_progress(message: str) -> None:
            status.update(label=message, state="running", expanded=True)
            status.write(message)

        report_progress("Extracting document text and OCRing only the pages that need it")
        payload, artifacts = _process_upload(
            upload_bytes,
            safe_filename,
            digest,
            processing_mode or "fast_document",
            progress_callback=report_progress,
        )
        st.session_state.pretrained_ocr_result = {
            "payload": payload,
            "artifacts": artifacts,
            "upload_name": uploaded_file.name,
            "identity": request_identity,
        }
        status.update(label="Recognition and translation complete", state="complete", expanded=False)
    except Exception as exc:
        status.update(label="Processing failed", state="error", expanded=True)
        st.error(
            f"Processing could not complete: {type(exc).__name__}: {exc}",
            icon=":material/error:",
        )

result = st.session_state.pretrained_ocr_result
if result is None:
    st.stop()

payload = result["payload"]
artifacts = result["artifacts"]
lines = payload.get("lines", [])
accepted_count = sum(bool(line.get("accepted")) for line in lines)
translated_count = sum(bool(str(line.get("english") or "").strip()) for line in lines)
review_count = sum(bool(line.get("review_required")) for line in lines)
translated_quality = [
    float(line.get("translation_quality") or 0.0)
    for line in lines
    if line.get("translation_status")
    in {"translated", "translated_low_confidence", "translated_with_warnings"}
]
script_counts = Counter(str(line.get("script", "unknown")) for line in lines)
provider_counts = Counter(
    str(line.get("translation_provider"))
    for line in lines
    if line.get("translation_provider")
)
residual_gurmukhi = sum(
    len(meaningful_residual_script_words(
        str(line.get("text") or ""), str(line.get("english") or ""),
        "gurmukhi", protect_translation_source(str(line.get("text") or "")),
    )) > 2
    for line in lines
)
residual_devanagari = sum(
    len(meaningful_residual_script_words(
        str(line.get("text") or ""), str(line.get("english") or ""),
        "devanagari", protect_translation_source(str(line.get("text") or "")),
    )) > 2
    for line in lines
)

st.subheader("Results")
with st.container(horizontal=True):
    st.metric("Detected lines", len(lines))
    st.metric("Accepted lines", accepted_count)
    st.metric("English output lines", translated_count)
    st.metric("Safety warnings", review_count)
    st.metric(
        "Translation QA",
        f"{sum(translated_quality) / len(translated_quality):.0%}"
        if translated_quality
        else "—",
    )

st.caption(
    "Translation QA checks English output shape, repetition, and exact preservation of "
    "dates, identifiers, numbers, and existing English. It is not a certified accuracy "
    "percentage. English output is always exported automatically; safety warnings only identify "
    "conservative source-preserving fallbacks or uncertain OCR in the optional audit."
)

with st.expander("Translation providers and language coverage", expanded=False):
    with st.container(horizontal=True):
        st.metric("Punjabi lines", script_counts["gurmukhi"])
        st.metric("Hindi lines", script_counts["devanagari"])
        st.metric("Mixed-script lines", script_counts["mixed"])
        st.metric(
            "OpenAI translations",
            provider_counts["openai"] + provider_counts["openai_direct"],
        )
        st.metric("Azure rescue", provider_counts["azure"])
        st.metric("NLLB rescue", provider_counts["nllb"])
        st.metric("Conservative fallbacks", sum(
            line.get("translation_status") == "preserved_source_with_warnings" for line in lines
        ))
    with st.container(horizontal=True):
        st.metric("Remaining Gurmukhi", residual_gurmukhi)
        st.metric("Remaining Devanagari", residual_devanagari)
        st.metric("Safety warnings", review_count)

# Do not make users infer why a line is held back from the four headline
# counters.  A source line can be accepted while its English output is held for
# a concrete safety reason (for example, protected identifiers were changed).
# Showing the aggregate reasons makes the next correction actionable without
# exposing a noisy error for every row.
review_reason_counts: Counter[str] = Counter()
for line in lines:
    if not line.get("review_required"):
        continue
    reasons = [
        *[str(reason) for reason in line.get("review_reasons", []) if reason],
        *[
            str(reason)
            for reason in line.get("translation_review_reasons", [])
            if reason
        ],
    ]
    if not reasons and line.get("translation_error"):
        reasons.append(str(line["translation_error"]))
    for reason in dict.fromkeys(reasons):
        review_reason_counts[reason] += 1

if review_reason_counts:
    with st.expander("Automatic safety notes (no action needed)", expanded=False):
        st.caption(
            "These notes never block the English output or download. They identify OCR or "
            "translation uncertainty retained in the audit metadata."
        )
        st.dataframe(
            [
                {"Reason": reason, "Lines": count}
                for reason, count in review_reason_counts.most_common()
            ],
            hide_index=True,
            width="stretch",
        )

if "detected_lines.jpg" in artifacts:
    with st.container(border=True):
        st.image(artifacts["detected_lines.jpg"], caption="Detected lines: green is high-confidence; orange has an automatic safety warning")

st.text_area(
    "Punjabi (Gurmukhi)/Hindi transcription",
    value=payload.get("transcription", ""),
    height=220,
    disabled=True,
    key=f"transcription_{result.get('identity', digest)}",
)
st.text_area(
    "English translation",
    value=payload.get("english_translation", ""),
    height=220,
    disabled=True,
    key=f"translation_{result.get('identity', digest)}",
)

rows = _result_rows(payload)
show_optional_audit = st.toggle(
    "Show optional audit and manual corrections",
    value=False,
    help="Not required for automatic translation or download.",
    key=f"optional_audit_{result.get('identity', digest)}",
)
if rows and show_optional_audit:
    st.subheader("Optional OCR audit")
    st.dataframe(rows, hide_index=True)

    st.subheader("Optional manual correction")
    st.caption(
        "Punjabi is written in Gurmukhi. Edit the source text exactly as it appears, "
        "tick confirmation, then submit. Confirmed Punjabi/Hindi lines are translated "
        "even when automatic OCR confidence was low; invalid Latin-only text is not sent "
        "to translation. You may also enter a reviewed English translation directly."
    )
    with st.form("review_and_translate", border=True):
        edited_rows = st.data_editor(
            _review_editor_rows(lines),
            key=f"line_editor_{result.get('identity', digest)}",
            hide_index=True,
            num_rows="fixed",
            disabled=["Key", "Page", "Line", "Language", "OCR confidence", "Reason"],
            column_config={
                "Punjabi/Hindi source (editable)": st.column_config.TextColumn(
                    "Punjabi/Hindi source (editable)", width="large"
                ),
                "Confirm source and translate": st.column_config.CheckboxColumn(
                    "Confirm source and translate",
                    help="Explicit confirmation allows a low-confidence source line to be translated.",
                ),
                "English override (optional)": st.column_config.TextColumn(
                    "English override (optional)", width="large"
                ),
                "OCR confidence": st.column_config.NumberColumn("OCR confidence", format="%.0f%%"),
            },
        )
        review_submitted = st.form_submit_button(
            "Apply corrections and create revised Word document",
            type="primary",
            icon=":material/translate:",
            width="stretch",
        )
    if review_submitted:
        corrected_page = artifacts.get("corrected_page.jpg")
        if corrected_page is None:
            st.error("The corrected page image is unavailable, so the Word document cannot be rebuilt.")
        else:
            with st.spinner("Translating confirmed corrections and rebuilding the Word document…"):
                try:
                    updated_payload, reviewed_docx, pending_count = _apply_review_edits(
                        payload, edited_rows, corrected_page
                    )
                    result["payload"] = updated_payload
                    result["artifacts"]["translated_en_reviewed.docx"] = reviewed_docx
                    st.session_state.pretrained_ocr_result = result
                    st.success(
                        f"Applied corrections. Translated {pending_count} newly confirmed source line(s).",
                        icon=":material/check_circle:",
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(
                        f"Could not apply review corrections: {type(exc).__name__}: {exc}",
                        icon=":material/error:",
                    )

st.subheader("Translated Word document")
word_artifact = "translated_en.docx"
if "translated_en_reviewed.docx" in artifacts:
    word_artifact = "translated_en_reviewed.docx"
word_content = artifacts.get(word_artifact)
if word_content is not None:
    st.download_button(
        label="Download translated Word document",
        data=word_content,
        file_name=_download_name(result["upload_name"], word_artifact),
        mime=DOWNLOAD_MIME_TYPES[".docx"],
        icon=":material/download:",
        type="primary",
        width="stretch",
        on_click="ignore",
        key=f"download_{result.get('identity', digest)}_{word_artifact}",
    )
    st.caption(
        f"{translated_count} of {len(lines)} detected lines translated. "
        "Every other detected line is retained as an explicit review entry; no text is invented."
    )
else:
    st.error("The translated Word artifact was not generated.", icon=":material/error:")

with st.expander("Additional downloads"):
    preferred_order = [
        "translation_en.txt",
        "transcription.txt",
        "result.json",
        "result.csv",
        "detected_lines.jpg",
        "corrected_page.jpg",
    ]
    for artifact_name in preferred_order:
        content = artifacts.get(artifact_name)
        if content is None:
            continue
        suffix = Path(artifact_name).suffix.casefold()
        st.download_button(
            label=f"Download {artifact_name}",
            data=content,
            file_name=_download_name(result["upload_name"], artifact_name),
            mime=DOWNLOAD_MIME_TYPES.get(suffix, "application/octet-stream"),
            icon=":material/download:",
            on_click="ignore",
            key=f"download_{result.get('identity', digest)}_{artifact_name}",
        )

with st.expander("Model and timing details"):
    st.json(
        {
            "models": payload.get("models", {}),
            "timings_seconds": payload.get("timings_seconds", {}),
            "preprocessing_operations": payload.get("preprocessing_operations", []),
            "accuracy_warning": payload.get("accuracy_warning"),
        },
        expanded=False,
    )
