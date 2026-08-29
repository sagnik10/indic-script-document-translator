"""Fast document routing for PDF, DOCX, and image inputs.

Native PDF text is extracted directly with PyMuPDF. Image-only pages are
rendered at a modest DPI and recognized with the installed Punjabi, Hindi, and
English Tesseract packs. In hybrid mode, only unresolved logical scan lines use
the large handwriting recognizers. Only script-consistent Punjabi/Hindi lines
enter the existing NLLB translation gate.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import cv2
import numpy as np
from PIL import Image

from pretrained_page_ocr.kaggle_app import (
    BBox,
    EnhancedPage,
    LineResult,
    ModelRegistry,
    RecognitionCandidate,
    _candidate_for_audit,
    _draw_detected_lines,
    _line_transcription,
    _line_translation,
    _opencv_detect,
    _prepare_tesseract,
    _tesseract_candidate,
    build_translated_docx,
    choose_best_recognition,
    correct_document,
    detect_script,
    enhance_page,
    load_models,
    normalize_text,
    recognize_gurmukhi,
    recognize_hindi,
    script_statistics,
    save_results,
    translate_text,
)


DocumentProcessingMode = Literal["fast_document", "fast_cpu", "line_accurate"]
ProgressCallback = Callable[[str], None]

PDF_RENDER_DPI = 150
PDF_PREVIEW_DPI = 110
MIN_NATIVE_PAGE_LETTERS = 18


@dataclass(slots=True)
class _ExtractedLine:
    text: str
    bbox: BBox
    confidence: float | None
    provider: str


@dataclass(slots=True)
class _ProcessedPdfPage:
    page_number: int
    page: EnhancedPage
    lines: list[LineResult]
    route: str


def _report(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _pixmap_to_rgb(pixmap: Any) -> np.ndarray:
    channels = int(pixmap.n)
    array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, channels
    )
    if channels == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    if channels == 1:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    return np.ascontiguousarray(array[:, :, :3])


def _render_pdf_page(page: Any, dpi: int) -> np.ndarray:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False, colorspace="rgb")
    return _pixmap_to_rgb(pixmap)


def _enhanced_pdf_page(rgb: np.ndarray, operation: str) -> EnhancedPage:
    """Keep the source render and use only mild grayscale enhancement for OCR."""

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
    enhanced_rgb = cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)
    threshold = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        13,
    )
    return EnhancedPage(
        corrected_rgb=rgb,
        enhanced_rgb=enhanced_rgb,
        enhanced_gray=clahe,
        threshold=threshold,
        operations=[operation, "mild_clahe"],
    )


def _clip_bbox(bbox: Sequence[float], width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, int(math.floor(x1)))),
        max(0, min(height - 1, int(math.floor(y1)))),
        max(1, min(width, int(math.ceil(x2)))),
        max(1, min(height, int(math.ceil(y2)))),
    )


def _native_pdf_lines(page: Any, rendered_width: int, rendered_height: int) -> list[_ExtractedLine]:
    """Extract native text lines and map their boxes to the rendered page."""

    document = page.get_text("dict", sort=True)
    page_width = max(1.0, float(page.rect.width))
    page_height = max(1.0, float(page.rect.height))
    scale_x = rendered_width / page_width
    scale_y = rendered_height / page_height
    lines: list[_ExtractedLine] = []
    for block in document.get("blocks", []):
        if int(block.get("type", 0)) != 0:
            continue
        for raw_line in block.get("lines", []):
            text = normalize_text(
                "".join(str(span.get("text", "")) for span in raw_line.get("spans", []))
            )
            if not text:
                continue
            raw_bbox = raw_line.get("bbox")
            if not raw_bbox:
                continue
            scaled = (
                float(raw_bbox[0]) * scale_x,
                float(raw_bbox[1]) * scale_y,
                float(raw_bbox[2]) * scale_x,
                float(raw_bbox[3]) * scale_y,
            )
            lines.append(
                _ExtractedLine(
                    text=text,
                    bbox=_clip_bbox(scaled, rendered_width, rendered_height),
                    confidence=0.99,
                    provider="pymupdf_native_text",
                )
            )
    return lines


def _native_page_is_usable(lines: Sequence[_ExtractedLine]) -> bool:
    letters = sum(int(script_statistics(line.text)["letters"]) for line in lines)
    return letters >= MIN_NATIVE_PAGE_LETTERS or (len(lines) >= 3 and letters >= 10)


def _pdf_image_coverage(page: Any) -> float:
    """Estimate how much of a PDF page is occupied by embedded raster images."""

    page_area = max(1.0, float(page.rect.width * page.rect.height))
    covered = 0.0
    seen: set[tuple[float, float, float, float]] = set()
    try:
        for image in page.get_images(full=True):
            for rectangle in page.get_image_rects(image[0]):
                key = tuple(round(float(value), 1) for value in rectangle)
                if key in seen:
                    continue
                seen.add(key)
                covered += max(0.0, float(rectangle.width * rectangle.height))
    except Exception:
        return 0.0
    return min(1.0, covered / page_area)


def _group_tesseract_lines(data: dict[str, list[Any]], width: int, height: int) -> list[_ExtractedLine]:
    groups: dict[tuple[int, int, int], dict[str, Any]] = {}
    count = len(data.get("text", []))
    for index in range(count):
        text = normalize_text(str(data["text"][index]))
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        left = int(data["left"][index])
        top = int(data["top"][index])
        word_width = int(data["width"][index])
        word_height = int(data["height"][index])
        key = (
            int(data.get("block_num", [0] * count)[index]),
            int(data.get("par_num", [0] * count)[index]),
            int(data.get("line_num", list(range(count)))[index]),
        )
        group = groups.setdefault(
            key,
            {
                "tokens": [],
                "confidences": [],
                "x1": left,
                "y1": top,
                "x2": left + word_width,
                "y2": top + word_height,
            },
        )
        group["tokens"].append(text)
        if confidence >= 0:
            group["confidences"].append(confidence / 100.0)
        group["x1"] = min(group["x1"], left)
        group["y1"] = min(group["y1"], top)
        group["x2"] = max(group["x2"], left + word_width)
        group["y2"] = max(group["y2"], top + word_height)

    output: list[_ExtractedLine] = []
    for group in groups.values():
        text = normalize_text(" ".join(group["tokens"]))
        if not text:
            continue
        confidences = group["confidences"]
        confidence = sum(confidences) / len(confidences) if confidences else None
        output.append(
            _ExtractedLine(
                text=text,
                bbox=_clip_bbox(
                    (group["x1"], group["y1"], group["x2"], group["y2"]),
                    width,
                    height,
                ),
                confidence=confidence,
                provider="tesseract_pdf_fast",
            )
        )
    return sorted(output, key=lambda line: (line.bbox[1], line.bbox[0]))


def _tesseract_pdf_lines(page: EnhancedPage) -> list[_ExtractedLine]:
    """Recognize a scanned page once with installed Indic/English packs."""

    pytesseract, _config = _prepare_tesseract(("pan", "hin", "eng"))
    data = pytesseract.image_to_data(
        page.enhanced_rgb,
        lang="pan+hin+eng",
        config="--oem 1 --psm 3 --dpi 150 -c preserve_interword_spaces=1",
        output_type=pytesseract.Output.DICT,
    )
    height, width = page.enhanced_gray.shape
    return _group_tesseract_lines(data, width, height)


def _overlap_fraction(first: BBox, second: BBox) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / min(first_area, second_area)


def _supplement_scanned_regions(
    page: EnhancedPage,
    extracted: Sequence[_ExtractedLine],
) -> list[_ExtractedLine]:
    """Add logical OpenCV lines that Tesseract detected spatially but could not read."""

    output = list(extracted)
    for region in _opencv_detect(page):
        if any(_overlap_fraction(region.bbox, item.bbox) >= 0.45 for item in output):
            continue
        output.append(
            _ExtractedLine(
                text="",
                bbox=region.bbox,
                confidence=None,
                provider="opencv_unread_scanned_line",
            )
        )
    return sorted(output, key=lambda line: (line.bbox[1], line.bbox[0]))


def _apply_candidate_to_line(
    line: LineResult,
    candidates: Sequence[RecognitionCandidate],
) -> bool:
    winner, accepted, reasons = choose_best_recognition(candidates)
    line.script = winner.detected_script
    line.text = winner.text
    line.script_purity = winner.script_purity
    line.confidence = winner.confidence
    line.text_quality = winner.text_quality
    line.recognition_model = winner.model_id
    line.provider_kind = winner.provider_kind
    line.accepted = accepted
    line.review_required = not accepted
    line.review_reasons = reasons
    line.candidates = [
        _candidate_for_audit(candidate) for candidate in candidates
    ]
    return accepted


def _candidate_from_line(line: LineResult) -> RecognitionCandidate | None:
    if not line.text or line.script not in {"gurmukhi", "devanagari"}:
        return None
    return RecognitionCandidate(
        expected_script=line.script,
        text=line.text,
        confidence=line.confidence,
        model_id=line.recognition_model,
        provider_kind=line.provider_kind,
    )


def _line_crop(page: EnhancedPage, bbox: BBox, padding: int = 8) -> Image.Image:
    height, width = page.enhanced_gray.shape
    x1, y1, x2, y2 = bbox
    return Image.fromarray(page.enhanced_rgb).crop(
        (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(width, x2 + padding),
            min(height, y2 + padding),
        )
    ).convert("RGB")


def _threshold_line_crop(
    page: EnhancedPage,
    bbox: BBox,
    padding: int = 8,
) -> Image.Image:
    """Return a second, coordinate-identical OCR view for faint/noisy lines."""

    height, width = page.threshold.shape
    x1, y1, x2, y2 = bbox
    crop = page.threshold[
        max(0, y1 - padding) : min(height, y2 + padding),
        max(0, x1 - padding) : min(width, x2 + padding),
    ]
    return Image.fromarray(crop).convert("RGB")


def _refine_with_script_tesseract(
    page: EnhancedPage,
    lines: Sequence[LineResult],
) -> None:
    """Use line-specific OCR packs only for unresolved scan lines."""

    accepted_scripts = [
        line.script for line in lines if line.accepted and line.script in {"gurmukhi", "devanagari"}
    ]
    dominant = (
        max(set(accepted_scripts), key=accepted_scripts.count)
        if accepted_scripts
        else None
    )
    for line in lines:
        if not line.review_required:
            continue
        crop = _line_crop(page, line.bbox)
        threshold_crop = _threshold_line_crop(page, line.bbox)
        routes = [dominant] if dominant else ["gurmukhi", "devanagari"]
        current = _candidate_from_line(line)
        candidates: list[RecognitionCandidate] = [current] if current else []
        for route in routes:
            if route == "gurmukhi":
                candidates.append(_tesseract_candidate(crop, "pan", "gurmukhi"))
            elif route == "devanagari":
                candidates.append(_tesseract_candidate(crop, "hin", "devanagari"))
        if not candidates:
            continue
        # Only spend a second Tesseract pass when the handwriting-safe gray
        # candidate still fails. The thresholded page is an OCR alternative,
        # never the authoritative source image.
        if _apply_candidate_to_line(line, candidates):
            continue
        for route in routes:
            if route == "gurmukhi":
                candidates.append(
                    _tesseract_candidate(threshold_crop, "pan", "gurmukhi")
                )
            elif route == "devanagari":
                candidates.append(
                    _tesseract_candidate(threshold_crop, "hin", "devanagari")
                )
        _apply_candidate_to_line(line, candidates)


def _refine_with_handwriting_models(
    page: EnhancedPage,
    lines: Sequence[LineResult],
    models: ModelRegistry,
    progress_callback: ProgressCallback | None,
) -> None:
    """Run neural HTR only for scan lines still unresolved after Tesseract."""

    target_indices = [index for index, line in enumerate(lines) if line.review_required]
    if not target_indices:
        return
    crops = [_line_crop(page, lines[index].bbox, padding=12) for index in target_indices]
    previous_candidates = [
        [candidate] if (candidate := _candidate_from_line(lines[index])) else []
        for index in target_indices
    ]
    _report(
        progress_callback,
        f"Handwriting refinement for {len(crops)} unresolved scanned lines",
    )
    gurmukhi = recognize_gurmukhi(crops, models, progress_callback, fast=False)
    unresolved: list[int] = []
    for local_index, candidate in enumerate(gurmukhi):
        line_index = target_indices[local_index]
        candidates = [*previous_candidates[local_index], candidate]
        if not _apply_candidate_to_line(lines[line_index], candidates):
            unresolved.append(local_index)
    if unresolved:
        hindi = recognize_hindi(
            [crops[index] for index in unresolved],
            models,
            progress_callback,
        )
        for local_index, candidate in zip(unresolved, hindi, strict=True):
            line_index = target_indices[local_index]
            prior = gurmukhi[local_index]
            _apply_candidate_to_line(
                lines[line_index],
                [*previous_candidates[local_index], prior, candidate],
            )
    models.release_recognizers()


def _meaningful_latin(text: str, confidence: float | None) -> bool:
    stats = script_statistics(text)
    tokens = re.findall(r"[A-Za-z]+", text)
    return bool(
        int(stats["letters"]) >= 3
        and float(stats["latin_ratio"]) >= 0.82
        and any(len(token) >= 2 for token in tokens)
        and (confidence is None or confidence >= 0.38)
    )


def _line_result(
    extracted: _ExtractedLine,
    *,
    page_number: int,
    line_number: int,
    crop_file: str,
) -> LineResult:
    text = normalize_text(extracted.text)
    script = detect_script(text)
    if script in {"gurmukhi", "devanagari"}:
        candidate = RecognitionCandidate(
            expected_script=script,
            text=text,
            confidence=extracted.confidence,
            model_id=("pymupdf:native-text" if extracted.provider == "pymupdf_native_text" else "tesseract:pan+hin+eng"),
            provider_kind=extracted.provider,
        )
        winner, accepted, reasons = choose_best_recognition([candidate])
        if extracted.provider == "pymupdf_native_text":
            stats = script_statistics(winner.text)
            source_letters = int(stats[script])
            # Native selectable Unicode is not OCR. Short headings and labels
            # should not inherit OCR-only confidence/length rejection rules.
            if (
                source_letters >= 1
                and winner.detected_script == script
                and winner.script_purity >= 0.90
            ):
                accepted = True
                reasons = []
        return LineResult(
            line=line_number,
            bbox=extracted.bbox,
            crop_file=crop_file,
            detection_confidence=extracted.confidence,
            detector=extracted.provider,
            script=winner.detected_script,
            text=winner.text,
            script_purity=winner.script_purity,
            confidence=winner.confidence,
            text_quality=winner.text_quality,
            recognition_model=winner.model_id,
            provider_kind=winner.provider_kind,
            accepted=accepted,
            review_required=not accepted,
            review_reasons=reasons,
            candidates=[_candidate_for_audit(winner)],
            page_number=page_number,
        )

    stats = script_statistics(text)
    structured = bool(text and not int(stats["letters"]) and any(char.isalnum() for char in text))
    preserved_english = _meaningful_latin(text, extracted.confidence)
    if extracted.provider == "pymupdf_native_text" and text:
        # Selectable PDF text is exact source content, not an OCR hypothesis.
        # Preserve short headings, initials, and labels without review noise.
        preserved_english = bool(int(stats["letters"]))
        structured = not preserved_english
        accepted = True
    else:
        accepted = preserved_english or structured
    reasons: list[str] = [] if accepted else ["no reliable Punjabi, Hindi, or English text"]
    return LineResult(
        line=line_number,
        bbox=extracted.bbox,
        crop_file=crop_file,
        detection_confidence=extracted.confidence,
        detector=extracted.provider,
        script="unknown",
        text=text,
        script_purity=float(stats["latin_ratio"]) if preserved_english else 0.0,
        confidence=extracted.confidence,
        text_quality=(
            min(1.0, 0.65 * float(stats["latin_ratio"]) + 0.35 * (extracted.confidence or 0.5))
            if preserved_english
            else 0.5 if structured else 0.0
        ),
        recognition_model=("pymupdf:native-text" if extracted.provider == "pymupdf_native_text" else "tesseract:pan+hin+eng"),
        provider_kind=extracted.provider,
        accepted=accepted,
        review_required=not accepted,
        review_reasons=reasons,
        english=text if accepted else "",
        translation_status=("preserved_english" if preserved_english else "preserved_structured" if structured else "not_attempted"),
        page_number=page_number,
    )


def _write_pdf_artifacts(
    *,
    input_path: Path,
    output_directory: Path,
    pages: Sequence[_ProcessedPdfPage],
    models: ModelRegistry,
    timings: dict[str, float],
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    corrected_directory = output_directory / "corrected_pages"
    detected_directory = output_directory / "detected_pages"
    corrected_directory.mkdir(exist_ok=True)
    detected_directory.mkdir(exist_ok=True)

    all_lines = [line for page in pages for line in page.lines]
    page_payload: list[dict[str, Any]] = []
    for processed in pages:
        page_name = f"page_{processed.page_number:03d}.jpg"
        corrected = cv2.cvtColor(processed.page.corrected_rgb, cv2.COLOR_RGB2BGR)
        detected = _draw_detected_lines(processed.page.corrected_rgb, processed.lines)
        corrected_path = corrected_directory / page_name
        detected_path = detected_directory / page_name
        if not cv2.imwrite(str(corrected_path), corrected, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise OSError(f"Could not write {corrected_path}")
        if not cv2.imwrite(str(detected_path), detected, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise OSError(f"Could not write {detected_path}")
        page_payload.append(
            {
                "page_number": processed.page_number,
                "route": processed.route,
                "line_count": len(processed.lines),
                "width": int(processed.page.corrected_rgb.shape[1]),
                "height": int(processed.page.corrected_rgb.shape[0]),
            }
        )

    if pages:
        shutil.copyfile(corrected_directory / "page_001.jpg", output_directory / "corrected_page.jpg")
        shutil.copyfile(detected_directory / "page_001.jpg", output_directory / "detected_lines.jpg")

    transcription = "\n".join(_line_transcription(line) for line in all_lines)
    english = "\n".join(
        value for value in (_line_translation(line) for line in all_lines) if value
    )
    (output_directory / "transcription.txt").write_text(transcription + "\n", encoding="utf-8")
    (output_directory / "translation_en.txt").write_text(english + "\n", encoding="utf-8")
    word_path = output_directory / "translated_en.docx"
    word_path.write_bytes(build_translated_docx(pages[0].page, all_lines))

    payload = {
        "schema_version": "1.1",
        "input_document": str(input_path),
        "input_image": str(input_path),
        "output_directory": str(output_directory),
        "inference_only": True,
        "models": models.status,
        "pages": page_payload,
        "lines": [asdict(line) for line in all_lines],
        "transcription": transcription,
        "english_translation": english,
        "primary_output": word_path.name,
        "timings_seconds": timings,
        "accuracy_warning": "Tesseract handwriting OCR is probabilistic; review uncertain lines.",
    }
    (output_directory / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "page_number", "line", "bbox", "script", "text", "script_purity",
        "confidence", "text_quality", "recognition_model", "provider_kind",
        "accepted", "review_required", "review_reasons", "english",
        "translation_confidence", "translation_quality",
        "translation_review_reasons", "translation_context_group",
        "translation_status", "translation_error", "crop_file",
    ]
    with (output_directory / "result.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for line in all_lines:
            row = asdict(line)
            row["bbox"] = json.dumps(row["bbox"])
            row["review_reasons"] = "; ".join(line.review_reasons)
            row["translation_review_reasons"] = "; ".join(
                line.translation_review_reasons
            )
            writer.writerow({field: row.get(field) for field in fields})
    return payload


def process_pdf(
    input_path: Path,
    output_directory: Path,
    progress_callback: ProgressCallback | None = None,
    processing_mode: DocumentProcessingMode = "fast_document",
) -> dict[str, Any]:
    """Process PDF pages with native extraction and selective scan recognition."""

    if processing_mode not in {"fast_document", "fast_cpu", "line_accurate"}:
        raise ValueError(f"Unsupported PDF processing mode: {processing_mode}")
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - dependency preflight covers this
        raise RuntimeError("PyMuPDF is required for PDF input; install pymupdf.") from exc

    started = time.perf_counter()
    models = load_models()
    pages: list[_ProcessedPdfPage] = []
    line_number = 0
    crop_directory = output_directory / "line_crops"
    crop_directory.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(input_path) as document:
        if document.needs_pass:
            raise ValueError("Password-protected PDFs are not supported by the fast route.")
        if document.page_count < 1:
            raise ValueError("The PDF contains no pages.")
        for page_index, pdf_page in enumerate(document, start=1):
            _report(progress_callback, f"Reading PDF page {page_index}/{document.page_count}")
            preview_rgb = _render_pdf_page(pdf_page, PDF_PREVIEW_DPI)
            native = _native_pdf_lines(
                pdf_page, preview_rgb.shape[1], preview_rgb.shape[0]
            )
            native_usable = _native_page_is_usable(native)
            raster_coverage = _pdf_image_coverage(pdf_page)
            if native_usable and raster_coverage < 0.15:
                enhanced = _enhanced_pdf_page(preview_rgb, "native_pdf_text_extraction")
                extracted = native
                route = "native_text"
            else:
                _report(
                    progress_callback,
                    f"Fast Tesseract OCR on scanned PDF page {page_index}/{document.page_count}",
                )
                scan_rgb = _render_pdf_page(pdf_page, PDF_RENDER_DPI)
                enhanced = _enhanced_pdf_page(scan_rgb, "scanned_pdf_tesseract")
                native_at_scan_size = (
                    _native_pdf_lines(
                        pdf_page,
                        scan_rgb.shape[1],
                        scan_rgb.shape[0],
                    )
                    if native_usable
                    else []
                )
                scanned_lines = [
                    item
                    for item in _tesseract_pdf_lines(enhanced)
                    if not any(
                        _overlap_fraction(item.bbox, native_item.bbox) >= 0.50
                        for native_item in native_at_scan_size
                    )
                ]
                extracted = _supplement_scanned_regions(
                    enhanced,
                    [*native_at_scan_size, *scanned_lines],
                )
                route = "mixed_native_scanned" if native_at_scan_size else "scanned_tesseract"

            page_lines: list[LineResult] = []
            for page_line, item in enumerate(extracted, start=1):
                line_number += 1
                crop_name = Path("line_crops") / f"page_{page_index:03d}_line_{page_line:03d}.jpg"
                x1, y1, x2, y2 = item.bbox
                crop = enhanced.corrected_rgb[y1:y2, x1:x2]
                if crop.size:
                    Image.fromarray(crop).save(output_directory / crop_name, quality=90)
                page_lines.append(
                    _line_result(
                        item,
                        page_number=page_index,
                        line_number=line_number,
                        crop_file=crop_name.as_posix(),
                    )
                )
            if route in {"scanned_tesseract", "mixed_native_scanned"}:
                _report(
                    progress_callback,
                    f"Script-specific OCR refinement on PDF page {page_index}",
                )
                _refine_with_script_tesseract(enhanced, page_lines)
                if processing_mode == "line_accurate":
                    _refine_with_handwriting_models(
                        enhanced,
                        page_lines,
                        models,
                        progress_callback,
                    )
                    route = "scanned_hybrid_handwriting"
            pages.append(
                _ProcessedPdfPage(
                    page_number=page_index,
                    page=enhanced,
                    lines=page_lines,
                    route=route,
                )
            )

    if not any(page.lines for page in pages):
        raise RuntimeError("No text lines were found in the PDF.")
    all_lines = [line for page in pages for line in page.lines]
    _report(progress_callback, "Translating validated Punjabi/Hindi PDF lines")
    translation_started = time.perf_counter()
    translate_text(
        all_lines,
        models,
        progress_callback,
        num_beams=(6 if processing_mode == "line_accurate" else 4),
    )
    timings = {
        "translation": time.perf_counter() - translation_started,
        "total": time.perf_counter() - started,
    }
    models.status["processing"] = {
        "mode": processing_mode,
        "pdf_render_dpi": PDF_RENDER_DPI,
        "large_handwriting_model_loaded": processing_mode == "line_accurate",
    }
    return _write_pdf_artifacts(
        input_path=input_path,
        output_directory=output_directory,
        pages=pages,
        models=models,
        timings=timings,
    )


def process_fast_image(
    input_path: Path,
    output_directory: Path,
    progress_callback: ProgressCallback | None = None,
    *,
    refine_handwriting: bool = False,
) -> dict[str, Any]:
    """Process an image with fast OCR and optional selective neural refinement."""

    started = time.perf_counter()
    _report(progress_callback, "Correcting and enhancing the scanned image")
    with Image.open(input_path) as source:
        corrected_rgb, operations = correct_document(source)
    page = enhance_page(corrected_rgb, operations)
    _report(progress_callback, "Fast Punjabi/Hindi/English OCR on the scanned image")
    extracted = _supplement_scanned_regions(page, _tesseract_pdf_lines(page))
    output_directory.mkdir(parents=True, exist_ok=True)
    crop_directory = output_directory / "line_crops"
    crop_directory.mkdir(parents=True, exist_ok=True)
    lines: list[LineResult] = []
    for index, item in enumerate(extracted, start=1):
        crop_name = Path("line_crops") / f"line_{index:03d}.jpg"
        _line_crop(page, item.bbox, padding=4).save(
            output_directory / crop_name,
            quality=92,
        )
        lines.append(
            _line_result(
                item,
                page_number=1,
                line_number=index,
                crop_file=crop_name.as_posix(),
            )
        )
    if not lines:
        raise RuntimeError("No credible text lines were found in the image.")
    _report(progress_callback, "Script-specific OCR refinement for unresolved lines")
    _refine_with_script_tesseract(page, lines)
    models = load_models()
    if refine_handwriting:
        _refine_with_handwriting_models(page, lines, models, progress_callback)
    _report(progress_callback, "Translating validated image text to English")
    translation_started = time.perf_counter()
    translate_text(
        lines,
        models,
        progress_callback,
        num_beams=6 if refine_handwriting else 4,
    )
    total = time.perf_counter() - started
    models.status["processing"] = {
        "mode": "line_accurate" if refine_handwriting else "fast_document",
        "large_handwriting_model_loaded": refine_handwriting,
    }
    return save_results(
        input_image=input_path,
        page=page,
        lines=lines,
        models=models,
        timings={
            "translation": time.perf_counter() - translation_started,
            "total": total,
        },
        output_directory=output_directory,
    )


def process_document(
    input_path: Path,
    output_directory: Path,
    progress_callback: ProgressCallback | None = None,
    processing_mode: DocumentProcessingMode = "fast_document",
) -> dict[str, Any]:
    """Route a supported document to its fastest safe local processing path."""

    input_path = Path(input_path)
    output_directory = Path(output_directory)
    suffix = input_path.suffix.casefold()
    if suffix == ".pdf":
        return process_pdf(
            input_path,
            output_directory,
            progress_callback,
            processing_mode,
        )
    if suffix == ".docx":
        from pretrained_page_ocr.docx_processor import process_docx

        return process_docx(
            input_path,
            output_directory,
            progress_callback=progress_callback,
            processing_mode=processing_mode,
        )
    if suffix in {".jpg", ".jpeg", ".png"}:
        return process_fast_image(
            input_path,
            output_directory,
            progress_callback,
            refine_handwriting=processing_mode == "line_accurate",
        )
    raise ValueError("Supported inputs are PDF, DOCX, JPG, JPEG, and PNG.")
