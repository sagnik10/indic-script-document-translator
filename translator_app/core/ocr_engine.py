"""Coordinate-aware OCR provider interfaces and a local Tesseract implementation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from PIL import Image

from ..config.settings import Settings
from ..exceptions import OCRUnavailableError
from ..schemas import (
    BlockType,
    BoundingBox,
    ReconstructionType,
    RegionType,
    TextBlock,
    UncertaintyState,
)
from .tesseract_runtime import configure_pytesseract, discover_tesseract_runtime

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OCRResult:
    blocks: list[TextBlock]
    warnings: list[str] = field(default_factory=list)
    engine: str = "unknown"
    languages_used: list[str] = field(default_factory=list)


class OCREngine(ABC):
    @abstractmethod
    def recognize(
        self,
        image: Image.Image,
        *,
        page_number: int,
        page_width: float,
        page_height: float,
        requested_languages: list[str],
        low_confidence_threshold: float,
        psm: int = 3,
    ) -> OCRResult:
        """Return logical OCR regions with source-page coordinates and confidence."""


class TesseractOCREngine(OCREngine):
    """Local OCR using pytesseract data output grouped into paragraph blocks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            import pytesseract

            self.pytesseract = pytesseract
            self.runtime = discover_tesseract_runtime(
                settings.tesseract_cmd,
                settings.tessdata_directory,
            )
            if self.runtime is None:
                raise OCRUnavailableError(
                    "Tesseract was not found in the configured, PATH, or standard installation locations"
                )
            configure_pytesseract(pytesseract, self.runtime)
            pytesseract.get_tesseract_version()
            LOGGER.info(
                "Using Tesseract executable %s with language data %s",
                self.runtime.executable,
                self.runtime.tessdata_directory or "installation default",
            )
        except OCRUnavailableError:
            raise
        except Exception as exc:
            raise OCRUnavailableError("Tesseract executable could not be started") from exc

    def _select_languages(self, requested: list[str]) -> tuple[list[str], list[str]]:
        try:
            available = set(self.pytesseract.get_languages(config=""))
        except Exception as exc:
            raise OCRUnavailableError("Could not query installed Tesseract language packs") from exc
        selected = [language for language in requested if language in available]
        warnings = []
        missing = [language for language in requested if language not in available]
        if missing:
            warnings.append(
                "Missing Tesseract language packs: " + ", ".join(sorted(missing))
            )
        if not selected and "eng" in available and "eng" in requested:
            selected = ["eng"]
            warnings.append("Using requested English OCR; other requested packs are unavailable.")
        if not selected:
            raise OCRUnavailableError(
                "None of the requested OCR languages are installed: " + ", ".join(requested)
            )
        return selected, warnings

    def recognize(
        self,
        image: Image.Image,
        *,
        page_number: int,
        page_width: float,
        page_height: float,
        requested_languages: list[str],
        low_confidence_threshold: float,
        psm: int = 3,
    ) -> OCRResult:
        languages, warnings = self._select_languages(requested_languages)
        try:
            output = self.pytesseract.image_to_data(
                image,
                lang="+".join(languages),
                config=(
                    f"--oem 1 --psm {int(psm)} preserve_interword_spaces=1"
                ),
                output_type=self.pytesseract.Output.DICT,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "error opening data file" in message or "failed loading language" in message:
                raise OCRUnavailableError("A configured Tesseract language pack failed to load") from exc
            raise OCRUnavailableError("Tesseract OCR failed") from exc
        scale_x = page_width / max(1, image.width)
        scale_y = page_height / max(1, image.height)
        grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
        count = len(output.get("text", []))
        for index in range(count):
            text = str(output["text"][index]).strip()
            try:
                confidence = float(output["conf"][index]) / 100.0
            except (TypeError, ValueError):
                confidence = -1.0
            if not text or confidence < 0:
                continue
            key = (
                int(output["block_num"][index]),
                int(output["par_num"][index]),
                int(output["line_num"][index]),
            )
            grouped[key].append(
                {
                    "text": text,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "left": int(output["left"][index]),
                    "top": int(output["top"][index]),
                    "width": int(output["width"][index]),
                    "height": int(output["height"][index]),
                    "word_number": int(output["word_num"][index]),
                }
            )
        blocks: list[TextBlock] = []
        parent_ids: dict[tuple[int, int], str] = {}
        for (block_number, paragraph_number, line_number), words in sorted(grouped.items()):
            parent_key = (block_number, paragraph_number)
            parent_id = parent_ids.setdefault(parent_key, uuid4().hex)
            x0 = min(word["left"] for word in words) * scale_x
            y0 = min(word["top"] for word in words) * scale_y
            x1 = max(word["left"] + word["width"] for word in words) * scale_x
            y1 = max(word["top"] + word["height"] for word in words) * scale_y
            total_chars = sum(max(1, len(word["text"])) for word in words)
            confidence = sum(
                word["confidence"] * max(1, len(word["text"])) for word in words
            ) / total_chars
            source_text = " ".join(word["text"] for word in words)
            uncertainty = (
                UncertaintyState.LOW_OCR_CONFIDENCE
                if confidence < low_confidence_threshold
                else UncertaintyState.CONFIRMED
            )
            word_metadata = [
                {
                    **word,
                    "bbox": {
                        "x0": word["left"] * scale_x,
                        "y0": word["top"] * scale_y,
                        "x1": (word["left"] + word["width"]) * scale_x,
                        "y1": (word["top"] + word["height"]) * scale_y,
                    },
                }
                for word in words
            ]
            blocks.append(
                TextBlock(
                    page_number=page_number,
                    block_type=BlockType.LINE,
                    source_bbox=BoundingBox(x0, y0, x1, y1),
                    source_text=source_text,
                    ocr_confidence=confidence,
                    uncertainty_state=uncertainty,
                    is_ocr=True,
                    region_type=RegionType.PRINTED_TEXT,
                    ocr_engine=f"tesseract_psm_{psm}",
                    reconstruction_type=ReconstructionType.OCR_EXTRACTED,
                    provenance=[f"Tesseract OEM 1 PSM {psm} languages={'+'.join(languages)}"],
                    parent_block_id=parent_id,
                    metadata={
                        "ocr_block_number": block_number,
                        "ocr_paragraph_number": paragraph_number,
                        "ocr_line_number": line_number,
                        "baseline": [x0, y1, x1, y1],
                        "words": word_metadata,
                    },
                )
            )
        paragraph_children: dict[str, list[str]] = defaultdict(list)
        for block in blocks:
            if block.parent_block_id:
                paragraph_children[block.parent_block_id].append(block.block_id)
        for block in blocks:
            siblings = paragraph_children.get(block.parent_block_id or "", [])
            block.metadata["ocr_paragraph_child_ids"] = siblings
            block.metadata["ocr_paragraph_is_virtual"] = True
        return OCRResult(blocks, warnings, f"tesseract_psm_{psm}", languages)
