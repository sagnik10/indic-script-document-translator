"""Strongly typed intermediate representation for document processing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class FileFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    PNG = "png"
    JPEG = "jpeg"


class ContentKind(StrEnum):
    NATIVE = "native"
    SCANNED = "scanned"
    MIXED = "mixed"


class BlockType(StrEnum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LINE = "line"
    WORD = "word"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"
    UNKNOWN = "unknown"


class RegionType(StrEnum):
    PRINTED_TEXT = "printed_text"
    HANDWRITING = "handwriting"
    TABLE_FORM = "table_form"
    STAMP_SEAL = "stamp_seal"
    SIGNATURE = "signature"
    GRAPHICAL_CONTENT = "graphical_content"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ScriptType(StrEnum):
    GURMUKHI = "gurmukhi"
    DEVANAGARI = "devanagari"
    LATIN = "latin"
    DIGITS = "digits"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class PageVisualType(StrEnum):
    PRINTED = "printed"
    HANDWRITING_HEAVY = "handwriting_heavy"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ReconstructionType(StrEnum):
    OCR_EXTRACTED = "OCR_EXTRACTED"
    OCR_CORRECTED = "OCR_CORRECTED"
    MODEL_INFERRED = "MODEL_INFERRED"
    UNREADABLE = "UNREADABLE"
    HUMAN_REVIEWED = "HUMAN_REVIEWED"
    MANUALLY_CONFIRMED = "MANUALLY_CONFIRMED"


class ReconstructionStatus(StrEnum):
    NOT_DETECTED = "NOT_DETECTED"
    DETECTED = "DETECTED"
    BLOCKED = "BLOCKED"
    CANDIDATE_REVIEW = "CANDIDATE_REVIEW"
    AUTO_ACCEPTED = "AUTO_ACCEPTED"
    MANUALLY_CONFIRMED = "MANUALLY_CONFIRMED"
    REJECTED = "REJECTED"
    UNAVAILABLE = "UNAVAILABLE"


class ReconstructionMode(StrEnum):
    CLEAN_REBUILD = "clean_rebuild"
    OVERLAY_TRANSLATION = "overlay_translation"
    TRANSLATION_ONLY_REPORT = "translation_only_report"


class ProcessingStatus(StrEnum):
    """Traceable safety/routing states accumulated by a logical source block."""

    OCR_CONFIRMED = "OCR_CONFIRMED"
    OCR_LOW_CONFIDENCE = "OCR_LOW_CONFIDENCE"
    SOURCE_VALIDATED = "SOURCE_VALIDATED"
    LANGUAGE_UNCERTAIN = "LANGUAGE_UNCERTAIN"
    HANDWRITING_UNSUPPORTED = "HANDWRITING_UNSUPPORTED"
    HTR_RECOGNIZED = "HTR_RECOGNIZED"
    HTR_LOW_CONFIDENCE = "HTR_LOW_CONFIDENCE"
    MANUALLY_CORRECTED = "MANUALLY_CORRECTED"
    HTR_UNAVAILABLE = "HTR_UNAVAILABLE"
    UNREADABLE = "UNREADABLE"
    RECONSTRUCTED_SOURCE = "RECONSTRUCTED_SOURCE"
    TRANSLATED = "TRANSLATED"
    TRANSLATION_SKIPPED = "TRANSLATION_SKIPPED"
    LAYOUT_OVERFLOW = "LAYOUT_OVERFLOW"
    MISSING_SPAN_DETECTED = "MISSING_SPAN_DETECTED"
    RECONSTRUCTION_REVIEW_REQUIRED = "RECONSTRUCTION_REVIEW_REQUIRED"
    RECONSTRUCTION_AUTO_ACCEPTED = "RECONSTRUCTION_AUTO_ACCEPTED"


class UncertaintyState(StrEnum):
    CONFIRMED = "confirmed"
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    CANDIDATE = "candidate"
    RECONSTRUCTED = "reconstructed"
    FLAGGED = "flagged"


class TranslationStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    TRANSLATED = "translated"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    SKIPPED = "skipped"


class LayoutStatus(StrEnum):
    PENDING = "pending"
    FIT = "fit"
    SHRUNK = "shrunk"
    EXPANDED = "expanded"
    OVERFLOW = "overflow"
    SKIPPED = "skipped"


class ProcessingStage(StrEnum):
    VALIDATING = "Validating the document"
    READING = "Reading the file"
    EXTRACTING = "Extracting pages"
    OCR = "Running OCR"
    LANGUAGE = "Detecting languages"
    RECONSTRUCTION = "Reconstructing uncertain text"
    REVIEW = "Reviewing uncertain OCR"
    TRANSLATION = "Translating"
    LAYOUT = "Rebuilding layout"
    RENDERING = "Rendering output"
    VALIDATING_OUTPUT = "Validating the result"
    COMPLETE = "Complete"


@dataclass(slots=True, frozen=True)
class BoundingBox:
    """Rectangle in source page points unless a page explicitly uses pixels."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("Bounding box coordinates must be ordered")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def intersects(self, other: "BoundingBox", padding: float = 0.0) -> bool:
        return not (
            self.x1 + padding <= other.x0
            or other.x1 + padding <= self.x0
            or self.y1 + padding <= other.y0
            or other.y1 + padding <= self.y0
        )

    def intersection_ratio(self, other: "BoundingBox") -> float:
        width = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        height = max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))
        intersection = width * height
        denominator = min(self.area, other.area)
        return intersection / denominator if denominator else 0.0

    def clamp(self, width: float, height: float) -> "BoundingBox":
        return BoundingBox(
            max(0.0, min(self.x0, width)),
            max(0.0, min(self.y0, height)),
            max(0.0, min(self.x1, width)),
            max(0.0, min(self.y1, height)),
        )


@dataclass(slots=True)
class FontMetadata:
    family: str = "Helvetica"
    size: float = 11.0
    bold: bool = False
    italic: bool = False
    underline: bool = False
    color: str = "#000000"
    line_spacing: float = 1.15


@dataclass(slots=True)
class TextLine:
    page_number: int
    line_id: str
    bbox: BoundingBox
    raw_text: str
    normalized_text: str = ""
    baseline: tuple[float, float] | None = None
    token_metadata: list[dict[str, Any]] = field(default_factory=list)
    reading_order: int = 0


@dataclass(slots=True)
class Region:
    page_number: int
    bbox: BoundingBox
    region_type: RegionType
    region_id: str = field(default_factory=lambda: uuid4().hex)
    classification_confidence: float = 0.0
    reading_order: int = 0
    block_ids: list[str] = field(default_factory=list)
    preserve_as_image: bool = False
    overlaps_critical_graphic: bool = False
    visual_script_candidate: ScriptType = ScriptType.UNKNOWN
    visual_script_confidence: float = 0.0
    recognized_unicode_script: ScriptType = ScriptType.UNKNOWN
    recognized_unicode_script_confidence: float = 0.0
    resolved_script: ScriptType = ScriptType.UNKNOWN
    script_resolution_reason: str = "not_resolved"
    expected_language_prior: str = "auto"
    resolved_language: str = "und"
    selected_recognition_engine: str = "unrouted"
    linguistic_evidence_score: float = 0.0
    rejected_as_noise: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TextBlock:
    """Text plus linguistic, uncertainty, and layout state for one logical region."""

    page_number: int
    block_type: BlockType
    source_bbox: BoundingBox
    source_text: str
    block_id: str = field(default_factory=lambda: uuid4().hex)
    normalized_text: str = ""
    detected_language: str = "und"
    language_confidence: float = 0.0
    ocr_confidence: float | None = None
    uncertainty_state: UncertaintyState = UncertaintyState.CONFIRMED
    reconstructed_text: str | None = None
    reconstruction_confidence: float | None = None
    english_translation: str | None = None
    font: FontMetadata = field(default_factory=FontMetadata)
    alignment: str = "left"
    background_color: str | None = None
    rotation: float = 0.0
    output_bbox: BoundingBox | None = None
    translation_status: TranslationStatus = TranslationStatus.PENDING
    layout_status: LayoutStatus = LayoutStatus.PENDING
    is_ocr: bool = False
    parent_block_id: str | None = None
    child_block_ids: list[str] = field(default_factory=list)
    source_reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    region_type: RegionType = RegionType.PRINTED_TEXT
    script: ScriptType = ScriptType.UNKNOWN
    ocr_engine: str = "native"
    is_handwritten: bool = False
    reconstruction_type: ReconstructionType = ReconstructionType.OCR_EXTRACTED
    protected_tokens: list[str] = field(default_factory=list)
    ocr_alternatives: list[dict[str, Any]] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    text_quality: float = 0.0
    source_validated: bool = False
    validation_reason: str = "not_validated"
    processing_statuses: list[ProcessingStatus] = field(default_factory=list)
    review_image_bytes: bytes | None = field(default=None, repr=False)
    missing_span_detected: bool = False
    missing_span_bbox: BoundingBox | None = None
    reconstruction_candidate: str | None = None
    reconstruction_method: str | None = None
    reconstruction_status: ReconstructionStatus = ReconstructionStatus.NOT_DETECTED
    readable_character_ratio: float = 0.0
    validated_context_token_count: int = 0
    protected_entity_detected: bool = False
    region_visual_script: ScriptType = ScriptType.UNKNOWN
    visual_script_confidence: float = 0.0
    recognized_unicode_script: ScriptType = ScriptType.UNKNOWN
    ocr_script_confidence: float = 0.0
    resolved_script: ScriptType = ScriptType.UNKNOWN
    resolved_language: str = "und"
    script_resolution_reason: str = "not_resolved"
    expected_language_prior: str = "auto"
    linguistic_evidence_score: float = 0.0

    @property
    def bbox(self) -> BoundingBox:
        return self.source_bbox

    @property
    def raw_text(self) -> str:
        return self.source_text

    @raw_text.setter
    def raw_text(self, value: str) -> None:
        self.source_text = value

    @property
    def language(self) -> str:
        return self.detected_language

    @language.setter
    def language(self, value: str) -> None:
        self.detected_language = value

    @property
    def is_uncertain(self) -> bool:
        return self.uncertainty_state != UncertaintyState.CONFIRMED

    @property
    def translation_confidence_or_status(self) -> str:
        return self.translation_status.value

    @property
    def font_size(self) -> float:
        return self.font.size

    @property
    def font_style(self) -> str:
        styles = [
            name
            for enabled, name in (
                (self.font.bold, "bold"),
                (self.font.italic, "italic"),
                (self.font.underline, "underline"),
            )
            if enabled
        ]
        return "+".join(styles) or "regular"

    @property
    def source_coordinates(self) -> BoundingBox:
        return self.source_bbox

    @property
    def output_coordinates(self) -> BoundingBox | None:
        return self.output_bbox

    @property
    def effective_source_text(self) -> str:
        if self.reconstruction_type in {
            ReconstructionType.HUMAN_REVIEWED,
            ReconstructionType.MANUALLY_CONFIRMED,
        } and self.reconstructed_text:
            return self.reconstructed_text
        if (
            self.reconstruction_type == ReconstructionType.UNREADABLE
            and self.missing_span_detected
            and self.reconstructed_text
        ):
            return self.reconstructed_text
        if (
            self.uncertainty_state == UncertaintyState.RECONSTRUCTED
            and self.reconstructed_text
        ):
            return self.reconstructed_text
        return self.normalized_text or self.source_text

    @property
    def output_text(self) -> str:
        if self.translation_status == TranslationStatus.TRANSLATED:
            return self.english_translation or self.effective_source_text
        return self.effective_source_text


@dataclass(slots=True)
class PageModel:
    page_number: int
    width: float
    height: float
    blocks: list[TextBlock] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    rotation: int = 0
    content_kind: ContentKind = ContentKind.NATIVE
    image_bytes: bytes | None = field(default=None, repr=False)
    enhanced_image_bytes: bytes | None = field(default=None, repr=False)
    visual_page_script: ScriptType = ScriptType.UNKNOWN
    visual_page_script_confidence: float = 0.0
    ocr_page_script: ScriptType = ScriptType.UNKNOWN
    ocr_page_script_confidence: float = 0.0
    resolved_page_script: ScriptType = ScriptType.UNKNOWN
    resolved_page_script_confidence: float = 0.0
    script_resolution_reason: str = "not_resolved"
    expected_language_prior: str = "auto"
    handwriting_probability: float = 0.0
    page_visual_type: PageVisualType = PageVisualType.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentModel:
    document_id: str
    source_filename: str
    file_format: FileFormat
    content_kind: ContentKind
    pages: list[PageModel]
    source_bytes: bytes = field(repr=False)
    mime_type: str = "application/octet-stream"
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def blocks(self) -> list[TextBlock]:
        return [block for page in self.pages for block in page.blocks]

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        """Serialize document metadata, excluding confidential source bytes and page images."""
        data: dict[str, Any] = {
            "document_id": self.document_id,
            "source_filename": self.source_filename,
            "file_format": self.file_format.value,
            "content_kind": self.content_kind.value,
            "mime_type": self.mime_type,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
            "pages": [],
        }
        for page_model in self.pages:
            page = asdict(page_model)
            page.pop("image_bytes", None)
            page.pop("enhanced_image_bytes", None)
            if not include_text:
                for block in page["blocks"]:
                    for key in (
                        "source_text",
                        "normalized_text",
                        "reconstructed_text",
                        "english_translation",
                    ):
                        if block.get(key):
                            block[key] = "[REDACTED]"
            page["content_kind"] = page_model.content_kind.value
            for block_model, block in zip(page_model.blocks, page["blocks"], strict=True):
                block.pop("review_image_bytes", None)
                block["block_type"] = block_model.block_type.value
                block["uncertainty_state"] = block_model.uncertainty_state.value
                block["translation_status"] = block_model.translation_status.value
                block["layout_status"] = block_model.layout_status.value
                block["region_type"] = block_model.region_type.value
                block["script"] = block_model.script.value
                block["region_visual_script"] = block_model.region_visual_script.value
                block["recognized_unicode_script"] = (
                    block_model.recognized_unicode_script.value
                )
                block["resolved_script"] = block_model.resolved_script.value
                block["reconstruction_type"] = block_model.reconstruction_type.value
                block["reconstruction_status"] = block_model.reconstruction_status.value
                block["processing_statuses"] = [
                    status.value for status in block_model.processing_statuses
                ]
            for region_model, region in zip(page_model.regions, page["regions"], strict=True):
                region["region_type"] = region_model.region_type.value
                region["visual_script_candidate"] = region_model.visual_script_candidate.value
                region["recognized_unicode_script"] = (
                    region_model.recognized_unicode_script.value
                )
                region["resolved_script"] = region_model.resolved_script.value
            page["visual_page_script"] = page_model.visual_page_script.value
            page["ocr_page_script"] = page_model.ocr_page_script.value
            page["resolved_page_script"] = page_model.resolved_page_script.value
            page["page_visual_type"] = page_model.page_visual_type.value
            data["pages"].append(page)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_bytes: bytes = b"") -> "DocumentModel":
        pages: list[PageModel] = []
        for page_data in data.get("pages", []):
            blocks: list[TextBlock] = []
            for block_data in page_data.get("blocks", []):
                copied = dict(block_data)
                copied["block_type"] = BlockType(copied["block_type"])
                copied["uncertainty_state"] = UncertaintyState(copied["uncertainty_state"])
                copied["translation_status"] = TranslationStatus(copied["translation_status"])
                copied["layout_status"] = LayoutStatus(copied["layout_status"])
                copied["region_type"] = RegionType(copied.get("region_type", "printed_text"))
                copied["script"] = ScriptType(copied.get("script", "unknown"))
                copied["region_visual_script"] = ScriptType(
                    copied.get("region_visual_script", "unknown")
                )
                copied["recognized_unicode_script"] = ScriptType(
                    copied.get("recognized_unicode_script", "unknown")
                )
                copied["resolved_script"] = ScriptType(
                    copied.get("resolved_script", copied.get("script", "unknown"))
                )
                copied["reconstruction_type"] = ReconstructionType(
                    copied.get("reconstruction_type", "OCR_EXTRACTED")
                )
                copied["reconstruction_status"] = ReconstructionStatus(
                    copied.get("reconstruction_status", "NOT_DETECTED")
                )
                copied["processing_statuses"] = [
                    ProcessingStatus(value)
                    for value in copied.get("processing_statuses", [])
                ]
                copied["source_bbox"] = BoundingBox(**copied["source_bbox"])
                if copied.get("output_bbox"):
                    copied["output_bbox"] = BoundingBox(**copied["output_bbox"])
                if copied.get("missing_span_bbox"):
                    copied["missing_span_bbox"] = BoundingBox(**copied["missing_span_bbox"])
                copied["font"] = FontMetadata(**copied.get("font", {}))
                blocks.append(TextBlock(**copied))
            page_copy = dict(page_data)
            page_copy["blocks"] = blocks
            page_copy["regions"] = [
                Region(
                    **{
                        **region,
                        "bbox": BoundingBox(**region["bbox"]),
                        "region_type": RegionType(region["region_type"]),
                        "visual_script_candidate": ScriptType(
                            region.get("visual_script_candidate", "unknown")
                        ),
                        "recognized_unicode_script": ScriptType(
                            region.get("recognized_unicode_script", "unknown")
                        ),
                        "resolved_script": ScriptType(
                            region.get("resolved_script", "unknown")
                        ),
                    }
                )
                for region in page_copy.get("regions", [])
            ]
            page_copy["content_kind"] = ContentKind(page_copy["content_kind"])
            page_copy["visual_page_script"] = ScriptType(
                page_copy.get("visual_page_script", "unknown")
            )
            page_copy["ocr_page_script"] = ScriptType(
                page_copy.get("ocr_page_script", "unknown")
            )
            page_copy["resolved_page_script"] = ScriptType(
                page_copy.get("resolved_page_script", "unknown")
            )
            page_copy["page_visual_type"] = PageVisualType(
                page_copy.get("page_visual_type", "unknown")
            )
            pages.append(PageModel(**page_copy))
        return cls(
            document_id=data["document_id"],
            source_filename=data["source_filename"],
            file_format=FileFormat(data["file_format"]),
            content_kind=ContentKind(data["content_kind"]),
            pages=pages,
            source_bytes=source_bytes,
            mime_type=data.get("mime_type", "application/octet-stream"),
            warnings=list(data.get("warnings", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class ProcessingOptions:
    target_language: str = "en"
    force_ocr: bool = False
    enable_preprocessing: bool = True
    enable_reconstruction: bool = True
    ocr_languages: list[str] = field(default_factory=list)
    ocr_low_confidence_threshold: float = 0.65
    reconstruction_accept_threshold: float = 0.82
    auto_reconstruct_threshold: float = 0.90
    review_reconstruct_threshold: float = 0.70
    min_context_quality: float = 0.80
    preserve_source_on_failure: bool = True
    preprocessing_profile: str = "auto"
    ocr_upscale_factor: float = 2.0
    enable_printed_ocr: bool = True
    enable_handwriting_ocr: bool = False
    handwriting_language_hint: str = "auto"
    expected_source_language: str = "auto"
    routing_debug: bool = False
    preserve_unreadable_handwriting_as_image: bool = True
    handwriting_confidence_threshold: float = 0.55
    reconstruction_mode: ReconstructionMode = ReconstructionMode.CLEAN_REBUILD
    review_before_render: bool = False
    debug_bounding_boxes: bool = False
    document_domain: str = "auto"
    protected_terms: list[str] = field(default_factory=list)
    glossary: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessingSummary:
    filename: str
    file_type: str
    page_count: int
    detected_languages: list[str]
    text_block_count: int
    ocr_block_count: int
    low_confidence_ocr_count: int
    reconstructed_block_count: int
    uncertain_block_count: int
    translation_count: int
    layout_overflow_count: int
    processing_duration_seconds: float
    output_filename: str
    region_count: int = 0
    handwriting_block_count: int = 0
    unreadable_block_count: int = 0
    preprocessing_profiles: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dominant_scripts: list[str] = field(default_factory=list)
    printed_region_count: int = 0
    handwritten_region_count: int = 0
    validated_punjabi_line_count: int = 0
    validated_hindi_line_count: int = 0
    translation_skipped_count: int = 0
    htr_recognized_line_count: int = 0
    manually_reviewed_line_count: int = 0
    rejected_handwriting_line_count: int = 0
    missing_span_count: int = 0
    auto_reconstructed_span_count: int = 0
    review_reconstruction_count: int = 0
    manually_confirmed_count: int = 0
    unresolved_missing_span_count: int = 0
    punjabi_htr_route_count: int = 0
    hindi_htr_route_count: int = 0
    printed_ocr_route_count: int = 0
    rejected_noise_region_count: int = 0
    page_visual_types: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProcessingResult:
    document: DocumentModel
    output_bytes: bytes
    output_filename: str
    output_mime_type: str
    summary: ProcessingSummary
    source_preview_images: list[bytes] = field(default_factory=list, repr=False)
    output_preview_images: list[bytes] = field(default_factory=list, repr=False)
    enhanced_preview_images: list[bytes] = field(default_factory=list, repr=False)
    debug_preview_images: list[bytes] = field(default_factory=list, repr=False)
    audit_json: bytes = field(default=b"", repr=False)


@dataclass(slots=True)
class AnalysisResult:
    """Reviewable state after OCR/language/reconstruction but before translation/rendering."""

    document: DocumentModel
    options: ProcessingOptions
    started_at: float
    source_preview_images: list[bytes] = field(default_factory=list, repr=False)
    enhanced_preview_images: list[bytes] = field(default_factory=list, repr=False)
    debug_preview_images: list[bytes] = field(default_factory=list, repr=False)
