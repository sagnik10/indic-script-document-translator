"""Typed application settings loaded from YAML and environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin, get_type_hints

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def _coerce(value: Any, annotation: Any) -> Any:
    """Coerce environment strings into a dataclass field's simple type."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list:
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]
        subtype = args[0] if args else str
        return [_coerce(item, subtype) for item in value]
    if origin is dict:
        return dict(value)
    if origin is not None and type(None) in args:
        if value in (None, "", "null", "None"):
            return None
        target = next(arg for arg in args if arg is not type(None))
        return _coerce(value, target)
    if annotation is bool and isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is str:
        return str(value)
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings. Paths are normalized relative to the project root."""

    max_upload_size: int = 52_428_800
    ocr_dpi: int = 300
    ocr_engine: str = "tesseract"
    ocr_languages: list[str] | None = None
    ocr_low_confidence_threshold: float = 0.65
    handwriting_confidence_threshold: float = 0.55
    ocr_upscale_factor: float = 2.0
    printed_ocr_psm_candidates: list[int] | None = None
    reconstruction_accept_threshold: float = 0.82
    auto_reconstruct_threshold: float = 0.90
    review_reconstruct_threshold: float = 0.70
    min_context_quality: float = 0.80
    min_validated_context_tokens: int = 3
    max_reconstruction_span_tokens: int = 3
    default_target_language: str = "en"
    min_output_font_size: float = 6.0
    translation_batch_size: int = 8
    temp_directory: Path = PROJECT_ROOT / ".runtime" / "tmp"
    output_directory: Path = PROJECT_ROOT / "outputs"
    device: str = "auto"
    log_level: str = "INFO"
    debug: bool = False
    log_document_text: bool = False
    translation_provider: str = "huggingface"
    translation_model: str = "facebook/nllb-200-distilled-600M"
    reconstruction_model: str | None = None
    hf_local_files_only: bool = False
    native_text_min_characters: int = 20
    max_model_input_characters: int = 1200
    enable_safe_block_expansion: bool = True
    max_block_expansion_points: float = 36.0
    tesseract_cmd: str | None = None
    tessdata_directory: Path | None = PROJECT_ROOT / ".runtime" / "tessdata"
    preprocessing_profile: str = "auto"
    reconstruction_mode: str = "clean_rebuild"
    preserve_unreadable_handwriting_as_image: bool = True
    handwriting_engine: str = "trocr"
    handwriting_language_hint: str = "auto"
    expected_source_language: str = "auto"
    handwriting_models: dict[str, str | None] | None = None
    htr_providers: list[dict[str, Any]] | None = None
    local_htr_model_directory: Path = PROJECT_ROOT / ".runtime" / "models"
    max_preview_pages: int = 8
    min_region_area: int = 180
    min_region_width: int = 28
    min_region_height: int = 9
    min_ocr_character_count: int = 3
    min_source_letter_count: int = 4
    min_text_quality_score: float = 0.48
    min_source_script_ratio: float = 0.55
    reconstruction_min_readable_ratio: float = 0.72
    catastrophic_unreadable_ratio: float = 0.60
    handwriting_page_threshold: float = 0.55
    visual_script_min_confidence: float = 0.52
    visual_indic_headline_threshold: float = 0.30
    handwriting_heavy_threshold: float = 0.58
    border_noise_max_fraction: float = 0.18
    region_merge_horizontal_gap_ratio: float = 2.5
    region_merge_vertical_overlap: float = 0.30
    paragraph_line_gap_ratio: float = 1.65
    background_reliability_threshold: float = 0.72
    enable_table_detection: bool = True
    document_domain: str = "auto"

    def __post_init__(self) -> None:
        if self.ocr_languages is None:
            object.__setattr__(self, "ocr_languages", ["eng", "hin", "pan"])
        if self.printed_ocr_psm_candidates is None:
            object.__setattr__(self, "printed_ocr_psm_candidates", [6, 11])
        if self.handwriting_models is None:
            object.__setattr__(
                self,
                "handwriting_models",
                {"en": "microsoft/trocr-base-handwritten", "hi": None, "pa": None},
            )
        if self.htr_providers is None:
            legacy_providers = []
            for language, model_id in (self.handwriting_models or {}).items():
                canonical = {"pan": "pa", "punjabi": "pa", "hin": "hi", "hindi": "hi"}.get(
                    str(language).casefold(), str(language).casefold()
                )
                script = {"pa": "gurmukhi", "hi": "devanagari", "en": "latin"}.get(
                    canonical, "unknown"
                )
                legacy_providers.append(
                    {
                        "provider_id": f"{canonical}_vision_encoder_decoder",
                        "backend": "transformers_vision_encoder_decoder",
                        "model_id": model_id,
                        "supported_languages": [canonical],
                        "supported_scripts": [script],
                        "confidence_capability": "sequence_probability",
                        "handwriting_validated": canonical == "en",
                    }
                )
            object.__setattr__(self, "htr_providers", legacy_providers)
        if not 0 <= self.ocr_low_confidence_threshold <= 1:
            raise ValueError("OCR_LOW_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if not 0 <= self.reconstruction_accept_threshold <= 1:
            raise ValueError("RECONSTRUCTION_ACCEPT_THRESHOLD must be between 0 and 1")
        for name, value in (
            ("AUTO_RECONSTRUCT_THRESHOLD", self.auto_reconstruct_threshold),
            ("REVIEW_RECONSTRUCT_THRESHOLD", self.review_reconstruct_threshold),
            ("MIN_CONTEXT_QUALITY", self.min_context_quality),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.review_reconstruct_threshold > self.auto_reconstruct_threshold:
            raise ValueError(
                "REVIEW_RECONSTRUCT_THRESHOLD cannot exceed AUTO_RECONSTRUCT_THRESHOLD"
            )
        if self.min_validated_context_tokens < 1 or self.max_reconstruction_span_tokens < 1:
            raise ValueError("Reconstruction context/span token limits must be positive")
        if self.max_upload_size <= 0 or self.ocr_dpi <= 0:
            raise ValueError("Upload size and OCR DPI must be positive")
        if self.min_output_font_size <= 0:
            raise ValueError("MIN_OUTPUT_FONT_SIZE must be positive")
        if not 0 <= self.handwriting_confidence_threshold <= 1:
            raise ValueError("HANDWRITING_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if not 1 <= self.ocr_upscale_factor <= 4:
            raise ValueError("OCR_UPSCALE_FACTOR must be between 1 and 4")
        if self.reconstruction_mode not in {
            "clean_rebuild",
            "overlay_translation",
            "translation_only_report",
        }:
            raise ValueError(
                "RECONSTRUCTION_MODE must be clean_rebuild, overlay_translation, or translation_only_report"
            )
        for name, value in (
            ("MIN_TEXT_QUALITY_SCORE", self.min_text_quality_score),
            ("MIN_SOURCE_SCRIPT_RATIO", self.min_source_script_ratio),
            ("RECONSTRUCTION_MIN_READABLE_RATIO", self.reconstruction_min_readable_ratio),
            ("CATASTROPHIC_UNREADABLE_RATIO", self.catastrophic_unreadable_ratio),
            ("HANDWRITING_PAGE_THRESHOLD", self.handwriting_page_threshold),
            ("REGION_MERGE_VERTICAL_OVERLAP", self.region_merge_vertical_overlap),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.handwriting_language_hint not in {"auto", "pa", "hi", "en"}:
            raise ValueError("HANDWRITING_LANGUAGE_HINT must be auto, pa, hi, or en")
        if self.expected_source_language not in {"auto", "pa", "hi", "pa+hi"}:
            raise ValueError(
                "EXPECTED_SOURCE_LANGUAGE must be auto, pa, hi, or pa+hi"
            )
        for name, value in (
            ("VISUAL_SCRIPT_MIN_CONFIDENCE", self.visual_script_min_confidence),
            ("VISUAL_INDIC_HEADLINE_THRESHOLD", self.visual_indic_headline_threshold),
            ("HANDWRITING_HEAVY_THRESHOLD", self.handwriting_heavy_threshold),
            ("BORDER_NOISE_MAX_FRACTION", self.border_noise_max_fraction),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

    @property
    def default_translation_batch_size(self) -> int:
        """Backward-compatible alias for callers written before config renaming."""
        return self.translation_batch_size

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Settings":
        normalized = {str(key).lower(): value for key, value in values.items()}
        kwargs: dict[str, Any] = {}
        annotations = get_type_hints(cls)
        for field_info in fields(cls):
            if field_info.name not in normalized:
                continue
            value = normalized[field_info.name]
            if field_info.name in {
                "temp_directory",
                "output_directory",
                "tessdata_directory",
                "local_htr_model_directory",
            } and value not in (None, "", "null", "None"):
                path = Path(str(value)).expanduser()
                kwargs[field_info.name] = path if path.is_absolute() else PROJECT_ROOT / path
            else:
                kwargs[field_info.name] = _coerce(value, annotations[field_info.name])
        return cls(**kwargs)

    def with_overrides(self, **changes: Any) -> "Settings":
        """Return a validated copy, useful for per-session UI controls and tests."""
        return replace(self, **changes)


def load_settings(config_path: Path | None = None) -> Settings:
    path = config_path or Path(os.getenv("DTX_CONFIG", DEFAULT_CONFIG_PATH))
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Configuration root must be a mapping: {path}")
            raw.update(loaded)
    valid_names = {field_info.name.upper() for field_info in fields(Settings)}
    for name in valid_names:
        env_name = f"DTX_{name}"
        if env_name in os.environ:
            raw[name] = os.environ[env_name]
    settings = Settings.from_mapping(raw)
    settings.temp_directory.mkdir(parents=True, exist_ok=True)
    settings.output_directory.mkdir(parents=True, exist_ok=True)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load application settings once for the current process."""
    return load_settings()
