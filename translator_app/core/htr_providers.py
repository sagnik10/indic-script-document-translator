"""Capability-declared, source-script-only handwriting recognition providers."""

from __future__ import annotations

import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from math import exp
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from ..config.settings import Settings
from ..schemas import ScriptType
from .source_validation import calculate_text_quality, normalize_language, script_ratio

LOGGER = logging.getLogger(__name__)


class HTRConfidenceCapability(StrEnum):
    """How a provider quantifies recognition confidence."""

    NONE = "none"
    SEQUENCE_PROBABILITY = "sequence_probability"


@dataclass(frozen=True, slots=True)
class HTRProviderCapabilities:
    """Machine-readable claims used before a provider may receive an image."""

    provider_id: str
    backend: str
    supported_languages: frozenset[str]
    supported_scripts: frozenset[ScriptType]
    confidence_capability: HTRConfidenceCapability
    source_language_output_only: bool = True
    local_processing: bool = True
    handwriting_validated: bool = True

    @property
    def returns_confidence(self) -> bool:
        return self.confidence_capability != HTRConfidenceCapability.NONE

    def supports(self, language: str, script: ScriptType) -> bool:
        canonical = normalize_language(language)
        language_supported = canonical in self.supported_languages
        script_supported = script == ScriptType.UNKNOWN or script in self.supported_scripts
        return (
            language_supported
            and script_supported
            and self.source_language_output_only
            and self.handwriting_validated
        )


@dataclass(slots=True)
class HTRPrediction:
    """One source-language line transcription and its traceable alternatives."""

    text: str
    confidence: float | None
    provider_id: str
    model_id: str
    alternatives: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class HTRProviderSpec:
    provider_id: str
    backend: str
    model_id: str | None
    supported_languages: tuple[str, ...]
    supported_scripts: tuple[ScriptType, ...]
    confidence_capability: HTRConfidenceCapability
    handwriting_validated: bool

    @property
    def configured(self) -> bool:
        return bool(self.model_id and self.model_id.strip())


class HandwritingRecognitionProvider(ABC):
    """Provider contract. Implementations may only return source-script text."""

    capabilities: HTRProviderCapabilities
    model_id: str

    @abstractmethod
    def recognize_line(self, image: Image.Image) -> HTRPrediction:
        raise NotImplementedError


_KNOWN_ENGLISH_ONLY_TROCR = {
    "microsoft/trocr-small-handwritten",
    "microsoft/trocr-base-handwritten",
    "microsoft/trocr-large-handwritten",
}


def _parse_script(value: object) -> ScriptType:
    try:
        return ScriptType(str(value).strip().casefold())
    except ValueError:
        return ScriptType.UNKNOWN


def _model_assets_exist(model_directory: Path) -> bool:
    """Return whether a local bundle has enough files for offline HF loading."""
    weights = (
        model_directory / "model.safetensors",
        model_directory / "pytorch_model.bin",
        model_directory / "model.safetensors.index.json",
        model_directory / "pytorch_model.bin.index.json",
    )
    tokenizer_assets = (
        model_directory / "tokenizer.json",
        model_directory / "tokenizer_config.json",
        model_directory / "vocab.json",
        model_directory / "sentencepiece.bpe.model",
    )
    return bool(
        (model_directory / "config.json").is_file()
        and (model_directory / "preprocessor_config.json").is_file()
        and any(path.is_file() for path in weights)
        and any(path.is_file() for path in tokenizer_assets)
    )


def _discover_local_htr_specs(settings: Settings) -> list[HTRProviderSpec]:
    """Discover locally installed, self-describing HTR bundles.

    A bundle cannot make capability claims through its filename alone.  It must
    include an explicit manifest and validation report.  Failed validation is
    still reported to model status, but remains non-routable.
    """
    root = settings.local_htr_model_directory.expanduser()
    if not root.is_dir():
        return []
    discovered: list[HTRProviderSpec] = []
    for manifest_path in sorted(root.glob("*/htr_manifest.json")):
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            model_subdirectory = str(raw.get("model_subdirectory") or ".")
            model_directory = (manifest_path.parent / model_subdirectory).resolve()
            if manifest_path.parent.resolve() not in {
                model_directory,
                *model_directory.parents,
            }:
                raise ValueError("model_subdirectory escapes the installed bundle")
            if not bool(raw.get("source_language_output_only", False)):
                raise ValueError("bundle does not declare source-language-only output")
            backend = str(raw.get("backend") or "").strip()
            if backend != "transformers_vision_encoder_decoder":
                raise ValueError(f"unsupported local HTR backend: {backend!r}")
            if not _model_assets_exist(model_directory):
                raise ValueError("bundle is missing model or processor assets")
            report_name = str(raw.get("validation_report") or "validation_report.json")
            report_path = manifest_path.parent / report_name
            report = json.loads(report_path.read_text(encoding="utf-8"))
            validation_passed = bool(report.get("passed", False))
            languages = tuple(
                dict.fromkeys(
                    normalize_language(str(value))
                    for value in raw.get("supported_languages", [])
                    if normalize_language(str(value)) != "und"
                )
            )
            scripts = tuple(
                script
                for script in dict.fromkeys(
                    _parse_script(value) for value in raw.get("supported_scripts", [])
                )
                if script != ScriptType.UNKNOWN
            )
            if not languages or not scripts:
                raise ValueError("bundle has no declared language/script capability")
            confidence = HTRConfidenceCapability(
                str(raw.get("confidence_capability", "none")).strip().casefold()
            )
            discovered.append(
                HTRProviderSpec(
                    provider_id=str(raw.get("provider_id") or manifest_path.parent.name),
                    backend=backend,
                    model_id=str(model_directory),
                    supported_languages=languages,
                    supported_scripts=scripts,
                    confidence_capability=confidence,
                    handwriting_validated=bool(
                        raw.get("handwriting_validated", False) and validation_passed
                    ),
                )
            )
        except Exception:
            LOGGER.warning(
                "Ignoring invalid local HTR bundle manifest=%s",
                manifest_path,
                exc_info=True,
            )
    return discovered


def discover_htr_provider_specs(settings: Settings) -> list[HTRProviderSpec]:
    """Validate configured providers without downloading or loading any model."""
    specs: list[HTRProviderSpec] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(settings.htr_providers or []):
        provider_id = str(raw.get("provider_id") or f"htr_provider_{index}").strip()
        if not provider_id or provider_id in seen_ids:
            raise ValueError(f"HTR provider IDs must be non-empty and unique: {provider_id!r}")
        seen_ids.add(provider_id)
        backend = str(raw.get("backend") or "transformers_vision_encoder_decoder").strip()
        model_value = raw.get("model_id")
        model_id = str(model_value).strip() if model_value not in {None, "", "null"} else None
        languages = tuple(
            dict.fromkeys(
                normalize_language(str(value))
                for value in raw.get("supported_languages", [])
                if normalize_language(str(value)) != "und"
            )
        )
        scripts = tuple(
            dict.fromkeys(_parse_script(value) for value in raw.get("supported_scripts", []))
        )
        scripts = tuple(script for script in scripts if script != ScriptType.UNKNOWN)
        try:
            confidence = HTRConfidenceCapability(
                str(raw.get("confidence_capability", "none")).strip().casefold()
            )
        except ValueError as exc:
            raise ValueError(
                f"Unsupported HTR confidence capability for {provider_id!r}"
            ) from exc
        if not languages or not scripts:
            raise ValueError(
                f"HTR provider {provider_id!r} must declare supported languages and scripts"
            )
        expected_scripts = {
            "pa": ScriptType.GURMUKHI,
            "hi": ScriptType.DEVANAGARI,
            "en": ScriptType.LATIN,
        }
        for language in languages:
            expected = expected_scripts.get(language)
            if expected is not None and expected not in scripts:
                raise ValueError(
                    f"HTR provider {provider_id!r} declares {language!r} without {expected.value!r} support"
                )
        if model_id and model_id.casefold() in _KNOWN_ENGLISH_ONLY_TROCR and languages != ("en",):
            raise ValueError(
                f"Generic English TrOCR checkpoint {model_id!r} cannot be declared as Punjabi/Hindi HTR"
            )
        output_mode = str(raw.get("output_mode", "source_transcription")).strip().casefold()
        if output_mode != "source_transcription":
            raise ValueError(
                f"HTR provider {provider_id!r} must output source transcriptions, not translations"
            )
        specs.append(
            HTRProviderSpec(
                provider_id=provider_id,
                backend=backend,
                model_id=model_id,
                supported_languages=languages,
                supported_scripts=scripts,
                confidence_capability=confidence,
                handwriting_validated=bool(raw.get("handwriting_validated", False)),
            )
        )
    positions = {spec.provider_id: index for index, spec in enumerate(specs)}
    for local_spec in _discover_local_htr_specs(settings):
        position = positions.get(local_spec.provider_id)
        if position is None:
            positions[local_spec.provider_id] = len(specs)
            specs.append(local_spec)
        elif not specs[position].configured:
            specs[position] = local_spec
    return specs


def _cache_directory_for(model_id: str) -> Path:
    safe = "models--" + model_id.replace("/", "--")
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE) / safe
    except ImportError:
        return Path.home() / ".cache" / "huggingface" / "hub" / safe


def model_location_status(model_id: str | None) -> str:
    """Report whether a model is a local path, cached, or requires an HF download."""
    if not model_id:
        return "unconfigured"
    path = Path(model_id).expanduser()
    if path.exists():
        return "local_path"
    if _cache_directory_for(model_id).exists():
        return "huggingface_cache"
    return "huggingface_download_required"


def select_source_script_candidate(
    candidates: list[dict[str, Any]], minimum_script_ratio: float
) -> dict[str, Any]:
    """Prefer a script-consistent beam before comparing model probability."""
    if not candidates:
        raise ValueError("At least one HTR candidate is required")
    consistent = [
        candidate
        for candidate in candidates
        if float(candidate.get("script_ratio") or 0.0) >= minimum_script_ratio
    ]
    eligible = consistent or candidates
    return max(
        eligible,
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            float(item.get("script_ratio") or 0.0),
            float(item.get("text_quality") or 0.0),
        ),
    )


class TransformersVisionEncoderDecoderHTRProvider(HandwritingRecognitionProvider):
    """Lazy line-level provider for explicitly capability-declared checkpoints."""

    def __init__(self, spec: HTRProviderSpec, settings: Settings, device: str) -> None:
        if not spec.model_id:
            raise ValueError(f"HTR provider {spec.provider_id!r} has no model_id")
        self.spec = spec
        self.settings = settings
        self.device = device
        self.model_id = spec.model_id
        self.capabilities = HTRProviderCapabilities(
            provider_id=spec.provider_id,
            backend=spec.backend,
            supported_languages=frozenset(spec.supported_languages),
            supported_scripts=frozenset(spec.supported_scripts),
            confidence_capability=spec.confidence_capability,
            handwriting_validated=spec.handwriting_validated,
        )
        self._processor: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        try:
            from transformers import AutoProcessor, VisionEncoderDecoderModel

            processor = AutoProcessor.from_pretrained(
                self.model_id,
                local_files_only=self.settings.hf_local_files_only,
            )
            model = VisionEncoderDecoderModel.from_pretrained(
                self.model_id,
                local_files_only=self.settings.hf_local_files_only,
            )
            model.to(self.device)
            model.eval()
            self._processor, self._model = processor, model
            return processor, model
        except Exception as exc:
            LOGGER.warning(
                "HTR provider failed to load provider=%s model=%s",
                self.capabilities.provider_id,
                self.model_id,
                exc_info=True,
            )
            raise RuntimeError(
                f"HTR provider unavailable: {self.capabilities.provider_id} ({self.model_id})"
            ) from exc

    def recognize_line(self, image: Image.Image) -> HTRPrediction:
        processor, model = self._load()
        import torch

        encoded = processor(images=image.convert("RGB"), return_tensors="pt")
        pixel_values = encoded.pixel_values.to(self.device)
        with torch.inference_mode():
            generated = model.generate(
                pixel_values,
                max_new_tokens=160,
                num_beams=4,
                num_return_sequences=3,
                return_dict_in_generate=True,
                output_scores=self.capabilities.returns_confidence,
            )
        texts = [
            text.strip()
            for text in processor.batch_decode(generated.sequences, skip_special_tokens=True)
        ]
        confidences: list[float | None] = [None] * len(texts)
        if self.capabilities.returns_confidence and generated.scores:
            try:
                transition = model.compute_transition_scores(
                    generated.sequences,
                    generated.scores,
                    generated.beam_indices,
                    normalize_logits=True,
                )
                for index, row in enumerate(transition):
                    valid = row[row < 0]
                    if valid.numel():
                        confidences[index] = max(
                            0.0, min(1.0, float(exp(float(valid.mean().item()))))
                        )
            except Exception:
                LOGGER.debug("Could not calculate HTR sequence confidence", exc_info=True)
        expected_scripts = tuple(self.capabilities.supported_scripts)
        candidates = [
            {
                "text": text,
                "confidence": confidence,
                "engine": self.capabilities.provider_id,
                "script_ratio": max(
                    (script_ratio(text, script) for script in expected_scripts),
                    default=0.0,
                ),
                "text_quality": max(
                    (calculate_text_quality(text, script) for script in expected_scripts),
                    default=0.0,
                ),
            }
            for text, confidence in zip(texts, confidences, strict=True)
            if text
        ]
        if not candidates:
            return HTRPrediction("", None, self.capabilities.provider_id, self.model_id)
        winner = select_source_script_candidate(
            candidates, self.settings.min_source_script_ratio
        )
        return HTRPrediction(
            text=str(winner["text"]),
            confidence=(
                float(winner["confidence"])
                if winner["confidence"] is not None
                else None
            ),
            provider_id=self.capabilities.provider_id,
            model_id=self.model_id,
            alternatives=[candidate for candidate in candidates if candidate is not winner],
        )


def build_htr_providers(
    settings: Settings, device: str
) -> tuple[list[HandwritingRecognitionProvider], list[HTRProviderSpec]]:
    """Instantiate configured local/HF providers while retaining unavailable specs."""
    specs = discover_htr_provider_specs(settings)
    providers: list[HandwritingRecognitionProvider] = []
    for spec in specs:
        if not spec.configured:
            continue
        if spec.backend != "transformers_vision_encoder_decoder":
            raise ValueError(
                f"Unsupported HTR provider backend {spec.backend!r} for {spec.provider_id!r}"
            )
        providers.append(TransformersVisionEncoderDecoderHTRProvider(spec, settings, device))
    return providers, specs


def provider_status_records(specs: Iterable[HTRProviderSpec]) -> list[dict[str, Any]]:
    return [
        {
            "provider_id": spec.provider_id,
            "backend": spec.backend,
            "model_id": spec.model_id,
            "supported_languages": list(spec.supported_languages),
            "supported_scripts": [script.value for script in spec.supported_scripts],
            "confidence_capability": spec.confidence_capability.value,
            "model_location": model_location_status(spec.model_id),
            "source_language_output_only": True,
            "handwriting_validated": spec.handwriting_validated,
        }
        for spec in specs
    ]
