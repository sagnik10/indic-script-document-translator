"""Central lazy-loading model registry with automatic CPU/CUDA selection."""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any

from ..config.settings import Settings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelStatus:
    name: str
    loaded: bool = False
    available: bool = True
    device: str = "cpu"
    detail: str = "Not loaded"


class ModelManager:
    """Load each heavy model at most once and report its non-sensitive status."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._select_device(settings.device)
        self._models: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._status = {
            name: ModelStatus(name=name, device=self.device)
            for name in (
                "ocr",
                "printed_ocr",
                "handwriting",
                "layout_detection",
                "visual_script",
                "page_ocr",
                "translation",
                "language_detection",
                "reconstruction",
            )
        }

    @staticmethod
    def _select_device(preference: str) -> str:
        if preference.lower() == "cpu":
            return "cpu"
        try:
            import torch

            if preference.lower().startswith("cuda") and not torch.cuda.is_available():
                LOGGER.warning("CUDA was requested but unavailable; using CPU")
                return "cpu"
            if preference.lower() == "auto":
                return "cuda" if torch.cuda.is_available() else "cpu"
            return preference.lower()
        except ImportError:
            return "cpu"

    def _load(self, name: str, factory: Any) -> Any:
        with self._lock:
            if name in self._models:
                return self._models[name]
            try:
                model = factory()
                self._models[name] = model
                self._status[name].loaded = True
                self._status[name].detail = "Loaded"
                return model
            except Exception as exc:
                self._status[name].available = False
                self._status[name].detail = f"Unavailable: {type(exc).__name__}"
                raise

    def get_ocr_engine(self) -> Any:
        if self.settings.ocr_engine.lower() != "tesseract":
            raise ValueError(f"Unsupported OCR engine: {self.settings.ocr_engine}")
        from ..core.ocr_engine import TesseractOCREngine

        return self._load("ocr", lambda: TesseractOCREngine(self.settings))

    def get_language_detector(self) -> Any:
        from ..core.language_detector import ScriptAwareLanguageDetector

        return self._load("language_detection", ScriptAwareLanguageDetector)

    def get_layout_detector(self) -> Any:
        from ..core.layout_detection import DocumentLayoutDetector

        return self._load(
            "layout_detection", lambda: DocumentLayoutDetector(self.settings)
        )

    def get_visual_script_classifier(self) -> Any:
        from ..core.visual_routing import HeuristicVisualScriptClassifier

        classifier = self._load(
            "visual_script", lambda: HeuristicVisualScriptClassifier(self.settings)
        )
        self._status["visual_script"].detail = (
            "Local visual morphology classifier loaded; heuristic confidence is auditable"
        )
        return classifier

    def get_printed_ocr(self) -> Any:
        from ..core.printed_ocr import PrintedTextOCR

        return self._load(
            "printed_ocr",
            lambda: PrintedTextOCR(self.settings, self.get_ocr_engine),
        )

    def get_handwriting_engine(self) -> Any:
        if self.settings.handwriting_engine.lower() not in {"trocr", "providers", "auto"}:
            raise ValueError(
                f"Unsupported handwriting engine: {self.settings.handwriting_engine}"
            )
        from ..core.handwriting_ocr import TrOCRHandwritingEngine

        engine = self._load(
            "handwriting",
            lambda: TrOCRHandwritingEngine(self.settings, self.device),
        )
        records = engine.provider_status()
        configured = sum(record.get("model_location") != "unconfigured" for record in records)
        gurmukhi = any(
            "gurmukhi" in record.get("supported_scripts", [])
            and record.get("model_location") != "unconfigured"
            and bool(record.get("handwriting_validated"))
            for record in records
        )
        self._status["handwriting"].detail = (
            f"Provider router loaded; {configured}/{len(records)} model(s) configured; "
            f"Gurmukhi HTR {'configured' if gurmukhi else 'unavailable (manual review fallback)'}"
        )
        return engine

    def get_page_ocr_pipeline(self) -> Any:
        from ..core.page_ocr import PageOCRPipeline

        return self._load(
            "page_ocr",
            lambda: PageOCRPipeline(
                self.get_layout_detector(),
                self.get_printed_ocr(),
                self.get_handwriting_engine(),
                self.get_language_detector(),
                self.get_visual_script_classifier(),
            ),
        )

    def get_translation_provider(self) -> Any:
        if self.settings.translation_provider.lower() != "huggingface":
            raise ValueError(
                f"Unsupported translation provider: {self.settings.translation_provider}"
            )
        from ..core.translation_engine import HuggingFaceNLLBProvider

        return self._load(
            "translation", lambda: HuggingFaceNLLBProvider(self.settings, self.device)
        )

    def get_reconstruction_model(self) -> Any | None:
        if not self.settings.reconstruction_model:
            self._status["reconstruction"].available = False
            self._status["reconstruction"].detail = "Disabled; conservative rules only"
            return None

        def factory() -> Any:
            from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

            pipeline_device = 0 if self.device.startswith("cuda") else -1
            tokenizer = AutoTokenizer.from_pretrained(
                self.settings.reconstruction_model,
                local_files_only=self.settings.hf_local_files_only,
            )
            model = AutoModelForMaskedLM.from_pretrained(
                self.settings.reconstruction_model,
                local_files_only=self.settings.hf_local_files_only,
            )
            return pipeline(
                "fill-mask",
                model=model,
                tokenizer=tokenizer,
                device=pipeline_device,
            )

        return self._load("reconstruction", factory)

    def get_source_reconstruction_provider(self) -> Any:
        """Return a lightweight source-span adapter; its heavy model remains lazy."""
        from ..core.source_reconstruction import LocalMaskedLanguageModelProvider

        return LocalMaskedLanguageModelProvider(self.get_reconstruction_model)

    def status(self) -> dict[str, dict[str, Any]]:
        payload = {name: asdict(status) for name, status in self._status.items()}
        handwriting = self._models.get("handwriting")
        if handwriting is not None and hasattr(handwriting, "provider_status"):
            payload["handwriting"]["providers"] = handwriting.provider_status()
        return payload
