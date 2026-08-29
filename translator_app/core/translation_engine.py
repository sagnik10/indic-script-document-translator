"""Local translation provider interface, batching, caching, and token preservation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable

from ..config.settings import Settings
from ..exceptions import TranslationUnavailableError
from ..schemas import ProcessingStatus, TextBlock, TranslationStatus
from ..utils.text_utils import chunk_text, normalize_text, restore_structured_tokens
from .terminology import ProtectedText, TerminologyProtector
from .source_validation import (
    PageLanguageContext,
    SUPPORTED_TRANSLATION_LANGUAGES,
    is_translatable_block,
    normalize_language,
    validate_source_block,
)

LOGGER = logging.getLogger(__name__)

NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "pa": "pan_Guru",
    "bn": "ben_Beng",
    "as": "asm_Beng",
    "gu": "guj_Gujr",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "or": "ory_Orya",
    "ne": "npi_Deva",
    "ur": "urd_Arab",
}


class TranslationProvider(ABC):
    """Replaceable provider contract for local or explicitly configured translators."""

    @abstractmethod
    def supports(self, source_language: str, target_language: str = "en") -> bool:
        raise NotImplementedError

    @abstractmethod
    def translate_batch(
        self, texts: list[str], source_language: str, target_language: str = "en"
    ) -> list[str]:
        raise NotImplementedError


class HuggingFaceNLLBProvider(TranslationProvider):
    """Standard Transformers implementation for the open NLLB model family."""

    def __init__(self, settings: Settings, device: str) -> None:
        self.settings = settings
        self.device = device
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.translation_model,
                local_files_only=settings.hf_local_files_only,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                settings.translation_model,
                local_files_only=settings.hf_local_files_only,
                # The default NLLB snapshot ships PyTorch weights. Explicitly select
                # them so Transformers does not start a second background conversion
                # download for model.safetensors after pytorch_model.bin is cached.
                use_safetensors=False,
                low_cpu_mem_usage=True,
            )
            self.model.to(device)
            self.model.eval()
            # Generation is bounded with max_new_tokens below. Clear the
            # checkpoint's legacy max_length=200 to avoid conflicting-limit
            # warnings in recent Transformers releases.
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.max_length = None
        except Exception as exc:
            raise TranslationUnavailableError(
                f"Failed to load local translation model {settings.translation_model}"
            ) from exc

    def supports(self, source_language: str, target_language: str = "en") -> bool:
        return (
            normalize_language(source_language) in SUPPORTED_TRANSLATION_LANGUAGES
            and target_language == "en"
        )

    def translate_batch(
        self, texts: list[str], source_language: str, target_language: str = "en"
    ) -> list[str]:
        source_language = normalize_language(source_language)
        if not self.supports(source_language, target_language):
            raise TranslationUnavailableError(
                f"Unsupported language route: {source_language} -> {target_language}"
            )
        import torch

        self.tokenizer.src_lang = NLLB_LANGUAGE_CODES[source_language]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(
            NLLB_LANGUAGE_CODES[target_language]
        )
        try:
            with torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=512,
                    num_beams=4,
                )
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise TranslationUnavailableError(
                "The translation model ran out of memory; use CPU or a smaller model."
            ) from exc
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)


class TranslationService:
    """Translate contextual chunks once per session and map results back to blocks."""

    def __init__(
        self,
        provider: TranslationProvider,
        settings: Settings,
        terminology_protector: TerminologyProtector | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.terminology_protector = terminology_protector or TerminologyProtector()
        self.cache: dict[tuple[str, str, str], str] = {}

    @staticmethod
    def _protect_mixed_english(block: TextBlock, text: str) -> tuple[str, dict[str, str]]:
        spans = block.metadata.get("language_spans", [])
        if not block.metadata.get("mixed_language") or not spans:
            return text, {}
        mapping: dict[str, str] = {}
        protected_parts: list[str] = []
        for span in spans:
            value = str(span.get("text", ""))
            if span.get("language") == "en" and any(character.isalpha() for character in value):
                index = len(mapping)
                token = f"__dtxenglish_{'x' * (index // 26)}{chr(ord('a') + index % 26)}__"
                mapping[token] = value
                protected_parts.append(token)
            else:
                protected_parts.append(value)
        return "".join(protected_parts), mapping

    def _translate_one(self, block: TextBlock, target_language: str) -> str:
        text = block.effective_source_text
        mixed_text, english_mapping = self._protect_mixed_english(block, text)
        protected = self.terminology_protector.protect(mixed_text)
        block.protected_tokens = list(protected.source_tokens)
        chunks = chunk_text(protected.text, self.settings.max_model_input_characters)
        translated_chunks: list[str] = []
        missing_chunks: list[str] = []
        missing_indices: list[int] = []
        for index, chunk in enumerate(chunks):
            key = (block.detected_language, target_language, chunk)
            if key in self.cache:
                translated_chunks.append(self.cache[key])
            else:
                translated_chunks.append("")
                missing_chunks.append(chunk)
                missing_indices.append(index)
        if missing_chunks:
            translated = self.provider.translate_batch(
                missing_chunks, block.detected_language, target_language
            )
            if len(translated) != len(missing_chunks):
                raise TranslationUnavailableError("Translation provider returned an invalid batch size")
            for index, source_chunk, translated_chunk in zip(
                missing_indices, missing_chunks, translated, strict=True
            ):
                normalized = normalize_text(translated_chunk)
                translated_chunks[index] = normalized
                self.cache[(block.detected_language, target_language, source_chunk)] = normalized
        output = " ".join(translated_chunks)
        output = self.terminology_protector.restore(output, protected)
        output = restore_structured_tokens(output, english_mapping)
        return normalize_text(output)

    def _translate_block_batch(
        self, blocks: list[TextBlock], target_language: str
    ) -> None:
        """Translate all missing chunks for a language/block batch in one provider call."""
        prepared: list[tuple[TextBlock, list[str], ProtectedText, dict[str, str]]] = []
        missing: dict[tuple[str, str, str], str] = {}
        for block in blocks:
            text = block.effective_source_text
            mixed_text, english_mapping = self._protect_mixed_english(block, text)
            protected = self.terminology_protector.protect(mixed_text)
            block.protected_tokens = list(protected.source_tokens)
            chunks = chunk_text(protected.text, self.settings.max_model_input_characters)
            prepared.append((block, chunks, protected, english_mapping))
            for chunk in chunks:
                key = (block.detected_language, target_language, chunk)
                if key not in self.cache:
                    missing[key] = chunk
        if missing:
            keys = list(missing)
            translated = self.provider.translate_batch(
                [missing[key] for key in keys], blocks[0].detected_language, target_language
            )
            if len(translated) != len(keys):
                raise TranslationUnavailableError("Translation provider returned an invalid batch size")
            for key, translated_chunk in zip(keys, translated, strict=True):
                self.cache[key] = normalize_text(translated_chunk)
        for block, chunks, protected, english_mapping in prepared:
            output = " ".join(
                self.cache[(block.detected_language, target_language, chunk)] for chunk in chunks
            )
            output = self.terminology_protector.restore(output, protected)
            output = restore_structured_tokens(output, english_mapping)
            block.english_translation = normalize_text(output)
            block.translation_status = TranslationStatus.TRANSLATED

    def translate_blocks(
        self, blocks: Iterable[TextBlock], target_language: str = "en"
    ) -> list[str]:
        warnings: list[str] = []
        skipped: dict[str, int] = defaultdict(int)
        grouped: dict[str, list[TextBlock]] = defaultdict(list)
        for block in blocks:
            block.detected_language = normalize_language(block.detected_language)
            if not block.source_validated:
                validate_source_block(
                    block,
                    self.settings,
                    PageLanguageContext(block.script, block.language_confidence),
                )
            if not block.effective_source_text.strip():
                block.translation_status = TranslationStatus.SKIPPED
                skipped["empty source text"] += 1
            elif block.detected_language == "en" and block.source_validated:
                block.english_translation = block.effective_source_text
                block.translation_status = TranslationStatus.NOT_REQUIRED
            elif not is_translatable_block(block):
                block.translation_status = TranslationStatus.SKIPPED
                skipped[block.validation_reason or "source validation failed"] += 1
                if ProcessingStatus.TRANSLATION_SKIPPED not in block.processing_statuses:
                    block.processing_statuses.append(ProcessingStatus.TRANSLATION_SKIPPED)
            elif not self.provider.supports(block.detected_language, target_language):
                block.translation_status = TranslationStatus.SKIPPED
                skipped["translation route unavailable"] += 1
            else:
                grouped[block.detected_language].append(block)
        for language, language_blocks in grouped.items():
            for start in range(0, len(language_blocks), self.settings.default_translation_batch_size):
                batch = language_blocks[start : start + self.settings.default_translation_batch_size]
                try:
                    self._translate_block_batch(batch, target_language)
                except Exception:
                    LOGGER.warning(
                        "Translation batch failed for %s; retrying blocks independently",
                        language,
                        exc_info=True,
                    )
                    for block in batch:
                        try:
                            block.english_translation = self._translate_one(block, target_language)
                            block.translation_status = TranslationStatus.TRANSLATED
                        except Exception as exc:
                            block.translation_status = TranslationStatus.FAILED
                            warnings.append(
                                f"Translation failed for block {block.block_id}; source text was retained."
                            )
                            LOGGER.warning(
                                "Translation failure for document block %s (%s): %s",
                                block.block_id,
                                language,
                                type(exc).__name__,
                                exc_info=True,
                            )
        if skipped:
            count = sum(skipped.values())
            warnings.append(
                f"{count} region(s) did not pass source-language translation gates and were preserved for review."
            )
        return warnings
