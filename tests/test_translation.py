from translator_app.config.settings import Settings
from translator_app.core.translation_engine import (
    HuggingFaceNLLBProvider,
    TranslationProvider,
    TranslationService,
)
from translator_app.schemas import (
    BlockType,
    BoundingBox,
    TextBlock,
    TranslationStatus,
)


class FakeProvider(TranslationProvider):
    def __init__(self) -> None:
        self.calls = 0

    def supports(self, source_language: str, target_language: str = "en") -> bool:
        return source_language == "hi" and target_language == "en"

    def translate_batch(
        self, texts: list[str], source_language: str, target_language: str = "en"
    ) -> list[str]:
        self.calls += 1
        return [f"translated {text}" for text in texts]


class FailingProvider(FakeProvider):
    def translate_batch(
        self, texts: list[str], source_language: str, target_language: str = "en"
    ) -> list[str]:
        self.calls += 1
        raise RuntimeError("local model unavailable")


def make_block(text: str, language: str = "hi") -> TextBlock:
    block = TextBlock(1, BlockType.LINE, BoundingBox(0, 0, 200, 20), text)
    block.normalized_text = text
    block.detected_language = language
    return block


def test_translation_preserves_structured_tokens_and_caches() -> None:
    provider = FakeProvider()
    service = TranslationService(provider, Settings(ocr_languages=["eng"]))
    text = "संपर्क test@example.com या +91 98765 43210 दिनांक 24/08/2026"
    first, second = make_block(text), make_block(text)
    warnings = service.translate_blocks([first, second])
    assert not warnings
    assert provider.calls == 1
    assert "test@example.com" in first.english_translation
    assert "+91 98765 43210" in first.english_translation
    assert "24/08/2026" in first.english_translation
    assert first.translation_status == TranslationStatus.TRANSLATED


def test_english_is_preserved_without_provider_call() -> None:
    provider = FakeProvider()
    block = make_block("Already in English", "en")
    TranslationService(provider, Settings(ocr_languages=["eng"])).translate_blocks([block])
    assert provider.calls == 0
    assert block.english_translation == "Already in English"
    assert block.translation_status == TranslationStatus.NOT_REQUIRED


def test_distinct_blocks_are_batched() -> None:
    provider = FakeProvider()
    first, second = make_block("पहला वाक्य"), make_block("दूसरा वाक्य")
    TranslationService(provider, Settings(ocr_languages=["eng"])).translate_blocks(
        [first, second]
    )
    assert provider.calls == 1
    assert first.translation_status == TranslationStatus.TRANSLATED
    assert second.translation_status == TranslationStatus.TRANSLATED


def test_translation_failure_retains_source_and_is_flagged() -> None:
    provider = FailingProvider()
    source = "स्रोत पाठ"
    block = make_block(source)
    warnings = TranslationService(provider, Settings(ocr_languages=["eng"])).translate_blocks(
        [block]
    )
    assert warnings
    assert block.translation_status == TranslationStatus.FAILED
    assert block.output_text == source


def test_nllb_loader_reuses_pytorch_weights_without_background_conversion(
    monkeypatch,
) -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_options: dict[str, object] = {}

    class FakeModel:
        def to(self, device: str) -> "FakeModel":
            return self

        def eval(self) -> "FakeModel":
            return self

    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: object()),
    )

    def fake_model_loader(*args, **kwargs):
        model_options.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(
        AutoModelForSeq2SeqLM,
        "from_pretrained",
        staticmethod(fake_model_loader),
    )

    HuggingFaceNLLBProvider(Settings(ocr_languages=["eng"]), "cpu")

    assert model_options["use_safetensors"] is False
    assert model_options["low_cpu_mem_usage"] is True
