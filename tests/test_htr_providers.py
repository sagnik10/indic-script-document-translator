from __future__ import annotations

from PIL import Image, ImageDraw
import pytest

from translator_app.config.settings import Settings
from translator_app.core.handwriting_ocr import TrOCRHandwritingEngine
from translator_app.core.htr_providers import (
    HTRConfidenceCapability,
    HTRPrediction,
    HTRProviderCapabilities,
    HandwritingRecognitionProvider,
    discover_htr_provider_specs,
)
from translator_app.core.source_validation import PageLanguageContext, validate_source_block
from translator_app.schemas import (
    BoundingBox,
    ProcessingOptions,
    ProcessingStatus,
    Region,
    RegionType,
    ScriptType,
)


class FakeGurmukhiHTR(HandwritingRecognitionProvider):
    model_id = "local/fake-gurmukhi-line-htr"
    capabilities = HTRProviderCapabilities(
        provider_id="fake_gurmukhi_htr",
        backend="test",
        supported_languages=frozenset({"pa"}),
        supported_scripts=frozenset({ScriptType.GURMUKHI}),
        confidence_capability=HTRConfidenceCapability.SEQUENCE_PROBABILITY,
    )

    def recognize_line(self, image: Image.Image) -> HTRPrediction:
        return HTRPrediction(
            "ਇਹ ਪੰਜਾਬੀ ਹੱਥ ਲਿਖਤ ਹੈ",
            0.94,
            self.capabilities.provider_id,
            self.model_id,
        )


def _handwriting_fixture() -> tuple[Image.Image, Region]:
    image = Image.new("L", (260, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.line((18, 45, 235, 40), fill="black", width=3)
    draw.line((30, 52, 220, 55), fill="black", width=2)
    region = Region(
        1,
        BoundingBox(10, 15, 250, 75),
        RegionType.HANDWRITING,
        metadata={"pixel_bbox": [10, 15, 250, 75], "dominant_script": "gurmukhi"},
    )
    return image, region


def test_capability_matched_gurmukhi_htr_returns_traceable_source_line() -> None:
    settings = Settings(htr_providers=[])
    engine = TrOCRHandwritingEngine(settings, "cpu", providers=[FakeGurmukhiHTR()])
    image, region = _handwriting_fixture()
    result = engine.recognize_regions(
        image,
        [region],
        {region.region_id: "pa"},
        page_number=1,
        options=ProcessingOptions(
            enable_handwriting_ocr=True,
            handwriting_confidence_threshold=0.55,
        ),
    )
    block = result.blocks[0]
    assert block.source_text == "ਇਹ ਪੰਜਾਬੀ ਹੱਥ ਲਿਖਤ ਹੈ"
    assert block.review_image_bytes and block.review_image_bytes.startswith(b"\x89PNG")
    assert ProcessingStatus.HTR_RECOGNIZED in block.processing_statuses
    assert block.metadata["source_origin"] == "htr"
    assert block.metadata["htr_source_language_output_only"] is True
    validation = validate_source_block(
        block,
        settings,
        PageLanguageContext(ScriptType.GURMUKHI, 0.95, 0.9),
    )
    assert validation.valid
    assert block.detected_language == "pa"


def test_generic_english_trocr_cannot_be_declared_as_punjabi() -> None:
    settings = Settings(
        htr_providers=[
            {
                "provider_id": "unsafe",
                "backend": "transformers_vision_encoder_decoder",
                "model_id": "microsoft/trocr-base-handwritten",
                "supported_languages": ["pa"],
                "supported_scripts": ["gurmukhi"],
                "confidence_capability": "sequence_probability",
            }
        ]
    )
    with pytest.raises(ValueError, match="Generic English TrOCR"):
        discover_htr_provider_specs(settings)


def test_provider_configuration_rejects_direct_image_translation() -> None:
    settings = Settings(
        htr_providers=[
            {
                "provider_id": "unsafe_translation",
                "backend": "transformers_vision_encoder_decoder",
                "model_id": "local/model",
                "supported_languages": ["pa"],
                "supported_scripts": ["gurmukhi"],
                "confidence_capability": "none",
                "output_mode": "english_translation",
            }
        ]
    )
    with pytest.raises(ValueError, match="source transcriptions"):
        discover_htr_provider_specs(settings)


def test_unconfigured_gurmukhi_provider_preserves_each_line_crop() -> None:
    settings = Settings(
        htr_providers=[
            {
                "provider_id": "gurmukhi_htr",
                "backend": "transformers_vision_encoder_decoder",
                "model_id": None,
                "supported_languages": ["pa"],
                "supported_scripts": ["gurmukhi"],
                "confidence_capability": "sequence_probability",
            }
        ]
    )
    engine = TrOCRHandwritingEngine(settings, "cpu")
    image, region = _handwriting_fixture()
    result = engine.recognize_regions(
        image,
        [region],
        {region.region_id: "pa"},
        page_number=1,
        options=ProcessingOptions(enable_handwriting_ocr=True),
    )
    assert result.blocks
    assert all(block.review_image_bytes for block in result.blocks)
    assert all(
        ProcessingStatus.HTR_UNAVAILABLE in block.processing_statuses
        for block in result.blocks
    )
    assert all(block.metadata["preserve_region_as_image"] for block in result.blocks)
    assert len(result.warnings) == 1


def test_unvalidated_gurmukhi_checkpoint_is_discoverable_but_not_routable() -> None:
    settings = Settings(
        htr_providers=[
            {
                "provider_id": "unvalidated_gurmukhi",
                "backend": "transformers_vision_encoder_decoder",
                "model_id": "local/not-evaluated-yet",
                "supported_languages": ["pa"],
                "supported_scripts": ["gurmukhi"],
                "confidence_capability": "sequence_probability",
                "handwriting_validated": False,
            }
        ]
    )
    engine = TrOCRHandwritingEngine(settings, "cpu")
    image, region = _handwriting_fixture()
    result = engine.recognize_regions(
        image,
        [region],
        {region.region_id: "pa"},
        page_number=1,
        options=ProcessingOptions(enable_handwriting_ocr=True),
    )
    assert ProcessingStatus.HTR_UNAVAILABLE in result.blocks[0].processing_statuses
    assert engine.provider_status()[0]["handwriting_validated"] is False
