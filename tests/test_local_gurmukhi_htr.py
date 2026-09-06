from __future__ import annotations

import json
from pathlib import Path

from translator_app.config.settings import Settings
from translator_app.core.htr_providers import (
    discover_htr_provider_specs,
    select_source_script_candidate,
)
from translator_app.core.source_validation import PageLanguageContext, validate_source_block
from translator_app.schemas import BlockType, BoundingBox, ScriptType, TextBlock


def _write_bundle(root: Path, *, passed: bool) -> Path:
    bundle = root / "gurmukhi_htr"
    bundle.mkdir(parents=True)
    for name in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "model.safetensors",
    ):
        (bundle / name).write_bytes(b"{}" if name.endswith(".json") else b"weights")
    (bundle / "validation_report.json").write_text(
        json.dumps({"passed": passed, "metrics": {"cer": 0.2}}), encoding="utf-8"
    )
    (bundle / "htr_manifest.json").write_text(
        json.dumps(
            {
                "provider_id": "gurmukhi_htr",
                "backend": "transformers_vision_encoder_decoder",
                "supported_languages": ["pa"],
                "supported_scripts": ["gurmukhi"],
                "confidence_capability": "sequence_probability",
                "source_language_output_only": True,
                "handwriting_validated": True,
                "validation_report": "validation_report.json",
            }
        ),
        encoding="utf-8",
    )
    return bundle


def test_validated_local_bundle_replaces_unconfigured_provider(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, passed=True)
    settings = Settings(
        local_htr_model_directory=tmp_path,
        htr_providers=[
            {
                "provider_id": "gurmukhi_htr",
                "model_id": None,
                "supported_languages": ["pa"],
                "supported_scripts": ["gurmukhi"],
                "confidence_capability": "sequence_probability",
                "handwriting_validated": False,
            }
        ],
    )
    spec = discover_htr_provider_specs(settings)[0]
    assert spec.model_id == str(bundle.resolve())
    assert spec.handwriting_validated is True


def test_failed_local_validation_remains_non_routable(tmp_path: Path) -> None:
    _write_bundle(tmp_path, passed=False)
    settings = Settings(local_htr_model_directory=tmp_path, htr_providers=[])
    spec = discover_htr_provider_specs(settings)[0]
    assert spec.configured
    assert spec.handwriting_validated is False


def test_script_consistent_beam_wins_over_higher_probability_latin_beam() -> None:
    winner = select_source_script_candidate(
        [
            {"text": "What is it?", "confidence": 0.99, "script_ratio": 0.0, "text_quality": 0.9},
            {
                "text": "\u0a07\u0a39 \u0a2a\u0a70\u0a1c\u0a3e\u0a2c\u0a40 \u0a39\u0a48",
                "confidence": 0.78,
                "script_ratio": 1.0,
                "text_quality": 0.9,
            },
        ],
        0.55,
    )
    assert winner["text"].startswith("\u0a07\u0a39")


def test_handwriting_uses_handwriting_confidence_threshold() -> None:
    text = "\u0a07\u0a39 \u0a2a\u0a70\u0a1c\u0a3e\u0a2c\u0a40 \u0a32\u0a3f\u0a16\u0a24 \u0a39\u0a48"
    block = TextBlock(1, BlockType.LINE, BoundingBox(0, 0, 200, 30), text)
    block.normalized_text = text
    block.detected_language = "pa"
    block.script = ScriptType.GURMUKHI
    block.ocr_confidence = 0.60
    block.is_ocr = True
    block.is_handwritten = True
    validation = validate_source_block(
        block,
        Settings(
            ocr_low_confidence_threshold=0.65,
            handwriting_confidence_threshold=0.55,
        ),
        PageLanguageContext(ScriptType.GURMUKHI, 0.95, 0.9),
    )
    assert validation.valid
    assert block.source_validated
