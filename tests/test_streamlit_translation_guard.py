from types import SimpleNamespace

from translator_app.schemas import ProcessingStatus
from translator_app.ui.app import (
    _requires_translated_output,
    _validated_indic_block_count,
)


def _analysis(*blocks: object) -> SimpleNamespace:
    return SimpleNamespace(document=SimpleNamespace(blocks=list(blocks)))


def _block(
    *,
    language: str = "und",
    validated: bool = False,
    statuses: list[ProcessingStatus] | None = None,
    metadata: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        block_id="block",
        detected_language=language,
        source_validated=validated,
        processing_statuses=statuses or [],
        metadata=metadata or {},
    )


def test_unresolved_htr_requires_a_real_translation_before_export() -> None:
    block = _block(
        language="pa",
        statuses=[ProcessingStatus.HTR_UNAVAILABLE],
        metadata={"htr_unavailable": True},
    )

    assert _requires_translated_output(_analysis(block))


def test_validated_indic_source_requires_a_real_translation_before_export() -> None:
    assert _requires_translated_output(_analysis(_block(language="pan_Guru", validated=True)))
    assert _requires_translated_output(_analysis(_block(language="hin_Deva", validated=True)))


def test_english_only_source_does_not_trigger_indic_translation_guard() -> None:
    assert not _requires_translated_output(_analysis(_block(language="en", validated=True)))


def test_only_validated_indic_source_allows_translation_attempt() -> None:
    analysis = _analysis(
        _block(language="pa", validated=False),
        _block(language="und", validated=True),
        _block(language="en", validated=True),
        _block(language="pan_Guru", validated=True),
    )

    assert _validated_indic_block_count(analysis) == 1
