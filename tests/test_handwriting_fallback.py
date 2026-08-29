from PIL import Image

from translator_app.config.settings import Settings
from translator_app.core.handwriting_ocr import TrOCRHandwritingEngine
from translator_app.schemas import (
    BoundingBox,
    ProcessingOptions,
    ReconstructionType,
    Region,
    RegionType,
    ProcessingStatus,
)


def test_unconfigured_punjabi_handwriting_is_preserved_and_marked_unreadable() -> None:
    settings = Settings(
        ocr_languages=["pan", "eng"], handwriting_models={"pa": None, "hi": None, "en": None}
    )
    engine = TrOCRHandwritingEngine(settings, "cpu")
    region = Region(
        1,
        BoundingBox(10, 10, 190, 60),
        RegionType.HANDWRITING,
        metadata={"pixel_bbox": [10, 10, 190, 60]},
    )
    options = ProcessingOptions(
        ocr_languages=["pan", "eng"],
        enable_handwriting_ocr=True,
        preserve_unreadable_handwriting_as_image=True,
    )
    result = engine.recognize_regions(
        Image.new("L", (200, 100), "white"),
        [region],
        {region.region_id: "pa"},
        page_number=1,
        options=options,
    )
    assert result.blocks[0].reconstruction_type == ReconstructionType.UNREADABLE
    assert result.blocks[0].is_handwritten
    assert result.blocks[0].metadata["preserve_region_as_image"] is True
    assert result.blocks[0].review_image_bytes is not None
    assert ProcessingStatus.HTR_UNAVAILABLE in result.blocks[0].processing_statuses
    assert region.preserve_as_image is True
    assert result.warnings
