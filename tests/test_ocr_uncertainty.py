from pathlib import Path

from PIL import Image

from translator_app.config.settings import Settings
from translator_app.core.ocr_engine import TesseractOCREngine
from translator_app.core.tesseract_runtime import TesseractRuntime
from translator_app.schemas import UncertaintyState


class FakeOutput:
    DICT = "dict"


class FakePytesseract:
    Output = FakeOutput

    @staticmethod
    def get_languages(config=""):
        return ["eng", "hin"]

    @staticmethod
    def image_to_data(*args, **kwargs):
        return {
            "text": ["नमस्ते", "दुनिया"],
            "conf": ["90", "30"],
            "left": [10, 55],
            "top": [20, 20],
            "width": [40, 45],
            "height": [15, 15],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 1],
            "word_num": [1, 2],
        }


def test_ocr_returns_coordinates_relationships_and_uncertainty() -> None:
    engine = object.__new__(TesseractOCREngine)
    engine.settings = Settings(ocr_languages=["eng", "hin"])
    engine.pytesseract = FakePytesseract()
    engine.runtime = TesseractRuntime(Path("tesseract"))
    result = engine.recognize(
        Image.new("RGB", (200, 100), "white"),
        page_number=2,
        page_width=100,
        page_height=50,
        requested_languages=["eng", "hin", "pan"],
        low_confidence_threshold=0.65,
    )
    assert len(result.blocks) == 1
    line = result.blocks[0]
    assert line.page_number == 2
    assert line.parent_block_id
    assert len(line.metadata["words"]) == 2
    assert line.source_bbox.x1 == 50
    assert line.uncertainty_state == UncertaintyState.LOW_OCR_CONFIDENCE
    assert any("pan" in warning for warning in result.warnings)
