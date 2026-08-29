from __future__ import annotations

import sys
import os
from types import SimpleNamespace

from translator_app.config.settings import Settings
from translator_app.core.ocr_engine import TesseractOCREngine
from translator_app.core.tesseract_runtime import discover_tesseract_runtime
from translator_app.exceptions import OCRUnavailableError
import pytest


class FakePytesseract:
    def __init__(self) -> None:
        self.pytesseract = SimpleNamespace(tesseract_cmd="tesseract")
        self.language_config = ""

    @staticmethod
    def get_tesseract_version() -> str:
        return "5.4.0"

    def get_languages(self, config: str = "") -> list[str]:
        self.language_config = config
        return ["eng", "hin", "pan"]


def test_runtime_uses_configured_executable_and_language_data(tmp_path) -> None:
    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"test executable")
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "eng.traineddata").write_bytes(b"language data")

    runtime = discover_tesseract_runtime(executable, tessdata)

    assert runtime is not None
    assert runtime.executable == executable.resolve()
    assert runtime.tessdata_directory == tessdata.resolve()
    assert "--tessdata-dir" in runtime.tessdata_config


def test_engine_passes_discovered_tessdata_to_language_query(
    tmp_path, monkeypatch
) -> None:
    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"test executable")
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "eng.traineddata").write_bytes(b"language data")
    fake = FakePytesseract()
    monkeypatch.setitem(sys.modules, "pytesseract", fake)

    engine = TesseractOCREngine(
        Settings(
            ocr_languages=["eng", "hin", "pan"],
            tesseract_cmd=str(executable),
            tessdata_directory=tessdata,
        )
    )
    selected, warnings = engine._select_languages(["eng", "hin", "pan"])

    assert fake.pytesseract.tesseract_cmd == str(executable.resolve())
    assert selected == ["eng", "hin", "pan"]
    assert not warnings
    assert fake.language_config == ""
    assert os.environ["TESSDATA_PREFIX"] == str(tessdata.resolve())


def test_missing_punjabi_pack_does_not_silently_fall_back_to_english() -> None:
    engine = object.__new__(TesseractOCREngine)
    fake = FakePytesseract()
    fake.get_languages = lambda config="": ["eng"]
    engine.pytesseract = fake
    with pytest.raises(OCRUnavailableError, match="pan"):
        engine._select_languages(["pan"])
