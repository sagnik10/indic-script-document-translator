"""Discover a local Tesseract executable and language-data directory."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..config.settings import PROJECT_ROOT


@dataclass(frozen=True, slots=True)
class TesseractRuntime:
    executable: Path
    tessdata_directory: Path | None = None

    @property
    def tessdata_config(self) -> str:
        if self.tessdata_directory is None:
            return ""
        return f'--tessdata-dir "{self.tessdata_directory}"'


def _existing_file(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_file() else None


def _language_data_directory(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_dir() and any(path.glob("*.traineddata")):
        return path.resolve()
    nested = path / "tessdata"
    if nested.is_dir() and any(nested.glob("*.traineddata")):
        return nested.resolve()
    return None


def discover_tesseract_runtime(
    configured_executable: str | Path | None = None,
    configured_tessdata: str | Path | None = None,
) -> TesseractRuntime | None:
    """Resolve configured, PATH, and common Windows Tesseract locations."""
    candidates: list[str | Path | None] = [configured_executable, shutil.which("tesseract")]
    if os.name == "nt":
        candidates.extend(
            [
                Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
                / "Tesseract-OCR"
                / "tesseract.exe",
                Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
                / "Tesseract-OCR"
                / "tesseract.exe",
                Path(os.environ.get("LOCALAPPDATA", ""))
                / "Programs"
                / "Tesseract-OCR"
                / "tesseract.exe",
            ]
        )
    executable = next(
        (resolved for candidate in candidates if (resolved := _existing_file(candidate))),
        None,
    )
    if executable is None:
        return None

    tessdata_candidates: list[str | Path | None] = [
        configured_tessdata,
        os.getenv("TESSDATA_PREFIX"),
        PROJECT_ROOT / ".runtime" / "tessdata",
        executable.parent / "tessdata",
    ]
    tessdata = next(
        (
            resolved
            for candidate in tessdata_candidates
            if (resolved := _language_data_directory(candidate))
        ),
        None,
    )
    return TesseractRuntime(executable, tessdata)


def configure_pytesseract(pytesseract_module: object, runtime: TesseractRuntime) -> None:
    """Point pytesseract at the discovered executable without changing system PATH."""
    pytesseract_module.pytesseract.tesseract_cmd = str(runtime.executable)  # type: ignore[attr-defined]
    if runtime.tessdata_directory is not None:
        # pytesseract's Windows config parser can retain quote characters in
        # --tessdata-dir paths. The native environment variable handles spaces.
        os.environ["TESSDATA_PREFIX"] = str(runtime.tessdata_directory)
