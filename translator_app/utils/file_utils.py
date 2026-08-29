"""Safe filename and temporary-workspace helpers."""

from __future__ import annotations

import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str, *, fallback: str = "document") -> str:
    """Return a basename safe for local output without path traversal."""
    name = Path(filename.replace("\\", "/")).name.strip().strip(".")
    stem = _UNSAFE_FILENAME.sub("_", Path(name).stem).strip("._") or fallback
    suffix = _UNSAFE_FILENAME.sub("", Path(name).suffix.lower())
    return f"{stem[:100]}{suffix[:10]}"


def translated_output_name(filename: str, extension: str) -> str:
    safe = sanitize_filename(filename)
    return f"{Path(safe).stem}_translated_en.{extension.lstrip('.').lower()}"


@contextmanager
def private_temporary_directory(base_directory: Path) -> Iterator[Path]:
    """Create and best-effort delete a task-specific local temporary directory."""
    base_directory.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="dtx_", dir=base_directory))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def safe_write_new(path: Path, data: bytes) -> Path:
    """Write bytes without overwriting an existing file, adding a numeric suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1
    with candidate.open("xb") as stream:
        stream.write(data)
    return candidate

