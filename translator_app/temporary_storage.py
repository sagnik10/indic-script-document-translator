"""Task-scoped secure temporary storage with deterministic cleanup."""

from __future__ import annotations

import shutil
import tempfile
import logging
from pathlib import Path
from uuid import uuid4


LOGGER = logging.getLogger(__name__)


class SecureTemporaryStore:
    """Store page intermediates under an application-controlled random directory."""

    def __init__(self, base_directory: Path) -> None:
        self.base_directory = base_directory
        self.path: Path | None = None

    def __enter__(self) -> "SecureTemporaryStore":
        self.base_directory.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix="document_", dir=self.base_directory))
        try:
            self.path.chmod(0o700)
        except OSError as exc:
            # Windows ACLs are inherited from the private application directory.
            LOGGER.debug("POSIX permission mode is unavailable for temporary storage: %s", exc)
        return self

    def write_bytes(self, data: bytes, suffix: str) -> Path:
        if self.path is None:
            raise RuntimeError("Temporary store must be used as a context manager")
        safe_suffix = suffix.lower() if suffix.lower() in {".png", ".jpg", ".json", ".bin"} else ".bin"
        destination = self.path / f"{uuid4().hex}{safe_suffix}"
        with destination.open("xb") as stream:
            stream.write(data)
        return destination

    @staticmethod
    def read_bytes(path: Path) -> bytes:
        with path.open("rb") as stream:
            return stream.read()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)
            self.path = None
