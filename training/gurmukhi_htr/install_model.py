"""Validate and install a trained Gurmukhi HTR bundle atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = PROJECT_ROOT / ".runtime" / "models" / "gurmukhi_htr"


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def safe_extract(archive: Path, destination: Path) -> None:
    with ZipFile(archive) as source:
        for member in source.infolist():
            if not _inside(destination / member.filename, destination):
                raise ValueError(f"Unsafe model archive member: {member.filename}")
        source.extractall(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_bundle_root(extracted: Path) -> Path:
    manifests = list(extracted.rglob("htr_manifest.json"))
    if len(manifests) != 1:
        raise ValueError("The archive must contain exactly one htr_manifest.json")
    return manifests[0].parent


def validate_bundle(bundle: Path) -> dict[str, object]:
    manifest = json.loads((bundle / "htr_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((bundle / "validation_report.json").read_text(encoding="utf-8"))
    if manifest.get("backend") != "transformers_vision_encoder_decoder":
        raise ValueError("Unsupported HTR backend")
    if manifest.get("source_language_output_only") is not True:
        raise ValueError("Direct image-to-English models cannot be installed")
    if "pa" not in manifest.get("supported_languages", []):
        raise ValueError("The bundle does not declare Punjabi support")
    if "gurmukhi" not in manifest.get("supported_scripts", []):
        raise ValueError("The bundle does not declare Gurmukhi support")
    if manifest.get("handwriting_validated") is not True or report.get("passed") is not True:
        raise ValueError("The held-out HTR validation gates did not pass")
    required = ("config.json", "preprocessor_config.json", "tokenizer_config.json")
    missing = [name for name in required if not (bundle / name).is_file()]
    weights = [
        bundle / "model.safetensors",
        bundle / "pytorch_model.bin",
        bundle / "model.safetensors.index.json",
        bundle / "pytorch_model.bin.index.json",
    ]
    if missing or not any(path.is_file() for path in weights):
        raise ValueError(f"Incomplete model bundle; missing={missing or ['model weights']}")
    checksum_path = bundle / "checksums.json"
    if checksum_path.is_file():
        expected = json.loads(checksum_path.read_text(encoding="utf-8"))
        mismatches = [
            name
            for name, digest in expected.items()
            if not (bundle / name).is_file() or sha256(bundle / name) != digest
        ]
        if mismatches:
            raise ValueError(f"Model bundle checksum mismatch: {', '.join(mismatches)}")
    return manifest


def install(archive: Path, target: Path = DEFAULT_TARGET) -> Path:
    archive = archive.expanduser().resolve()
    target = target.expanduser().resolve()
    if not archive.is_file() or archive.suffix.casefold() != ".zip":
        raise FileNotFoundError(f"Model ZIP was not found: {archive}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gurmukhi-htr-install-") as temporary:
        extracted = Path(temporary) / "extracted"
        extracted.mkdir()
        safe_extract(archive, extracted)
        bundle = find_bundle_root(extracted)
        validate_bundle(bundle)
        staged = Path(temporary) / target.name
        shutil.copytree(bundle, staged)
        validate_bundle(staged)
        if target.exists():
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup = target.with_name(f"{target.name}.backup-{timestamp}")
            target.replace(backup)
            print(f"Previous model preserved at: {backup}")
        shutil.move(str(staged), str(target))
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    installed = install(args.archive, args.target)
    print(f"Validated Gurmukhi HTR model installed at: {installed}")
    print("Restart Streamlit; the model will be discovered automatically.")


if __name__ == "__main__":
    main()
