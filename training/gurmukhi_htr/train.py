"""Fine-tune a source-language Gurmukhi TrOCR model on a free Kaggle GPU.

The official IIIT data is word-level.  This trainer retains word samples and
also composes deterministic multi-word lines so the exported checkpoint matches
the application's line-level HTR contract.  The artifact is never marked
validated unless held-out CER, Unicode-script purity, and non-empty coverage
all pass their configured gates.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import shutil
import unicodedata
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageEnhance, ImageOps


TRAINING_ARCHIVE_URL = (
    "https://ilocr.iiit.ac.in/ihtr/assets/dataset/trainingset/gurumukhi.zip"
)
BASE_MODEL = "microsoft/trocr-small-stage1"
GURMUKHI_START = 0x0A00
GURMUKHI_END = 0x0A7F


@dataclass(frozen=True, slots=True)
class Sample:
    image_path: Path
    text: str


def normalize_source_text(value: str) -> str:
    """Normalize spacing and Unicode without changing source-language content."""
    return " ".join(unicodedata.normalize("NFC", value).split())


def gurmukhi_ratio(value: str) -> float:
    letters = [character for character in value if character.isalpha()]
    if not letters:
        return 0.0
    return sum(GURMUKHI_START <= ord(character) <= GURMUKHI_END for character in letters) / len(
        letters
    )


def _within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a public dataset archive without permitting path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = destination / member.filename
            if not _within(target, destination):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        source.extractall(destination)


def download_with_resume(url: str, destination: Path) -> Path:
    """Download once; completed archives are reused across notebook reruns."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "OCRModelApp-HTR/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(destination)
    return destination


def _resolve_image(
    relative_name: str,
    label_file: Path,
    data_root: Path,
    basename_index: dict[str, Path] | None,
) -> tuple[Path | None, dict[str, Path] | None]:
    normalized = relative_name.replace("\\", "/").lstrip("./")
    candidates = (label_file.parent / normalized, data_root / normalized)
    for candidate in candidates:
        if candidate.is_file():
            return candidate, basename_index
    if basename_index is None:
        basename_index = {
            path.name: path
            for path in data_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
        }
    return basename_index.get(Path(normalized).name), basename_index


def load_samples(data_root: Path) -> list[Sample]:
    """Load the official ``path transcription`` manifest defensively."""
    label_files = sorted(data_root.rglob("train.txt"))
    if not label_files:
        label_files = sorted(
            path
            for path in data_root.rglob("*.txt")
            if "train" in path.name.casefold()
        )
    if not label_files:
        raise FileNotFoundError(f"No train.txt was found below {data_root}")
    preferred = sorted(
        label_files,
        key=lambda path: (
            "gur" not in path.as_posix().casefold() and "pun" not in path.as_posix().casefold(),
            len(path.parts),
        ),
    )[0]
    samples: list[Sample] = []
    basename_index: dict[str, Path] | None = None
    for raw_line in preferred.read_text(encoding="utf-8-sig").splitlines():
        parts = raw_line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        image_path, basename_index = _resolve_image(
            parts[0], preferred, data_root, basename_index
        )
        text = normalize_source_text(parts[1])
        if image_path and text and gurmukhi_ratio(text) >= 0.70:
            samples.append(Sample(image_path, text))
    if len(samples) < 1_000:
        raise ValueError(
            f"Only {len(samples)} valid Gurmukhi samples were found; refusing to train"
        )
    return samples


def split_samples(
    samples: Sequence[Sample], validation_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(500, min(3_000, round(len(shuffled) * validation_fraction)))
    return shuffled[validation_size:], shuffled[:validation_size]


def _trim_word(image: Image.Image) -> Image.Image:
    gray = ImageOps.autocontrast(image.convert("L"))
    inverted = ImageOps.invert(gray)
    bbox = inverted.point(lambda value: 255 if value > 18 else 0).getbbox()
    return gray.crop(bbox) if bbox else gray


def compose_line(
    samples: Sequence[Sample],
    indices: Sequence[int],
    rng: random.Random,
    *,
    augment: bool,
) -> tuple[Image.Image, str]:
    words: list[Image.Image] = []
    labels: list[str] = []
    target_height = rng.randint(52, 72)
    for index in indices:
        sample = samples[index]
        with Image.open(sample.image_path) as source:
            word = _trim_word(source)
        scale = target_height / max(1, word.height)
        width = max(8, round(word.width * scale))
        word = word.resize((width, target_height), Image.Resampling.LANCZOS)
        words.append(word)
        labels.append(sample.text)
    gaps = [rng.randint(12, 30) for _ in range(max(0, len(words) - 1))]
    margin = 16
    canvas_width = sum(word.width for word in words) + sum(gaps) + 2 * margin
    canvas = Image.new("L", (max(64, canvas_width), target_height + 2 * margin), 255)
    x = margin
    for position, word in enumerate(words):
        y = margin + rng.randint(-3, 3)
        canvas.paste(word, (x, y))
        x += word.width + (gaps[position] if position < len(gaps) else 0)
    if augment:
        canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(0.78, 1.22))
        canvas = canvas.rotate(
            rng.uniform(-1.5, 1.5),
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=255,
        )
    return canvas.convert("RGB"), " ".join(labels)


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def corpus_character_error_rate(predictions: Iterable[str], references: Iterable[str]) -> float:
    errors = 0
    characters = 0
    for prediction, reference in zip(predictions, references, strict=True):
        errors += levenshtein(prediction, reference)
        characters += len(reference)
    return errors / max(1, characters)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_checkpoint(directory: Path) -> Path | None:
    candidates = [path for path in directory.glob("checkpoint-*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: int(path.name.rsplit("-", 1)[-1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/gurmukhi_htr"))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--train-examples", type=int, default=90_000)
    parser.add_argument("--validation-examples", type=int, default=1_000)
    parser.add_argument("--max-steps", type=int, default=6_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-label-length", type=int, default=96)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--max-validation-cer", type=float, default=0.35)
    parser.add_argument("--min-script-purity", type=float, default=0.90)
    parser.add_argument("--min-nonempty-rate", type=float, default=0.95)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    archive = args.work_dir / "downloads" / "gurumukhi.zip"
    data_root = args.dataset_dir or args.work_dir / "data"
    if args.dataset_dir is None:
        download_with_resume(TRAINING_ARCHIVE_URL, archive)
        marker = data_root / ".extracted"
        if not marker.exists():
            safe_extract(archive, data_root)
            marker.write_text("ok\n", encoding="utf-8")

    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        default_data_collator,
        set_seed,
    )

    set_seed(args.seed)
    samples = load_samples(data_root)
    train_samples, validation_samples = split_samples(
        samples, args.validation_fraction, args.seed
    )
    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)
    tokenizer = processor.tokenizer
    model.config.decoder_start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id
    model.config.max_length = args.max_label_length
    model.config.num_beams = 4
    model.config.no_repeat_ngram_size = 0
    model.config.early_stopping = True

    class LineDataset(Dataset[Any]):
        def __init__(
            self,
            source: Sequence[Sample],
            length: int,
            seed: int,
            augment: bool,
        ) -> None:
            self.source = source
            self.length = length
            self.seed = seed
            self.augment = augment

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            rng = random.Random(self.seed + index * 104_729)
            word_count = 1 if rng.random() < 0.20 else rng.randint(2, 5)
            indices = [rng.randrange(len(self.source)) for _ in range(word_count)]
            image, text = compose_line(self.source, indices, rng, augment=self.augment)
            pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]
            token_ids = tokenizer(
                text,
                padding="max_length",
                max_length=args.max_label_length,
                truncation=True,
            ).input_ids
            labels = torch.tensor(
                [token if token != tokenizer.pad_token_id else -100 for token in token_ids],
                dtype=torch.long,
            )
            return {"pixel_values": pixel_values, "labels": labels}

    train_dataset = LineDataset(
        train_samples, args.train_examples, args.seed, augment=True
    )
    validation_dataset = LineDataset(
        validation_samples,
        min(args.validation_examples, max(500, len(validation_samples))),
        args.seed + 9_000_001,
        augment=False,
    )

    def decode_metrics(prediction_output: Any) -> dict[str, float]:
        prediction_ids = prediction_output.predictions
        if isinstance(prediction_ids, tuple):
            prediction_ids = prediction_ids[0]
        label_ids = np.array(prediction_output.label_ids)
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        predicted_text = [
            normalize_source_text(value)
            for value in tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        ]
        reference_text = [
            normalize_source_text(value)
            for value in tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        ]
        nonempty = [value for value in predicted_text if value]
        return {
            "cer": corpus_character_error_rate(predicted_text, reference_text),
            "script_purity": (
                sum(gurmukhi_ratio(value) for value in nonempty) / max(1, len(nonempty))
            ),
            "nonempty_rate": len(nonempty) / max(1, len(predicted_text)),
        }

    checkpoints = args.work_dir / "checkpoints"
    training_kwargs: dict[str, Any] = {
        "output_dir": str(checkpoints),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "warmup_ratio": 0.05,
        "logging_steps": 50,
        "save_steps": args.save_steps,
        "eval_steps": args.save_steps,
        "save_strategy": "steps",
        "predict_with_generate": True,
        "generation_max_length": args.max_label_length,
        "generation_num_beams": 4,
        "load_best_model_at_end": True,
        "metric_for_best_model": "cer",
        "greater_is_better": False,
        "save_total_limit": 3,
        "remove_unused_columns": False,
        "fp16": bool(torch.cuda.is_available()),
        "dataloader_num_workers": 2,
        "report_to": [],
        "seed": args.seed,
    }
    argument_names = inspect.signature(Seq2SeqTrainingArguments).parameters
    training_kwargs[
        "eval_strategy" if "eval_strategy" in argument_names else "evaluation_strategy"
    ] = "steps"
    training_args = Seq2SeqTrainingArguments(**training_kwargs)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=default_data_collator,
        compute_metrics=decode_metrics,
    )
    resume = args.resume_from or latest_checkpoint(checkpoints)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    prediction = trainer.predict(validation_dataset)
    metrics = {
        key.removeprefix("test_"): float(value)
        for key, value in prediction.metrics.items()
        if isinstance(value, (int, float))
    }
    passed = bool(
        metrics.get("cer", 1.0) <= args.max_validation_cer
        and metrics.get("script_purity", 0.0) >= args.min_script_purity
        and metrics.get("nonempty_rate", 0.0) >= args.min_nonempty_rate
    )

    artifact = args.work_dir / "artifact" / "gurmukhi_htr"
    artifact.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(artifact, safe_serialization=True)
    processor.save_pretrained(artifact)
    report = {
        "schema_version": "1.0",
        "passed": passed,
        "metrics": metrics,
        "gates": {
            "max_cer": args.max_validation_cer,
            "min_script_purity": args.min_script_purity,
            "min_nonempty_rate": args.min_nonempty_rate,
        },
        "validation_examples": len(validation_dataset),
        "word_level_source_samples": len(samples),
        "synthetic_line_validation": True,
        "domain_warning": (
            "The public corpus is word-level; synthetic lines do not prove accuracy on "
            "historical legal or medical pages. Runtime source validation remains mandatory."
        ),
    }
    (artifact / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "provider_id": "gurmukhi_htr",
        "backend": "transformers_vision_encoder_decoder",
        "model_subdirectory": ".",
        "supported_languages": ["pa"],
        "supported_scripts": ["gurmukhi"],
        "confidence_capability": "sequence_probability",
        "source_language_output_only": True,
        "handwriting_validated": passed,
        "validation_report": "validation_report.json",
        "base_model": args.base_model,
        "training_dataset": {
            "name": "IIIT-INDIC-HW-WORDS Gurumukhi",
            "url": TRAINING_ARCHIVE_URL,
            "source_level": "word",
        },
        "training_arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"dataset_dir", "resume_from"}
            and isinstance(value, (str, int, float, bool, type(None)))
        },
    }
    (artifact / "htr_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checksums = {
        path.name: sha256(path)
        for path in artifact.iterdir()
        if path.is_file() and path.name != "checksums.json"
    }
    (artifact / "checksums.json").write_text(
        json.dumps(checksums, indent=2), encoding="utf-8"
    )
    archive_path = shutil.make_archive(
        str(args.work_dir / "gurmukhi_htr_model"), "zip", artifact
    )
    print(json.dumps({"artifact": archive_path, "validation": report}, indent=2))
    if not passed:
        raise SystemExit(
            "Training completed, but validation gates failed. The bundle was exported "
            "for diagnosis and will not be auto-routed by the application."
        )


if __name__ == "__main__":
    main()
