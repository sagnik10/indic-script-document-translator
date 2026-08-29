"""Named preprocessing profiles for common difficult-document conditions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreprocessingProfile:
    name: str
    shadow_correction: bool
    clahe_clip_limit: float
    denoise_strength: int
    adaptive_threshold: bool
    sharpen_amount: float
    morphology_strength: int
    background_cleanup: bool
    local_enhancement: bool
    geometry_correction: bool


PROFILES: dict[str, PreprocessingProfile] = {
    "clean_scan": PreprocessingProfile(
        "clean_scan", False, 1.4, 3, False, 0.35, 0, False, False, False
    ),
    "photocopy": PreprocessingProfile(
        "photocopy", True, 2.2, 9, True, 0.75, 1, True, True, False
    ),
    "mobile_photo": PreprocessingProfile(
        "mobile_photo", True, 2.0, 7, True, 0.65, 1, True, True, True
    ),
    "faded_document": PreprocessingProfile(
        "faded_document", True, 3.0, 5, True, 0.9, 1, False, True, False
    ),
    "handwriting_heavy": PreprocessingProfile(
        "handwriting_heavy", True, 1.8, 5, False, 0.45, 0, True, True, True
    ),
}


def get_profile(name: str) -> PreprocessingProfile:
    if name not in PROFILES:
        raise ValueError(f"Unknown preprocessing profile: {name}")
    return PROFILES[name]

