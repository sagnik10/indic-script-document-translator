import numpy as np
from PIL import Image, ImageDraw

from translator_app.core.image_processor import ImagePreprocessor
from translator_app.core.preprocessing_profiles import PROFILES
from translator_app.core.quality_analysis import ImageQualityAnalyzer, ImageQualityMetrics


def metrics(**changes) -> ImageQualityMetrics:
    values = dict(
        width=1200,
        height=1600,
        brightness=0.8,
        contrast=0.8,
        blur_score=0.8,
        noise_score=0.1,
        shadow_score=0.05,
        background_noise=0.1,
        skew_angle=0.0,
        border_confidence=0.98,
        perspective_distortion=0.0,
        handwriting_likelihood=0.2,
    )
    values.update(changes)
    return ImageQualityMetrics(**values)


def test_named_profiles_exist_and_auto_selection_covers_degraded_modes() -> None:
    assert set(PROFILES) == {
        "clean_scan",
        "photocopy",
        "mobile_photo",
        "faded_document",
        "handwriting_heavy",
    }
    selector = ImageQualityAnalyzer.select_profile
    assert selector(metrics()) == "clean_scan"
    assert selector(metrics(noise_score=0.7)) == "photocopy"
    assert selector(metrics(contrast=0.2, brightness=0.8)) == "faded_document"
    assert selector(metrics(perspective_distortion=0.2)) == "mobile_photo"
    assert selector(metrics(handwriting_likelihood=0.9)) == "handwriting_heavy"


def test_manual_photocopy_profile_produces_quality_metadata_without_changing_input() -> None:
    image = Image.new("RGB", (700, 900), "#d9d7cf")
    draw = ImageDraw.Draw(image)
    draw.text((80, 100), "Faded photocopy line 31.03.2016", fill="#777777")
    before = np.asarray(image).copy()
    result = ImagePreprocessor().preprocess(
        image, profile="photocopy", allow_geometry=False, upscale_factor=2
    )
    assert result.profile == "photocopy"
    assert result.quality is not None
    assert result.image.width >= image.width
    assert np.array_equal(np.asarray(image), before)

