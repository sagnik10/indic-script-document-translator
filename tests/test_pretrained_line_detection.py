"""Regression coverage for the Windows pretrained OCR line detector."""

import cv2
import numpy as np

from pretrained_page_ocr.kaggle_app import EnhancedPage, _opencv_detect


def _synthetic_page() -> EnhancedPage:
    image = np.full((300, 600, 3), 255, dtype=np.uint8)
    # Gurmukhi-like connected headline plus descending strokes.  The old rule
    # suppression removed this as a horizontal form line before OCR.
    for y in (90, 180):
        for x in range(45, 500, 65):
            cv2.line(image, (x, y), (x + 52, y), (0, 0, 0), 2)
            cv2.line(image, (x + 16, y), (x + 16, y + 31), (0, 0, 0), 2)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    return EnhancedPage(image, image, gray, threshold)


def test_opencv_detector_keeps_gurmukhi_like_headlines_as_text_lines() -> None:
    regions = _opencv_detect(_synthetic_page())

    assert len(regions) >= 2
    assert any(region.bbox[1] <= 95 <= region.bbox[3] for region in regions)
    assert any(region.bbox[1] <= 185 <= region.bbox[3] for region in regions)
