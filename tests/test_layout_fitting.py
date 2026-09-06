from translator_app.core.layout_engine import collides, fit_text
from translator_app.schemas import BoundingBox, LayoutStatus


def test_short_text_fits_at_original_size() -> None:
    result = fit_text(
        "Short heading",
        BoundingBox(0, 0, 200, 30),
        original_font_size=14,
        min_font_size=6,
        page_width=600,
        page_height=800,
    )
    assert not result.overflow
    assert result.font_size == 14
    assert result.status == LayoutStatus.FIT


def test_long_text_reports_overflow_when_expansion_disabled() -> None:
    result = fit_text(
        "many translated words " * 30,
        BoundingBox(0, 0, 80, 18),
        original_font_size=12,
        min_font_size=8,
        page_width=600,
        page_height=800,
        allow_expansion=False,
    )
    assert result.overflow
    assert result.status == LayoutStatus.OVERFLOW


def test_collision_detection() -> None:
    assert collides(BoundingBox(0, 0, 20, 20), [BoundingBox(10, 10, 30, 30)])
    assert not collides(BoundingBox(0, 0, 20, 20), [BoundingBox(30, 30, 40, 40)])

