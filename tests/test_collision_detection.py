from __future__ import annotations

import pytest

from translator_app.core.collision_detection import CollisionDetector
from translator_app.schemas import BoundingBox


@pytest.mark.parametrize(
    "source",
    [
        BoundingBox(460, 100, 500, 140),
        BoundingBox(100, 760, 180, 800),
        BoundingBox(480, 780, 500, 800),
        BoundingBox(490, 790, 520, 830),
    ],
)
def test_safe_region_handles_page_edge_sources(source: BoundingBox) -> None:
    result = CollisionDetector().find_safe_region(
        source,
        desired_height=36,
        page_width=500,
        page_height=800,
        obstacles=[],
    )

    if result is not None:
        assert 0 <= result.x0 <= result.x1 <= 500
        assert 0 <= result.y0 <= result.y1 <= 800
        assert result.width >= 40
        assert result.height >= 12


def test_safe_region_returns_none_for_invalid_page_geometry() -> None:
    result = CollisionDetector().find_safe_region(
        BoundingBox(0, 0, 10, 10),
        desired_height=20,
        page_width=0,
        page_height=800,
        obstacles=[],
    )

    assert result is None
