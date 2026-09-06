"""Collision checks and safe annotation-space search."""

from __future__ import annotations

from ..schemas import BoundingBox


class CollisionDetector:
    @staticmethod
    def intersects_any(
        candidate: BoundingBox, obstacles: list[BoundingBox], padding: float = 2.0
    ) -> bool:
        return any(candidate.intersects(obstacle, padding=padding) for obstacle in obstacles)

    def find_safe_region(
        self,
        source: BoundingBox,
        *,
        desired_height: float,
        page_width: float,
        page_height: float,
        obstacles: list[BoundingBox],
        margin: float = 4.0,
    ) -> BoundingBox | None:
        if page_width <= 0 or page_height <= 0:
            return None

        anchor = source.clamp(page_width, page_height)
        width = max(72.0, source.width)
        height = max(18.0, min(desired_height, 120.0))
        raw_candidates = [
            (anchor.x0, anchor.y1 + margin, anchor.x0 + width, anchor.y1 + margin + height),
            (anchor.x0, anchor.y0 - margin - height, anchor.x0 + width, anchor.y0 - margin),
            (anchor.x1 + margin, anchor.y0, anchor.x1 + margin + width, anchor.y0 + height),
            (anchor.x0 - margin - width, anchor.y0, anchor.x0 - margin, anchor.y0 + height),
        ]

        for x0, y0, x1, y1 in raw_candidates:
            left = max(0.0, min(x0, page_width))
            top = max(0.0, min(y0, page_height))
            right = max(0.0, min(x1, page_width))
            bottom = max(0.0, min(y1, page_height))
            if right - left < 40.0 or bottom - top < 12.0:
                continue
            candidate = BoundingBox(left, top, right, bottom)
            if not self.intersects_any(candidate, obstacles):
                return candidate
        return None
