"""Vùng giám sát (Zone) và vạch đếm (CountingLine).

Khớp trực tiếp với ``CountingZone`` ở frontend (``src/types.ts``):

  * ``type='zone'`` → đa giác; ``role`` = ``monitor`` (AI giám sát) hoặc
    ``exclude`` (vùng ngoại lệ AI bỏ qua).
  * ``type='line'`` → vạch cắt hai chiều (đếm vào/ra).

Toạ độ trong sản phẩm là **phần trăm 0..100** của khung hình (người dùng vẽ trên
video). ``to_pixels`` quy đổi sang pixel theo (w, h) để tính hình học.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .geometry import line_crossing, point_in_polygon, polygon_area

Point = Tuple[float, float]

__all__ = ["Zone", "CountingLine", "ZoneRole"]

ZoneRole = str  # "monitor" | "exclude"


@dataclass
class Zone:
    """Vùng đa giác giám sát (toạ độ phần trăm 0..100 như frontend)."""

    id: str
    name: str
    points_pct: Sequence[Point]
    role: ZoneRole = "monitor"
    max_limit: Optional[int] = None  # ngưỡng đông người/vật để cảnh báo quá tải

    def to_pixels(self, w: int, h: int) -> List[Point]:
        return [(px / 100.0 * w, py / 100.0 * h) for px, py in self.points_pct]

    def contains(self, point_px: Point, w: int, h: int) -> bool:
        return point_in_polygon(point_px, self.to_pixels(w, h))

    def area_px(self, w: int, h: int) -> float:
        return polygon_area(self.to_pixels(w, h))

    def is_exclude(self) -> bool:
        return self.role == "exclude"


@dataclass
class CountingLine:
    """Vạch đếm hai chiều (toạ độ phần trăm 0..100)."""

    id: str
    name: str
    start_pct: Point
    end_pct: Point
    in_label: str = "IN"
    out_label: str = "OUT"
    in_count: int = 0
    out_count: int = 0

    def endpoints(self, w: int, h: int) -> Tuple[Point, Point]:
        (sx, sy), (ex, ey) = self.start_pct, self.end_pct
        return (sx / 100.0 * w, sy / 100.0 * h), (ex / 100.0 * w, ey / 100.0 * h)

    def update(self, prev_px: Point, curr_px: Point, w: int, h: int) -> Optional[str]:
        """Cập nhật bộ đếm nếu track cắt vạch giữa hai frame; trả 'in'/'out'/None."""
        s, e = self.endpoints(w, h)
        direction = line_crossing(prev_px, curr_px, s, e)
        if direction == "in":
            self.in_count += 1
        elif direction == "out":
            self.out_count += 1
        return direction

    @property
    def total(self) -> int:
        return self.in_count + self.out_count
