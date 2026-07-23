"""Kiểu dữ liệu nền tảng cho tầng nhận diện của VisionOS.

Toàn bộ hệ thống trong sản phẩm CV_product xoay quanh **một đối tượng được phát
hiện** (`Detection`) và **ba chế độ giám sát** (`MonitoringMode`). Module này giữ
những kiểu đó ở dạng *thuần Python* (chỉ phụ thuộc numpy để tính hình học) nên
import được ở mọi môi trường — không cần torch / supervision / GPU.

Ánh xạ sang sản phẩm:
  * ``MonitoringMode.STANDARD``  → mô hình YOLO-NAS (từ vựng cố định: người/xe).
  * ``MonitoringMode.SMART``     → LocateAnything-3B (ngôn ngữ tự nhiên, open-vocab).
  * ``MonitoringMode.DEFECT``    → so mẫu chuẩn SSIM + CNN + OCR (phát hiện lỗi).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

__all__ = [
    "MonitoringMode",
    "BoundingBox",
    "Detection",
    "DetectorResult",
    "Point",
]

Point = Tuple[float, float]


class MonitoringMode(str, Enum):
    """Ba chế độ giám sát của sản phẩm (khớp ``monitoringMode`` ở frontend)."""

    STANDARD = "standard"          # YOLO-NAS, phát hiện đối tượng từ vựng cố định
    SMART = "smart"                # LocateAnything-3B, mô tả ngôn ngữ tự nhiên
    DEFECT = "defect_detection"    # so mẫu chuẩn (SSIM/CNN/OCR)

    @classmethod
    def from_str(cls, value: str) -> "MonitoringMode":
        for m in cls:
            if m.value == value:
                return m
        raise ValueError(f"monitoring mode không hợp lệ: {value!r}")


@dataclass
class BoundingBox:
    """Hộp bao theo pixel: ``(x1, y1)`` góc trên-trái, ``(x2, y2)`` góc dưới-phải."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        # Chuẩn hoá để x1<=x2, y1<=y2 (một số model trả ngược thứ tự).
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1

    # ---- thuộc tính hình học thường dùng ---------------------------------- #
    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Point:
        """Điểm chạm sàn — anchor ổn định nhất cho người/xe (khớp scenarios)."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def anchor(self, name: str = "CENTER") -> Point:
        """Trả về điểm neo theo tên Position của supervision (CENTER, BOTTOM_CENTER…)."""
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0
        xs = {"LEFT": self.x1, "CENTER": cx, "RIGHT": self.x2}
        ys = {"TOP": self.y1, "CENTER": cy, "BOTTOM": self.y2}
        vert, _, horiz = name.upper().partition("_")
        if not horiz:  # dạng "CENTER" đơn
            return (cx, cy)
        return (xs.get(horiz, cx), ys.get(vert, cy))

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-Union với hộp khác (0..1)."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, point: Point) -> bool:
        px, py = point
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

    def distance_to(self, other: "BoundingBox") -> float:
        """Khoảng cách Euclid giữa hai tâm hộp (dùng cho tracker centroid)."""
        (ax, ay), (bx, by) = self.center, other.center
        return math.hypot(ax - bx, ay - by)


@dataclass
class Detection:
    """Một đối tượng được phát hiện trong 1 frame.

    ``label``   : nhãn lớp ("person", "car", "hộp carton"…). Ở chế độ SMART đây
                  chính là cụm mô tả ngôn ngữ tự nhiên.
    ``track_id``: gán sau khi qua tracker (None khi chưa track).
    ``attributes``: thuộc tính phụ do model chuyên biệt gắn (vd PPE: has_helmet).
    """

    bbox: BoundingBox
    label: str
    confidence: float = 0.85
    class_id: int = 0
    track_id: Optional[int] = None
    attributes: dict = field(default_factory=dict)

    @classmethod
    def from_xyxy(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        label: str,
        confidence: float = 0.85,
        **kw,
    ) -> "Detection":
        return cls(BoundingBox(x1, y1, x2, y2), label, confidence, **kw)


@dataclass
class DetectorResult:
    """Kết quả một lần suy luận: danh sách phát hiện + text thô + độ trễ.

    ``raw`` giữ output nguyên bản của model (đặc biệt hữu ích với LocateAnything,
    nơi bbox được parse từ text — xem ``la_counting.parsing``).
    """

    detections: List[Detection] = field(default_factory=list)
    raw: str = ""
    latency_ms: float = 0.0
    model_name: str = ""

    def __len__(self) -> int:
        return len(self.detections)

    def labels(self) -> List[str]:
        return [d.label for d in self.detections]

    def filter_confidence(self, threshold: float) -> "DetectorResult":
        keep = [d for d in self.detections if d.confidence >= threshold]
        return DetectorResult(keep, self.raw, self.latency_ms, self.model_name)
