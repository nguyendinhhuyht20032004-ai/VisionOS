"""Detector giả lập + tiện ích sinh kịch bản chuyển động cho kiểm thử.

Nhờ ``CountingPipeline`` nhận detector duck-typed (chỉ cần ``detect``), ta thay
mô hình nặng vài GB bằng ``ScriptedDetector`` chạy tức thì trên CPU, để kiểm thử
**toàn bộ luồng detect → track → đếm** mà không cần GPU. Kịch bản chuyển động là
tất định nên biết trước đáp số → đo được *hiệu quả đếm* một cách khách quan.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..base import BoundingBox, Detection, DetectorResult

__all__ = [
    "ScriptedDetector",
    "linear_track",
    "merge_tracks",
    "blank_frames",
    "box_moving",
]

Box = Tuple[float, float, float, float]


class ScriptedDetector:
    """Trả về bbox theo kịch bản dựng sẵn, mỗi lần gọi tiến 1 frame.

    ``frames_boxes[i]`` là danh sách box (x1,y1,x2,y2) cho frame thứ i. Hết kịch
    bản thì trả rỗng. Bỏ qua nội dung ảnh; echo prompt vào ``label``.
    """

    def __init__(self, frames_boxes: Sequence[Sequence[Box]], confidence: float = 0.85):
        self.frames_boxes = frames_boxes
        self.confidence = confidence
        self.calls = 0
        self.prompts_seen: List[str] = []

    def detect(self, frame, prompt: str) -> DetectorResult:
        self.prompts_seen.append(prompt)
        boxes = (
            self.frames_boxes[self.calls] if self.calls < len(self.frames_boxes) else []
        )
        self.calls += 1
        dets = [
            Detection(BoundingBox(*[float(v) for v in b]), prompt, self.confidence)
            for b in boxes
        ]
        return DetectorResult(dets, raw="scripted", model_name="ScriptedDetector")

    def reset(self) -> None:
        self.calls = 0
        self.prompts_seen.clear()


def linear_track(start: Box, end: Box, n_frames: int) -> List[Box]:
    """Nội suy tuyến tính box ``start`` → ``end`` qua ``n_frames`` frame.

    Chuyển động mượt để tracker giữ nguyên track_id xuyên suốt (không nhảy id).
    """
    out: List[Box] = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        out.append(tuple(s + (e - s) * t for s, e in zip(start, end)))
    return out


def merge_tracks(*tracks: Sequence[Box]) -> List[List[Box]]:
    """Gộp nhiều track cùng độ dài thành kịch bản đa vật (frame i = box thứ i)."""
    n = len(tracks[0])
    assert all(len(t) == n for t in tracks), "các track phải cùng số frame"
    return [[t[i] for t in tracks] for i in range(n)]


def box_moving(
    cx0: float, cy0: float, cx1: float, cy1: float, w: float, h: float, n_frames: int
) -> List[Box]:
    """Box kích thước (w,h) có tâm đi từ (cx0,cy0) → (cx1,cy1) qua n_frames."""
    boxes: List[Box] = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        cx = cx0 + (cx1 - cx0) * t
        cy = cy0 + (cy1 - cy0) * t
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def blank_frames(n: int, w: int = 1280, h: int = 720) -> List[np.ndarray]:
    """n frame BGR đen kích thước (w,h) — nội dung không quan trọng với fake."""
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]
