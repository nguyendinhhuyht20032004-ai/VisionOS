"""Detector giả lập + tiện ích sinh kịch bản chuyển động cho test.

Nhờ ``CountingPipeline`` nhận detector duck-typed, ta thay mô hình 6GB bằng một
``ScriptedDetector`` chạy tức thì trên CPU, để test toàn bộ luồng
detect -> track -> count mà không cần GPU.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from la_counting.parsing import Detection

Box = Tuple[float, float, float, float]


class ScriptedDetector:
    """Trả về bbox theo kịch bản dựng sẵn, mỗi lần gọi lấy 1 frame.

    frames_boxes[i] là danh sách box (x1,y1,x2,y2) cho frame thứ i. Hết kịch bản
    thì trả rỗng. Bỏ qua nội dung ảnh và prompt (chỉ echo prompt vào class_name).
    """

    def __init__(self, frames_boxes: Sequence[Sequence[Box]]):
        self.frames_boxes = frames_boxes
        self.calls = 0
        self.prompts_seen: List[str] = []

    def detect_frame(self, frame, prompt, max_new_tokens=None):
        self.prompts_seen.append(prompt)
        boxes = (
            self.frames_boxes[self.calls]
            if self.calls < len(self.frames_boxes)
            else []
        )
        self.calls += 1
        dets = [
            Detection(tuple(int(round(v)) for v in b), prompt, 0.85) for b in boxes
        ]
        return dets, "scripted"


def linear_track(
    start: Box, end: Box, n_frames: int
) -> List[Box]:
    """Sinh chuỗi box nội suy tuyến tính từ ``start`` -> ``end`` qua n_frames.

    Chuyển động mượt để ByteTrack giữ nguyên track_id xuyên suốt.
    """
    out: List[Box] = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        out.append(tuple(s + (e - s) * t for s, e in zip(start, end)))
    return out


def merge_tracks(*tracks: Sequence[Box]) -> List[List[Box]]:
    """Gộp nhiều track (mỗi track là list box theo frame) thành kịch bản đa vật.

    Các track phải cùng độ dài; frame i gồm box thứ i của mọi track.
    """
    n = len(tracks[0])
    assert all(len(t) == n for t in tracks), "các track phải cùng số frame"
    return [[t[i] for t in tracks] for i in range(n)]


def blank_frames(n: int, w: int = 1280, h: int = 720) -> List[np.ndarray]:
    """n frame BGR đen kích thước (w,h) — nội dung không quan trọng với fake."""
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]
