"""Pipeline đếm: detect -> track (ByteTrack) -> đếm cắt vạch (LineZone).

``CountingPipeline`` nhận vào một *detector* bất kỳ có phương thức::

    detect_frame(frame_bgr, prompt) -> (List[Detection], raw_text)

Nhờ vậy pipeline hoàn toàn tách khỏi mô hình: khi chạy thật thì truyền
``LocateAnythingDetector`` (GPU), còn khi unit-test thì truyền một detector giả
lập (``tests/fakes.py``) chạy được không cần GPU. Toàn bộ phần theo vết và đếm
dùng **supervision thật**, nên test phản ánh đúng hành vi production.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

import numpy as np

from .parsing import Detection
from .scenarios import Scenario, build_line_zone

__all__ = ["CountResult", "CountingPipeline"]


@dataclass
class CountResult:
    """Kết quả đếm của một lần chạy scenario."""

    scenario_key: str
    frames: int = 0
    in_count: int = 0
    out_count: int = 0
    total_detections: int = 0
    elapsed_s: float = 0.0
    # nhãn hiển thị lấy từ scenario để scorecard đọc được ngay
    in_label: str = "IN"
    out_label: str = "OUT"

    @property
    def total_crossings(self) -> int:
        return self.in_count + self.out_count

    @property
    def avg_detections(self) -> float:
        return self.total_detections / self.frames if self.frames else 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def as_row(self) -> dict:
        """Một dòng cho bảng scorecard."""
        return {
            "scenario": self.scenario_key,
            "frames": self.frames,
            self.in_label: self.in_count,
            self.out_label: self.out_count,
            "total": self.total_crossings,
            "avg_det/frame": round(self.avg_detections, 2),
            "fps": round(self.fps, 2),
        }


def detections_to_sv(dets: List[Detection]):
    """Đổi list :class:`Detection` -> ``supervision.Detections``.

    Trả về ``Detections.empty()`` khi không có gì (đúng như notebook).
    """
    import supervision as sv

    if not dets:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array([d.bbox for d in dets], dtype=np.float32),
        confidence=np.array([d.confidence for d in dets], dtype=np.float32),
        class_id=np.zeros(len(dets), dtype=int),
    )


class CountingPipeline:
    """Chạy detect -> track -> đếm cho một scenario."""

    def __init__(
        self,
        detector,
        scenario: Scenario,
        frame_rate: int = 30,
        resize: bool = True,
    ):
        import supervision as sv

        scenario.validate()
        self.detector = detector
        self.scenario = scenario
        self.resize = resize
        self.w, self.h = scenario.resolution
        self.tracker = sv.ByteTrack(frame_rate=frame_rate)
        self.line_zone = build_line_zone(scenario, self.w, self.h)
        self.result = CountResult(
            scenario_key=scenario.key,
            in_label=scenario.in_label,
            out_label=scenario.out_label,
        )

    def _prep(self, frame_bgr):
        import cv2

        if self.resize and frame_bgr.shape[1::-1] != (self.w, self.h):
            return cv2.resize(frame_bgr, (self.w, self.h))
        return frame_bgr

    def process_frame(self, frame_bgr):
        """Xử lý 1 frame; trả về ``(sv_detections_tracked, raw_text)``.

        Cập nhật bộ đếm nội bộ. Có thể gọi trực tiếp để tự vẽ annotation.
        """
        frame = self._prep(frame_bgr)
        dets, raw = self.detector.detect_frame(frame, self.scenario.prompt)
        self.result.total_detections += len(dets)

        sv_d = detections_to_sv(dets)
        sv_d = self.tracker.update_with_detections(sv_d)
        self.line_zone.trigger(sv_d)

        self.result.frames += 1
        self.result.in_count = int(self.line_zone.in_count)
        self.result.out_count = int(self.line_zone.out_count)
        return sv_d, raw

    def run(
        self,
        frames: Iterable,
        max_frames: Optional[int] = None,
        on_frame: Optional[Callable] = None,
    ) -> CountResult:
        """Chạy pipeline trên một iterable frame BGR.

        max_frames: giới hạn số frame (mặc định lấy ``scenario.max_frames``).
        on_frame(frame_bgr, sv_detections, pipeline): callback tuỳ chọn để vẽ /
            ghi video ở tầng gọi mà không nhồi logic vào pipeline.
        """
        limit = max_frames or self.scenario.max_frames
        t0 = time.time()
        for i, frame in enumerate(frames):
            if i >= limit:
                break
            sv_d, _ = self.process_frame(frame)
            if on_frame is not None:
                on_frame(self._prep(frame), sv_d, self)
        self.result.elapsed_s = time.time() - t0
        return self.result


def iter_video_frames(video_path: str):
    """Generator đọc frame BGR từ video bằng OpenCV (dùng khi chạy thật)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()
