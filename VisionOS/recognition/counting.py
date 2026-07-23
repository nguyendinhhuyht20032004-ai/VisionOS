"""Pipeline đếm: detect → track → đếm (cắt vạch hoặc chiếm vùng).

``CountingPipeline`` nhận vào một *detector* bất kỳ có phương thức::

    detect(frame, prompt) -> DetectorResult

nên hoàn toàn tách khỏi mô hình: chạy thật thì truyền ``YoloNasDetector`` /
``LocateAnythingDetector`` (GPU), còn kiểm thử thì truyền ``ScriptedDetector``
(CPU, tức thì). Toàn bộ tracking + đếm là thuần Python (``recognition.tracking``,
``recognition.zones``) nên **test được hiệu quả đếm ở mọi máy** — đúng mục tiêu
"làm bài toán đếm trước để kiểm thử".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Set

from .base import Detection, DetectorResult
from .scenarios import CountScenario
from .tracking import CentroidTracker
from .zones import CountingLine, Zone

__all__ = ["CountResult", "CountingPipeline"]


@dataclass
class CountResult:
    """Kết quả đếm của một lần chạy bài toán — một dòng scorecard."""

    scenario_key: str
    counting_type: str
    frames: int = 0
    in_count: int = 0
    out_count: int = 0
    zone_current: int = 0            # số đối tượng đang trong vùng (frame cuối)
    zone_peak: int = 0               # đỉnh chiếm vùng trong cả lượt chạy
    unique_tracks: int = 0           # tổng số danh tính đã xuất hiện
    total_detections: int = 0
    elapsed_s: float = 0.0
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
        if self.counting_type == "line":
            # Cột IN/OUT dùng tên chung để mọi bài thẳng hàng; nhãn riêng của từng
            # bài (Vào/Ra, Chiều tới/lui…) vẫn giữ ở ``in_label``/``out_label``.
            return {
                "scenario": self.scenario_key,
                "frames": self.frames,
                "IN": self.in_count,
                "OUT": self.out_count,
                "chiều": f"{self.in_label}/{self.out_label}",
                "total": self.total_crossings,
                "tracks": self.unique_tracks,
                "det/frame": round(self.avg_detections, 2),
                "fps": round(self.fps, 1),
            }
        return {
            "scenario": self.scenario_key,
            "frames": self.frames,
            "trong_vùng": self.zone_current,
            "đỉnh_vùng": self.zone_peak,
            "tracks": self.unique_tracks,
            "det/frame": round(self.avg_detections, 2),
            "fps": round(self.fps, 1),
        }


class CountingPipeline:
    """Chạy detect → track → đếm cho một ``CountScenario``."""

    def __init__(
        self,
        detector,
        scenario: CountScenario,
        tracker: Optional[CentroidTracker] = None,
    ):
        scenario.validate()
        self.detector = detector
        self.scenario = scenario
        self.w, self.h = scenario.resolution
        self.tracker = tracker or CentroidTracker()
        self.line: Optional[CountingLine] = (
            scenario.build_line() if scenario.counting_type == "line" else None
        )
        self.zone: Optional[Zone] = (
            scenario.build_zone() if scenario.counting_type == "zone" else None
        )
        # Vị trí anchor pixel trước đó của mỗi track, để xét cắt vạch.
        self._prev_anchor: dict = {}
        self._seen_tracks: Set[int] = set()
        self.result = CountResult(
            scenario_key=scenario.key,
            counting_type=scenario.counting_type,
            in_label=scenario.in_label,
            out_label=scenario.out_label,
        )

    def _detect(self, frame) -> DetectorResult:
        """Gọi detector qua interface thống nhất (hỗ trợ cả detect_frame cũ)."""
        if hasattr(self.detector, "detect"):
            return self.detector.detect(frame, self.scenario.prompt)
        # Tương thích ngược với detector kiểu la_counting (detect_frame → tuple).
        dets, raw = self.detector.detect_frame(frame, self.scenario.prompt)
        conv = [
            d if isinstance(d, Detection) else Detection.from_xyxy(*d.bbox, d.class_name, d.confidence)
            for d in dets
        ]
        return DetectorResult(conv, raw)

    def process_frame(self, frame) -> List[Detection]:
        """Xử lý 1 frame; cập nhật bộ đếm; trả list phát hiện đã gán track_id."""
        res = self._detect(frame)
        self.result.total_detections += len(res)
        tracked = self.tracker.update(res.detections)

        for d in tracked:
            if d.track_id is not None:
                self._seen_tracks.add(d.track_id)

        if self.line is not None:
            anchor = self.scenario.zone_anchor
            for d in tracked:
                if d.track_id is None:
                    continue
                curr = d.bbox.anchor(anchor)
                prev = self._prev_anchor.get(d.track_id)
                if prev is not None:
                    self.line.update(prev, curr, self.w, self.h)
                self._prev_anchor[d.track_id] = curr
            self.result.in_count = self.line.in_count
            self.result.out_count = self.line.out_count

        if self.zone is not None:
            anchor = self.scenario.zone_anchor
            occupants = {
                d.track_id
                for d in tracked
                if d.track_id is not None
                and self.zone.contains(d.bbox.anchor(anchor), self.w, self.h)
            }
            self.result.zone_current = len(occupants)
            self.result.zone_peak = max(self.result.zone_peak, len(occupants))

        self.result.frames += 1
        self.result.unique_tracks = len(self._seen_tracks)
        return tracked

    def run(
        self,
        frames: Iterable,
        max_frames: Optional[int] = None,
        on_frame: Optional[Callable] = None,
    ) -> CountResult:
        """Chạy pipeline trên iterable frame; trả ``CountResult`` (scorecard)."""
        limit = max_frames or self.scenario.max_frames
        t0 = time.time()
        for i, frame in enumerate(frames):
            if i >= limit:
                break
            tracked = self.process_frame(frame)
            if on_frame is not None:
                on_frame(frame, tracked, self)
        self.result.elapsed_s = time.time() - t0
        return self.result

    def sanity_ok(self) -> bool:
        """Kiểm tra kết quả có "hợp lý" so với kỳ vọng tối thiểu của scenario."""
        if self.scenario.counting_type == "line":
            return self.result.total_crossings >= self.scenario.expect_min
        return self.result.zone_peak >= self.scenario.expect_min
