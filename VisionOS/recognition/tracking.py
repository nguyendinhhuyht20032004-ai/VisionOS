"""Theo vết đối tượng (tracking) — thuần Python, chạy được không cần GPU.

Đếm chính xác đòi hỏi **giữ được danh tính** của mỗi đối tượng qua các frame, nếu
không một người đứng yên sẽ bị đếm lại mỗi frame. Module này cài một
``CentroidTracker`` nhẹ (gán track theo khoảng cách tâm hộp gần nhất, kiểu
SORT/ByteTrack rút gọn) để toàn bộ pipeline đếm test được ở mọi máy chỉ với numpy.

Khi chạy production trên GPU có thể thay bằng ``supervision.ByteTrack`` (xem
``la_counting.counting``); interface ``update(detections) -> tracked`` giữ nguyên
nên pipeline không phải sửa gì.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .base import Detection, Point

__all__ = ["Track", "CentroidTracker"]


@dataclass
class Track:
    """Trạng thái một track sống: id, vị trí hiện tại, lịch sử tâm, tuổi."""

    track_id: int
    label: str
    centroid: Point
    history: List[Point] = field(default_factory=list)
    age: int = 0                 # số frame đã tồn tại
    missed: int = 0              # số frame liên tiếp không khớp phát hiện nào
    first_frame: int = 0
    last_frame: int = 0

    def dwell_frames(self) -> int:
        """Số frame track đã hiện diện (để tính thời gian lưu lại — dwell time)."""
        return self.last_frame - self.first_frame + 1


class CentroidTracker:
    """Tracker gán danh tính theo khoảng cách tâm gần nhất (greedy).

    max_distance : ngưỡng khoảng cách (pixel) để coi hai hộp là cùng một đối tượng.
    max_missed   : số frame vắng mặt tối đa trước khi xoá track (chịu che khuất).
    """

    def __init__(self, max_distance: float = 80.0, max_missed: int = 30):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._next_id = 0
        self._frame = -1
        self.tracks: Dict[int, Track] = {}

    def _distance(self, a: Point, b: Point) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Gán ``track_id`` cho từng phát hiện của frame hiện tại.

        Trả về chính danh sách ``detections`` (đã set ``track_id``), giữ những
        cái được gán track hợp lệ. Cập nhật lịch sử tâm cho mỗi track để dùng
        đếm cắt vạch và tính dwell.
        """
        self._frame += 1
        centroids = [d.bbox.center for d in detections]

        # Ghép greedy: duyệt mọi cặp (track, detection) theo khoảng cách tăng dần.
        unmatched_dets = set(range(len(detections)))
        pairs: List[Tuple[float, int, int]] = []
        for tid, tr in self.tracks.items():
            for di, c in enumerate(centroids):
                dist = self._distance(tr.centroid, c)
                if dist <= self.max_distance:
                    pairs.append((dist, tid, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks = set()
        matched_dets = set()
        for _dist, tid, di in pairs:
            if tid in matched_tracks or di in matched_dets:
                continue
            tr = self.tracks[tid]
            c = centroids[di]
            tr.history.append(tr.centroid)
            tr.centroid = c
            tr.age += 1
            tr.missed = 0
            tr.last_frame = self._frame
            detections[di].track_id = tid
            matched_tracks.add(tid)
            matched_dets.add(di)
            unmatched_dets.discard(di)

        # Track không được ghép frame này -> tăng missed, xoá nếu vắng quá lâu.
        for tid, tr in list(self.tracks.items()):
            if tid not in matched_tracks:
                tr.missed += 1
                if tr.missed > self.max_missed:
                    del self.tracks[tid]

        # Phát hiện chưa khớp -> tạo track mới.
        for di in sorted(unmatched_dets):
            tid = self._next_id
            self._next_id += 1
            c = centroids[di]
            self.tracks[tid] = Track(
                track_id=tid,
                label=detections[di].label,
                centroid=c,
                history=[],
                age=1,
                missed=0,
                first_frame=self._frame,
                last_frame=self._frame,
            )
            detections[di].track_id = tid

        return detections

    def get_track(self, track_id: int) -> Optional[Track]:
        return self.tracks.get(track_id)

    def reset(self) -> None:
        self._next_id = 0
        self._frame = -1
        self.tracks.clear()
