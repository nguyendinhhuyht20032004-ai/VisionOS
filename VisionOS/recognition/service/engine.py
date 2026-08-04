"""``StreamingCounter`` — engine đếm THEO LUỒNG (xử lý TỪNG frame, có trạng thái).

Khác ``sv_counting.sv_run`` (chạy 1 lượt hết video), lớp này giữ trạng thái để
service AI nạp frame lẻ từ camera → trả về (frame đã annotate, số đếm) NGAY. Tái
dùng đúng các bước đã kiểm chứng: detect → ByteTrack → DetectionsSmoother (chống
nhấp nháy) → LineZone/PolygonZone → annotator supervision.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

from ..counting import CountResult
from ..sv_counting import _annotate_sv, _draw, _make_polygon_zone, _to_sv

__all__ = ["StreamingCounter"]


class StreamingCounter:
    """Đếm theo luồng cho MỘT scenario + MỘT detector. Gọi :meth:`process` mỗi frame."""

    def __init__(self, scenario, detector, resolution: Optional[Tuple[int, int]] = None,
                 track_thresh: float = 0.1, smoother_len: int = 8):
        import warnings

        import numpy as np

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # ẩn FutureWarning ByteTrack
            import supervision as sv

        self._sv, self._np = sv, np
        self.scenario = scenario
        self.detector = detector
        self.w, self.h = resolution or scenario.resolution

        # ByteTrack (fallback theo phiên bản) — bám dai để đỡ đứt track = đỡ bỏ sót.
        self.tracker = None
        for kw in (dict(track_activation_threshold=track_thresh, minimum_consecutive_frames=1,
                        lost_track_buffer=120),
                   dict(track_thresh=track_thresh), {}):
            try:
                self.tracker = sv.ByteTrack(**kw)
                break
            except TypeError:
                continue
        # Smoother: giữ box qua frame detect trượt → hết nhấp nháy.
        try:
            self.smoother = sv.DetectionsSmoother(length=smoother_len)
        except Exception:  # noqa: BLE001
            self.smoother = None

        # Vạch / vùng.
        self.line = None
        self.polys = []
        self.zones_px = (
            [np.array([[int(x), int(y)] for x, y in z.to_pixels(self.w, self.h)], dtype=np.int32)
             for z in scenario.build_zones()]
            if scenario.counting_type == "zone" else [])
        if scenario.counting_type == "line":
            (sx, sy), (ex, ey) = scenario.build_line().endpoints(self.w, self.h)
            self.line = sv.LineZone(start=sv.Point(float(sx), float(sy)),
                                    end=sv.Point(float(ex), float(ey)))
        elif scenario.counting_type == "zone":
            for z in scenario.build_zones():
                poly = np.array([[int(x), int(y)] for x, y in z.to_pixels(self.w, self.h)], dtype=int)
                self.polys.append(_make_polygon_zone(sv, poly, self.w, self.h))
        # fullscreen: không vạch, không vùng — đếm toàn khung.

        # Annotator supervision (box bo góc + nhãn + trace + bộ đếm). Lỗi API → vẽ cv2.
        self.annos = None
        try:
            self.annos = {
                "box": sv.RoundBoxAnnotator(color_lookup=sv.ColorLookup.CLASS, thickness=2),
                "label": sv.LabelAnnotator(color_lookup=sv.ColorLookup.CLASS, text_scale=0.45),
                "trace": sv.TraceAnnotator(color_lookup=sv.ColorLookup.CLASS, thickness=2, trace_length=30),
                "line": sv.LineZoneAnnotator(thickness=2, text_scale=0.7) if self.line is not None else None,
                "zones": [sv.PolygonZoneAnnotator(zone=pz, color=sv.Color.GREEN, thickness=2)
                          for pz in self.polys],
            }
        except Exception:  # noqa: BLE001
            self.annos = None

        self.result = CountResult(scenario_key=scenario.key, counting_type=scenario.counting_type,
                                  in_label=scenario.in_label, out_label=scenario.out_label)
        self._seen: set = set()
        self._t0 = time.time()
        self._last_latency_ms = 0.0
        # Frame + detections của lần process GẦN NHẤT (cho service crop vật đã đếm → vector DB).
        self.last_frame = None
        self.last_det = None

    # ------------------------------------------------------------------ #
    def process(self, frame_bgr):
        """Xử lý 1 frame BGR → trả frame ĐÃ ANNOTATE (BGR). Cập nhật số đếm nội bộ."""
        import cv2

        sv, np = self._sv, self._np
        if frame_bgr.shape[1::-1] != (self.w, self.h):
            frame_bgr = cv2.resize(frame_bgr, (self.w, self.h))

        t0 = time.time()
        dr = self.detector.detect(frame_bgr, self.scenario.prompt)
        self._last_latency_ms = (time.time() - t0) * 1000
        self.result.total_detections += len(dr.detections)

        det = _to_sv(dr.detections, sv, np)
        det = self.tracker.update_with_detections(det)
        if self.smoother is not None:
            try:
                det = self.smoother.update_with_detections(det)
            except Exception:  # noqa: BLE001
                pass

        if det.tracker_id is not None:
            for tid in det.tracker_id:
                if tid is not None:
                    self._seen.add(int(tid))

        if self.line is not None:
            self.line.trigger(det)
            self.result.in_count = int(self.line.in_count)
            self.result.out_count = int(self.line.out_count)
        if self.polys:
            inside = np.zeros(len(det), dtype=bool)
            for pz in self.polys:
                inside = inside | np.asarray(pz.trigger(det), dtype=bool)
            cur = int(inside.sum())
            self.result.zone_current = cur
            self.result.zone_peak = max(self.result.zone_peak, cur)
        if self.scenario.counting_type == "fullscreen":
            # TOÀN MÀN HÌNH: đếm MỌI vật đang trong khung (hiện tại/đỉnh); tổng = tracks.
            cur = int(len(det))
            self.result.zone_current = cur
            self.result.zone_peak = max(self.result.zone_peak, cur)

        self.result.frames += 1
        self.result.unique_tracks = len(self._seen)
        self.result.elapsed_s = time.time() - self._t0
        self.last_frame, self.last_det = frame_bgr, det   # cho service crop vật → vector DB

        out = None
        if self.annos is not None:
            try:
                out = _annotate_sv(frame_bgr, det, self.annos, self.line, self.result,
                                   self.w, self.h, cv2, sv)
            except Exception:  # noqa: BLE001
                out = None
        if out is None:
            out = _draw(frame_bgr, det, self.scenario, self.line, self.zones_px, self.result,
                        self.w, self.h, cv2, np)
        return out

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Số đếm hiện tại (JSON-friendly) cho API."""
        r = self.result
        d = {
            "scenario": self.scenario.key,
            "prompt": self.scenario.prompt,
            "counting_type": self.scenario.counting_type,
            "frames": r.frames,
            "tracks": r.unique_tracks,           # số vật KHÁC NHAU đã thấy (≈ tổng)
            "det_per_frame": round(r.avg_detections, 2),
            "fps": round(r.fps, 2),
            "latency_ms": round(self._last_latency_ms, 1),
        }
        if self.scenario.counting_type == "line":
            d.update({"in": r.in_count, "out": r.out_count, "total": r.total_crossings,
                      "in_label": r.in_label, "out_label": r.out_label})
        elif self.scenario.counting_type == "fullscreen":
            # TOÀN MÀN HÌNH: in_frame = đang trong khung, peak = đông nhất, total = tổng vật khác nhau.
            d.update({"in_frame": r.zone_current, "peak": r.zone_peak, "total": r.unique_tracks})
        else:
            d.update({"in_zone": r.zone_current, "zone_peak": r.zone_peak})
        return d
