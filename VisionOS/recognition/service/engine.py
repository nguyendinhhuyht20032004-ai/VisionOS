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
                 track_thresh: float = 0.1, smoother_len: int = 8, detect_every: int = 1,
                 merge_label: Optional[str] = None):
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
            start, end = sv.Point(float(sx), float(sy)), sv.Point(float(ex), float(ey))
            # Đếm theo 1 ĐIỂM NEO (tâm) thay vì cả 4 góc → xe TO (làn ngoài, gần camera)
            # cũng đếm được, không sót làn nào. Fallback nếu bản supervision cũ không có tham số.
            anchor = getattr(sv.Position, scenario.zone_anchor, sv.Position.CENTER)
            try:
                self.line = sv.LineZone(start=start, end=end, triggering_anchors=(anchor,))
            except TypeError:
                self.line = sv.LineZone(start=start, end=end)
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
        # detect_every: chỉ chạy YOLO mỗi N frame (tăng throughput CPU); frame giữa vẽ lại box cũ.
        self.detect_every = max(1, int(detect_every))
        self._frame_i = 0
        self._track_cls: dict = {}          # track_id → {lớp: số lần} (bình chọn lớp ổn định)
        self._track_pos: dict = {}          # track_id → (cx, cy, t) → tính VẬN TỐC để dự đoán box
        # merge_label: gộp MỌI vật về 1 nhãn (vd "vehicle") — tùy chọn khi muốn đếm gộp phương
        # tiện thành 1 loại (bỏ tick trên web = giữ phân loại car/truck/bus, mỗi loại 1 màu).
        self.merge_label = merge_label

    # ------------------------------------------------------------------ #
    def detect(self, frame_bgr):
        """PHÁT HIỆN + bám + ĐẾM cho 1 frame (phần NẶNG — chạy YOLO). Trả ``det`` (sv.Detections).

        KHÔNG vẽ. Tách riêng để service chạy ở LUỒNG NỀN, còn luồng hiển thị chỉ ``render``
        (nhẹ) mỗi frame ở tốc độ gốc → video mượt như bản gốc dù YOLO trên CPU chậm hơn.
        """
        import cv2

        sv, np = self._sv, self._np
        if frame_bgr.shape[1::-1] != (self.w, self.h):
            frame_bgr = cv2.resize(frame_bgr, (self.w, self.h))

        self._frame_i += 1
        # detect_every>1: chỉ chạy YOLO mỗi N frame (dùng ở nhánh đồng bộ). Nhánh nền để =1.
        do_detect = (self._frame_i % self.detect_every == 0) or self.last_det is None

        if do_detect:
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
            self._stabilize_class(det)   # nhãn/màu ổn định theo track (chống car↔truck nhấp nháy)

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
            self._attach_velocity(det)   # gắn vận tốc/track → luồng hiển thị DỰ ĐOÁN vị trí (bù trễ)
            self.last_det = det          # cho service crop vật → vector DB + luồng hiển thị vẽ lại
        else:
            det = self.last_det          # frame BỎ QUA detect: dùng box gần nhất

        self.result.frames += 1
        self.result.unique_tracks = len(self._seen)
        self.result.elapsed_s = time.time() - self._t0
        self.last_frame = frame_bgr
        return det

    def render(self, frame_bgr, det=None, age_s: float = 0.0):
        """VẼ (phần NHẸ) — annotate ``det`` (mặc định lấy det gần nhất) lên frame. KHÔNG đếm.

        Dùng ở luồng hiển thị: gọi MỖI frame ở tốc độ gốc để video mượt; box lấy từ luồng
        detect nền. ``age_s`` = thời gian TỪ lúc det được tính → dịch box theo VẬN TỐC track để
        DỰ ĐOÁN vị trí hiện tại (xe nhanh không bị box chạy sau). det=None → trả frame gốc.
        """
        import cv2

        sv, np = self._sv, self._np
        if frame_bgr.shape[1::-1] != (self.w, self.h):
            frame_bgr = cv2.resize(frame_bgr, (self.w, self.h))
        if det is None:
            det = self.last_det
        if det is None:
            return frame_bgr             # chưa có detect nào → hiện frame gốc (vài chục ms đầu)

        draw_det = self._extrapolate(det, age_s)

        out = None
        if self.annos is not None:
            try:
                out = _annotate_sv(frame_bgr, draw_det, self.annos, self.line, self.result,
                                   self.w, self.h, cv2, sv)
            except Exception:  # noqa: BLE001
                out = None
        if out is None:
            out = _draw(frame_bgr, draw_det, self.scenario, self.line, self.zones_px, self.result,
                        self.w, self.h, cv2, np)
        return out

    def _extrapolate(self, det, age_s: float):
        """Trả BẢN SAO det với box đã DỊCH theo vận tốc × (age + độ trễ detect) → bù trễ luồng nền.

        Giới hạn 0.25s để không vọt quá xa; vật đứng yên/đi chậm (v≈0) → gần như không dịch.
        """
        np = self._np
        if not (getattr(det, "data", None) and "vx" in det.data) or len(det) == 0:
            return det
        a = age_s + self._last_latency_ms / 1000.0        # bù cả thời gian YOLO chạy
        a = max(0.0, min(a, 0.25))
        if a <= 0.0:
            return det
        try:
            import copy as _copy
            dx = det.data["vx"] * a
            dy = det.data["vy"] * a
            shift = np.column_stack([dx, dy, dx, dy]).astype(det.xyxy.dtype)
            draw_det = _copy.copy(det)
            draw_det.xyxy = det.xyxy + shift
            return draw_det
        except Exception:  # noqa: BLE001
            return det

    def _attach_velocity(self, det):
        """Gắn VẬN TỐC (px/s) mỗi box theo track vào ``det.data`` → luồng hiển thị dịch box tới
        vị trí dự đoán (bù độ trễ detect). Vật mới xuất hiện → vận tốc 0 (chưa có mốc trước)."""
        np = self._np
        n = len(det)
        vx = np.zeros(n, dtype="float32")
        vy = np.zeros(n, dtype="float32")
        now_t = time.time()
        if det.tracker_id is not None:
            for i, tid in enumerate(det.tracker_id):
                if tid is None:
                    continue
                tid = int(tid)
                x1, y1, x2, y2 = (float(v) for v in det.xyxy[i])
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                prev = self._track_pos.get(tid)
                if prev is not None:
                    pcx, pcy, pt = prev
                    dt = now_t - pt
                    if dt > 1e-3:
                        vx[i] = (cx - pcx) / dt
                        vy[i] = (cy - pcy) / dt
                self._track_pos[tid] = (cx, cy, now_t)
        det.data["vx"] = vx
        det.data["vy"] = vy

    def process(self, frame_bgr):
        """detect + render trong 1 lượt (đồng bộ) — cho test/notebook và nguồn không cần tách luồng."""
        det = self.detect(frame_bgr)
        return self.render(frame_bgr, det)

    # ------------------------------------------------------------------ #
    def _stabilize_class(self, det):
        """Gán nhãn LỚP theo track = lớp XUẤT HIỆN NHIỀU NHẤT của track đó.

        YOLO có thể gán CÙNG một xe lúc 'car' lúc 'truck' qua các frame (nhất là khi
        chạy chậm/ảnh nhỏ) → nhãn + màu + thống kê theo lớp nhấp nháy. Bình chọn đa số
        theo track cho ra nhãn ỔN ĐỊNH. (Số ĐẾM tổng không đổi — vẫn đếm theo track.)
        """
        np = self._np
        if det.tracker_id is None or not getattr(det, "data", None) or "class_name" not in det.data:
            return
        from ..sv_counting import _class_id

        # GỘP nhãn: mọi vật về 1 lớp (vd "vehicle") — 1 màu, không còn "sai loại".
        if self.merge_label:
            n = len(det.data["class_name"])
            det.data["class_name"] = np.array([self.merge_label] * n)
            det.class_id = np.array([_class_id(self.merge_label)] * n, dtype=int)
            return

        names = list(det.data["class_name"])
        ids = list(det.class_id) if det.class_id is not None else [0] * len(names)
        for i, tid in enumerate(det.tracker_id):
            if tid is None:
                continue
            counts = self._track_cls.setdefault(int(tid), {})
            counts[str(names[i])] = counts.get(str(names[i]), 0) + 1
            best = max(counts, key=counts.get)          # lớp thấy nhiều nhất tới giờ
            names[i], ids[i] = best, _class_id(best)
        det.data["class_name"] = np.array(names)
        det.class_id = np.array(ids, dtype=int)

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
