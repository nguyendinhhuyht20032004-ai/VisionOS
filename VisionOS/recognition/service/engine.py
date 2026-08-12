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
                 track_thresh: float = 0.25, smoother_len: int = 8, detect_every: int = 1,
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
        for kw in (dict(track_activation_threshold=track_thresh, minimum_consecutive_frames=3,
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

        # ReID nối track ĐỨT: vật bị che/detect trượt rồi hiện lại được ByteTrack cấp id MỚI →
        # gán về id CŨ theo ngoại hình + vị trí + thời gian → hết đếm trùng / phình "số vật".
        # Tắt/chỉnh qua env: REID_STITCH=0 (tắt), REID_SIM, REID_GAP, REID_DIST.
        import os

        from .reid import TrackStitcher
        self._stitcher = TrackStitcher(
            sim_thresh=float(os.environ.get("REID_SIM", "0.5")),
            max_gap=int(os.environ.get("REID_GAP", "60")),
            max_dist_frac=float(os.environ.get("REID_DIST", "0.3")),
            enabled=os.environ.get("REID_STITCH", "1").lower() not in ("0", "false", "no", ""),
        )

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
        # merge_label: gộp MỌI vật về 1 nhãn (vd "vehicle") — tùy chọn khi muốn đếm gộp phương
        # tiện thành 1 loại (bỏ tick trên web = giữ phân loại car/truck/bus, mỗi loại 1 màu).
        self.merge_label = merge_label

    # ------------------------------------------------------------------ #
    def process(self, frame_bgr):
        """Xử lý 1 frame BGR → trả frame ĐÃ ANNOTATE (BGR). Cập nhật số đếm nội bộ."""
        import cv2

        sv, np = self._sv, self._np
        if frame_bgr.shape[1::-1] != (self.w, self.h):
            frame_bgr = cv2.resize(frame_bgr, (self.w, self.h))

        self._frame_i += 1
        # detect_every>1: chỉ chạy YOLO mỗi N frame (nặng nhất) → throughput cao hơn trên CPU.
        # Frame bỏ qua: vẽ lại box GẦN NHẤT (không đếm lại) → video vẫn mượt.
        do_detect = (self._frame_i % self.detect_every == 0) or self.last_det is None

        if do_detect:
            t0 = time.time()
            dr = self.detector.detect(frame_bgr, self.scenario.prompt)
            self._last_latency_ms = (time.time() - t0) * 1000
            self.result.total_detections += len(dr.detections)

            det = _to_sv(dr.detections, sv, np)
            det = self.tracker.update_with_detections(det)
            det = self._stitch(det, frame_bgr)   # ReID: nối lại track bị đứt (giữ 1 id/vật)
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
                in_match, out_match = self.line.trigger(det)
                self.result.in_count = int(self.line.in_count)
                self.result.out_count = int(self.line.out_count)
                # Lưu lại các sự kiện cắt vạch để StreamManager đọc và publish JSON
                if not hasattr(det, "cross_events"):
                    det.cross_events = []
                for i, is_in in enumerate(in_match):
                    if is_in and det.tracker_id[i] is not None:
                        det.cross_events.append((det.tracker_id[i], "IN"))
                for i, is_out in enumerate(out_match):
                    if is_out and det.tracker_id[i] is not None:
                        det.cross_events.append((det.tracker_id[i], "OUT"))
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
            self.last_det = det          # cho service crop vật → vector DB + frame bỏ-detect dùng lại
        else:
            det = self.last_det          # frame BỎ QUA detect: vẽ lại box gần nhất

        self.result.frames += 1
        self.result.unique_tracks = len(self._seen)
        self.result.elapsed_s = time.time() - self._t0
        self.last_frame = frame_bgr

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

        # ---- THÊM LOG TERMINAL ĐỂ DEMO BÁO CÁO ----
        if self._frame_i % 10 == 0:
            s = self.stats()
            if self.scenario.counting_type == "line":
                print(f"[AI Service Log] Frame {self._frame_i:04d} | Đang theo dõi (Tracks): {s.get('tracks')} | IN: {s.get('in')} | OUT: {s.get('out')} | Tốc độ: {s.get('fps')} fps")
            else:
                print(f"[AI Service Log] Frame {self._frame_i:04d} | Đang theo dõi (Tracks): {s.get('tracks')} | Đang có trên màn hình: {s.get('in_zone', s.get('in_frame', 0))} | Tốc độ: {s.get('fps')} fps")

        return out

    # ------------------------------------------------------------------ #
    def _stitch(self, det, frame_bgr):
        """Remap ``tracker_id`` qua ReID (nối track đứt). Giữ nguyên nếu tắt/không có track."""
        st = getattr(self, "_stitcher", None)
        if st is None or not st.enabled or det.tracker_id is None or len(det) == 0:
            return det
        np = self._np
        from .vectordb import embed_crop

        boxes = det.xyxy
        if getattr(det, "data", None) and "class_name" in det.data:
            classes = [str(c) for c in det.data["class_name"]]
        else:
            classes = [None] * len(det)
        embs = []
        for i in range(len(det)):
            x1, y1, x2, y2 = (int(v) for v in boxes[i])
            crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            embs.append(embed_crop(crop))
        raw = [int(t) if t is not None else None for t in det.tracker_id]
        stable = st.remap(raw, boxes, embs, classes, (self.w, self.h))
        det.tracker_id = np.array([s if s is not None else -1 for s in stable], dtype=int)
        return det

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
            
        # Thêm chi tiết output của supervision (tracker_id, class, conf, xyxy)
        det_list = []
        if self.last_det is not None:
            for i in range(len(self.last_det)):
                tid = int(self.last_det.tracker_id[i]) if self.last_det.tracker_id is not None else None
                cls_name = str(self.last_det.data["class_name"][i]) if (getattr(self.last_det, "data", None) and "class_name" in self.last_det.data) else "obj"
                conf = round(float(self.last_det.confidence[i]), 3) if self.last_det.confidence is not None else 0.0
                # Lấy toạ độ xyxy
                box = self.last_det.xyxy[i]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                det_list.append({
                    "track_id": tid,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
        d["detections"] = det_list
        
        return d
