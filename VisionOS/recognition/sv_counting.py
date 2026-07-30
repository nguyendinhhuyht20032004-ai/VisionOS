"""Đếm bằng thư viện **supervision** (Roboflow): ByteTrack + LineZone/PolygonZone.

Tracker **ByteTrack** mạnh hơn ``CentroidTracker`` tự viết rất nhiều → giữ danh tính
ổn định qua frame, nên đếm **cắt vạch** / **chiếm vùng** chính xác hơn. Overlay + banner
vẽ bằng ``cv2`` (KHÔNG phụ thuộc annotator của supervision để tránh lệch API giữa các
phiên bản). Chỉ cần supervision core: ``ByteTrack, LineZone, PolygonZone, Detections, Point``.

Trả về ``CountResult`` (giống pipeline tự viết) nên scorecard/print dùng chung.
"""

from __future__ import annotations

import time
import unicodedata

from .counting import CountResult

__all__ = ["sv_available", "sv_run"]


def sv_available() -> bool:
    """True nếu import được supervision (để engine 'auto' tự chọn)."""
    try:
        import supervision  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _ascii(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _to_sv(dets, sv, np):
    """List Detection (của mình) → sv.Detections (xyxy/confidence/class_id).

    QUAN TRỌNG: báo confidence CAO (0.9) cho ByteTrack. Vì detector ĐÃ lọc theo
    ``--confidence`` rồi → mọi box trả về đều là vật ta MUỐN bám. ByteTrack có ngưỡng
    nội bộ (det_thresh ≈ activation+0.1, mặc định ~0.35) sẽ LOẠI vật conf thấp → không
    tạo track → không vẽ/đếm. Nâng conf ở đây để ByteTrack bám MỌI vật đã detect.
    """
    if not dets:
        return sv.Detections.empty()
    xyxy = np.array([[float(v) for v in d.bbox.as_xyxy()] for d in dets], dtype=float)
    conf = np.full(len(dets), 0.9, dtype=float)
    # GIỮ tên lớp thật (car/truck/tomato…) để ĐÁNH NHÃN; class_id theo lớp → màu theo lớp.
    names = [(getattr(d, "label", None) or "object") for d in dets]
    uniq = {n: i for i, n in enumerate(sorted(set(names)))}
    cls = np.array([uniq[n] for n in names], dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls,
                         data={"class_name": np.array(names)})


def _make_polygon_zone(sv, poly, w, h):
    """Tạo PolygonZone chịu được khác biệt API giữa các bản supervision."""
    try:
        return sv.PolygonZone(polygon=poly)                      # supervision ≥0.19
    except TypeError:
        return sv.PolygonZone(polygon=poly, frame_resolution_wh=(w, h))  # bản cũ


def sv_run(frames, scenario, detector, resolution, max_frames=300, writer=None, track_thresh=0.1):
    """Đếm 1 video bằng supervision. Trả về ``CountResult``.

    frames: iterable frame BGR (đã resize về ``resolution``).
    writer: nếu có (cv2.VideoWriter) → vẽ overlay + ghi video output.
    track_thresh: ngưỡng KÍCH HOẠT track của ByteTrack — phải **≤ ngưỡng detect** kẻo vật
      conf thấp (ĐÃ detect) bị ByteTrack loại → không vẽ/đếm. Mặc định 0.1 để bám MỌI vật.
    """
    import warnings

    import cv2
    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # ẩn FutureWarning ByteTrack deprecated
        import supervision as sv

    w, h = resolution
    # track_activation_threshold THẤP (mặc định sv 0.25 — CAO hơn conf detect nên hay rớt
    # vật) + minimum_consecutive_frames=1 (xác nhận ngay) + lost_track_buffer lớn (giữ
    # track qua che khuất, ít đứt-nối ID). Fallback dần cho bản supervision cũ.
    tracker = None
    for kwargs in (
        dict(track_activation_threshold=track_thresh, minimum_consecutive_frames=1, lost_track_buffer=60),
        dict(track_thresh=track_thresh),
        {},
    ):
        try:
            tracker = sv.ByteTrack(**kwargs)
            break
        except TypeError:
            continue

    line = None
    polys = []
    zones_px = [np.array([[int(x), int(y)] for x, y in z.to_pixels(w, h)], dtype=np.int32)
                for z in scenario.build_zones()]
    if scenario.counting_type == "line":
        (sx, sy), (ex, ey) = scenario.build_line().endpoints(w, h)
        line = sv.LineZone(start=sv.Point(float(sx), float(sy)),
                           end=sv.Point(float(ex), float(ey)))
    else:
        for z in scenario.build_zones():
            poly = np.array([[int(x), int(y)] for x, y in z.to_pixels(w, h)], dtype=int)
            polys.append(_make_polygon_zone(sv, poly, w, h))

    # Annotator supervision (box bo góc + nhãn + vệt + bộ đếm trên vạch/vùng) — dựng
    # 1 lần (TraceAnnotator tích luỹ vệt qua frame). Lỗi API → rơi về vẽ cv2.
    annos = None
    if writer is not None:
        try:
            annos = {
                "box": sv.RoundBoxAnnotator(color_lookup=sv.ColorLookup.TRACK, thickness=2),
                "label": sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK, text_scale=0.45),
                "trace": sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK, thickness=2, trace_length=30),
                "line": sv.LineZoneAnnotator(thickness=2, text_scale=0.7) if line is not None else None,
                "zones": [sv.PolygonZoneAnnotator(zone=pz, color=sv.Color.GREEN, thickness=2)
                          for pz in polys],
            }
        except Exception:  # noqa: BLE001
            annos = None

    res = CountResult(scenario_key=scenario.key, counting_type=scenario.counting_type,
                      in_label=scenario.in_label, out_label=scenario.out_label)
    seen: set = set()
    t0 = time.time()
    for i, frame in enumerate(frames):
        if i >= max_frames:
            break
        dr = detector.detect(frame, scenario.prompt)
        res.total_detections += len(dr.detections)
        det = _to_sv(dr.detections, sv, np)
        det = tracker.update_with_detections(det)

        if det.tracker_id is not None:
            for tid in det.tracker_id:
                if tid is not None:
                    seen.add(int(tid))

        if line is not None:
            line.trigger(det)
            res.in_count = int(line.in_count)
            res.out_count = int(line.out_count)
        if polys:
            inside = np.zeros(len(det), dtype=bool)
            for pz in polys:
                inside = inside | np.asarray(pz.trigger(det), dtype=bool)   # OR: trong BẤT KỲ vùng nào
            cur = int(inside.sum())
            res.zone_current = cur
            res.zone_peak = max(res.zone_peak, cur)

        res.frames += 1
        res.unique_tracks = len(seen)
        if writer is not None:
            out = None
            if annos is not None:
                try:
                    out = _annotate_sv(frame, det, annos, line, res, w, h, cv2, sv)
                except Exception:  # noqa: BLE001 — annotator lỗi → vẽ cv2
                    out = None
            writer.write(out if out is not None
                         else _draw(frame, det, scenario, line, zones_px, res, w, h, cv2, np))
    res.elapsed_s = time.time() - t0
    return res


def _annotate_sv(frame, det, annos, line, res, w, h, cv2, sv):
    """Vẽ output bằng annotator supervision (đẹp, đúng chất sv) + banner tóm tắt."""
    import numpy as np

    f = frame.copy()
    for za in annos["zones"]:
        f = za.annotate(scene=f)                       # vùng + số trong vùng
    if len(det):
        f = annos["trace"].annotate(f, det)            # vệt chuyển động
        f = annos["box"].annotate(f, det)              # box bo góc theo track
        names = det.data.get("class_name") if getattr(det, "data", None) else None
        tids = det.tracker_id
        labels = []
        for i in range(len(det)):
            nm = _ascii(str(names[i])) if names is not None else ""
            tid = tids[i] if tids is not None else None
            labels.append(f"{nm} #{int(tid)}".strip() if tid is not None else (nm or "?"))
        f = annos["label"].annotate(f, det, labels=labels)  # NHÃN: tên vật + #track
    if line is not None and annos["line"] is not None:
        f = annos["line"].annotate(f, line)            # bộ đếm in/out trên vạch
    # Banner tóm tắt trên cùng (ASCII).
    if line is not None:
        txt = f"{res.in_label}:{res.in_count}  {res.out_label}:{res.out_count}  frame:{res.frames}"
    else:
        txt = f"trong vung:{res.zone_current}  dinh:{res.zone_peak}  frame:{res.frames}"
    f = np.ascontiguousarray(f)
    cv2.rectangle(f, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(f, _ascii(txt), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return f


def _draw(frame, det, scenario, line, zones_px, res, w, h, cv2, np):
    """Vẽ vạch/vùng + box + #track-id + banner số đếm (ASCII) — soi mắt thường."""
    img = frame
    YELLOW, GREEN, CYAN, WHITE = (0, 255, 255), (0, 200, 0), (255, 255, 0), (255, 255, 255)
    if line is not None:
        (sx, sy), (ex, ey) = scenario.build_line().endpoints(w, h)
        cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)), YELLOW, 3)
    for pts in zones_px:
        ov = img.copy()
        cv2.fillPoly(ov, [pts], (0, 170, 0))
        cv2.addWeighted(ov, 0.25, img, 0.75, 0, img)
        cv2.polylines(img, [pts], True, GREEN, 2)
    n = len(det)
    xyxy = det.xyxy if n else []
    tids = det.tracker_id if (n and det.tracker_id is not None) else [None] * n
    names = det.data.get("class_name") if (n and getattr(det, "data", None)) else None
    for idx, ((x1, y1, x2, y2), tid) in enumerate(zip(xyxy, tids)):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), CYAN, 2)
        nm = _ascii(str(names[idx])) if names is not None else ""
        lbl = f"{nm} #{int(tid)}".strip() if tid is not None else (nm or "#?")
        cv2.putText(img, lbl, (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
    if line is not None:
        txt = f"{scenario.in_label}:{res.in_count}  {scenario.out_label}:{res.out_count}  frame:{res.frames}"
    else:
        txt = f"trong vung:{res.zone_current}  dinh:{res.zone_peak}  frame:{res.frames}"
    cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(img, _ascii(txt), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    return img
