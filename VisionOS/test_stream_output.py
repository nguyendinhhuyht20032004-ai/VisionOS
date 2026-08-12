# -*- coding: utf-8 -*-
"""Test NHANH phần output của StreamWorker — không cần Docker, GPU, camera hay Redis.

Chạy:  python test_stream_output.py

Phủ 5 thứ:
  1. Khung hình TRỐNG vẫn phải bắn track_event "end" sau 2 giây.
  2. PATCH chỉ gửi 1 trường không được xoá mất các trường còn lại.
  3. PATCH dựng lại tracker → phải đóng vòng đời track cũ trước.
  4. track_event / stream_status vào cả stream chính lẫn stream bền.
  5. classes nhiều lớp phải ra đủ mọi lớp, không im lặng bỏ bớt.

Thư viện nặng (supervision/ultralytics/torch) được thay bằng stub nếu máy chưa cài —
test này chỉ đụng tới phần logic thuần Python.
"""
import json
import os
import sys
import time
import types
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for _name in ("supervision", "ultralytics", "torch", "cv2", "redis"):
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = MagicMock()

from recognition.service.stream_manager import (                        # noqa: E402
    StreamParams, StreamWorker,
)


# --------------------------------------------------------------------------- #
# Đồ giả
# --------------------------------------------------------------------------- #
class FakeRedis:
    """Ghi lại mọi lần XADD để test soi được (nhớ cả stream đích)."""

    def __init__(self):
        self.msgs = []
        self.by_stream = {}

    def xadd(self, key, fields, **kw):
        msg = json.loads(fields["data"])
        self.by_stream.setdefault(key, []).append(msg)
        if key == "VISIONOS_RESULTS":
            self.msgs.append(msg)

    def of_type(self, t):
        return [m for m in self.msgs if m.get("type") == t]


class FakeDet:
    """Giả lập supervision.Detections — chỉ cần __len__, tracker_id, xyxy, data."""

    def __init__(self, n=0):
        self.tracker_id = None if n == 0 else list(range(n))
        self.xyxy = [[10.0, 20.0, 60.0, 120.0]] * n
        self.confidence = [0.9] * n
        self.data = {"class_name": ["person"] * n} if n else {}

    def __len__(self):
        return len(self.xyxy)


def make_worker(redis_cli=None):
    """Dựng StreamWorker KHÔNG chạy __init__ (tránh nạp YOLO)."""
    w = object.__new__(StreamWorker)
    w.stream_id, w.camera_id = "s1", "cam-01"
    w.redis = redis_cli or FakeRedis()
    w.maxlen, w.stream_key = 1000, "VISIONOS_RESULTS"
    w._reported_tracks, w._track_last_seen, w._track_classes = set(), {}, {}
    w.event_stream_key, w.event_maxlen = "VISIONOS_EVENTS", 50000
    w.counter = types.SimpleNamespace(last_det=None, stats=lambda: {})
    return w


# --------------------------------------------------------------------------- #
def test_end_event_khi_khung_hinh_trong():
    """Mọi người rời khỏi khung → vẫn phải bắn "end" sau 2 giây.

    Lỗi cũ: vòng quét timeout nằm TRONG `if det and ...`, mà Detections có __len__
    nên khung trống là falsy → không track nào kết thúc, backend treo vĩnh viễn.
    """
    r = FakeRedis()
    w = make_worker(r)

    w.counter.last_det = FakeDet(2)
    w._publish_result(None)
    starts = [m for m in r.of_type("track_event") if m["event"] == "start"]
    assert len(starts) == 2, f"mong 2 'start', nhận {len(starts)}"

    # Khung hình trống + đã mất dấu 3 giây
    w.counter.last_det = FakeDet(0)
    for tid in w._track_last_seen:
        w._track_last_seen[tid] = time.time() - 3.0
    r.msgs.clear()
    w._publish_result(None)

    ends = [m for m in r.of_type("track_event") if m["event"] == "end"]
    assert len(ends) == 2, f"mong 2 'end', nhận {len(ends)}"
    assert all(m["class"] == "person" for m in ends), ends
    assert w._track_last_seen == {} and w._reported_tracks == set()
    print("  ✅ khung hình trống vẫn bắn đủ 2 sự kiện 'end'")


def test_patch_khong_xoa_truong_khac():
    """PATCH {"conf": ...} không được thổi bay classes/roi đang chạy."""
    w = make_worker()
    w.params = StreamParams(classes=["person"], conf=0.3, roi=[[0, 50], [100, 50]])
    w._build_counter = lambda: None                      # bỏ qua nạp YOLO

    w.update_params(StreamParams(**json.loads('{"conf": 0.5}')))
    assert w.params.conf == 0.5
    assert w.params.classes == ["person"], "classes bị xoá"
    assert w.params.roi == [[0, 50], [100, 50]], "roi bị xoá → luồng nhảy về fullscreen"
    print("  ✅ PATCH một trường giữ nguyên các trường còn lại")

    # Gửi tường minh null vẫn phải xoá được
    w.update_params(StreamParams(**json.loads('{"roi": null}')))
    assert w.params.roi is None and w.params.classes == ["person"]
    print('  ✅ gửi tường minh {"roi": null} vẫn xoá được roi')


def test_patch_dong_track_cu():
    """PATCH dựng lại tracker → phải đóng vòng đời track cũ trước.

    Tracker mới đánh track_id lại từ 1. Không đóng track cũ thì backend treo chúng vĩnh
    viễn, rồi lại nhận đúng những id đó từ tracker mới — trùng id, sai dữ liệu.
    """
    r = FakeRedis()
    w = make_worker(r)
    w.params = StreamParams(classes=["person"], conf=0.3, roi=[[0, 50], [100, 50]])
    w._build_counter = lambda: None
    w._reported_tracks = {7, 9}
    w._track_classes = {7: "person", 9: "car"}
    w._track_last_seen = {7: 1.0, 9: 2.0}
    w.update_params(StreamParams(**json.loads('{"conf": 0.5}')))

    ends = [m for m in r.of_type("track_event") if m["event"] == "end"]
    assert {m["track_id"] for m in ends} == {"7", "9"}, ends
    assert {m["class"] for m in ends} == {"person", "car"}, "phải giữ đúng lớp của track cũ"
    assert w._reported_tracks == set() and w._track_classes == {} and w._track_last_seen == {}
    print(f"  ✅ đóng {len(ends)} track cũ bằng sự kiện 'end', đúng lớp")


def test_tin_nghiep_vu_vao_stream_ben():
    """track_event / stream_status phải vào CẢ stream chính lẫn stream bền.

    Stream chính bị `frame` (10 tin/giây) đẩy văng sau ~2 phút. Tin nghiệp vụ mất là
    sai số liệu, nên phải có bản sao ở stream MAXLEN lớn.
    """
    r = FakeRedis()
    w = make_worker(r)

    w.counter.last_det = FakeDet(1)
    w._publish_result(None)                 # → 1 frame + 1 track_event(start)
    w._publish_status("source_lost", "mất kết nối RTSP")

    main = r.by_stream["VISIONOS_RESULTS"]
    durable = r.by_stream["VISIONOS_EVENTS"]

    assert {m["type"] for m in main} == {"frame", "track_event", "stream_status"}, \
        "stream chính phải giữ ĐỦ mọi loại tin (backend cũ không phải sửa gì)"
    assert {m["type"] for m in durable} == {"track_event", "stream_status"}, \
        "stream bền chỉ chứa tin nghiệp vụ, KHÔNG chứa frame"
    assert not [m for m in durable if m["type"] == "frame"]
    print(f"  ✅ stream chính: {len(main)} tin (đủ 3 loại)")
    print(f"  ✅ stream bền  : {len(durable)} tin nghiệp vụ, không lẫn 'frame'")

    st = [m for m in durable if m["type"] == "stream_status"][-1]
    assert st["status"] == "source_lost" and st["camera_id"] == "cam-01"
    assert set(st) == {"type", "camera_id", "stream_id", "status", "detail", "timestamp"}
    print(f"  ✅ tin trạng thái: status={st['status']} detail={st['detail']!r}")


def test_loc_nhieu_lop():
    """classes=['person','car'] phải ra CẢ HAI, không phải chỉ lớp đầu bảng alias."""
    try:
        from recognition.detectors.ultralytics_yolo import UltralyticsYoloDetector as D
    except Exception as e:                       # thiếu ultralytics thật → bỏ qua
        print(f"  ⏭  bỏ qua (không import được detector: {type(e).__name__})")
        return

    d = D.__new__(D)
    d.want = None
    cases = {
        ("person",):          {"person"},
        ("person", "car"):    {"person", "car"},
        ("car", "person"):    {"person", "car"},     # thứ tự không được ảnh hưởng
        ("car", "truck"):     {"car", "truck"},
    }
    for classes, expect in cases.items():
        got = d._wanted_classes(",".join(classes))
        assert got == expect, f"classes={list(classes)} → {sorted(got)}, mong {sorted(expect)}"
        print(f"  ✅ classes={str(list(classes)):<22} → {sorted(got)}")

    # prompt ngôn ngữ tự nhiên (một cụm, không dấu phẩy) vẫn phải chạy như cũ
    assert d._wanted_classes("đếm người đi bộ") == {"person"}
    print("  ✅ prompt tự nhiên 'đếm người đi bộ' → ['person'] (không vỡ)")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [
        ("Sự kiện 'end' khi khung hình trống", test_end_event_khi_khung_hinh_trong),
        ("PATCH không xoá trường khác",        test_patch_khong_xoa_truong_khac),
        ("PATCH đóng track cũ",                test_patch_dong_track_cu),
        ("Tin nghiệp vụ vào stream bền",       test_tin_nghiep_vu_vao_stream_ben),
        ("Lọc nhiều lớp vật thể",              test_loc_nhieu_lop),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"  ❌ HỎNG: {e}")
    print()
    if failed:
        print(f"❌ {failed}/{len(tests)} nhóm test hỏng.")
        sys.exit(1)
    print(f"🎉 Cả {len(tests)} nhóm test đều đạt.")
