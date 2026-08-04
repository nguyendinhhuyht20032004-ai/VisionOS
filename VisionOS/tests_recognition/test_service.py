"""Test service AI đếm-theo-camera (KHÔNG cần GPU; fastapi tùy chọn).

Engine + builder + camera dùng supervision/cv2 (đã cài); detector là GIẢ.
"""

import numpy as np
import pytest

from recognition.base import BoundingBox, Detection, DetectorResult
from recognition.service import FrameSource, StreamingCounter, make_scenario, parse_source
from recognition.service.builder import wants_yolo


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def test_wants_yolo_routing():
    # TỪ-LỚP đơn thuần → YOLO
    assert wants_yolo("person") and wants_yolo("car") and wants_yolo("a car") and wants_yolo("truck")
    assert wants_yolo("xe tải")                       # cụm lớp COCO
    # có MÀU / MÔ TẢ → LocateAnything (open-vocab) — mấu chốt để nhận MÀU XE
    assert not wants_yolo("a red car") and not wants_yolo("a white car")
    assert not wants_yolo("a large truck") and not wants_yolo("a person wearing a backpack")
    # không thuộc COCO → LocateAnything
    assert not wants_yolo("object") and not wants_yolo("carton box") and not wants_yolo("tomato")


def test_color_query_routes_to_locate():
    _, kind = make_scenario("a red car", "line")
    assert kind == "locate"                           # màu xe → LocateAnything
    _, kind = make_scenario("car", "line")
    assert kind == "yolo"                             # đếm mọi xe → YOLO (nhanh)


def test_make_scenario_line_yolo():
    sc, kind = make_scenario("person", "line", line=[0, 50, 100, 50], resolution=(320, 180))
    assert kind == "yolo"
    assert sc.counting_type == "line" and sc.resolution == (320, 180)
    assert sc.line_start_pct == (0.0, 50.0) and sc.line_end_pct == (100.0, 50.0)
    sc.validate()


def test_make_scenario_zone_locate():
    sc, kind = make_scenario("object", "zone", zone=[[10, 10], [90, 10], [90, 90], [10, 90]])
    assert kind == "locate" and sc.counting_type == "zone"
    assert len(sc.zone_points_pct) == 4
    sc.validate()


def test_make_scenario_model_override():
    _, kind = make_scenario("person", model="locate")
    assert kind == "locate"
    _, kind = make_scenario("object", model="yolo")
    assert kind == "yolo"


def test_parse_source():
    assert parse_source("0") == 0 and parse_source("1") == 1
    assert parse_source("rtsp://x") == "rtsp://x"
    assert parse_source("/a/b.mp4") == "/a/b.mp4"


# --------------------------------------------------------------------------- #
# engine (StreamingCounter) với detector GIẢ
# --------------------------------------------------------------------------- #
class _FakeDet:
    """1 box đi TỪ TRÊN XUỐNG → cắt vạch ngang giữa khung."""

    def __init__(self, n):
        self.i, self.n = 0, n

    def detect(self, frame, prompt):
        h, w = frame.shape[:2]
        y = int((self.i / max(self.n - 1, 1)) * (h - 40))
        self.i += 1
        d = [Detection(BoundingBox(w * 0.4, float(y), w * 0.4 + 40, float(y + 40)), "object", 0.9)]
        return DetectorResult(d, raw="", latency_ms=0.0, model_name="fake")


def test_streaming_counter_counts_crossing_and_annotates():
    n = 14
    sc, _ = make_scenario("object", "line", line=[0, 50, 100, 50], resolution=(320, 180))
    sc_counter = StreamingCounter(sc, _FakeDet(n), resolution=(320, 180))
    out = None
    for _ in range(n):
        out = sc_counter.process(np.zeros((180, 320, 3), dtype="uint8"))
    assert out.shape == (180, 320, 3)                 # trả frame đã annotate
    st = sc_counter.stats()
    assert st["frames"] == n
    assert st["total"] >= 1                           # vật đã cắt vạch (đếm được)
    assert st["tracks"] >= 1
    assert st["counting_type"] == "line"


def test_make_scenario_fullscreen():
    sc, kind = make_scenario("person", "fullscreen", resolution=(320, 180))
    assert sc.counting_type == "fullscreen"
    sc.validate()                                      # fullscreen KHÔNG cần vạch/vùng


def test_streaming_counter_fullscreen_counts_whole_frame():
    # TOÀN MÀN HÌNH: đếm mọi vật trong khung (in_frame/peak) + tổng track.
    sc, _ = make_scenario("object", "fullscreen", resolution=(320, 180))
    sc_counter = StreamingCounter(sc, _FakeDet(8), resolution=(320, 180))
    out = None
    for _ in range(8):
        out = sc_counter.process(np.zeros((180, 320, 3), dtype="uint8"))
    assert out.shape == (180, 320, 3)
    st = sc_counter.stats()
    assert st["counting_type"] == "fullscreen"
    assert st["in_frame"] >= 1 and st["peak"] >= 1     # có vật trong khung
    assert st["total"] >= 1                             # tổng vật khác nhau
    assert sc_counter.line is None and not sc_counter.polys   # không dựng vạch/vùng


def test_streaming_counter_zone_stats():
    sc, _ = make_scenario("object", "zone",
                          zone=[[0, 0], [100, 0], [100, 100], [0, 100]], resolution=(320, 180))
    sc_counter = StreamingCounter(sc, _FakeDet(6), resolution=(320, 180))
    for _ in range(6):
        sc_counter.process(np.zeros((180, 320, 3), dtype="uint8"))
    st = sc_counter.stats()
    assert st["counting_type"] == "zone"
    assert "in_zone" in st and "zone_peak" in st
    assert st["zone_peak"] >= 1                        # vùng phủ cả khung → có vật trong vùng


def test_frame_source_missing_file_sets_error():
    fs = FrameSource("/khong/ton/tai_abc.mp4", reconnect=False).start()
    import time
    for _ in range(20):
        if not fs.alive:
            break
        time.sleep(0.1)
    assert fs.read() is None
    fs.stop()


# --------------------------------------------------------------------------- #
# API (chỉ chạy nếu có fastapi)
# --------------------------------------------------------------------------- #
def test_api_smoke():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from recognition.service.app import app

    client = TestClient(app)
    assert client.get("/").status_code == 200          # trang web
    assert client.get("/api/jobs").json() == []        # chưa có job
    assert client.get("/api/jobs/khong-co").status_code == 404
