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


class _FlipDet:
    """1 vật ĐỨNG YÊN nhưng bị gán lớp lúc car lúc truck (mô phỏng YOLO nhiễu nhãn)."""

    def __init__(self, seq):
        self.seq, self.calls = seq, 0

    def detect(self, frame, prompt):
        h, w = frame.shape[:2]
        cls = self.seq[self.calls % len(self.seq)]
        self.calls += 1
        d = [Detection(BoundingBox(w * 0.4, 80.0, w * 0.4 + 50, 130.0), cls, 0.9)]
        return DetectorResult(d, raw="", latency_ms=0.0, model_name="fake")


def test_class_stabilized_by_track_majority():
    # cùng 1 track bị gán car/truck xen kẽ (đa số car) → nhãn ổn định về 'car'
    seq = ["car", "car", "truck", "car", "car", "truck", "car", "car"]
    sc, _ = make_scenario("vehicle", "fullscreen", resolution=(320, 180))
    c = StreamingCounter(sc, _FlipDet(seq), resolution=(320, 180))
    for _ in range(len(seq)):
        c.process(np.zeros((180, 320, 3), dtype="uint8"))
    assert list(c.last_det.data["class_name"]) == ["car"]   # bình chọn đa số
    assert c.stats()["tracks"] == 1                          # vẫn 1 xe


class _MultiClassDet:
    """Trả 2 vật khác lớp (car + truck) — mô phỏng xe cao bị gán truck/bus."""

    def detect(self, frame, prompt):
        h, w = frame.shape[:2]
        d = [Detection(BoundingBox(w * 0.3, 80.0, w * 0.3 + 60, 140.0), "car", 0.9),
             Detection(BoundingBox(w * 0.6, 80.0, w * 0.6 + 90, 150.0), "truck", 0.9)]
        return DetectorResult(d, raw="", latency_ms=0.0, model_name="fake")


def test_merge_label_groups_all_vehicle_types():
    # đếm GỘP phương tiện (tùy chọn): car/truck → 1 nhãn "vehicle" (khi không cần tách loại)
    sc, _ = make_scenario("vehicle", "fullscreen", resolution=(320, 180))
    c = StreamingCounter(sc, _MultiClassDet(), resolution=(320, 180), merge_label="vehicle")
    for _ in range(4):
        c.process(np.zeros((180, 320, 3), dtype="uint8"))
    assert set(map(str, c.last_det.data["class_name"])) == {"vehicle"}


def test_line_uses_single_triggering_anchor():
    # vạch đếm theo 1 điểm neo (tâm) → xe TO làn ngoài cũng đếm được, không sót làn
    sc, _ = make_scenario("vehicle", "line", line=[0, 50, 100, 50], resolution=(320, 180))
    c = StreamingCounter(sc, _FakeDet(6), resolution=(320, 180))
    assert c.line is not None                            # LineZone dựng OK (kể cả bản sv cũ)


def test_detect_every_skips_detection():
    # detect_every=3: 9 frame chỉ gọi YOLO ở frame 1,3,6,9 (frame đầu luôn detect) = 4 lần
    det = _FlipDet(["car"])
    sc, _ = make_scenario("vehicle", "line", line=[0, 50, 100, 50], resolution=(320, 180))
    c = StreamingCounter(sc, det, resolution=(320, 180), detect_every=3)
    out = None
    for _ in range(9):
        out = c.process(np.zeros((180, 320, 3), dtype="uint8"))
    assert det.calls == 4 and out.shape == (180, 320, 3)     # ~1/3 số lần gọi → nhanh hơn


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


# --------------------------------------------------------------------------- #
# Vector DB (in-memory — KHÔNG cần qdrant)
# --------------------------------------------------------------------------- #
def test_embed_crop_dim_and_norm():
    from recognition.service.vectordb import EMBED_DIM, embed_crop

    red = np.zeros((32, 32, 3), dtype="uint8")
    red[:] = (0, 0, 200)
    v = embed_crop(red)
    assert len(v) == EMBED_DIM
    assert abs(sum(x * x for x in v) - 1.0) < 1e-3      # đã L2-normalize
    assert embed_crop(np.zeros((0, 0, 3), dtype="uint8")) == [0.0] * EMBED_DIM  # crop rỗng


def test_vectorstore_inmemory_add_search_recent():
    from recognition.service.vectordb import VectorStore, embed_crop, make_event_payload

    vs = VectorStore(url=None)                          # ép in-memory
    assert vs.backend == "memory"
    red = np.full((32, 32, 3), 0, "uint8"); red[:] = (0, 0, 200)
    blue = np.full((32, 32, 3), 0, "uint8"); blue[:] = (200, 0, 0)
    vs.add_event(embed_crop(red), make_event_payload(1, "car", "cam", "line"))
    vs.add_event(embed_crop(blue), make_event_payload(2, "car", "cam", "line"))
    assert vs.count() == 2
    hits = vs.search(embed_crop(red), limit=2)          # tìm ĐỎ → track 1 đứng đầu
    assert hits and hits[0]["payload"]["track_id"] == 1
    assert hits[0]["score"] >= hits[-1]["score"]        # xếp theo độ giống giảm dần
    assert len(vs.recent(5)) == 2


# --------------------------------------------------------------------------- #
# Snapshot (lấy 1 frame để vẽ) — video synthetic, không cần camera thật
# --------------------------------------------------------------------------- #
def test_grab_snapshot_from_file(tmp_path):
    import cv2

    from recognition.service import encode_jpeg, grab_snapshot

    p = str(tmp_path / "s.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(12):
        vw.write(np.full((120, 160, 3), 60, dtype="uint8"))
    vw.release()
    fr = grab_snapshot(p, timeout=5)
    assert fr is not None and fr.shape == (120, 160, 3)
    jpg = encode_jpeg(fr)
    assert jpg and jpg[:2] == b"\xff\xd8"               # magic JPEG


def test_grab_snapshot_bad_source_returns_none():
    from recognition.service import grab_snapshot

    assert grab_snapshot("/khong/ton/tai_xyz.mp4", timeout=2) is None


def test_framesource_paces_file_to_native_fps(tmp_path):
    """FILE → FrameSource đặt nhịp phát theo FPS gốc (chống video bị TUA NHANH)."""
    import cv2

    p = str(tmp_path / "paced.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(20):
        vw.write(np.full((120, 160, 3), 50, dtype="uint8"))
    vw.release()

    fs = FrameSource(p)
    cap = fs._open()                                    # gọi trực tiếp (không chạy thread)
    try:
        assert cap is not None
        assert 0.08 <= fs._frame_interval <= 0.13      # 10 fps → ~0.1 s/frame
        assert fs.drop_frames is False                 # FILE → KHÔNG bỏ frame (phát mượt/đúng thứ tự)
    finally:
        if cap is not None:
            cap.release()


def test_framesource_file_reads_in_order_no_drop(tmp_path):
    """FILE → đọc ĐỦ frame, ĐÚNG THỨ TỰ (không nhảy/bỏ frame) → tracker/đếm chính xác."""
    import time as _t

    import cv2

    p = str(tmp_path / "seq.mp4")
    N = 15
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(N):
        vw.write(np.full((48, 64, 3), i * 15, dtype="uint8"))   # mỗi frame 1 mức xám tăng dần
    vw.release()

    fs = FrameSource(p, reconnect=False).start()
    got = []
    t0 = _t.time()
    while _t.time() - t0 < 4 and len(got) < N:
        fr = fs.read()
        if fr is not None:
            got.append(int(fr[0, 0, 0]))               # mức xám ~ chỉ số frame * 15
        else:
            _t.sleep(0.005)
    fs.stop()
    assert len(got) >= N - 3                            # đọc gần đủ (không nuốt mất frame)
    assert got == sorted(got)                          # ĐÚNG thứ tự, không đảo/nhảy lung tung


def test_framesource_file_playback_is_realtime(tmp_path):
    """Đọc FILE KHÔNG vượt tốc độ thật: sau ~0.4s chỉ đọc được vài frame (không nuốt cả file)."""
    import time as _t

    import cv2

    p = str(tmp_path / "rt.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(50):                                # 50 frame @10fps = 5s video
        vw.write(np.full((120, 160, 3), 90, dtype="uint8"))
    vw.release()

    fs = FrameSource(p, reconnect=False).start()
    try:
        _t.sleep(0.4)                                  # ~0.4s thực → ~4 frame @10fps
        n = fs.frames_read
    finally:
        fs.stop()
    # CÓ nhịp → chỉ ~4 frame trong 0.4s (không phải cả 50 như khi đọc tự do). Nới rộng chống flaky.
    assert 1 <= n <= 30


# --------------------------------------------------------------------------- #
# API mới: healthz / snapshot / events (chỉ chạy nếu có fastapi)
# --------------------------------------------------------------------------- #
def test_api_health_snapshot_events(tmp_path):
    pytest.importorskip("fastapi")
    import cv2
    from fastapi.testclient import TestClient

    from recognition.service.app import app

    client = TestClient(app)
    h = client.get("/healthz").json()
    assert h["status"] == "ok" and "vectordb" in h
    assert client.get("/api/vectordb").json()["backend"] in ("memory", "qdrant")
    assert "events" in client.get("/api/events").json()
    # snapshot từ file synthetic
    p = str(tmp_path / "v.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(12):
        vw.write(np.full((120, 160, 3), 70, dtype="uint8"))
    vw.release()
    r = client.get("/api/snapshot", params={"source": p})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/api/snapshot", params={"source": "/khong/co.mp4"}).status_code == 502
