# -*- coding: utf-8 -*-
"""Test THẬT vòng đời camera: chết → sống → chết → sống lại.

Chuẩn bị:
    docker compose up -d
    (máy host cần ffmpeg)

Chạy:
    python test_camera_lifecycle_e2e.py

Vì sao cần test này — hai lỗi đã bắt được bằng tay và phải chặn tái phát:

  1. ``FrameSource`` giữ frame CŨ trong bộ nhớ và tự kết nối lại ngầm. Camera chết
     thì ``read()`` vẫn trả ảnh (đứng hình) và ``.alive`` vẫn True → AI vẫn chạy
     nhận diện trên ảnh đông cứng và bắn tin đều đặn. Đo thực tế: camera tắt lúc
     04:30:10 mà tin vẫn chảy tới 04:30:55, lặp lại y hệt 31 box ở nguyên chỗ cũ. Backend
     hiển thị "31 người" vĩnh viễn trên một camera đã chết.

  2. Camera sai URL / chưa bật thì backend chỉ nhận ``started`` rồi im lặng —
     không phân biệt được với "camera chạy nhưng không có ai đi qua".
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import redis
import requests

AI_URL     = os.getenv("AI_URL", "http://localhost:8000")
REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379")
MAIN_KEY   = os.getenv("REDIS_STREAM_KEY", "VISIONOS_RESULTS")
EVENT_KEY  = os.getenv("REDIS_EVENT_STREAM_KEY", "VISIONOS_EVENTS")

STREAM_ID = "e2e-lifecycle"
RTSP_PATH = "e2e-lifecycle"
VIDEO     = "data/people-walking.mp4"

OFF_BEFORE_S = 12    # để camera tắt bao lâu trước khi bật (phải > SOURCE_STALE_SEC=5)
ON_S         = 25    # để camera chạy bao lâu
OFF_S        = 20    # tắt đột ngột bao lâu
BACK_S       = 20    # bật lại rồi chờ bao lâu

_ff = []


def die(msg):
    print(f"\n❌ {msg}")
    stop_camera()
    requests.delete(f"{AI_URL}/streams/{STREAM_ID}", timeout=5)
    sys.exit(1)


def start_camera():
    _ff.append(subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1",
         "-i", VIDEO, "-c", "copy", "-f", "rtsp", "-rtsp_transport", "tcp",
         f"rtsp://localhost:8554/{RTSP_PATH}"]))


def stop_camera():
    while _ff:
        p = _ff.pop()
        p.kill()
        p.wait(timeout=5)


def read_all(r, key):
    """Đọc tin của RIÊNG luồng này.

    Mọi camera cùng ghi vào một stream Redis, nên không lọc theo ``stream_id`` là
    lẫn tin của luồng khác — đúng như tài liệu dặn backend phải làm.
    """
    msgs = (json.loads(f["data"]) for _, f in r.xrange(key, "-", "+"))
    return [m for m in msgs if m.get("stream_id") == STREAM_ID]


# --------------------------------------------------------------------------- #
if not shutil.which("ffmpeg"):
    die("Máy host chưa có ffmpeg (macOS: `brew install ffmpeg`).")
if not os.path.exists(VIDEO):
    die(f"Thiếu video mẫu {VIDEO}.")
try:
    requests.get(f"{AI_URL}/healthz", timeout=5).raise_for_status()
except Exception as e:
    die(f"AI Service chưa chạy ({e}). `docker compose up -d` trước.")

r = redis.from_url(REDIS_URL, decode_responses=True)
r.delete(MAIN_KEY, EVENT_KEY)
requests.delete(f"{AI_URL}/streams/{STREAM_ID}", timeout=5)

try:
    print("[1] Gán camera trong khi camera đang TẮT")
    resp = requests.post(f"{AI_URL}/streams/{STREAM_ID}", timeout=15, json={
        "camera_id": "cam-lifecycle",
        "rtsp_url": f"rtsp://mediamtx:8554/{RTSP_PATH}",
        "params": {"classes": ["person"], "conf": 0.3, "roi": [[0, 50], [100, 50]]},
    })
    if resp.status_code != 200:
        die(f"Không tạo được stream: {resp.text}")
    time.sleep(OFF_BEFORE_S)

    print("[2] BẬT camera")
    start_camera()
    time.sleep(ON_S)

    print("[3] TẮT camera đột ngột")
    stop_camera()
    time.sleep(OFF_S)

    print("[4] BẬT lại camera")
    start_camera()
    time.sleep(BACK_S)

    # ------------------------------------------------------------------ kiểm
    print("\n=== Vòng đời camera nhận được ===")
    statuses = [m for m in read_all(r, EVENT_KEY) if m["type"] == "stream_status"]
    for m in statuses:
        print(f"  {m['timestamp']}  {m['status']:<12} {m['detail']}")

    seq = [m["status"] for m in statuses]
    expect = ["started", "source_lost", "source_ok", "source_lost", "reconnected"]
    if seq != expect:
        die(f"Chuỗi trạng thái sai.\n   nhận : {seq}\n   mong : {expect}")
    print(f"\n  ✅ đúng chuỗi {' → '.join(expect)}")

    if "không kết nối được" not in statuses[1]["detail"]:
        die(f"Lần rớt đầu phải nói rõ chưa kết nối được, nhận: {statuses[1]['detail']!r}")
    print(f"  ✅ phân biệt được 'chưa kết nối được' với 'mất kết nối giữa chừng'")

    # Lỗi ảnh ĐỨNG HÌNH: khi camera chết phải NGỪNG bắn tin frame
    print("\n=== Khi camera chết có ngừng bắn tin không? ===")
    frames = [m for m in read_all(r, MAIN_KEY) if m["type"] == "frame"]
    if not frames:
        die("Không có tin frame nào — camera chưa từng chạy?")

    def t(m):
        return datetime.fromisoformat(m["frame_timestamp"][:-1])

    gaps = [(t(b) - t(a)).total_seconds() for a, b in zip(frames, frames[1:])]
    biggest = max(gaps) if gaps else 0.0
    print(f"  {len(frames)} tin frame, khoảng ngừng dài nhất: {biggest:.0f}s")
    if biggest < OFF_S * 0.5:
        die(f"Camera chết {OFF_S}s mà tin frame chỉ ngừng {biggest:.0f}s — "
            f"nhiều khả năng đang bắn nhận diện trên ảnh ĐỨNG HÌNH.")
    print(f"  ✅ ngừng {biggest:.0f}s đúng lúc camera chết — không bắn ảnh đứng hình")

    # Tin nghiệp vụ phải có ở CẢ hai stream
    print("\n=== Hai stream ===")
    main_types = {m["type"] for m in read_all(r, MAIN_KEY)}
    ev_types = {m["type"] for m in read_all(r, EVENT_KEY)}
    print(f"  {MAIN_KEY:<20} chứa {sorted(main_types)}")
    print(f"  {EVENT_KEY:<20} chứa {sorted(ev_types)}")
    if "frame" not in main_types or "stream_status" not in main_types:
        die("Stream chính phải chứa đủ mọi loại tin.")
    if "frame" in ev_types:
        die("Stream bền KHÔNG được chứa frame (sẽ đẩy văng tin nghiệp vụ).")
    print("  ✅ stream bền chỉ giữ tin nghiệp vụ")

    print("\n🎉 ĐẠT — vòng đời camera báo đúng, không bắn nhận diện trên ảnh chết.")

finally:
    stop_camera()
    try:
        requests.delete(f"{AI_URL}/streams/{STREAM_ID}", timeout=5)
    except Exception:
        pass
    print("\n(đã dọn dẹp)")
