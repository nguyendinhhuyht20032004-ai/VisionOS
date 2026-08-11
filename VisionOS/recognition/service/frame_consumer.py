# -*- coding: utf-8 -*-
"""
FrameConsumer – Module BỔ SUNG (không thay thế bất kỳ file nào).

Chạy SONG SONG với hệ thống hiện tại. Lắng nghe Redis stream
``visionos:frames:in`` để nhận frame từ Backend, xử lý bằng YOLO,
rồi publish kết quả detection + counting ngược lại vào
``VISIONOS_RESULTS``.

Hệ thống hiện tại (RTSP → Job → StreamingCounter) vẫn hoạt động
bình thường, không bị ảnh hưởng.

Cách dùng:
  - Import và gọi ``start_consumer()`` khi server khởi động.
  - Hoặc chạy trực tiếp: ``python -m recognition.service.frame_consumer``
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger("frame_consumer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[FrameConsumer] %(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Config từ biến môi trường (giống docker-compose.yml hiện tại)
# ---------------------------------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
INPUT_STREAM = os.environ.get("FRAME_INPUT_STREAM", "visionos:frames:in")
OUTPUT_STREAM = os.environ.get("REDIS_STREAM_KEY", "VISIONOS_RESULTS")
OUTPUT_MAXLEN = int(os.environ.get("REDIS_STREAM_MAXLEN", "1000"))
PUBLISH_FPS = int(os.environ.get("OVERLAY_PUBLISH_FPS", "10"))
CONSUMER_GROUP = os.environ.get("CONSUMER_GROUP", "ai-workers")
CONSUMER_NAME = os.environ.get("CONSUMER_NAME", f"worker-{os.getpid()}")


# ---------------------------------------------------------------------------
# Registry: lưu cấu hình job (vạch, vùng, prompt) mà Backend đã đăng ký
# Cách đăng ký: vẫn gọi POST /api/jobs như cũ, HOẶC Backend gửi kèm
# config trong message Redis.
# ---------------------------------------------------------------------------
_CONSUMER_JOBS: Dict[str, "_ConsumerJob"] = {}
_CONSUMER_LOCK = threading.Lock()


class _ConsumerJob:
    """Một phiên đếm được quản lý bởi FrameConsumer."""

    def __init__(self, job_id: str, config: dict):
        from .builder import get_detector, make_scenario
        from .engine import StreamingCounter

        self.id = job_id
        self.config = config

        prompt = config.get("prompt", "person")
        counting_type = config.get("counting_type", "line")
        line = config.get("line", [0, 50, 100, 50])
        zone = config.get("zone")
        resolution = tuple(config.get("resolution", [960, 540]))
        in_label = config.get("in_label", "IN")
        out_label = config.get("out_label", "OUT")
        anchor = config.get("anchor")
        model = config.get("model", "yolo")
        confidence = config.get("confidence", 0.25)
        detect_every = config.get("detect_every", 1)

        self.scenario, self.kind = make_scenario(
            prompt, counting_type, line, zone, resolution,
            in_label, out_label, anchor, model,
        )
        detector = get_detector(self.kind, confidence)
        self.counter = StreamingCounter(
            self.scenario, detector,
            resolution=resolution,
            detect_every=detect_every,
            smoother_len=int(os.environ.get("SMOOTHER_LEN", "2")),
        )
        self.last_publish_time = 0.0
        logger.info("Job '%s' registered (prompt=%s, type=%s)", job_id, prompt, counting_type)


def register_job(job_id: str, config: dict):
    """Đăng ký cấu hình một job mới cho Consumer (gọi từ API hoặc Redis message)."""
    with _CONSUMER_LOCK:
        _CONSUMER_JOBS[job_id] = _ConsumerJob(job_id, config)


def unregister_job(job_id: str):
    """Huỷ đăng ký job."""
    with _CONSUMER_LOCK:
        _CONSUMER_JOBS.pop(job_id, None)


# ---------------------------------------------------------------------------
# Core: Xử lý 1 message frame từ Redis
# ---------------------------------------------------------------------------
def _process_message(redis_client, msg_id: str, fields: dict):
    """Xử lý 1 frame message: decode → inference → publish kết quả."""

    # --- Bước 1: Đọc các trường bắt buộc ---
    job_id = fields.get(b"job_id") or fields.get("job_id")
    image_bytes = fields.get(b"image") or fields.get("image")
    frame_id = fields.get(b"frame_id") or fields.get("frame_id", b"0")
    ts = fields.get(b"ts") or fields.get("ts")

    if isinstance(job_id, bytes):
        job_id = job_id.decode("utf-8")
    if isinstance(frame_id, bytes):
        frame_id = frame_id.decode("utf-8")
    if isinstance(ts, bytes):
        ts = ts.decode("utf-8")

    if not job_id or not image_bytes:
        logger.warning("Message %s thiếu job_id hoặc image → skip.", msg_id)
        return

    # --- Bước 2: Kiểm tra cấu hình trong message (auto-register) ---
    config_raw = fields.get(b"config") or fields.get("config")
    if config_raw:
        if isinstance(config_raw, bytes):
            config_raw = config_raw.decode("utf-8")
        try:
            config = json.loads(config_raw)
            with _CONSUMER_LOCK:
                if job_id not in _CONSUMER_JOBS:
                    register_job(job_id, config)
        except json.JSONDecodeError:
            pass

    # --- Bước 3: Lookup job ---
    with _CONSUMER_LOCK:
        job = _CONSUMER_JOBS.get(job_id)
    if job is None:
        logger.warning("Job '%s' chưa đăng ký → skip frame %s.", job_id, frame_id)
        return

    # --- Bước 4: Decode JPEG → BGR matrix ---
    if isinstance(image_bytes, str):
        image_bytes = image_bytes.encode("latin-1")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        logger.warning("Job '%s' frame %s: JPEG decode thất bại → skip.", job_id, frame_id)
        return

    # --- Bước 5: Chạy AI (inference) ---
    annotated = job.counter.process(frame_bgr)

    # --- Bước 6: Publish frame_result (throttle theo PUBLISH_FPS) ---
    now = time.time()
    interval = 1.0 / PUBLISH_FPS if PUBLISH_FPS > 0 else 0.0
    if now - job.last_publish_time >= interval:
        job.last_publish_time = now
        stats = job.counter.stats()
        result_msg = {
            "event_type": "frame_result",
            "job_id": job_id,
            "ts": ts or str(now),
            "frame_id": str(frame_id),
            "payload": json.dumps(stats, ensure_ascii=False),
        }
        try:
            redis_client.xadd(OUTPUT_STREAM, result_msg, maxlen=OUTPUT_MAXLEN)
        except Exception as e:
            logger.error("Publish frame_result lỗi: %s", e)

    # --- Bước 7: Publish track_event (nếu có vật cắt vạch) ---
    det = job.counter.last_det
    if det is not None and hasattr(det, "cross_events") and det.cross_events:
        for track_id, direction in det.cross_events:
            # Tìm index của track_id trong det để lấy class_name, confidence, bbox
            cls_name = "object"
            conf = 0.0
            bbox = [0, 0, 0, 0]
            if det.tracker_id is not None:
                for i in range(len(det)):
                    tid = det.tracker_id[i]
                    if tid is not None and int(tid) == int(track_id):
                        if getattr(det, "data", None) and "class_name" in det.data:
                            cls_name = str(det.data["class_name"][i])
                        if det.confidence is not None:
                            conf = round(float(det.confidence[i]), 3)
                        box = det.xyxy[i]
                        bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                        break

            event_msg = {
                "event_type": "track_event",
                "job_id": job_id,
                "ts": str(now),
                "payload": json.dumps({
                    "track_id": int(track_id),
                    "direction": direction,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": bbox,
                    "source": job.config.get("source", ""),
                }, ensure_ascii=False),
            }
            try:
                redis_client.xadd(OUTPUT_STREAM, event_msg, maxlen=OUTPUT_MAXLEN)
            except Exception as e:
                logger.error("Publish track_event lỗi: %s", e)

        # Xoá cross_events sau khi đã publish (tránh gửi trùng)
        det.cross_events.clear()


# ---------------------------------------------------------------------------
# Consumer loop: XREAD blocking từ Redis stream
# ---------------------------------------------------------------------------
def _consumer_loop():
    """Vòng lặp chính: lắng nghe stream Redis và xử lý frame."""
    import redis as redis_lib

    logger.info("Khởi động FrameConsumer (stream=%s, output=%s)", INPUT_STREAM, OUTPUT_STREAM)

    while True:
        try:
            r = redis_lib.from_url(REDIS_URL, decode_responses=False,
                                    socket_timeout=30, socket_connect_timeout=10)
            r.ping()
            logger.info("Kết nối Redis OK (%s)", REDIS_URL)

            # Tạo consumer group (nếu chưa có)
            try:
                r.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
                logger.info("Đã tạo consumer group '%s' trên stream '%s'",
                            CONSUMER_GROUP, INPUT_STREAM)
            except redis_lib.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    logger.info("Consumer group '%s' đã tồn tại, sử dụng lại.", CONSUMER_GROUP)
                else:
                    raise

            # Vòng đọc chính
            last_id = ">"  # Chỉ đọc message MỚI (chưa ai consume)
            while True:
                try:
                    messages = r.xreadgroup(
                        CONSUMER_GROUP, CONSUMER_NAME,
                        {INPUT_STREAM: last_id},
                        count=1,       # Xử lý 1 frame mỗi lần
                        block=5000,    # Block 5 giây, tự thức khi có message
                    )
                except redis_lib.ConnectionError:
                    logger.warning("Mất kết nối Redis, reconnect sau 2 giây...")
                    time.sleep(2)
                    break

                if not messages:
                    continue  # Timeout, tiếp tục chờ

                for stream_name, msg_list in messages:
                    for msg_id, fields in msg_list:
                        try:
                            _process_message(r, msg_id, fields)
                            # ACK message đã xử lý xong
                            r.xack(INPUT_STREAM, CONSUMER_GROUP, msg_id)
                        except Exception as e:
                            logger.error("Lỗi xử lý message %s: %s", msg_id, e, exc_info=True)

        except Exception as e:
            logger.error("FrameConsumer lỗi kết nối: %s — retry sau 3 giây", e)
            time.sleep(3)


# ---------------------------------------------------------------------------
# Public API: Khởi động consumer (gọi từ app.py hoặc chạy standalone)
# ---------------------------------------------------------------------------
_consumer_thread: Optional[threading.Thread] = None


def start_consumer():
    """Khởi động FrameConsumer chạy ngầm (gọi 1 lần duy nhất)."""
    global _consumer_thread
    if _consumer_thread is not None and _consumer_thread.is_alive():
        logger.info("FrameConsumer đã chạy, bỏ qua.")
        return
    _consumer_thread = threading.Thread(target=_consumer_loop, daemon=True, name="FrameConsumer")
    _consumer_thread.start()
    logger.info("FrameConsumer thread đã khởi động.")


# ---------------------------------------------------------------------------
# Standalone: chạy trực tiếp bằng python -m recognition.service.frame_consumer
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    start_consumer()
    # Giữ main thread sống
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("FrameConsumer dừng.")
