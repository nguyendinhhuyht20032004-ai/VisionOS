# -*- coding: utf-8 -*-
"""
StreamManager – quản lý luồng RTSP động cho AI Service.
Mỗi stream được đại diện bởi một Thread (StreamWorker) thực hiện:
  * Kết nối RTSP (cv2.VideoCapture)
  * Chạy pipeline YOLO + Supervision (StreamingCounter)
  * Publish JSON lên Redis Stream (type="frame"/"track_event")
"""

import os, json, time, datetime, threading, base64
from typing import Dict, Optional
import warnings

# Tắt các cảnh báo spam từ YOLO / PyTorch như "half is deprecated"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import redis
from pydantic import BaseModel, Field

from .engine import StreamingCounter
from .builder import get_detector, make_scenario
from .camera import FrameSource

# ---------------------------------------------------------------------------
# Pydantic models cho Control API
# ---------------------------------------------------------------------------
class StreamParams(BaseModel):
    """Các tham số tùy chọn khi tạo / cập nhật stream."""
    roi: Optional[list] = None                 # [[x1,y1], [x2,y2]] – không dùng trong demo hiện tại
    conf: Optional[float] = None               # confidence threshold
    classes: Optional[list] = None             # ví dụ ["person","car"]
    detect_every: Optional[int] = Field(3, description="Chạy AI mỗi N frame (tăng để mượt/nhẹ CPU, giảm để chính xác)")
    track_timeout: Optional[float] = Field(2.0, description="Thời gian (giây) mất dấu trước khi bắn sự kiện end")
    publish_fps: Optional[float] = Field(12.0, description="Tần số gửi kết quả lên Redis (khung hình / giây)")

class StreamControlRequest(BaseModel):
    camera_id: str = Field(..., description="ID camera (định danh người dùng)")
    rtsp_url: str = Field(..., description="URL RTSP tới MediaMTX")
    params: Optional[StreamParams] = None

# ---------------------------------------------------------------------------
# Worker – thực thi một luồng RTSP
# ---------------------------------------------------------------------------
class StreamWorker(threading.Thread):
    def __init__(self, stream_id: str, req: StreamControlRequest,
                 redis_cli: redis.Redis):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.camera_id = req.camera_id
        self.rtsp_url = req.rtsp_url
        self.params = req.params or StreamParams()
        self.redis = redis_cli
        self.running = True
        self._reported_tracks = set()
        self._track_last_seen = {}  # dict to store last seen time of each track_id
        self._prev_centers = {}    # track_id -> (cx, cy, timestamp) for velocity
        self._pending_cross_events = []

        # env vars (đã định nghĩa trong AI_SERVICE_INTEGRATION.md)
        self.reconnect_interval = float(os.getenv("RTSP_RECONNECT_INTERVAL_SEC", "5"))
        self.maxlen = int(os.getenv("REDIS_STREAM_MAXLEN", "1000"))
        self.stream_key = os.getenv("REDIS_STREAM_KEY", "VISIONOS_RESULTS")

        self._build_counter()
        self.fs: Optional[FrameSource] = None
        self.current_fps = 0.0

    # -------------------------------------------------------------------
    def get_status(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "camera_id": self.camera_id,
            "rtsp_url": self.rtsp_url,
            "fps": round(self.current_fps, 1),
            "params": self.params.model_dump()
        }

    # -------------------------------------------------------------------
    def _build_counter(self):
        # Prompt = các class được join bằng ','
        prompt = ",".join(self.params.classes) if self.params.classes else "person"
        conf = self.params.conf or 0.3

        # Determine counting type from ROI
        counting_type = "fullscreen"
        line_pts = None
        zone_pts = None

        if self.params.roi:
            if len(self.params.roi) == 2:
                counting_type = "line"
                # Flatten [[x1, y1], [x2, y2]] -> [x1, y1, x2, y2]
                line_pts = [float(c) for pt in self.params.roi for c in pt]
            elif len(self.params.roi) > 2:
                counting_type = "zone"
                zone_pts = [[float(c) for c in pt] for pt in self.params.roi]

        # Tạo scenario
        scenario, kind = make_scenario(
            prompt, counting_type, line_pts, zone_pts,
            (960, 540), "IN", "OUT", None, "yolo"
        )
        detector = get_detector(kind, confidence=conf)
        self.counter = StreamingCounter(
            scenario,
            detector,
            resolution=scenario.resolution,
            detect_every=int(os.getenv("DETECT_EVERY",
                                       str(self.params.detect_every if self.params.detect_every is not None else 3))),
            smoother_len=int(os.getenv("SMOOTHER_LEN", "2")),
        )

    # -------------------------------------------------------------------
    def _open_source(self):
        if self.rtsp_url.startswith("rtsp"):
            transport = os.getenv("RTSP_TRANSPORT", "tcp")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
        self.fs = FrameSource(self.rtsp_url).start()

    # -------------------------------------------------------------------
    def run(self):
        self._open_source()
        last_pub = 0.0
        _yolo_samples = []
        _tuned = False
        _slow_cpu = False
        _last_latency = 0.0

        while self.running:
            fps = self.params.publish_fps if self.params.publish_fps is not None else float(os.getenv("OVERLAY_PUBLISH_FPS", "12"))
            interval = 1.0 / max(fps, 1e-3)
            if self.fs is None:
                time.sleep(self.reconnect_interval)
                self._open_source()
                continue

            frame = self.fs.read()
            if frame is None:
                if not self.fs.alive:
                    print(f"[StreamWorker] {self.stream_id} thread dead, reconnecting...", flush=True)
                    self.fs.stop()
                    self.fs = None
                    time.sleep(self.reconnect_interval)
                else:
                    time.sleep(0.1)
                continue

            loop_start = time.time()
            # ----- AI pipeline -----
            t_ai = time.time()
            out = self.counter.process(frame)
            ai_ms = (time.time() - t_ai) * 1000

            # Collect cross events (IN/OUT) before next detection overwrites them
            det_ev = self.counter.last_det
            if det_ev and hasattr(det_ev, "cross_events") and det_ev.cross_events:
                self._pending_cross_events.extend(det_ev.cross_events)
                det_ev.cross_events = []

            # ----- Auto-tune detect_every from actual YOLO speed -----
            cur_latency = self.counter._last_latency_ms
            if not _tuned and cur_latency != _last_latency and cur_latency > 0:
                _last_latency = cur_latency
                _yolo_samples.append(cur_latency)
                if len(_yolo_samples) >= 5:
                    avg_ms = sum(_yolo_samples) / len(_yolo_samples)
                    if avg_ms > 150:
                        self.counter.detect_every = 1
                        _slow_cpu = True
                        print(f"[StreamWorker] {self.stream_id} YOLO avg={avg_ms:.0f}ms → detect_every=1 (CPU mode)", flush=True)
                    else:
                        print(f"[StreamWorker] {self.stream_id} YOLO avg={avg_ms:.0f}ms → keeping detect_every={self.counter.detect_every}", flush=True)
                    _tuned = True

            # ----- Publish (giới hạn FPS) -----
            now = time.time()
            if now - last_pub >= interval:
                t_pub = time.time()
                self._publish_result(out)
                pub_ms = (time.time() - t_pub) * 1000
                last_pub = now
            else:
                pub_ms = 0.0

            # ----- Benchmark log (mỗi 30 frame) -----
            total_ms = (time.time() - loop_start) * 1000
            if not hasattr(self, '_bench_count'):
                self._bench_count = 0
            self._bench_count += 1
            if self._bench_count % 30 == 0:
                self.current_fps = 1000.0 / max(total_ms, 1e-3)
                print(f"[Benchmark] {self.stream_id} | AI: {ai_ms:.0f}ms | Pub: {pub_ms:.0f}ms | Total: {total_ms:.0f}ms ({self.current_fps:.1f} fps)", flush=True)

            # ----- Loop pacing -----
            if not _slow_cpu:
                elapsed = time.time() - now
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # clean up when stopped
        if self.fs:
            self.fs.stop()

    # -------------------------------------------------------------------
    def _publish_result(self, annotated_bgr):
        # 1. Frame result
        boxes = []
        det = self.counter.last_det
        current_tids = set()
        if det and getattr(det, "data", None):
            if not hasattr(self, "_track_classes"):
                self._track_classes = {}
            for i, tid in enumerate(det.tracker_id):
                if tid is None:
                    continue
                # Convert (x1, y1, x2, y2) → (x, y, w, h)
                x1 = float(det.xyxy[i][0])
                y1 = float(det.xyxy[i][1])
                x2 = float(det.xyxy[i][2])
                y2 = float(det.xyxy[i][3])
                w = x2 - x1
                h = y2 - y1
                box_info = {
                    "track_id": str(tid),
                    "class": str(det.data.get("class_name")[i]) if "class_name" in det.data and len(det.data["class_name"]) > i else "",
                    "confidence": float(det.confidence[i]) if hasattr(det, "confidence") and det.confidence is not None else 0.0,
                    "bbox": [x1, y1, w, h],
                }
                cx, cy = x1 + w / 2, y1 + h / 2
                now_v = time.time()
                prev = self._prev_centers.get(str(tid))
                if prev is not None:
                    dt = now_v - prev[2]
                    if dt > 0.01:
                        box_info["velocity"] = [round((cx - prev[0]) / dt, 1),
                                                round((cy - prev[1]) / dt, 1)]
                    else:
                        box_info["velocity"] = [0.0, 0.0]
                else:
                    box_info["velocity"] = [0.0, 0.0]
                self._prev_centers[str(tid)] = (cx, cy, now_v)
                boxes.append(box_info)
                current_tids.add(tid)
                self._track_last_seen[tid] = time.time()
                self._track_classes[tid] = box_info["class"]

                # Emit start event when a new track appears
                if tid not in self._reported_tracks:
                    self._reported_tracks.add(tid)
                    self._publish_track_event(str(tid), boxes[-1]["class"], "start")
            
        # Emit end events for tracks not seen for 2 seconds
        now_ts = time.time()
        ended_tids = []
        for tid, last_seen in list(self._track_last_seen.items()):
            timeout = self.params.track_timeout if self.params.track_timeout is not None else 2.0
            if now_ts - last_seen > timeout:
                ended_tids.append(tid)
        
        for tid in ended_tids:
            cls_name = self._track_classes.get(tid, "unknown")
            self._publish_track_event(str(tid), cls_name, "end")
            del self._track_last_seen[tid]
            if tid in self._track_classes:
                del self._track_classes[tid]
            if tid in self._reported_tracks:
                self._reported_tracks.remove(tid)
            self._prev_centers.pop(str(tid), None)

        # Publish IN/OUT crossing events to Redis
        if self._pending_cross_events:
            for tid, direction in self._pending_cross_events:
                cls_name = self._track_classes.get(tid, "unknown") if hasattr(self, '_track_classes') else "unknown"
                self._publish_track_event(str(tid), cls_name, direction)
            self._pending_cross_events = []

        frame_msg = {
            "type": "frame",
            "camera_id": self.camera_id,
            "stream_id": self.stream_id,
            "frame_timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "resolution": {"width": self.counter.w, "height": self.counter.h},
            "boxes": boxes,
        }
        
        try:
            self.redis.xadd(
                self.stream_key,
                {"data": json.dumps(frame_msg)},
                maxlen=self.maxlen,
                approximate=True,
            )
            if len(boxes) > 0:
                sample = boxes[0]
                bbox = sample['bbox']
                print(f"[StreamWorker] {self.stream_id} published {len(boxes)} boxes. Sample Box ID {sample['track_id']}: x={bbox[0]:.1f}, y={bbox[1]:.1f}, v={sample['velocity']}", flush=True)
            else:
                print(f"[StreamWorker] {self.stream_id} published frame with 0 boxes", flush=True)
        except Exception as e:
            print(f"[StreamWorker] {self.stream_id} failed to publish to Redis: {e}", flush=True)

    # -------------------------------------------------------------------
    def _publish_track_event(self, track_id: str, class_name: str, event: str):
        ev = {
            "type": "track_event",
            "camera_id": self.camera_id,
            "stream_id": self.stream_id,
            "track_id": track_id,
            "class": class_name,
            "event": event,
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        }
        self.redis.xadd(
            self.stream_key,
            {"data": json.dumps(ev)},
            maxlen=self.maxlen,
            approximate=True,
        )

    # -------------------------------------------------------------------
    def stop(self):
        self.running = False
        if self.fs:
            self.fs.stop()

    # -------------------------------------------------------------------
    def update_params(self, params: StreamParams):
        self.params = params
        self._build_counter()

# ---------------------------------------------------------------------------
# StreamManager – singleton quản lý các StreamWorker
# ---------------------------------------------------------------------------
class StreamManager:
    def __init__(self, redis_cli: redis.Redis):
        self.redis = redis_cli
        self.workers: Dict[str, StreamWorker] = {}
        self.lock = threading.Lock()

    def start(self, stream_id: str, req: StreamControlRequest):
        with self.lock:
            if stream_id in self.workers:
                worker = self.workers[stream_id]
                # Nếu URL không đổi, cập nhật động thông số mà không cần khởi động lại
                if worker.rtsp_url == req.rtsp_url:
                    print(f"[StreamManager] Stream {stream_id} exists with same URL. Updating params dynamically...", flush=True)
                    worker.update_params(req.params or StreamParams())
                    return {"status": "updated", "stream_id": stream_id}
                else:
                    # Nếu URL thay đổi, bắt buộc phải tắt đi bật lại
                    print(f"[StreamManager] Stream {stream_id} URL changed. Stopping old stream to restart...", flush=True)
                    worker.stop()
            
            worker = StreamWorker(stream_id, req, self.redis)
            self.workers[stream_id] = worker
            worker.start()
            return {"status": "started", "stream_id": stream_id}

    def update(self, stream_id: str, params: StreamParams):
        with self.lock:
            worker = self.workers.get(stream_id)
            if not worker:
                raise KeyError(f"Stream {stream_id} not found")
            worker.update_params(params)
            return {"status": "updated", "stream_id": stream_id}

    def get_active_streams(self):
        with self.lock:
            return [worker.get_status() for worker in self.workers.values()]

    def stop(self, stream_id: str):
        with self.lock:
            worker = self.workers.pop(stream_id, None)
            if not worker:
                raise KeyError(f"Stream {stream_id} not found")
            worker.stop()
            return {"status": "stopped", "stream_id": stream_id}
