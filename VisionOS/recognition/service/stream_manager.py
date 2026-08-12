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
    publish_fps: Optional[float] = Field(10.0, description="Tần số gửi kết quả lên Redis (khung hình / giây)")

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

        # env vars (đã định nghĩa trong AI_SERVICE_INTEGRATION.md)
        self.reconnect_interval = float(os.getenv("RTSP_RECONNECT_INTERVAL_SEC", "5"))
        self.maxlen = int(os.getenv("REDIS_STREAM_MAXLEN", "1000"))
        self.stream_key = os.getenv("REDIS_STREAM_KEY", "VISIONOS_RESULTS")

        self._build_counter()
        self.fs: Optional[FrameSource] = None

    # -------------------------------------------------------------------
    def _build_counter(self):
        # Prompt = các class được join bằng ','
        prompt = ",".join(self.params.classes) if self.params.classes else "person"
        conf = self.params.conf or 0.25

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
            detect_every=self.params.detect_every if self.params.detect_every is not None else 3,
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

        while self.running:
            fps = self.params.publish_fps if self.params.publish_fps is not None else float(os.getenv("OVERLAY_PUBLISH_FPS", "10"))
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
                    time.sleep(0.1) # Wait for first frame
                continue

            # ----- AI pipeline -----
            out = self.counter.process(frame)
            
            # ----- Publish (giới hạn FPS) -----
            now = time.time()
            if now - last_pub >= interval:
                self._publish_result(out)
                last_pub = now

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
                boxes.append({
                    "track_id": str(tid),
                    "class": str(det.data.get("class_name")[i]) if "class_name" in det.data and len(det.data["class_name"]) > i else "",
                    "confidence": float(det.confidence[i]) if hasattr(det, "confidence") and det.confidence is not None else 0.0,
                    "bbox": [x1, y1, w, h],
                })
                current_tids.add(tid)
                self._track_last_seen[tid] = time.time()
                self._track_classes[tid] = boxes[-1]["class"]

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
        
        # Clear line crossing events — IN/OUT không nằm trong enum start|end của spec
        if det and hasattr(det, "cross_events"):
            det.cross_events = []

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
            print(f"[StreamWorker] {self.stream_id} published frame with {len(boxes)} boxes", flush=True)
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
                raise ValueError(f"Stream {stream_id} already exists")
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

    def stop(self, stream_id: str):
        with self.lock:
            worker = self.workers.pop(stream_id, None)
            if not worker:
                raise KeyError(f"Stream {stream_id} not found")
            worker.stop()
            return {"status": "stopped", "stream_id": stream_id}
