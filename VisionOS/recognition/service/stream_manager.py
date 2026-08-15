# -*- coding: utf-8 -*-
"""
StreamManager -- quan ly luong RTSP dong cho AI Service.
Moi stream duoc dai dien boi mot Thread (StreamWorker) thuc hien:
  * Ket noi RTSP (cv2.VideoCapture)
  * Chay pipeline YOLO + Supervision (StreamingCounter)
  * Publish OverlayFrame + DetectionEvent qua gRPC (thay Redis)

Theo AI_SERVICE_CONTRACT.md:
  * POST /streams/{jobKey} tao moi HOAC cap nhat nong job dang chay
  * DELETE /streams/{jobKey} dung job, giai phong RTSP
  * Ket qua day qua grpc_publisher (OverlayPublisher)
"""

import os
import time
import datetime
import threading
import uuid
from typing import Dict, Optional
import warnings

# Tat cac canh bao spam tu YOLO / PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
from pydantic import BaseModel, Field

from .engine import StreamingCounter
from .builder import get_detector, make_scenario
from .camera import FrameSource

# ---------------------------------------------------------------------------
# Pydantic models cho Control API (theo AI_SERVICE_CONTRACT.md muc 3)
# ---------------------------------------------------------------------------
class StreamParams(BaseModel):
    """Cac tham so tuy chon khi tao / cap nhat stream."""
    classes: Optional[list] = Field(None, description="Ten lop COCO, >=1 phan tu")
    conf: Optional[float] = Field(None, description="Confidence threshold 0..1")
    roi: Optional[list] = Field(None, description="[[x,y],...] phan tram 0..100. 2 diem = vach, >2 = da giac, null = toan khung")
    publish_fps: Optional[float] = Field(30.0, description="Tran fps day qua gRPC")
    detect_every: Optional[int] = Field(3, description="Chay detector moi N frame, con lai noi suy")
    track_timeout: Optional[float] = Field(2.0, description="Giay giu track khi vat bi che khuat")

class StreamControlRequest(BaseModel):
    camera_id: str = Field(..., description="ID camera (so nguyen dang chuoi)")
    rtsp_url: str = Field(..., description="URL RTSP toi MediaMTX")
    params: Optional[StreamParams] = None

# ---------------------------------------------------------------------------
# Worker -- thuc thi mot luong RTSP
# ---------------------------------------------------------------------------
class StreamWorker(threading.Thread):
    def __init__(self, stream_id: str, req: StreamControlRequest,
                 grpc_publisher):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.camera_id = req.camera_id
        self.rtsp_url = req.rtsp_url
        self.params = req.params or StreamParams()
        self.publisher = grpc_publisher
        self.running = True
        self._reported_tracks = set()
        self._track_last_seen = {}
        self._prev_centers = {}
        self._pending_cross_events = []

        # env vars
        self.reconnect_interval = float(os.getenv("RTSP_RECONNECT_INTERVAL_SEC", "5"))

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
        prompt = ",".join(self.params.classes) if self.params.classes else "person"
        conf = self.params.conf or 0.3

        # Determine counting type from ROI
        counting_type = "fullscreen"
        line_pts = None
        zone_pts = None

        if self.params.roi:
            if len(self.params.roi) == 2:
                counting_type = "line"
                line_pts = [float(c) for pt in self.params.roi for c in pt]
            elif len(self.params.roi) > 2:
                counting_type = "zone"
                zone_pts = [[float(c) for c in pt] for pt in self.params.roi]

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
            fps = self.params.publish_fps if self.params.publish_fps is not None else float(os.getenv("OVERLAY_PUBLISH_FPS", "30"))
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

            # Ghi lai thoi diem DOC DUOC frame (truoc inference) theo Contract
            captured_at_ms = int(time.time() * 1000)

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
                        print(f"[StreamWorker] {self.stream_id} YOLO avg={avg_ms:.0f}ms -> detect_every=1 (CPU mode)", flush=True)
                    else:
                        print(f"[StreamWorker] {self.stream_id} YOLO avg={avg_ms:.0f}ms -> keeping detect_every={self.counter.detect_every}", flush=True)
                    _tuned = True

            # ----- Publish qua gRPC (gioi han FPS) -----
            now = time.time()
            if now - last_pub >= interval:
                t_pub = time.time()
                self._publish_result(captured_at_ms)
                pub_ms = (time.time() - t_pub) * 1000
                last_pub = now
            else:
                pub_ms = 0.0

            # ----- Benchmark log (moi 30 frame) -----
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
    def _publish_result(self, captured_at_ms: int):
        """Publish OverlayFrame + DetectionEvent qua gRPC."""
        if not self.publisher:
            return

        # 1. Build boxes list
        boxes = []
        det = self.counter.last_det
        current_tids = set()
        if det and getattr(det, "data", None):
            if not hasattr(self, "_track_classes"):
                self._track_classes = {}
            for i, tid in enumerate(det.tracker_id):
                if tid is None:
                    continue
                # Toa do pixel: (x, y, w, h) goc tren-trai (theo Contract)
                x1 = float(det.xyxy[i][0])
                y1 = float(det.xyxy[i][1])
                x2 = float(det.xyxy[i][2])
                y2 = float(det.xyxy[i][3])
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                str_tid = str(tid)

                vx = 0.0
                vy = 0.0
                now_t = time.time()

                if not hasattr(self, "_prev_centers"):
                    self._prev_centers = {}

                if str_tid in self._prev_centers:
                    prev_cx, prev_cy, prev_t = self._prev_centers[str_tid]
                    dt = now_t - prev_t
                    if dt > 0.01:
                        vx = (cx - prev_cx) / dt
                        vy = (cy - prev_cy) / dt

                self._prev_centers[str_tid] = (cx, cy, now_t)

                box_info = {
                    "track_id": str_tid,
                    "class_name": str(det.data.get("class_name")[i]) if "class_name" in det.data and len(det.data["class_name"]) > i else "",
                    "confidence": float(det.confidence[i]) if hasattr(det, "confidence") and det.confidence is not None else 0.0,
                    "x": x1,
                    "y": y1,
                    "w": w,
                    "h": h,
                    "velocity_x": vx,
                    "velocity_y": vy,
                }
                boxes.append(box_info)
                current_tids.add(tid)
                self._track_last_seen[tid] = time.time()
                self._track_classes[tid] = box_info["class_name"]

                # Emit START event when a new track appears
                if tid not in self._reported_tracks:
                    self._reported_tracks.add(tid)
                    self.publisher.publish_event(
                        camera_id=self.camera_id,
                        job_key=self.stream_id,
                        track_id=str(tid),
                        class_name=box_info["class_name"],
                        kind="START",
                        occurred_at_ms=captured_at_ms,
                    )

        # 2. Publish frame qua gRPC (ke ca khi boxes rong, de client xoa box cu)
        self.publisher.publish_frame(
            camera_id=self.camera_id,
            job_key=self.stream_id,
            captured_at_ms=captured_at_ms,
            width=self.counter.w,
            height=self.counter.h,
            boxes=boxes,
        )

        if len(boxes) > 0 and self._bench_count and self._bench_count % 30 == 0:
            sample = boxes[0]
            print(f"[StreamWorker] {self.stream_id} published {len(boxes)} boxes via gRPC", flush=True)

        # 3. Emit end events for tracks not seen for track_timeout seconds
        now_ts = time.time()
        ended_tids = []
        for tid, last_seen in list(self._track_last_seen.items()):
            timeout = self.params.track_timeout if self.params.track_timeout is not None else 2.0
            if now_ts - last_seen > timeout:
                ended_tids.append(tid)

        for tid in ended_tids:
            # Note: Contract khong co event "end", chi co START/IN/OUT
            # Nhung van clean up internal state
            del self._track_last_seen[tid]
            if tid in self._track_classes:
                del self._track_classes[tid]
            if tid in self._reported_tracks:
                self._reported_tracks.remove(tid)
            self._prev_centers.pop(str(tid), None)

        # 4. Publish IN/OUT crossing events qua gRPC
        if self._pending_cross_events:
            for tid, direction in self._pending_cross_events:
                cls_name = self._track_classes.get(tid, "unknown") if hasattr(self, '_track_classes') else "unknown"
                self.publisher.publish_event(
                    camera_id=self.camera_id,
                    job_key=self.stream_id,
                    track_id=str(tid),
                    class_name=cls_name,
                    kind=direction,  # "IN" or "OUT"
                    occurred_at_ms=captured_at_ms,
                )
            self._pending_cross_events = []

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
# StreamManager -- singleton quan ly cac StreamWorker
# ---------------------------------------------------------------------------
class StreamManager:
    def __init__(self, grpc_publisher):
        self.publisher = grpc_publisher
        self.workers: Dict[str, StreamWorker] = {}
        self.lock = threading.Lock()

    def start(self, stream_id: str, req: StreamControlRequest):
        with self.lock:
            if stream_id in self.workers:
                worker = self.workers[stream_id]
                # Neu URL khong doi, cap nhat dong thong so ma khong can khoi dong lai
                if worker.rtsp_url == req.rtsp_url:
                    print(f"[StreamManager] Stream {stream_id} exists with same URL. Updating params dynamically...", flush=True)
                    worker.update_params(req.params or StreamParams())
                    return {"status": "updated", "stream_id": stream_id}
                else:
                    # Neu URL thay doi, bat buoc phai tat di bat lai
                    print(f"[StreamManager] Stream {stream_id} URL changed. Stopping old stream to restart...", flush=True)
                    worker.stop()

            worker = StreamWorker(stream_id, req, self.publisher)
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

    def stop_all(self):
        """Dung tat ca worker (dung khi shutdown)."""
        with self.lock:
            for worker in self.workers.values():
                worker.stop()
            self.workers.clear()
