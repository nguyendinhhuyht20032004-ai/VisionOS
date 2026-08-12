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
def _now_iso() -> str:
    """Mốc thời gian UTC ISO-8601 có mili-giây: ``2026-08-12T09:15:32.450Z``."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _dump(model: BaseModel, **kw) -> dict:
    """Đổi model → dict. Hỗ trợ cả pydantic v2 (``model_dump``) lẫn v1 (``dict``)."""
    return model.model_dump(**kw) if hasattr(model, "model_dump") else model.dict(**kw)


class StreamParams(BaseModel):
    """Các tham số tùy chọn khi tạo / cập nhật stream."""
    roi: Optional[list] = None                 # [[x1,y1], [x2,y2]] – không dùng trong demo hiện tại
    conf: Optional[float] = None               # confidence threshold
    classes: Optional[list] = None             # ví dụ ["person","car"]

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
        self._track_classes = {}    # track_id → class_name (giữ để bắn "end" đúng lớp)

        # env vars (đã định nghĩa trong AI_SERVICE_INTEGRATION.md)
        self.reconnect_interval = float(os.getenv("RTSP_RECONNECT_INTERVAL_SEC", "5"))
        self.publish_fps = float(os.getenv("OVERLAY_PUBLISH_FPS", "10"))
        self.maxlen = int(os.getenv("REDIS_STREAM_MAXLEN", "1000"))
        self.stream_key = os.getenv("REDIS_STREAM_KEY", "VISIONOS_RESULTS")
        # Stream RIÊNG cho tin nghiệp vụ (track_event, stream_status). Stream chính bị
        # `frame` 10 tin/giây đẩy văng chỉ sau ~2 phút, mà `frame` mất được còn tin
        # nghiệp vụ thì không. Stream chính VẪN nhận đủ mọi loại tin như cũ.
        self.event_stream_key = os.getenv("REDIS_EVENT_STREAM_KEY", "VISIONOS_EVENTS")
        self.event_maxlen = int(os.getenv("REDIS_EVENT_STREAM_MAXLEN", "50000"))

        self._build_counter()
        self.fs: Optional[FrameSource] = None

    # -------------------------------------------------------------------
    def _build_counter(self):
        # Prompt = các class được join bằng ','
        prompt = ",".join(self.params.classes) if self.params.classes else "person"
        # `or 0.25` cũ vừa bỏ qua env YOLO_CONF vừa hiểu sai conf=0.0 thành 0.25.
        conf = self.params.conf if self.params.conf is not None else float(os.getenv("YOLO_CONF", "0.3"))

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
            detect_every=1,
            smoother_len=int(os.getenv("SMOOTHER_LEN", "2")),
            min_confidence=conf,      # detector dùng chung → phải lọc lại theo từng luồng
        )

    # -------------------------------------------------------------------
    def _open_source(self):
        if self.rtsp_url.startswith("rtsp"):
            transport = os.getenv("RTSP_TRANSPORT", "tcp")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
        self.fs = FrameSource(self.rtsp_url).start()

    # -------------------------------------------------------------------
    def run(self):
        self._publish_status("started")
        self._open_source()
        interval = 1.0 / max(self.publish_fps, 1e-3)
        last_pub = 0.0
        source_ok = False        # đang nhận được frame MỚI hay không
        ever_ok = False          # đã từng nhận được frame (để phân biệt lần đầu ↔ nối lại)
        lost_announced = False   # đã báo source_lost cho lần rớt này chưa (chống spam)
        # FrameSource giữ frame CŨ trong bộ nhớ và tự kết nối lại ngầm, nên khi camera
        # chết thì read() vẫn trả ảnh (đứng hình) và .alive vẫn True → không thể dựa vào
        # hai thứ đó. Bám theo bộ đếm frames_read mới biết có ảnh MỚI thật hay không.
        stale_after = float(os.getenv("SOURCE_STALE_SEC", "5"))
        last_frames_read, last_new_t = -1, time.time()

        while self.running:
            if self.fs is None:
                time.sleep(self.reconnect_interval)
                self._open_source()
                continue

            frame = self.fs.read()
            if frame is None:
                if not self.fs.alive:
                    print(f"[StreamWorker] {self.stream_id} thread dead, reconnecting...", flush=True)
                    # Báo MỘT lần cho mỗi lần rớt — kể cả khi chưa từng kết nối được.
                    # Nếu chỉ báo khi đã từng có frame thì camera sai URL / chưa bật sẽ
                    # im lặng mãi, backend không biết vì sao không có dữ liệu.
                    if not lost_announced:
                        lost_announced = True
                        source_ok = False
                        self._publish_status(
                            "source_lost",
                            "mất kết nối RTSP" if ever_ok else "không kết nối được RTSP")
                    self.fs.stop()
                    self.fs = None
                    time.sleep(self.reconnect_interval)
                else:
                    time.sleep(0.1) # Wait for first frame
                if not lost_announced and time.time() - last_new_t > stale_after:
                    lost_announced = True
                    self._publish_status("source_lost", "không kết nối được RTSP")
                continue

            # ----- Ảnh này có MỚI không? -----
            fr_read = getattr(self.fs, "frames_read", 0)
            now = time.time()
            if fr_read != last_frames_read:
                last_frames_read, last_new_t = fr_read, now
                if not source_ok:                       # camera (lại) chạy
                    source_ok, lost_announced = True, False
                    self._publish_status("reconnected" if ever_ok else "source_ok")
                    ever_ok = True
            elif source_ok and now - last_new_t > stale_after:
                source_ok = False
                if not lost_announced:
                    lost_announced = True
                    self._publish_status("source_lost",
                                         f"không có frame mới trong {stale_after:.0f}s")

            if not source_ok:
                # Ảnh đã ĐỨNG HÌNH — tuyệt đối không publish detection nữa, nếu không
                # backend thấy "31 người" đứng yên vĩnh viễn trên một camera đã chết.
                time.sleep(0.1)
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
        self._publish_status("stopped")

    # -------------------------------------------------------------------
    def _publish_result(self, annotated_bgr):
        # 1. Frame result
        boxes = []
        det = self.counter.last_det
        if det is not None and len(det) > 0 and det.tracker_id is not None:
            has_cls = bool(getattr(det, "data", None)) and "class_name" in det.data
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
                    "class": str(det.data["class_name"][i]) if has_cls and len(det.data["class_name"]) > i else "",
                    "confidence": float(det.confidence[i]) if hasattr(det, "confidence") and det.confidence is not None else 0.0,
                    "bbox": [x1, y1, w, h],
                })
                self._track_last_seen[tid] = time.time()
                self._track_classes[tid] = boxes[-1]["class"]

                # Emit start event when a new track appears
                if tid not in self._reported_tracks:
                    self._reported_tracks.add(tid)
                    self._publish_track_event(str(tid), boxes[-1]["class"], "start")

        # Emit end events for tracks not seen for 2 seconds.
        # PHẢI nằm NGOÀI khối trên: khung hình TRỐNG (len(det)==0) mới đúng là lúc cần
        # bắn "end" nhất. Để bên trong thì mọi vật rời khỏi khung = không track nào kết
        # thúc, backend treo track vĩnh viễn.
        now_ts = time.time()
        ended_tids = [tid for tid, last_seen in list(self._track_last_seen.items())
                      if now_ts - last_seen > 2.0]
        for tid in ended_tids:
            cls_name = self._track_classes.pop(tid, "unknown")
            self._publish_track_event(str(tid), cls_name, "end")
            del self._track_last_seen[tid]
            self._reported_tracks.discard(tid)

        # Clear line crossing events — IN/OUT không nằm trong enum start|end của spec
        if det is not None and hasattr(det, "cross_events"):
            det.cross_events = []

        frame_msg = {
            "type": "frame",
            "camera_id": self.camera_id,
            "stream_id": self.stream_id,
            "frame_timestamp": _now_iso(),
            "boxes": boxes,
        }
        
        self._publish(frame_msg)
        print(f"[StreamWorker] {self.stream_id} published frame with {len(boxes)} boxes", flush=True)

    # -------------------------------------------------------------------
    def _publish(self, msg: dict, durable: bool = False):
        """Đẩy 1 message lên Redis. ``durable`` = gửi kèm sang stream bền.

        Stream chính (``REDIS_STREAM_KEY``) nhận MỌI loại tin, y như trước — backend
        đang chạy không phải sửa gì. Tin nghiệp vụ đi THÊM vào stream bền để không bị
        `frame` đẩy văng sau vài phút.
        """
        payload = {"data": json.dumps(msg)}
        targets = [(self.stream_key, self.maxlen)]
        if durable:
            targets.append((self.event_stream_key, self.event_maxlen))
        for key, maxlen in targets:
            try:
                self.redis.xadd(key, payload, maxlen=maxlen, approximate=True)
            except Exception as e:
                print(f"[StreamWorker] {self.stream_id} publish lên '{key}' lỗi: {e}", flush=True)

    # -------------------------------------------------------------------
    def _publish_status(self, status: str, detail: str = ""):
        """Báo vòng đời luồng: started · source_ok · source_lost · reconnected · stopped.

        Không có tin này thì backend không phân biệt được "camera chết" với "đang
        không có ai đi qua" — cả hai đều chỉ là tin ngừng chảy.
        """
        print(f"[StreamWorker] {self.stream_id} status={status} {detail}".rstrip(), flush=True)
        self._publish({
            "type": "stream_status",
            "camera_id": self.camera_id,
            "stream_id": self.stream_id,
            "status": status,
            "detail": detail,
            "timestamp": _now_iso(),
        }, durable=True)

    # -------------------------------------------------------------------
    def _publish_track_event(self, track_id: str, class_name: str, event: str):
        self._publish({
            "type": "track_event",
            "camera_id": self.camera_id,
            "stream_id": self.stream_id,
            "track_id": track_id,
            "class": class_name,
            "event": event,
            "timestamp": _now_iso(),
        }, durable=True)

    # -------------------------------------------------------------------
    def stop(self):
        self.running = False
        if self.fs:
            self.fs.stop()

    # -------------------------------------------------------------------
    def update_params(self, params: StreamParams):
        """PATCH = cập nhật MỘT PHẦN — chỉ ghi đè trường backend thực sự gửi lên.

        Gán đè nguyên khối sẽ xoá mất ``classes``/``roi`` cũ khi backend chỉ gửi mỗi
        ``conf`` → luồng âm thầm nhảy từ đếm-vạch về fullscreen, prompt về "person".
        BACKEND_INTEGRATION.md §4 hứa "chỉ cần truyền những trường muốn thay đổi".
        Gửi tường minh ``null`` (vd ``{"roi": null}``) vẫn xoá được — vì
        ``exclude_unset`` phân biệt "không gửi" với "gửi null".
        """
        merged = _dump(self.params)
        merged.update(_dump(params, exclude_unset=True))
        self.params = StreamParams(**merged)

        # Dựng lại counter = tracker MỚI TINH, track_id đánh lại từ 1. Phải đóng vòng đời
        # track cũ trước — không thì backend treo track không bao giờ kết thúc, rồi lại
        # nhận đúng những track_id đó từ tracker mới (trùng id, sai dữ liệu).
        for tid in list(self._reported_tracks):
            self._publish_track_event(str(tid), self._track_classes.get(tid, "unknown"), "end")
        self._reported_tracks.clear()
        self._track_last_seen.clear()
        self._track_classes.clear()
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
