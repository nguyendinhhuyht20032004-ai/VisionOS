"""``FrameSource`` — đọc frame từ camera/stream trong LUỒNG NỀN.

Hỗ trợ: RTSP (``rtsp://``), HTTP(S) stream / file (``http…``, đường dẫn), webcam
(chuỗi số ``"0"`` → chỉ số thiết bị). Đọc ở thread riêng + tự kết nối lại khi rớt
→ service luôn lấy được frame MỚI NHẤT mà không bị nghẽn theo tốc độ model.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

__all__ = ["FrameSource", "parse_source", "grab_snapshot", "encode_jpeg"]


def parse_source(source):
    """'0'/'1' → int (webcam); còn lại giữ nguyên chuỗi (rtsp/http/file)."""
    if isinstance(source, int):
        return source
    s = str(source).strip()
    return int(s) if s.isdigit() else s


def grab_snapshot(source, timeout: float = 8.0, warmup: int = 3):
    """Lấy 1 FRAME từ nguồn (để người dùng VẼ vạch/vùng lên đó). Trả ndarray BGR hoặc None.

    Đọc vài frame đầu (warmup) cho camera ổn định rồi trả frame mới nhất. Dùng
    ``FrameSource`` (chịu được RTSP/HTTP/file/webcam + tự reconnect trong ``timeout``).
    """
    import time as _t

    fs = FrameSource(source, reconnect=True).start()
    frame, got = None, 0
    t0 = _t.time()
    try:
        while _t.time() - t0 < timeout:
            fr = fs.read()
            if fr is not None:
                frame = fr
                got += 1
                if got >= warmup:
                    break
            elif fs.error and not fs.alive:
                break
            _t.sleep(0.05)
    finally:
        fs.stop()
    return frame


def encode_jpeg(frame_bgr, quality: int = 85):
    """ndarray BGR → bytes JPEG."""
    import cv2

    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return buf.tobytes() if ok else None


class FrameSource:
    def __init__(self, source, reconnect: bool = True, reconnect_delay: float = 1.0,
                 drop_frames: Optional[bool] = None, queue_size: int = 8):
        self.source = parse_source(source)
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay
        # drop_frames: True = chỉ giữ frame MỚI NHẤT (camera trực tiếp → realtime, bỏ frame cũ);
        #   False = KHÔNG bỏ frame, trả ĐÚNG THỨ TỰ (file → phát mượt, đếm chính xác);
        #   None = tự chọn khi mở (file có frame_count>0 → False; stream/rtsp/webcam → True).
        self.drop_frames = drop_frames
        self._queue_size = max(1, int(queue_size))
        self._q: Optional[queue.Queue] = None
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._run = False
        self._thread: Optional[threading.Thread] = None
        self.ok = False
        self.error: Optional[str] = None
        self.frames_read = 0
        self._frame_interval = 0.0          # >0 với FILE → phát đúng tốc độ gốc (xem _open)

    def start(self) -> "FrameSource":
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _open(self):
        import cv2

        cap = cv2.VideoCapture(self.source)
        try:                                  # giảm trễ RTSP: buffer nhỏ (bỏ qua nếu không hỗ trợ)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass
        if not cap.isOpened():
            return None
        # Phân biệt FILE vs STREAM qua TỔNG số frame: file có frame_count>0 (đọc hết được);
        # stream trực tiếp (rtsp/webcam/mjpeg) trả <=0.
        is_file = False
        self._frame_interval = 0.0
        try:
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if n and n > 0:
                is_file = True
                if fps and 1.0 <= fps <= 120.0:
                    # FILE → phát ĐÚNG FPS gốc (nếu không cv2 đọc nhanh hết cỡ → video TUA NHANH).
                    self._frame_interval = 1.0 / float(fps)
        except Exception:  # noqa: BLE001
            pass
        # Chốt chế độ giữ/bỏ frame (chỉ lần mở ĐẦU): file → KHÔNG bỏ frame (phát mượt, đúng
        # thứ tự → tracker/đếm chính xác); stream → giữ frame mới nhất (realtime, bỏ frame cũ).
        if self.drop_frames is None:
            self.drop_frames = not is_file
        if not self.drop_frames and self._q is None:
            self._q = queue.Queue(maxsize=self._queue_size)
        return cap

    def _loop(self):
        next_t = time.time()
        while self._run:
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    self.ok = False
                    self.error = f"Không mở được nguồn: {self.source!r}"
                    if not self.reconnect:
                        break
                    time.sleep(self.reconnect_delay)
                    continue
                self.error = None
                next_t = time.time()          # mở/replay xong → đặt lại lịch phát
            ok, fr = self._cap.read()
            if not ok:
                self._cap.release()
                self._cap = None
                self.ok = False
                if not self.reconnect:        # file hết → dừng
                    self.error = "Hết luồng (end of stream)."
                    break
                time.sleep(self.reconnect_delay)
                continue
            self.ok = True
            self.frames_read += 1
            if self.drop_frames:
                with self._lock:                  # camera trực tiếp: chỉ giữ frame mới nhất
                    self._frame = fr
            else:
                # FILE: đẩy vào hàng đợi theo THỨ TỰ, KHÔNG bỏ frame. Đầy → CHỜ consumer
                # (backpressure) → tự khớp tốc độ, phát mượt, không nhảy frame.
                while self._run:
                    try:
                        self._q.put(fr, timeout=0.2)
                        break
                    except queue.Full:
                        continue
            # FILE → ngủ cho đủ 1 nhịp frame = phát ĐÚNG tốc độ gốc. Stream trực tiếp:
            # _frame_interval=0 → không ngủ. Lịch CỘNG DỒN (next_t += interval) tránh trôi giờ.
            if self._frame_interval:
                next_t += self._frame_interval
                delay = next_t - time.time()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -1.0:            # tụt quá xa (giật/model chậm) → đồng bộ lại
                    next_t = time.time()
        if self._cap is not None:
            self._cap.release()

    def read(self):
        """Trả 1 frame để xử lý (copy) hoặc None nếu chưa có.

        - Camera trực tiếp (drop_frames=True): frame MỚI NHẤT (bỏ frame cũ → realtime).
        - FILE (drop_frames=False): frame KẾ TIẾP theo ĐÚNG thứ tự trong hàng đợi (phát mượt).
        """
        if self._q is not None:
            try:
                return self._q.get(timeout=0.1)
            except queue.Empty:
                return None
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def alive(self) -> bool:
        return bool(self._run and self._thread and self._thread.is_alive())
