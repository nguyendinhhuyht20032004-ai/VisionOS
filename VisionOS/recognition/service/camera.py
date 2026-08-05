"""``FrameSource`` — đọc frame từ camera/stream trong LUỒNG NỀN.

Hỗ trợ: RTSP (``rtsp://``), HTTP(S) stream / file (``http…``, đường dẫn), webcam
(chuỗi số ``"0"`` → chỉ số thiết bị). Đọc ở thread riêng + tự kết nối lại khi rớt
→ service luôn lấy được frame MỚI NHẤT mà không bị nghẽn theo tốc độ model.
"""

from __future__ import annotations

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
    def __init__(self, source, reconnect: bool = True, reconnect_delay: float = 1.0):
        self.source = parse_source(source)
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._run = False
        self._thread: Optional[threading.Thread] = None
        self.ok = False
        self.error: Optional[str] = None
        self.frames_read = 0
        self._is_file = False               # True = file (pace theo timeline thật, xem _loop)
        self._frame_interval = 0.0          # 1/fps — fallback pace khi POS_MSEC không có

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
        # FILE (đọc được TỔNG số frame) → phát theo TIMELINE THẬT (POS_MSEC) trong _loop, để MỌI
        # video đúng tốc độ (không phụ thuộc metadata FPS hay sai). Stream trực tiếp (rtsp/webcam/
        # mjpeg) có frame_count<=0 → chạy tự do (nguồn tự giới hạn tốc độ).
        self._is_file = False
        self._frame_interval = 0.0
        try:
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if n and n > 0:
                self._is_file = True
                if fps and 1.0 <= fps <= 120.0:
                    self._frame_interval = 1.0 / float(fps)   # fallback khi POS_MSEC không có
        except Exception:  # noqa: BLE001
            pass
        return cap

    def _loop(self):
        import cv2

        start_wall = None                    # mốc đồng hồ khi bắt đầu phát (đặt ở frame đầu)
        base_msec = 0.0                      # POS_MSEC của frame đầu
        idx = 0
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
                start_wall = None            # mở/replay xong → đặt lại mốc thời gian
                idx = 0
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
            idx += 1
            with self._lock:
                self._frame = fr
            # FILE → phát ĐÚNG TỐC ĐỘ THẬT: khớp đồng hồ với TIMELINE video (POS_MSEC = mốc thời
            # gian thật của frame). KHÔNG phụ thuộc metadata FPS (hay sai/thiếu → video lúc nhanh
            # lúc chậm). POS_MSEC không có → fallback đếm frame × interval (theo FPS). Stream
            # trực tiếp: _is_file=False → chạy tự do.
            if self._is_file:
                if start_wall is None:
                    start_wall = time.time()
                    base_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
                msec = self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
                vid_t = (msec - base_msec) / 1000.0
                if vid_t <= 0.0 and self._frame_interval:
                    vid_t = idx * self._frame_interval
                wall_t = time.time() - start_wall
                if vid_t - wall_t > 0.0:
                    time.sleep(min(vid_t - wall_t, 1.0))      # đi trước lịch → chờ (cap 1s an toàn)
                elif wall_t - vid_t > 2.0:
                    start_wall = time.time() - vid_t          # tụt quá xa → đặt lại mốc (khỏi tua bù)
        if self._cap is not None:
            self._cap.release()

    def read(self):
        """Trả frame MỚI NHẤT (copy) hoặc None nếu chưa có."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def alive(self) -> bool:
        return bool(self._run and self._thread and self._thread.is_alive())
