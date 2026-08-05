"""``FrameSource`` — đọc frame từ camera/stream trong LUỒNG NỀN.

Hỗ trợ: RTSP (``rtsp://``), HTTP(S) stream / file (``http…``, đường dẫn), webcam
(chuỗi số ``"0"`` → chỉ số thiết bị). Đọc ở thread riêng + tự kết nối lại khi rớt
→ service luôn lấy được frame MỚI NHẤT mà không bị nghẽn theo tốc độ model.
"""

from __future__ import annotations

import os
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


def _grab_once(source, reconnect, timeout, warmup):
    """1 lần thử lấy frame (dùng FrameSource, đọc ở thread nền nên KHÔNG treo quá timeout)."""
    import time as _t

    fs = FrameSource(source, reconnect=reconnect).start()
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


def grab_snapshot(source, timeout: Optional[float] = None, warmup: int = 2):
    """Lấy 1 FRAME từ nguồn (để người dùng VẼ vạch/vùng lên đó). Trả ndarray BGR hoặc None.

    - File LOCAL thiếu → None NGAY (khỏi chờ).
    - RTSP → THỬ CẢ TCP LẪN UDP (không ép cứng 1 loại): Docker hay cần TCP (NAT), vài camera
      chỉ chạy UDP → thử lần lượt, cái nào ra frame thì lấy. (User đặt sẵn env thì tôn trọng.)
    - Stream mở chậm hơn file → timeout dài hơn (mặc định 20s).
    """
    s = parse_source(source)
    is_file = isinstance(s, str) and not s.startswith(("rtsp://", "http://", "https://", "rtmp://"))
    if is_file and not os.path.exists(s):
        return None
    if timeout is None:
        timeout = 8.0 if is_file else 20.0

    is_rtsp = isinstance(s, str) and s.lower().startswith("rtsp://")
    user_env = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")   # user tự đặt → không tự đổi
    transports = ["tcp", "udp"] if (is_rtsp and not user_env) else [None]
    per = timeout / len(transports)
    try:
        for tr in transports:
            if tr:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{tr}"
            fr = _grab_once(source, reconnect=not is_file, timeout=per, warmup=warmup)
            if fr is not None:
                return fr
        return None
    finally:
        if is_rtsp and not user_env:                            # khôi phục: đừng dính udp cho lần sau
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)


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

    def start(self) -> "FrameSource":
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _open(self):
        import cv2

        # RTSP + chưa ai đặt transport → mặc định TCP (bền trong Docker/NAT). User/snapshot
        # đặt env trước thì tôn trọng (không đè).
        if (isinstance(self.source, str) and self.source.lower().startswith("rtsp://")
                and not os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.source)
        try:                                  # giảm trễ RTSP: buffer nhỏ (bỏ qua nếu không hỗ trợ)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass
        return cap if cap.isOpened() else None

    def _loop(self):
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
            with self._lock:
                self._frame = fr
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
