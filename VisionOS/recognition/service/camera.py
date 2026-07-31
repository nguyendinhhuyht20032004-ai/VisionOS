"""``FrameSource`` — đọc frame từ camera/stream trong LUỒNG NỀN.

Hỗ trợ: RTSP (``rtsp://``), HTTP(S) stream / file (``http…``, đường dẫn), webcam
(chuỗi số ``"0"`` → chỉ số thiết bị). Đọc ở thread riêng + tự kết nối lại khi rớt
→ service luôn lấy được frame MỚI NHẤT mà không bị nghẽn theo tốc độ model.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

__all__ = ["FrameSource", "parse_source"]


def parse_source(source):
    """'0'/'1' → int (webcam); còn lại giữ nguyên chuỗi (rtsp/http/file)."""
    if isinstance(source, int):
        return source
    s = str(source).strip()
    return int(s) if s.isdigit() else s


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
