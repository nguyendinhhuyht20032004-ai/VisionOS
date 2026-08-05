"""Service AI đếm theo LUỒNG CAMERA.

Đưa nguồn camera (RTSP/HTTP/file/webcam) vào → detect + track + đếm →
xuất luồng MJPEG đã annotate + số đếm realtime qua REST API + trang web.

  * :class:`StreamingCounter` — engine đếm từng frame (có trạng thái).
  * :class:`FrameSource`      — đọc frame nền, tự kết nối lại.
  * :func:`make_scenario` / :func:`get_detector` — dựng cấu hình + detector.
  * ``recognition.service.app`` — ứng dụng FastAPI (import khi chạy service).

App FastAPI KHÔNG import ở đây để ``import recognition.service`` nhẹ (không cần
fastapi). Chạy service: ``python run_service.py`` (hoặc ``python -m recognition.service``).
"""

from .builder import get_detector, make_scenario, wants_yolo
from .camera import FrameSource, encode_jpeg, grab_snapshot, parse_source
from .engine import StreamingCounter
from .vectordb import VectorStore, embed_crop

__all__ = [
    "StreamingCounter", "FrameSource", "parse_source", "grab_snapshot", "encode_jpeg",
    "make_scenario", "get_detector", "wants_yolo",
    "VectorStore", "embed_crop",
]
