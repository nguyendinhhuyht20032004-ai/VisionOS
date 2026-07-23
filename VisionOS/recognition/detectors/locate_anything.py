"""``LocateAnythingDetector`` — mô hình chế độ SMART (open-vocabulary).

Đây là **adapter** bọc lại ``la_counting.detector.LocateAnythingDetector`` đã có
sẵn trong repo (bọc ``nvidia/LocateAnything-3B``), quy đổi output sang kiểu
``recognition.base.Detection`` / ``DetectorResult`` để dùng chung interface
``detect(frame, prompt)`` với YOLO-NAS và detector giả.

Nhờ tái sử dụng, ta không viết lại phần nạp model + parse toạ độ (đã được unit
-test kỹ ở ``tests/test_parsing.py``); ``torch``/``transformers`` vẫn import lazy.
"""

from __future__ import annotations

import time
from typing import Optional

from ..base import BoundingBox, Detection, DetectorResult

__all__ = ["LocateAnythingDetector"]


class LocateAnythingDetector:
    """Wrapper open-vocab: đếm bất kỳ vật gì mô tả bằng ngôn ngữ tự nhiên."""

    def __init__(self, model_dir: str = "nvidia/LocateAnything-3B", max_new_tokens: int = 1024):
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self._impl = None

    def load(self):
        # Import lazy: chỉ cần khi chạy model thật (kéo theo torch/transformers).
        from la_counting.detector import LocateAnythingDetector as _LA

        self._impl = _LA(self.model_dir, self.max_new_tokens)
        self._impl.load()
        return self

    def detect(self, frame, prompt: str, max_new_tokens: Optional[int] = None) -> DetectorResult:
        """Phát hiện mọi thực thể khớp mô tả ``prompt`` trong 1 frame BGR."""
        if self._impl is None:
            self.load()
        t0 = time.time()
        la_dets, raw = self._impl.detect_frame(frame, prompt, max_new_tokens)
        dets = [
            Detection(
                BoundingBox(*[float(v) for v in d.bbox]),
                d.class_name,
                float(d.confidence),
            )
            for d in la_dets
        ]
        return DetectorResult(
            dets, raw=raw, latency_ms=(time.time() - t0) * 1000,
            model_name="LocateAnything-3B",
        )

    # Tương thích ngược với code gọi kiểu la_counting (detect_frame → tuple).
    def detect_frame(self, frame, prompt, max_new_tokens=None):
        res = self.detect(frame, prompt, max_new_tokens)
        return res.detections, res.raw
