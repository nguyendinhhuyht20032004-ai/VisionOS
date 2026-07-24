"""``UltralyticsYoloDetector`` — detector YOLO chế độ STANDARD, cài đặt ổn định.

Giống ``YoloNasDetector`` về vai trò (phát hiện người/xe từ vựng cố định cho các
bài toán đếm), nhưng dùng **ultralytics YOLOv8** thay cho super-gradients. Lý do:
super-gradients ghim nhiều phiên bản cũ, hay xung đột và cài lỗi trên Kaggle/Colab;
ultralytics cài trong vài giây, không xung đột. Cách đếm phía sau không đổi.

``ultralytics`` được import **lazy** trong ``load()`` để module này import được ở
môi trường không có sẵn (phần test dùng detector giả / result giả).
"""

from __future__ import annotations

import time
from typing import List, Optional, Set

from ..base import BoundingBox, Detection, DetectorResult
from .yolo_nas import COCO_ALIASES  # dùng chung bảng ánh xạ prompt → lớp COCO

__all__ = ["UltralyticsYoloDetector"]


class UltralyticsYoloDetector:
    """Detector YOLOv8 (ultralytics) cho đối tượng COCO: người, xe…"""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou: float = 0.5,
        want: Optional[Set[str]] = None,
        device: Optional[str] = None,
    ):
        self.weights = weights
        self.confidence = confidence
        self.iou = iou
        self.want = set(want) if want else None  # None = suy ra từ prompt
        self.device = device
        self._model = None
        self._names = None

    def load(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            import subprocess
            import sys

            print("📦 ultralytics chưa cài → đang cài...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "ultralytics"],
                check=True,
            )
            from ultralytics import YOLO

        t0 = time.time()
        self._model = YOLO(self.weights)      # tự tải weight lần đầu
        if self.device:
            self._model.to(self.device)
        self._names = self._model.names       # dict {id: 'person', ...}
        print(f"✅ YOLOv8 ({self.weights}) loaded in {time.time() - t0:.1f}s")
        return self

    def _wanted_classes(self, prompt: str) -> Set[str]:
        """Suy ra tập lớp COCO cần giữ từ prompt (khớp COCO_ALIASES của YOLO-NAS)."""
        if self.want:
            return self.want
        p = prompt.lower().strip()
        for key, classes in COCO_ALIASES.items():
            if key in p:
                return set(classes)
        return {p}

    @staticmethod
    def _boxes_to_detections(boxes, names, want: Set[str]) -> List[Detection]:
        """Quy đổi ``result.boxes`` của ultralytics → list ``Detection`` (đã lọc lớp).

        Tách riêng, thuần Python để **unit-test được không cần cài ultralytics**
        (chỉ cần dựng object 'boxes' giả với .cls/.conf/.xyxy).
        """
        dets: List[Detection] = []
        for b in boxes:
            cls_id = int(b.cls[0])
            name = names[cls_id]
            if want and name not in want:
                continue
            conf = float(b.conf[0])
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            dets.append(Detection(BoundingBox(x1, y1, x2, y2), name, conf))
        return dets

    def detect(self, frame, prompt: str) -> DetectorResult:
        """Phát hiện đối tượng khớp ``prompt`` trong 1 frame BGR (numpy)."""
        if self._model is None:
            self.load()
        t0 = time.time()
        want = self._wanted_classes(prompt)
        result = self._model(frame, conf=self.confidence, iou=self.iou, verbose=False)[0]
        dets = self._boxes_to_detections(result.boxes, self._names, want)
        return DetectorResult(
            dets,
            raw=f"{len(dets)} dets",
            latency_ms=(time.time() - t0) * 1000,
            model_name=f"YOLOv8/ultralytics ({self.weights})",
        )
