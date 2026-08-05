"""``YoloNasDetector`` — mô hình chế độ STANDARD (từ vựng cố định).

Bọc YOLO-NAS-S của super-gradients cho các bài toán đếm người/xe (uc-people-count,
uc-vehicle). ``torch`` / ``super_gradients`` được import **lazy** trong ``load()``
để module này import được ở môi trường không GPU (phần test dùng detector giả).

Chỉ dùng khi thực sự chạy mô hình (GPU). Interface ``detect(frame, prompt)``
đồng nhất với các detector khác, nên ``CountingPipeline`` không cần biết là model
nào đứng sau.
"""

from __future__ import annotations

import time
from typing import List, Optional

from ..base import BoundingBox, Detection, DetectorResult

__all__ = ["YoloNasDetector", "COCO_ALIASES"]

# Tập LỚP XE cho prompt nhóm "vehicle" — dùng đúng các lớp XE của COCO mà YOLO nhận diện:
# car / motorcycle / truck / bus. (YOLO chỉ phân loại các loại xe này.)
_VEHICLE = ["car", "motorcycle", "truck", "bus"]

# Ánh xạ prompt tiếng Việt/thông dụng → lớp COCO mà YOLO nhận diện.
COCO_ALIASES = {
    "person": ["person"],
    "người": ["person"],
    "car": ["car"],
    "ô tô": ["car"],
    "oto": ["car"],
    # "vehicle"/"phương tiện"/"xe" → MỌI loại xe (car/motorcycle/truck/bus)
    "vehicle": _VEHICLE,
    "phương tiện": _VEHICLE,
    "phuong tien": _VEHICLE,
    "xe": _VEHICLE,
    "xe máy": ["motorcycle"],
    "motorcycle": ["motorcycle"],
    "truck": ["truck"],
    "xe tải": ["truck"],
    "tải": ["truck"],
    "bus": ["bus"],
    "xe buýt": ["bus"],
}


class YoloNasDetector:
    """Detector YOLO-NAS-S cho đối tượng từ vựng cố định (COCO)."""

    def __init__(
        self,
        model_name: str = "yolo_nas_s",
        confidence: float = 0.65,
        iou: float = 0.45,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self._loaded = False
        self.model = None

    def load(self):
        import torch
        from super_gradients.training import models

        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        self.model = models.get(self.model_name, pretrained_weights="coco")
        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"✅ YOLO-NAS ({self.model_name}) loaded in {time.time() - t0:.1f}s "
              f"on {self.device}")
        return self

    def _wanted_classes(self, prompt: str) -> List[str]:
        p = prompt.lower().strip()
        for key, classes in COCO_ALIASES.items():
            if key in p:
                return classes
        return [p]  # để nguyên nếu là tên lớp COCO trực tiếp

    def detect(self, frame, prompt: str) -> DetectorResult:
        """Phát hiện đối tượng khớp ``prompt`` trong 1 frame BGR (numpy)."""
        if not self._loaded:
            self.load()
        import numpy as np  # noqa: F401  (đảm bảo có numpy khi chạy thật)

        t0 = time.time()
        wanted = set(self._wanted_classes(prompt))
        preds = self.model.predict(frame, conf=self.confidence, iou=self.iou)

        dets: List[Detection] = []
        # super-gradients trả về đối tượng có .prediction (bboxes_xyxy, labels, confidence)
        pred = preds[0] if hasattr(preds, "__getitem__") else preds
        p = pred.prediction
        names = pred.class_names
        for bbox, label_id, conf in zip(p.bboxes_xyxy, p.labels, p.confidence):
            cls = names[int(label_id)]
            if wanted and cls not in wanted:
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox)
            dets.append(Detection(BoundingBox(x1, y1, x2, y2), cls, float(conf)))

        return DetectorResult(
            dets, raw=str(len(dets)) + " dets",
            latency_ms=(time.time() - t0) * 1000, model_name=self.model_name,
        )
