"""``UltralyticsYoloDetector`` — detector YOLO chế độ STANDARD, cài đặt ổn định.

Giống ``YoloNasDetector`` về vai trò (phát hiện người/xe từ vựng cố định cho các
bài toán đếm), nhưng dùng **ultralytics YOLOv8** thay cho super-gradients. Lý do:
super-gradients ghim nhiều phiên bản cũ, hay xung đột và cài lỗi trên Kaggle/Colab;
ultralytics cài trong vài giây, không xung đột. Cách đếm phía sau không đổi.

``ultralytics`` được import **lazy** trong ``load()`` để module này import được ở
môi trường không có sẵn (phần test dùng detector giả / result giả).
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Set

from ..base import BoundingBox, Detection, DetectorResult
from .yolo_nas import COCO_ALIASES  # dùng chung bảng ánh xạ prompt → lớp COCO

__all__ = ["UltralyticsYoloDetector"]


class UltralyticsYoloDetector:
    """Detector YOLOv8 (ultralytics) cho đối tượng COCO: người, xe…

    ĐỂ ĐẾM ÍT BỎ SÓT NGƯỜI/XE (recall cao), mặc định mạnh tay:
      * **yolov8x** — bản LỚN NHẤT họ YOLOv8 (recall cao nhất; nano/m hay bỏ sót
        người xa/nhỏ, xe khuất). Env ``YOLO_WEIGHTS`` đổi (yolov8m.pt cho nhanh,
        hoặc yolo11x.pt/yolov9e.pt nếu muốn mới hơn).
      * **conf=0.15** — ngưỡng THẤP để bắt cả vật mờ/khuất (bỏ sót giảm; nhiễu thừa
        do tracker + min-track lọc bớt). Env ``YOLO_CONF``.
      * **imgsz=1280** — ảnh lớn → vật NHỎ/ở xa rõ hơn → bắt được. Env ``YOLO_IMGSZ``.
      * **max_det=1000** — cảnh ĐÔNG (đám đông, kẹt xe) không bị cắt ở 300. Env ``YOLO_MAX_DET``.
      * **augment (TTA)** — bật ``YOLO_AUGMENT=1`` để tăng recall thêm (chậm hơn ~2-3×).
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        confidence: Optional[float] = None,
        iou: float = 0.5,
        want: Optional[Set[str]] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
    ):
        # yolov8x (lớn nhất) — RECALL cao nhất, giảm BỎ SÓT người/xe. Đổi bằng YOLO_WEIGHTS.
        self.weights = weights or os.environ.get("YOLO_WEIGHTS", "yolov8x.pt")
        self.confidence = (confidence if confidence is not None
                           else float(os.environ.get("YOLO_CONF", "0.15")))
        self.iou = iou
        self.imgsz = int(imgsz or os.environ.get("YOLO_IMGSZ", "1280"))
        # max_det: cảnh đông không bị chặn ở 300 (mặc định ultralytics). augment=TTA.
        self.max_det = int(os.environ.get("YOLO_MAX_DET", "1000"))
        self.augment = os.environ.get("YOLO_AUGMENT", "0") == "1"
        # Lớp cần giữ: ưu tiên tham số, rồi env YOLO_CLASSES (cho model tuỳ biến như
        # VisDrone có tên lớp khác COCO), None = suy ra từ prompt.
        if want:
            self.want = set(want)
        elif os.environ.get("YOLO_CLASSES"):
            self.want = {c.strip().lower() for c in os.environ["YOLO_CLASSES"].split(",") if c.strip()}
        else:
            self.want = None
        self.device = device
        # TILED inference (SAHI): cắt ảnh thành ô nhỏ rồi chạy YOLO từng ô → vật NHỎ
        # (xe top-down/aerial) to hơn trong ô nên detect được. Bật qua YOLO_TILE=1.
        self.tile = os.environ.get("YOLO_TILE", "0") == "1"
        self.tile_wh = int(os.environ.get("YOLO_TILE_WH", "640"))
        self.tile_overlap = int(os.environ.get("YOLO_TILE_OVERLAP", "128"))
        self._model = None
        self._names = None
        self._slicer = None

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
        w = self._resolve_weights(self.weights)   # hf://… → tải về; URL/path để nguyên
        # RT-DETR (detector transformer của supervision demo) dùng class riêng; YOLO cho phần còn lại.
        if "rtdetr" in w.lower() or "rt-detr" in w.lower():
            try:
                from ultralytics import RTDETR

                self._model = RTDETR(w)
            except Exception:  # noqa: BLE001 — ultralytics cũ → thử YOLO()
                self._model = YOLO(w)
        else:
            self._model = YOLO(w)             # tự tải weight (kể cả http URL) lần đầu
        if self.device:
            self._model.to(self.device)
        self._names = self._model.names       # dict {id: 'person', ...}
        print(f"✅ YOLOv8 ({self.weights}, imgsz={self.imgsz}, conf={self.confidence}, "
              f"max_det={self.max_det}, TTA={self.augment}) loaded in {time.time() - t0:.1f}s")
        return self

    @staticmethod
    def _resolve_weights(w: str) -> str:
        """``hf://repo_id/đường/dẫn.pt`` → tải từ HuggingFace về path cục bộ. Còn lại
        (tên yolo, path, http URL) để nguyên cho ultralytics tự xử lý (nó tải URL được).
        """
        if w.startswith("hf://"):
            from huggingface_hub import hf_hub_download

            parts = w[len("hf://"):].split("/")
            repo_id = "/".join(parts[:2])          # owner/name
            filename = "/".join(parts[2:]) or "best.pt"
            print(f"⬇️  tải model từ HuggingFace: {repo_id} / {filename}")
            return hf_hub_download(repo_id=repo_id, filename=filename)
        return w

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
            if want and name.lower() not in want:
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
        if self.tile:
            dets = self._detect_tiled(frame, want)
            mode = "tiled"
        else:
            result = self._model(frame, conf=self.confidence, iou=self.iou,
                                 imgsz=self.imgsz, max_det=self.max_det,
                                 augment=self.augment, verbose=False)[0]
            dets = self._boxes_to_detections(result.boxes, self._names, want)
            mode = "full"
        return DetectorResult(
            dets,
            raw=f"{len(dets)} dets ({mode})",
            latency_ms=(time.time() - t0) * 1000,
            model_name=f"YOLOv8/ultralytics ({self.weights}{'+tiled' if self.tile else ''})",
        )

    def _detect_tiled(self, frame, want: Set[str]) -> List[Detection]:
        """Chạy YOLO **toàn ảnh + trên nhiều Ô cắt** rồi gộp NMS (supervision).

        - Ô cắt (InferenceSlicer): bắt vật NHỎ/ở xa (aerial, đám đông xa) — nhỏ trong
          ảnh gốc nhưng to trong từng ô.
        - Toàn ảnh: bắt vật TO ở gần (tiling thuần dễ cắt đôi vật lớn hơn 1 ô).
        Gộp 2 nguồn + NMS → phủ cả gần lẫn xa. Chậm hơn ~(số ô + 1) lần.
        """
        import supervision as sv

        if self._slicer is None:
            def _cb(img_slice):
                r = self._model(img_slice, conf=self.confidence, iou=self.iou,
                                max_det=self.max_det, augment=self.augment, verbose=False)[0]
                return sv.Detections.from_ultralytics(r)

            wh = (self.tile_wh, self.tile_wh)
            try:                                    # supervision mới: overlap_wh (pixel)
                self._slicer = sv.InferenceSlicer(
                    callback=_cb, slice_wh=wh,
                    overlap_wh=(self.tile_overlap, self.tile_overlap))
            except TypeError:                       # bản cũ: overlap_ratio_wh (tỉ lệ)
                self._slicer = sv.InferenceSlicer(
                    callback=_cb, slice_wh=wh, overlap_ratio_wh=(0.2, 0.2))

        full = self._model(frame, conf=self.confidence, iou=self.iou,
                           imgsz=self.imgsz, max_det=self.max_det,
                           augment=self.augment, verbose=False)[0]
        det = sv.Detections.merge([sv.Detections.from_ultralytics(full), self._slicer(frame)])
        if len(det):
            det = det.with_nms(threshold=self.iou)

        dets: List[Detection] = []
        for i in range(len(det)):
            name = self._names[int(det.class_id[i])]
            if want and name.lower() not in want:
                continue
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[i])
            conf = float(det.confidence[i]) if det.confidence is not None else 0.85
            dets.append(Detection(BoundingBox(x1, y1, x2, y2), name, conf))
        return dets
