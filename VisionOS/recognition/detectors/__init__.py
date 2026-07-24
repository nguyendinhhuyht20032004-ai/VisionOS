"""Các mô hình nhận diện (detector) — interface thống nhất ``detect(frame, prompt)``.

  * ``ScriptedDetector``       : detector giả cho kiểm thử (CPU, không GPU).
  * ``YoloNasDetector``        : YOLO-NAS-S (super-gradients), chế độ STANDARD.
  * ``UltralyticsYoloDetector``: YOLOv8 (ultralytics), chế độ STANDARD — cài ổn
    định trên Kaggle/Colab, dùng thay khi không có super-gradients.
  * ``LocateAnythingDetector`` : LocateAnything-3B, chế độ SMART (open-vocab).

Chỉ ``ScriptedDetector`` import được ở mọi môi trường; các detector thật kéo theo
torch/super-gradients/ultralytics/transformers nên được import **lazy**.

Dùng ``load_standard_detector()`` để lấy detector YOLO mà không cần quan tâm
backend nào có sẵn — nó tự chọn super-gradients nếu cài được, không thì ultralytics.
"""

from .fake import (
    ScriptedDetector,
    linear_track,
    merge_tracks,
    blank_frames,
    box_moving,
)

__all__ = [
    "ScriptedDetector",
    "linear_track",
    "merge_tracks",
    "blank_frames",
    "box_moving",
    "load_yolo_nas",
    "load_ultralytics_yolo",
    "load_standard_detector",
    "load_locate_anything",
]


def load_yolo_nas(**kwargs):
    """Factory lazy cho ``YoloNasDetector`` (super-gradients)."""
    from .yolo_nas import YoloNasDetector

    return YoloNasDetector(**kwargs)


def load_ultralytics_yolo(**kwargs):
    """Factory lazy cho ``UltralyticsYoloDetector`` (YOLOv8)."""
    from .ultralytics_yolo import UltralyticsYoloDetector

    return UltralyticsYoloDetector(**kwargs)


def load_locate_anything(**kwargs):
    """Factory lazy cho ``LocateAnythingDetector`` (LocateAnything-3B, open-vocab).

    Detector chế độ SMART: đếm bất kỳ vật gì mô tả bằng ngôn ngữ tự nhiên. Model
    (torch/transformers) chỉ được nạp khi gọi ``detect`` lần đầu.
    """
    from .locate_anything import LocateAnythingDetector

    return LocateAnythingDetector(**kwargs)


def load_standard_detector(backend: str = "auto", **kwargs):
    """Trả detector YOLO chế độ STANDARD, tự chọn backend sẵn có.

    backend:
      * ``"auto"``            (mặc định): thử super-gradients trước, không có thì
        rơi về ultralytics — nhờ vậy ``run_counting.py`` chạy được ngay trên
        Kaggle/Colab mà không cần cài super-gradients.
      * ``"super_gradients"`` : ép dùng YOLO-NAS (lỗi nếu chưa cài).
      * ``"ultralytics"``     : ép dùng YOLOv8.

    Chỉ nhận các tham số chung (``confidence``, ``iou``, ``device``).
    """
    if backend in ("auto", "super_gradients"):
        try:
            import super_gradients  # noqa: F401  (chỉ kiểm tra có cài không)

            return load_yolo_nas(**kwargs)
        except Exception:
            if backend == "super_gradients":
                raise
            print("ℹ️  super-gradients không có → dùng ultralytics YOLOv8.")
    return load_ultralytics_yolo(**kwargs)
