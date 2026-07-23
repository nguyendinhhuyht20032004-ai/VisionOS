"""Các mô hình nhận diện (detector) — interface thống nhất ``detect(frame, prompt)``.

  * ``ScriptedDetector``      : detector giả cho kiểm thử (CPU, không GPU).
  * ``YoloNasDetector``       : YOLO-NAS-S, chế độ STANDARD (người/xe, từ vựng cố định).
  * ``LocateAnythingDetector``: LocateAnything-3B, chế độ SMART (open-vocab).

Chỉ ``ScriptedDetector`` import được ở mọi môi trường; hai detector thật kéo theo
torch/super-gradients/transformers nên được import **lazy** (chỉ khi ``load()``).
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
    "load_locate_anything",
]


def load_yolo_nas(**kwargs):
    """Factory lazy cho ``YoloNasDetector`` (tránh import torch khi chỉ test)."""
    from .yolo_nas import YoloNasDetector

    return YoloNasDetector(**kwargs)


def load_locate_anything(**kwargs):
    """Factory lazy cho ``LocateAnythingDetector`` (open-vocab, GPU)."""
    from .locate_anything import LocateAnythingDetector

    return LocateAnythingDetector(**kwargs)
