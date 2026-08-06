"""Dựng ``CountScenario`` + chọn detector cho service từ cấu hình request.

Tách riêng (thuần dữ liệu) để test được KHÔNG cần fastapi/GPU.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..base import MonitoringMode
from ..coco import COCO_ALIASES
from ..scenarios import CountScenario

__all__ = ["wants_yolo", "make_scenario", "get_detector", "clear_detector_cache"]

# Prompt map được về lớp COCO → dùng YOLO (nhanh, realtime); còn lại (mô tả mở, sản
# phẩm không thuộc COCO) → LocateAnything-3B (open-vocab, chậm).
_YOLO_WORDS = {"person", "people", "nguoi", "car", "vehicle", "xe", "truck", "bus",
               "motorcycle", "motorbike", "bicycle", "bottle"}


_ARTICLES = {"a", "an", "the", "một", "các", "những"}


def wants_yolo(prompt: str) -> bool:
    """True → YOLO (lớp COCO cố định, nhanh); False → LocateAnything (open-vocab).

    CHỈ định tuyến YOLO khi prompt là TỪ-LỚP ĐƠN THUẦN (car / truck / person / 'xe tải').
    Có thêm MÔ TẢ (màu 'a red car', tính từ 'a large truck', 'wearing a backpack') →
    LocateAnything, vì YOLO bỏ qua màu/mô tả (chỉ biết lớp). Đây là điểm mấu chốt để
    nhận diện MÀU XE / đặc điểm.
    """
    p = (prompt or "").lower().strip()
    if not p:
        return False
    if p in COCO_ALIASES:                       # cụm lớp COCO (vd 'ô tô', 'xe tải', 'xe máy')
        return True
    words = [w for w in p.split() if w not in _ARTICLES]   # bỏ mạo từ
    single = _YOLO_WORDS | {k for k in COCO_ALIASES if " " not in k}
    return len(words) == 1 and words[0] in single          # đúng 1 từ-lớp → YOLO; còn lại → LA


def make_scenario(prompt: str, counting_type: str = "line",
                  line: Optional[List[float]] = None,
                  zone: Optional[List[List[float]]] = None,
                  resolution: Tuple[int, int] = (960, 540),
                  in_label: str = "IN", out_label: str = "OUT",
                  anchor: Optional[str] = None,
                  model: str = "auto") -> Tuple[CountScenario, str]:
    """Trả (scenario, kind) với kind ∈ {'yolo','locate'}.

    model: 'yolo' | 'locate' | 'auto' (auto suy từ prompt).
    """
    counting_type = counting_type if counting_type in ("line", "zone", "fullscreen") else "line"
    kind = model if model in ("yolo", "locate") else ("yolo" if wants_yolo(prompt) else "locate")
    model_name = "YOLO-NAS-S" if kind == "yolo" else "LocateAnything-3B"
    mode = MonitoringMode.STANDARD if kind == "yolo" else MonitoringMode.SMART

    kw = dict(key="stream", title="Camera stream", usecase_id="uc-stream", mode=mode,
              model=model_name, prompt=prompt, counting_type=counting_type,
              resolution=(int(resolution[0]), int(resolution[1])),
              in_label=in_label, out_label=out_label,
              zone_anchor=anchor or ("CENTER" if counting_type == "line" else "BOTTOM_CENTER"))
    if counting_type == "line":
        ln = line or [0.0, 50.0, 100.0, 50.0]
        kw.update(line_start_pct=(float(ln[0]), float(ln[1])),
                  line_end_pct=(float(ln[2]), float(ln[3])))
    elif counting_type == "zone":
        zn = zone or [[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]]
        kw.update(zone_points_pct=tuple((float(x), float(y)) for x, y in zn))
    # fullscreen: KHÔNG cần vạch/vùng — đếm toàn khung.

    sc = CountScenario(**kw)
    sc.validate()
    return sc, kind


# --------------------------------------------------------------------------- #
# Detector dùng chung (nạp 1 lần, cache). Service chỉ dùng YOLO (người/xe COCO).
# --------------------------------------------------------------------------- #
_DET_CACHE: dict = {}


def get_detector(kind: str = "yolo", confidence: float = 0.25):
    """Nạp (hoặc lấy từ cache) detector YOLO (super-gradients → fallback ultralytics).

    Service chỉ dùng YOLO. Tham số ``kind`` giữ để tương thích chữ ký cũ (bỏ qua giá trị).
    """
    if "yolo" not in _DET_CACHE:
        from ..detectors import load_standard_detector

        det = load_standard_detector(confidence=confidence)
        det.load()
        _DET_CACHE["yolo"] = det
    return _DET_CACHE["yolo"]


def clear_detector_cache():
    _DET_CACHE.clear()
