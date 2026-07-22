"""Parse text output của LocateAnything-3B thành danh sách bounding box.

Đây là phần *dễ sai nhất* của toàn bộ harness: mô hình trả về toạ độ dưới dạng
text theo nhiều định dạng khác nhau, và ta phải dò regex + quy đổi toạ độ chuẩn
hoá (0..1000) về pixel. Vì vậy nó được tách riêng thành hàm thuần Python để
unit-test kỹ mà không cần nạp mô hình.

Logic ở đây được giữ **đúng y hệt** phương thức ``LocateAnythingDetector._parse``
trong notebook, chỉ khác là viết lại thành hàm độc lập ``parse_boxes``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

__all__ = ["Detection", "parse_boxes", "rescale_box"]

# Mô hình phát ra toạ độ đã chuẩn hoá về thang [0, 1000].
NORM_SCALE = 1000

# Regex cho 3 định dạng đầu ra mà model có thể sinh ra.
_RE_BOX_TAG = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
_RE_BRACKET = re.compile(
    r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
)
_RE_LOC = re.compile(r"<loc_(\d+)>")


@dataclass
class Detection:
    """Một bbox đã giải mã.

    bbox là ``(x1, y1, x2, y2)`` theo **pixel** trong hệ toạ độ có kích thước
    ``(w, h)`` được truyền vào ``parse_boxes``.
    """

    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float


def _add(dets: List[Detection], x1, y1, x2, y2, w, h, cls, conf, scale):
    """Quy đổi toạ độ chuẩn hoá -> pixel, chỉ giữ box hợp lệ (x2>x1, y2>y1)."""
    x1, x2 = int(x1 / scale * w), int(x2 / scale * w)
    y1, y2 = int(y1 / scale * h), int(y2 / scale * h)
    if x2 > x1 and y2 > y1:
        dets.append(Detection((x1, y1, x2, y2), cls, conf))


def parse_boxes(
    text: str,
    class_name: str,
    w: int,
    h: int,
    default_conf: float = 0.85,
) -> List[Detection]:
    """Giải mã text của model ra danh sách :class:`Detection`.

    Thử lần lượt 3 định dạng; định dạng sau chỉ được thử nếu định dạng trước
    KHÔNG cho ra box nào (đúng theo hành vi của notebook gốc):

    1. ``<box><x1><y1><x2><y2></box>`` — toạ độ chuẩn hoá 0..1000.
    2. ``[x1, y1, x2, y2]`` — tự đoán thang: nếu giá trị lớn nhất > 1.5 coi là
       thang 0..1000, ngược lại coi là 0..1 (chuẩn hoá theo tỉ lệ).
    3. ``<loc_n>`` lặp lại theo bộ 4 — toạ độ chuẩn hoá 0..1000.

    Trả về list rỗng nếu không khớp định dạng nào (ví dụ model trả lời bằng
    câu chữ, hoặc không phát hiện đối tượng).
    """
    dets: List[Detection] = []

    # --- Pattern 1: <box><...></box> (thang 0..1000) ---
    for m in _RE_BOX_TAG.findall(text):
        x1, y1, x2, y2 = (int(v) for v in m)
        _add(dets, x1, y1, x2, y2, w, h, class_name, default_conf, NORM_SCALE)

    # --- Pattern 2: [x1, y1, x2, y2] (tự đoán thang) ---
    if not dets:
        for m in _RE_BRACKET.findall(text):
            v = [float(x) for x in m]
            scale = NORM_SCALE if max(v) > 1.5 else 1
            _add(dets, v[0], v[1], v[2], v[3], w, h, class_name, default_conf, scale)

    # --- Pattern 3: <loc_n> theo bộ 4 (thang 0..1000) ---
    if not dets:
        locs = _RE_LOC.findall(text)
        for i in range(0, len(locs) - 3, 4):
            x1, y1, x2, y2 = (int(v) for v in locs[i : i + 4])
            _add(dets, x1, y1, x2, y2, w, h, class_name, default_conf, NORM_SCALE)

    return dets


def rescale_box(
    bbox: Tuple[float, float, float, float],
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """Đổi bbox từ hệ ``from_size`` (w,h) sang ``to_size`` (w,h).

    Dùng khi model chạy trên frame đã resize (vd 1280x720) nhưng cần vẽ / so
    khớp ground-truth trên ảnh gốc (đúng như bước rescale trong cell đánh giá
    mAP của notebook).
    """
    fw, fh = from_size
    tw, th = to_size
    x1, y1, x2, y2 = bbox
    return (
        x1 * tw / fw,
        y1 * th / fh,
        x2 * tw / fw,
        y2 * th / fh,
    )
