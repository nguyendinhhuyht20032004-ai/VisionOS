"""Nối lại track bị ĐỨT (ReID nhẹ) — giữ **một** id cho cùng một vật.

ByteTrack bám theo chuyển động (Kalman + IoU). Khi vật bị che hoặc detect trượt vài
frame rồi hiện lại ở chỗ lệch so với dự đoán, ByteTrack coi đó là vật MỚI và cấp
``track_id`` khác. Hệ quả: số "vật khác nhau" phình lên và vạch có thể đếm lại → đếm
nhầm.

``TrackStitcher`` xử lý phần đó: mỗi khi một id thô (raw) MỚI xuất hiện, thử gán nó về
một danh tính vừa BIẾN MẤT gần đây dựa trên NGOẠI HÌNH (embedding màu) cộng với hai
điều kiện chặn để không nhập nhầm hai vật khác nhau:

  * id cũ phải đang KHÔNG xuất hiện ở frame này (hai vật cùng thấy thì chắc chắn khác nhau);
  * chỉ nối khi khoảng cách thời gian ngắn, vị trí gần chỗ biến mất, và cùng nhóm lớp.

Thà thỉnh thoảng bỏ sót một lần nối (vật nhận id mới) còn hơn nối nhầm hai vật (đếm
thiếu). Vì vậy các ngưỡng mặc định thiên về AN TOÀN; chỉnh qua env ở ``StreamingCounter``.

Lớp này thuần Python + numpy (không phụ thuộc cv2/supervision) để test được không cần GPU.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

__all__ = ["TrackStitcher"]

_PERSON = {"person", "nguoi", "người"}
_VEHICLE = {"car", "truck", "bus", "motorcycle", "motorbike", "bicycle", "vehicle",
            "ô tô", "oto", "xe", "xe máy", "xe tải"}


def _group(cls: Optional[str]) -> str:
    """Gộp lớp về NHÓM thô (person / vehicle / tên gốc) — chống car↔truck nhấp nháy chặn nhầm."""
    c = (cls or "").strip().lower()
    if c in _PERSON:
        return "person"
    if c in _VEHICLE:
        return "vehicle"
    return c


def _cos(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na = float(np.linalg.norm(a)) or 1.0
    nb = float(np.linalg.norm(b)) or 1.0
    return float(a @ b / (na * nb))


class TrackStitcher:
    """Ánh xạ id thô của ByteTrack → id ỔN ĐỊNH, nối lại danh tính bị đứt.

    Tham số:
      * ``sim_thresh``     : cosine ngoại hình tối thiểu để coi là "cùng vật" (0..1).
      * ``max_gap``        : số frame tối đa kể từ lúc mất tới lúc được nối lại.
      * ``max_dist_frac``  : khoảng cách tâm tối đa (theo đường chéo khung) giữa chỗ biến
                             mất và chỗ hiện lại.
      * ``emb_ema``        : hệ số làm mượt embedding trong gallery (0 = luôn lấy mới nhất).
      * ``enabled``        : False → trả nguyên id thô (tắt hẳn ReID).
    """

    def __init__(self, sim_thresh: float = 0.5, max_gap: int = 60,
                 max_dist_frac: float = 0.3, emb_ema: float = 0.5, enabled: bool = True):
        self.sim_thresh = float(sim_thresh)
        self.max_gap = int(max_gap)
        self.max_dist_frac = float(max_dist_frac)
        self.emb_ema = float(emb_ema)
        self.enabled = bool(enabled)
        self._map: dict = {}        # raw id → stable id
        self._gallery: dict = {}    # stable id → {emb, cx, cy, frame, grp}
        self._frame = 0

    # ------------------------------------------------------------------ #
    def remap(self, raw_ids: Sequence[Optional[int]], boxes, embs: Sequence,
              classes: Sequence[Optional[str]], frame_wh) -> List[Optional[int]]:
        """Trả list id ỔN ĐỊNH tương ứng từng phần tử của ``raw_ids``.

        ``boxes[i]`` = (x1,y1,x2,y2) pixel; ``embs[i]`` = vector ngoại hình (list/ndarray);
        ``classes[i]`` = tên lớp (hoặc None); ``frame_wh`` = (w, h) khung.
        """
        self._frame += 1
        n = len(raw_ids)
        if not self.enabled:
            return [int(r) if r is not None else None for r in raw_ids]

        w, h = frame_wh
        diag = math.hypot(float(w), float(h)) or 1.0

        # Các stable id ĐANG hiện ở frame này (id thô đã biết) — cấm nối vật mới vào chúng.
        active = set()
        for r in raw_ids:
            if r is not None and int(r) in self._map:
                active.add(self._map[int(r)])

        out: List[Optional[int]] = [None] * n
        for i, r in enumerate(raw_ids):
            if r is None:
                continue
            r = int(r)
            if r in self._map:
                s = self._map[r]
            else:
                s = self._try_match(embs[i], boxes[i], classes[i], diag, active)
                if s is None:
                    s = r        # id thô của ByteTrack tăng dần → dùng luôn làm stable id mới, không đụng id cũ
                self._map[r] = s
            active.add(s)
            out[i] = s
            self._touch(s, embs[i], boxes[i], classes[i])

        self._prune()
        return out

    # ------------------------------------------------------------------ #
    def _try_match(self, emb, box, cls, diag: float, active: set) -> Optional[int]:
        """Tìm stable id đã biến mất khớp nhất với vật mới (None nếu không đủ tin)."""
        cx, cy = (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0
        grp = _group(cls)
        best_s, best_sim = None, self.sim_thresh
        for s, g in self._gallery.items():
            if s in active:                                   # id cũ đang hiện → chắc chắn khác vật
                continue
            if self._frame - g["frame"] > self.max_gap:       # mất quá lâu → thôi
                continue
            if g["grp"] != grp:                               # khác nhóm lớp (người vs xe)
                continue
            if math.hypot(cx - g["cx"], cy - g["cy"]) / diag > self.max_dist_frac:
                continue                                       # hiện lại quá xa chỗ biến mất
            sim = _cos(emb, g["emb"])
            if sim >= best_sim:
                best_sim, best_s = sim, s
        return best_s

    def _touch(self, s: int, emb, box, cls):
        """Cập nhật gallery cho stable id (vị trí + frame mới nhất, embedding làm mượt EMA)."""
        import numpy as np

        cx, cy = (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0
        e = np.asarray(emb, dtype="float32")
        g = self._gallery.get(s)
        if g is not None and g.get("emb") is not None and self.emb_ema > 0:
            e = self.emb_ema * np.asarray(g["emb"], dtype="float32") + (1.0 - self.emb_ema) * e
        self._gallery[s] = {"emb": e, "cx": cx, "cy": cy, "frame": self._frame, "grp": _group(cls)}

    def _prune(self):
        """Bỏ khỏi gallery các danh tính đã mất quá lâu (khỏi phình bộ nhớ)."""
        cutoff = self.max_gap * 4
        dead = [s for s, g in self._gallery.items() if self._frame - g["frame"] > cutoff]
        for s in dead:
            del self._gallery[s]
