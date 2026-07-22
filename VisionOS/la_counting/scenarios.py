"""Định nghĩa các *bài toán đếm* (test scenarios).

Mỗi :class:`Scenario` gói toàn bộ thứ cần để chạy một bài toán:
prompt cho model, cách đặt vạch đếm, nhãn hiển thị (vd "Vào"/"Ra"), nguồn video
và ngưỡng số frame. Đây chính là phần "nhiều bài toán" trong yêu cầu:

  * ``PEOPLE_IN_OUT`` — đếm người **ra / vào** toà nhà (tách riêng 2 chiều).
  * ``CONVEYOR``      — đếm **sản phẩm** đi qua vạch trên băng chuyền.
  * ``VEHICLES``      — đếm **phương tiện** qua lại.

Config là dữ liệu thuần (không phụ thuộc supervision) nên import & test được ở
mọi môi trường; ``build_line_zone`` mới cần supervision và được import lazy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

__all__ = [
    "LineConfig",
    "Scenario",
    "PEOPLE_IN_OUT",
    "CONVEYOR",
    "VEHICLES",
    "SCENARIOS",
    "build_line_zone",
]

# Các vị trí anchor hợp lệ của supervision mà harness dùng để quyết định thời
# điểm một track "cắt" vạch. Dùng để validate config (test) mà không cần import
# supervision.
VALID_ANCHORS = {
    "TOP_LEFT", "TOP_CENTER", "TOP_RIGHT",
    "CENTER_LEFT", "CENTER", "CENTER_RIGHT",
    "BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT",
}

VALID_ORIENTATIONS = {"horizontal", "vertical"}


@dataclass(frozen=True)
class LineConfig:
    """Cấu hình vạch đếm.

    orientation: "horizontal" (vạch ngang) hoặc "vertical" (vạch dọc).
    position:    vị trí vạch theo tỉ lệ 0..1 của chiều tương ứng
                 (với vạch ngang là theo y, với vạch dọc là theo x).
    anchor:      điểm neo trên bbox dùng để xét cắt vạch (tên Position của sv).
    """

    orientation: str
    position: float
    anchor: str = "CENTER"

    def points(self, w: int, h: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Trả về (start, end) của vạch ở độ phân giải (w, h) theo pixel."""
        if self.orientation == "horizontal":
            y = int(self.position * h)
            return (0, y), (w, y)
        x = int(self.position * w)
        return (x, 0), (x, h)


@dataclass(frozen=True)
class Scenario:
    """Một bài toán đếm hoàn chỉnh."""

    key: str
    title: str
    prompt: str
    line: LineConfig
    resolution: Tuple[int, int] = (1280, 720)  # (w, h) frame đưa vào model
    in_label: str = "IN"
    out_label: str = "OUT"
    max_frames: int = 150
    # Nguồn video: URL tải tự động (nếu công khai) và/hoặc gợi ý đường dẫn local
    # trên Kaggle. Người dùng có thể override khi chạy benchmark.
    video_url: Optional[str] = None
    video_path_hint: Optional[str] = None
    # Kỳ vọng "sanity" — không phải nhãn chính xác, chỉ để cảnh báo khi kết quả
    # vô lý (vd đếm ra 0 hoặc quá nhiều). Dùng trong scorecard.
    expect_min_crossings: int = 0
    notes: str = ""

    def validate(self) -> None:
        """Kiểm tra config hợp lệ; raise ValueError nếu sai."""
        if not self.key or not self.key.isidentifier():
            raise ValueError(f"key không hợp lệ: {self.key!r}")
        if not self.prompt.strip():
            raise ValueError(f"[{self.key}] prompt rỗng")
        if self.line.orientation not in VALID_ORIENTATIONS:
            raise ValueError(
                f"[{self.key}] orientation {self.line.orientation!r} không hợp lệ"
            )
        if not (0.0 < self.line.position < 1.0):
            raise ValueError(
                f"[{self.key}] position phải trong (0,1), gặp {self.line.position}"
            )
        if self.line.anchor not in VALID_ANCHORS:
            raise ValueError(f"[{self.key}] anchor {self.line.anchor!r} không hợp lệ")
        w, h = self.resolution
        if w <= 0 or h <= 0:
            raise ValueError(f"[{self.key}] resolution không hợp lệ: {self.resolution}")
        if self.max_frames <= 0:
            raise ValueError(f"[{self.key}] max_frames phải > 0")


# --------------------------------------------------------------------------- #
# 3 BÀI TOÁN CHUẨN
# --------------------------------------------------------------------------- #

# 1) Đếm người ra/vào toà nhà.
#    Vạch NGANG ở giữa khung; lấy anchor ở chân (BOTTOM_CENTER) vì người đi bộ
#    được định vị chính xác nhất bởi điểm chạm sàn. Tách riêng 2 chiều: người
#    đi xuống qua vạch = "Vào", đi lên = "Ra" (tuỳ bố trí camera có thể đảo).
PEOPLE_IN_OUT = Scenario(
    key="people",
    title="Đếm người ra/vào toà nhà",
    prompt="person",
    line=LineConfig(orientation="horizontal", position=0.50, anchor="BOTTOM_CENTER"),
    in_label="Vào (IN)",
    out_label="Ra (OUT)",
    max_frames=150,
    video_path_hint="/kaggle/input/**/people*.mp4",
    expect_min_crossings=1,
    notes="Camera cửa ra vào; đổi in/out nếu chiều bị ngược so với thực tế.",
)

# 2) Đếm sản phẩm trên băng chuyền.
#    Sản phẩm chạy ngang theo băng chuyền -> dùng vạch DỌC giữa khung. Anchor
#    CENTER vì vật thể nhỏ, gọn, tâm bbox là điểm ổn định nhất.
CONVEYOR = Scenario(
    key="conveyor",
    title="Đếm sản phẩm trên băng chuyền",
    prompt="object",
    line=LineConfig(orientation="vertical", position=0.50, anchor="CENTER"),
    in_label="Qua vạch",
    out_label="Ngược (loại)",
    max_frames=150,
    video_path_hint="/kaggle/input/**/*.mp4",
    expect_min_crossings=1,
    notes="Prompt 'object'/'product'/'box' tuỳ loại hàng; vạch dọc chặn dòng chảy ngang.",
)

# 3) Đếm phương tiện qua lại.
#    Video mẫu công khai của Roboflow. Vạch NGANG ở nửa dưới khung, anchor
#    BOTTOM_CENTER (bánh xe chạm đường).
VEHICLES = Scenario(
    key="vehicles",
    title="Đếm phương tiện qua lại",
    prompt="car",
    line=LineConfig(orientation="horizontal", position=0.55, anchor="BOTTOM_CENTER"),
    in_label="Chiều tới",
    out_label="Chiều lui",
    max_frames=150,
    video_url="https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
    video_path_hint="/kaggle/working/vehicles.mp4",
    expect_min_crossings=1,
    notes="Video mẫu Roboflow tự tải; đổi prompt 'vehicle'/'truck' nếu cần.",
)

# Tra cứu theo key, giữ thứ tự cho scorecard.
SCENARIOS: Dict[str, Scenario] = {
    s.key: s for s in (PEOPLE_IN_OUT, CONVEYOR, VEHICLES)
}


def build_line_zone(scenario: Scenario, w: Optional[int] = None, h: Optional[int] = None):
    """Dựng ``supervision.LineZone`` từ scenario (import supervision lazy).

    w, h mặc định lấy theo ``scenario.resolution`` — chính là kích thước frame
    thực tế đưa vào model, để vạch nằm đúng chỗ.
    """
    import supervision as sv  # lazy: chỉ cần khi thật sự chạy pipeline

    if w is None or h is None:
        w, h = scenario.resolution
    (sx, sy), (ex, ey) = scenario.line.points(w, h)
    anchor = getattr(sv.Position, scenario.line.anchor)
    return sv.LineZone(
        start=sv.Point(sx, sy),
        end=sv.Point(ex, ey),
        triggering_anchors=(anchor,),
    )
