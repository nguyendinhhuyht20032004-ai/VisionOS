"""Các *bài toán đếm* của sản phẩm — khớp use-case ở frontend (``AI_USECASES``).

Mỗi ``CountScenario`` gói đủ thứ để chạy một bài toán đếm end-to-end: chọn mô
hình nào (YOLO-NAS hay LocateAnything), đối tượng/mô tả cần đếm, kiểu đếm (vạch
hay vùng) và cấu hình vạch/vùng. Đây là 4 bài toán đếm cốt lõi:

  * ``PEOPLE_IN_OUT`` — đếm người **vào/ra** (uc-people-count) — YOLO-NAS, vạch.
  * ``VEHICLES``      — đếm **xe** qua trạm (uc-vehicle) — YOLO-NAS, vạch.
  * ``PACKAGES``      — đếm **kiện hàng** trên băng chuyền (uc-package) — LocateAnything, vạch.
  * ``QUEUE``         — đếm **người xếp hàng** trong vùng (uc-queue) — YOLO-NAS, vùng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .base import MonitoringMode
from .zones import CountingLine, Zone

__all__ = [
    "CountScenario",
    "PEOPLE_IN_OUT",
    "VEHICLES",
    "PACKAGES",
    "QUEUE",
    "COUNT_SCENARIOS",
]


@dataclass(frozen=True)
class CountScenario:
    """Một bài toán đếm hoàn chỉnh (dữ liệu thuần, test được mọi nơi)."""

    key: str
    title: str
    usecase_id: str                      # id use-case tương ứng ở frontend
    mode: MonitoringMode
    model: str
    prompt: str                          # đối tượng (YOLO) hoặc mô tả (LocateAnything)
    counting_type: str                   # "line" | "zone"
    resolution: Tuple[int, int] = (1280, 720)
    # Vạch đếm (khi counting_type == "line"), toạ độ phần trăm 0..100.
    line_start_pct: Tuple[float, float] = (0.0, 50.0)
    line_end_pct: Tuple[float, float] = (100.0, 50.0)
    in_label: str = "IN"
    out_label: str = "OUT"
    # Vùng đếm (khi counting_type == "zone"), toạ độ phần trăm 0..100.
    zone_points_pct: Tuple[Tuple[float, float], ...] = ()
    # NHIỀU vùng đếm CHUNG trong 1 bài (đếm vật trong BẤT KỲ vùng nào) — vd nhiều
    # làn/ô. Nếu set thì dùng cái này thay cho zone_points_pct đơn lẻ.
    zones_pct: Tuple[Tuple[Tuple[float, float], ...], ...] = ()
    zone_anchor: str = "BOTTOM_CENTER"   # điểm neo xét thuộc vùng / cắt vạch
    max_frames: int = 150
    expect_min: int = 1                  # kỳ vọng "sanity" cho scorecard
    notes: str = ""

    def validate(self) -> None:
        if not self.key.isidentifier():
            raise ValueError(f"key không hợp lệ: {self.key!r}")
        if not self.prompt.strip():
            raise ValueError(f"[{self.key}] prompt rỗng")
        if self.counting_type not in ("line", "zone"):
            raise ValueError(f"[{self.key}] counting_type sai: {self.counting_type}")
        if self.counting_type == "zone":
            if self.zones_pct:
                for i, pts in enumerate(self.zones_pct):
                    if len(pts) < 3:
                        raise ValueError(f"[{self.key}] vùng #{i + 1} cần ≥3 đỉnh")
            elif len(self.zone_points_pct) < 3:
                raise ValueError(f"[{self.key}] vùng đếm cần ≥3 đỉnh")
        w, h = self.resolution
        if w <= 0 or h <= 0:
            raise ValueError(f"[{self.key}] resolution không hợp lệ")

    def build_line(self) -> CountingLine:
        return CountingLine(
            id=f"line-{self.key}",
            name=self.title,
            start_pct=self.line_start_pct,
            end_pct=self.line_end_pct,
            in_label=self.in_label,
            out_label=self.out_label,
        )

    def build_zone(self) -> Zone:
        return Zone(
            id=f"zone-{self.key}",
            name=self.title,
            points_pct=list(self.zone_points_pct),
            role="monitor",
        )

    def build_zones(self) -> List[Zone]:
        """Trả về DANH SÁCH vùng (đa-vùng nếu ``zones_pct`` set, ngược lại 1 vùng)."""
        if self.zones_pct:
            return [
                Zone(id=f"zone-{self.key}-{i + 1}", name=f"{self.title} #{i + 1}",
                     points_pct=list(pts), role="monitor")
                for i, pts in enumerate(self.zones_pct)
            ]
        if self.zone_points_pct:
            return [self.build_zone()]
        return []


# --------------------------------------------------------------------------- #
# 4 BÀI TOÁN ĐẾM CHUẨN
# --------------------------------------------------------------------------- #

# 1) Đếm người vào/ra — YOLO-NAS phát hiện "person", vạch ngang giữa khung,
#    anchor chân người (BOTTOM_CENTER) là điểm định vị ổn định nhất.
PEOPLE_IN_OUT = CountScenario(
    key="people",
    title="Đếm người vào/ra",
    usecase_id="uc-people-count",
    mode=MonitoringMode.STANDARD,
    model="YOLO-NAS-S",
    prompt="person",
    counting_type="line",
    line_start_pct=(0.0, 50.0),
    line_end_pct=(100.0, 50.0),
    in_label="Vào",
    out_label="Ra",
    expect_min=1,
    notes="Camera cửa ra vào; đảo in/out nếu chiều ngược thực tế.",
)

# 2) Đếm xe qua trạm — YOLO-NAS ("car"), vạch ngang nửa dưới khung.
VEHICLES = CountScenario(
    key="vehicles",
    title="Đếm xe ra vào",
    usecase_id="uc-vehicle",
    mode=MonitoringMode.STANDARD,
    model="YOLO-NAS-S",
    prompt="car",
    counting_type="line",
    line_start_pct=(0.0, 55.0),
    line_end_pct=(100.0, 55.0),
    in_label="Chiều tới",
    out_label="Chiều lui",
    expect_min=1,
    notes="Đổi prompt 'motorcycle'/'truck' cho loại xe khác.",
)

# 3) Đếm kiện hàng trên băng chuyền — LocateAnything (open-vocab), vạch DỌC
#    chặn dòng chảy ngang; anchor tâm hộp (vật nhỏ, gọn).
PACKAGES = CountScenario(
    key="packages",
    title="Đếm hàng hoá trên chuyền",
    usecase_id="uc-package",
    mode=MonitoringMode.SMART,
    model="LocateAnything-3B",
    prompt="hộp carton, kiện hàng",
    counting_type="line",
    line_start_pct=(50.0, 0.0),
    line_end_pct=(50.0, 100.0),
    in_label="Qua vạch",
    out_label="Ngược (loại)",
    zone_anchor="CENTER",
    expect_min=1,
    notes="Prompt đổi theo loại hàng: 'thùng carton màu vàng', 'chai nước'…",
)

# 4) Phân tích hàng chờ — đếm người trong VÙNG (occupancy), YOLO-NAS.
QUEUE = CountScenario(
    key="queue",
    title="Phân tích hàng chờ",
    usecase_id="uc-queue",
    mode=MonitoringMode.STANDARD,
    model="YOLO-NAS-S",
    prompt="person",
    counting_type="zone",
    zone_points_pct=((20.0, 30.0), (80.0, 30.0), (80.0, 90.0), (20.0, 90.0)),
    zone_anchor="BOTTOM_CENTER",
    expect_min=0,
    notes="Đếm số người đang đứng trong vùng chờ; cảnh báo khi vượt max_limit.",
)

COUNT_SCENARIOS: Dict[str, CountScenario] = {
    s.key: s for s in (PEOPLE_IN_OUT, VEHICLES, PACKAGES, QUEUE)
}
