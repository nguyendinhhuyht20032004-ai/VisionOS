"""Catalog VIDEO CÔNG KHAI (tải trực tiếp) để test đếm trên NHIỀU kịch bản thật.

Bổ sung cho bộ test đếm NGƯỜI đã có (xem Test_Model.xlsx): thêm **dây chuyền sản
xuất** và **phương tiện vào/ra**, mỗi bài nhiều video — để đánh giá pipeline
detect → track → đếm trên cảnh thật, không chỉ kịch bản giả lập.

Nguồn video: **supervision video-examples** của Roboflow — CDN ổn định, tải trực
tiếp, KHÔNG cần API key:

    https://media.roboflow.com/supervision/video-examples/<file>

Cách tải (ưu tiên API chính thức, có kiểm tra hash; fallback tải thẳng CDN):
    from recognition.video_catalog import CATALOG, download_video
    path = download_video(CATALOG[0])   # -> đường dẫn file .mp4 cục bộ

Mỗi :class:`VideoScenario` gói 1 video + cấu hình vạch đếm ĐÃ chỉnh cho video đó,
quy về :class:`~recognition.scenarios.CountScenario` để chạy pipeline như thường.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from .base import MonitoringMode
from .scenarios import CountScenario

__all__ = ["VideoScenario", "CATALOG", "by_task", "download_video", "RB_CDN"]

# CDN Roboflow (ổn định) — filename trùng giá trị supervision.assets.VideoAssets.
RB_CDN = "https://media.roboflow.com/supervision/video-examples/"


@dataclass(frozen=True)
class VideoScenario:
    """1 video công khai + cấu hình đếm cho video đó."""

    name: str                # tên hiển thị trong scorecard
    filename: str            # tên file trên CDN (= VideoAssets value)
    asset: str               # tên hằng VideoAssets (để download_assets có hash-check)
    task: str                # "vehicles" | "conveyor" | "people"
    scenario: CountScenario  # cấu hình vạch/prompt/model cho video này
    source: str              # mô tả nguồn + đặc điểm cảnh
    tips: str = ""           # gợi ý chỉnh vạch nếu đếm lệch

    @property
    def url(self) -> str:
        return RB_CDN + self.filename


def _line(key, title, prompt, model, mode, ls, le, in_lbl, out_lbl, anchor, res, note):
    return CountScenario(
        key=key, title=title, usecase_id=f"uc-{key}", mode=mode, model=model,
        prompt=prompt, counting_type="line",
        line_start_pct=ls, line_end_pct=le, in_label=in_lbl, out_label=out_lbl,
        zone_anchor=anchor, resolution=res, notes=note,
    )


# --------------------------------------------------------------------------- #
# 🚗 PHƯƠNG TIỆN VÀO/RA — YOLO (car/truck/bus/motorcycle là lớp COCO, đếm ổn).
#    Xe chạy dọc khung → vạch NGANG giữa; xuống = một chiều, lên = chiều kia.
# --------------------------------------------------------------------------- #
_VEHICLES = [
    VideoScenario(
        name="Cao tốc (vehicles.mp4)",
        filename="vehicles.mp4", asset="VEHICLES", task="vehicles",
        source="Roboflow supervision · cao tốc quay top-down, xe chạy dọc, ~1280×720",
        scenario=_line(
            "vehicles", "Đếm xe cao tốc", "car", "YOLO-NAS-S", MonitoringMode.STANDARD,
            (0.0, 50.0), (100.0, 50.0), "Xuống", "Lên", "CENTER", (1280, 720),
            "Vạch ngang giữa khung; đổi prompt 'truck'/'bus' để đếm loại khác.",
        ),
        tips="Nếu xe đi ngang: đổi vạch sang DỌC (50,0)->(50,100).",
    ),
    VideoScenario(
        name="Giao lộ (vehicles-2.mp4)",
        filename="vehicles-2.mp4", asset="VEHICLES_2", task="vehicles",
        source="Roboflow supervision · góc quay khác, nhiều làn",
        scenario=_line(
            "vehicles2", "Đếm xe giao lộ", "car", "YOLO-NAS-S", MonitoringMode.STANDARD,
            (0.0, 55.0), (100.0, 55.0), "Chiều tới", "Chiều lui", "CENTER", (1280, 720),
            "Vạch ngang hơi dưới giữa để bắt xe ở gần camera.",
        ),
        tips="Chỉnh y của vạch (50->65) nếu xe bị đếm quá sớm/muộn.",
    ),
]

# --------------------------------------------------------------------------- #
# 📦 DÂY CHUYỀN SẢN XUẤT — chai chạy NGANG trên băng chuyền → vạch DỌC giữa.
#    'bottle' là lớp COCO nên YOLO đếm được; đổi model='LocateAnything-3B' +
#    prompt mô tả để đếm hàng KHÔNG thuộc COCO (thùng carton, linh kiện…).
# --------------------------------------------------------------------------- #
_CONVEYOR = [
    VideoScenario(
        name="Dây chuyền chiết chai (milk-bottling-plant.mp4)",
        filename="milk-bottling-plant.mp4", asset="MILK_BOTTLING_PLANT", task="conveyor",
        source="Roboflow supervision · nhà máy chiết sữa, chai chạy ngang băng chuyền",
        scenario=_line(
            "conveyor", "Đếm chai trên chuyền", "bottle", "YOLO-NAS-S", MonitoringMode.STANDARD,
            (50.0, 0.0), (50.0, 100.0), "Qua vạch", "Ngược (loại)", "CENTER", (1280, 720),
            "Vạch DỌC giữa khung chặn dòng chai chảy ngang.",
        ),
        tips="Hàng không phải COCO → dùng --model locate, prompt='thùng carton'.",
    ),
]

# --------------------------------------------------------------------------- #
# 🚶 NGƯỜI VÀO/RA — bổ sung để đối chiếu với bộ test người đã có trong Excel.
# --------------------------------------------------------------------------- #
_PEOPLE = [
    VideoScenario(
        name="Người đi bộ (people-walking.mp4)",
        filename="people-walking.mp4", asset="PEOPLE_WALKING", task="people",
        source="Roboflow supervision · lối đi bộ top-down, mật độ vừa",
        scenario=_line(
            "people", "Đếm người vào/ra", "person", "YOLO-NAS-S", MonitoringMode.STANDARD,
            (0.0, 50.0), (100.0, 50.0), "Vào", "Ra", "BOTTOM_CENTER", (1280, 720),
            "Vạch ngang giữa; anchor chân người ổn định nhất.",
        ),
    ),
    VideoScenario(
        name="Quảng trường (market-square.mp4)",
        filename="market-square.mp4", asset="MARKET_SQUARE", task="people",
        source="Roboflow supervision · quảng trường đông, nhiều hướng đi",
        scenario=_line(
            "people_sq", "Đếm người quảng trường", "person", "YOLO-NAS-S", MonitoringMode.STANDARD,
            (0.0, 50.0), (100.0, 50.0), "Vào", "Ra", "BOTTOM_CENTER", (1280, 720),
            "Cảnh đông + đi nhiều hướng → recall thấp là bình thường.",
        ),
        tips="Cảnh rất đông: tăng --max-frames, cân nhắc conf thấp hơn.",
    ),
    VideoScenario(
        name="Ga tàu điện (subway.mp4)",
        filename="subway.mp4", asset="SUBWAY", task="people",
        source="Roboflow supervision · ga tàu điện ngầm, luồng người 2 chiều",
        scenario=_line(
            "people_sub", "Đếm người ga tàu", "person", "YOLO-NAS-S", MonitoringMode.STANDARD,
            (0.0, 55.0), (100.0, 55.0), "Vào", "Ra", "BOTTOM_CENTER", (1280, 720),
            "Luồng 2 chiều rõ → hợp để kiểm IN/OUT.",
        ),
    ),
]

CATALOG: List[VideoScenario] = [*_VEHICLES, *_CONVEYOR, *_PEOPLE]


def by_task(task: Optional[str] = None) -> List[VideoScenario]:
    """Lọc catalog theo bài toán ('vehicles' | 'conveyor' | 'people'); None = tất cả."""
    if not task:
        return list(CATALOG)
    return [v for v in CATALOG if v.task == task]


def download_video(vs: VideoScenario, dest_dir: str = "videos") -> str:
    """Tải video của 1 :class:`VideoScenario` về máy, trả đường dẫn file .mp4.

    Ưu tiên ``supervision.assets.download_assets`` (chính thức, KIỂM TRA HASH). Nếu
    không có, tải thẳng từ CDN bằng ``urllib``. Đã có file thì dùng lại (không tải lại).
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, vs.filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

    # (1) API chính thức của supervision — có hash-check, đáng tin nhất.
    try:
        from supervision.assets import VideoAssets, download_assets

        asset = getattr(VideoAssets, vs.asset, None)
        if asset is not None:
            cwd = os.getcwd()
            os.chdir(dest_dir)  # download_assets lưu vào CWD theo tên file
            try:
                path = download_assets(asset)
            finally:
                os.chdir(cwd)
            full = path if os.path.isabs(path) else os.path.join(dest_dir, os.path.basename(path))
            if os.path.exists(full):
                return full
    except Exception as e:  # noqa: BLE001 — fallback bên dưới
        print(f"ℹ️  supervision.assets không dùng được ({e}); tải thẳng CDN…")

    # (2) Fallback: tải thẳng từ CDN Roboflow.
    import urllib.request

    print(f"⬇️  Tải {vs.filename} từ {vs.url} …")
    urllib.request.urlretrieve(vs.url, dest)
    return dest
