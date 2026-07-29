"""Catalog VIDEO CÔNG KHAI (tải trực tiếp) để test đếm trên NHIỀU kịch bản/GÓC QUAY.

Bổ sung cho bộ test đếm NGƯỜI đã có (Test_Model.xlsx): thêm **phương tiện vào/ra**
và **dây chuyền sản xuất**, mỗi bài NHIỀU video với góc quay khác nhau — để đánh
giá pipeline detect → track → đếm trên cảnh thật.

Ba nguồn video, đều tải trực tiếp KHÔNG cần API key:
  * ``asset``      — supervision video-examples (Roboflow), tải qua download_assets
    (kiểm tra hash). VD: VEHICLES, MILK_BOTTLING_PLANT, PEOPLE_WALKING…
  * ``url``        — link .mp4 đầy đủ (Pexels video-files) đã xác minh.
  * ``pexels_id``  — chỉ có ID Pexels → downloader tự dò hậu tố chất lượng
    (hd_1920_1080_30fps, sd_640_360_25fps…) đến khi tải được.

Dùng:
    from recognition.video_catalog import CATALOG, by_task, download_video
    path = download_video(CATALOG[0])     # -> đường dẫn .mp4 cục bộ
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from .base import MonitoringMode
from .scenarios import CountScenario

__all__ = ["VideoScenario", "CATALOG", "by_task", "download_video", "RB_CDN", "PEXELS_CDN"]

RB_CDN = "https://media.roboflow.com/supervision/video-examples/"
PEXELS_CDN = "https://videos.pexels.com/video-files/"

# Hậu tố chất lượng Pexels hay gặp — downloader thử lần lượt (ưu tiên HD gọn nhẹ).
_PEXELS_QUALITIES = (
    "hd_1920_1080_30fps", "hd_1920_1080_25fps", "hd_1920_1080_24fps",
    "hd_1280_720_30fps", "hd_1280_720_25fps",
    "sd_960_540_30fps", "sd_640_360_30fps", "sd_640_360_25fps", "sd_640_360_24fps",
    "uhd_2560_1440_30fps", "uhd_2560_1440_25fps",
    "uhd_3840_2160_30fps", "uhd_3840_2160_25fps", "uhd_3840_2160_24fps",
)


@dataclass(frozen=True)
class VideoScenario:
    """1 video công khai + cấu hình đếm cho video đó."""

    name: str
    task: str                          # "vehicles" | "conveyor" | "people"
    scenario: CountScenario
    source: str
    filename: str                      # tên file lưu cục bộ
    asset: Optional[str] = None        # tên hằng supervision VideoAssets
    url: Optional[str] = None          # link .mp4 đầy đủ
    pexels_id: Optional[str] = None    # chỉ ID Pexels (tự dò hậu tố)
    tips: str = ""


def _sc(key, title, prompt, model, ls, le, in_lbl, out_lbl, anchor, res, note):
    return CountScenario(
        key=key, title=title, usecase_id=f"uc-{key}",
        mode=(MonitoringMode.SMART if not model.startswith("YOLO") else MonitoringMode.STANDARD),
        model=model, prompt=prompt, counting_type="line",
        line_start_pct=ls, line_end_pct=le, in_label=in_lbl, out_label=out_lbl,
        zone_anchor=anchor, resolution=res, notes=note,
    )


def _veh(key, title, res=(1280, 720), y=50.0):
    return _sc(key, title, "car", "YOLO-NAS-S", (0.0, y), (100.0, y),
               "Chiều A", "Chiều B", "CENTER", res, "Xe chạy dọc → vạch NGANG.")


def _ppl(key, title, res=(1280, 720), y=50.0):
    return _sc(key, title, "person", "YOLO-NAS-S", (0.0, y), (100.0, y),
               "Vào", "Ra", "BOTTOM_CENTER", res, "Người → vạch ngang, neo chân.")


def _conv_yolo(key, title, prompt="bottle", res=(1280, 720)):
    return _sc(key, title, prompt, "YOLO-NAS-S", (50.0, 0.0), (50.0, 100.0),
               "Qua vạch", "Ngược", "CENTER", res, "Hàng chạy ngang → vạch DỌC.")


def _conv_locate(key, title, prompt, res=(1280, 720)):
    return _sc(key, title, prompt, "LocateAnything-3B", (50.0, 0.0), (50.0, 100.0),
               "Qua vạch", "Ngược", "CENTER", res,
               "Hàng KHÔNG thuộc COCO → open-vocab; vạch DỌC.")


# --------------------------------------------------------------------------- #
# 🚗 PHƯƠNG TIỆN VÀO/RA — 6 góc quay (top-down cao tốc, giao lộ, phố…)
# --------------------------------------------------------------------------- #
_VEHICLES = [
    VideoScenario("Cao tốc top-down (supervision)", "vehicles",
        _veh("veh_hw", "Xe cao tốc"), "Roboflow supervision · cao tốc quay dọc",
        "vehicles.mp4", asset="VEHICLES", tips="Kinh điển cho đếm xe."),
    VideoScenario("Giao lộ nhiều làn (supervision)", "vehicles",
        _veh("veh_junc", "Xe giao lộ", y=55.0), "Roboflow supervision · góc khác, nhiều làn",
        "vehicles-2.mp4", asset="VEHICLES_2"),
    VideoScenario("Cao tốc 1080p (Pexels 2103099)", "vehicles",
        _veh("veh_px1", "Xe cao tốc 1080p", (1920, 1080)),
        "Pexels · highway traffic 1920×1080", "traffic_pexels_2103099.mp4",
        url=PEXELS_CDN + "2103099/2103099-hd_1920_1080_30fps.mp4"),
    VideoScenario("Phố thành phố (Pexels 1093662)", "vehicles",
        _veh("veh_px2", "Xe trong phố", (1920, 1080)),
        "Pexels · city traffic 1920×1080", "traffic_pexels_1093662.mp4",
        url=PEXELS_CDN + "1093662/1093662-hd_1920_1080_30fps.mp4"),
    VideoScenario("Đường đông (Pexels 5750647)", "vehicles",
        _veh("veh_px3", "Xe đường đông", (1920, 1080)),
        "Pexels · busy street 1920×1080", "traffic_pexels_5750647.mp4",
        url=PEXELS_CDN + "5750647/5750647-hd_1920_1080_24fps.mp4"),
    VideoScenario("Giao lộ bận (Pexels 2053100)", "vehicles",
        _veh("veh_px4", "Xe giao lộ bận", (640, 360)),
        "Pexels · busy intersection (SD)", "traffic_pexels_2053100.mp4",
        pexels_id="2053100", tips="Xe+buýt qua ngã tư; đổi prompt 'bus'/'truck'."),
]

# --------------------------------------------------------------------------- #
# 📦 DÂY CHUYỀN SẢN XUẤT — chai (YOLO) + thùng/hàng hoá (open-vocab LocateAnything)
# --------------------------------------------------------------------------- #
_CONVEYOR = [
    VideoScenario("Nhà máy chiết chai (supervision)", "conveyor",
        _conv_yolo("conv_milk", "Đếm chai"), "Roboflow supervision · dây chuyền chiết sữa, chai chạy ngang",
        "milk-bottling-plant.mp4", asset="MILK_BOTTLING_PLANT",
        tips="'bottle' là lớp COCO → YOLO đếm được."),
    VideoScenario("Kiện hàng trên chuyền (Pexels 4156510)", "conveyor",
        _conv_locate("conv_pkg", "Đếm kiện hàng", "cardboard box on a conveyor belt"),
        "Pexels · packages moving on conveyor belt", "conveyor_packages_4156510.mp4",
        pexels_id="4156510", tips="Thùng carton KHÔNG thuộc COCO → chạy --model locate."),
    VideoScenario("Băng chuyền nhà máy (Pexels 4473250)", "conveyor",
        _conv_locate("conv_fac", "Đếm vật trên chuyền", "item on the conveyor belt"),
        "Pexels · factory conveyor belt", "conveyor_factory_4473250.mp4",
        pexels_id="4473250", tips="Đổi prompt theo mặt hàng thật của bạn."),
    VideoScenario("Băng chuyền cận cảnh (Pexels 4473187)", "conveyor",
        _conv_locate("conv_black", "Đếm vật băng chuyền", "product on the conveyor belt"),
        "Pexels · black conveyor belt close-up", "conveyor_black_4473187.mp4",
        pexels_id="4473187"),
    VideoScenario("Dây chuyền hiện đại (Pexels 30715848)", "conveyor",
        _conv_locate("conv_line", "Đếm sản phẩm dây chuyền", "product on the production line"),
        "Pexels · modern factory production line", "conveyor_line_30715848.mp4",
        pexels_id="30715848"),
]

# --------------------------------------------------------------------------- #
# 🚶 NGƯỜI VÀO/RA — 5 cảnh (lối đi, quảng trường, ga tàu, siêu thị, vỉa hè)
# --------------------------------------------------------------------------- #
_PEOPLE = [
    VideoScenario("Lối đi bộ top-down (supervision)", "people",
        _ppl("ppl_walk", "Người đi bộ"), "Roboflow supervision · lối đi bộ, mật độ vừa",
        "people-walking.mp4", asset="PEOPLE_WALKING"),
    VideoScenario("Quảng trường đông (supervision)", "people",
        _ppl("ppl_square", "Người quảng trường"), "Roboflow supervision · quảng trường, nhiều hướng",
        "market-square.mp4", asset="MARKET_SQUARE", tips="Cảnh đông → recall thấp là bình thường."),
    VideoScenario("Ga tàu điện (supervision)", "people",
        _ppl("ppl_subway", "Người ga tàu", y=55.0), "Roboflow supervision · ga tàu, luồng 2 chiều",
        "subway.mp4", asset="SUBWAY"),
    VideoScenario("Siêu thị (supervision)", "people",
        _ppl("ppl_store", "Người siêu thị"), "Roboflow supervision · lối đi siêu thị",
        "grocery-store.mp4", asset="GROCERY_STORE"),
    VideoScenario("Vỉa hè đông người (Pexels 3121459)", "people",
        _ppl("ppl_side", "Người vỉa hè", (640, 360)),
        "Pexels · crowded city sidewalk (SD)", "people_sidewalk_3121459.mp4",
        pexels_id="3121459"),
]

CATALOG: List[VideoScenario] = [*_VEHICLES, *_CONVEYOR, *_PEOPLE]


def by_task(task: Optional[str] = None) -> List[VideoScenario]:
    """Lọc catalog theo bài ('vehicles' | 'conveyor' | 'people'); None = tất cả."""
    if not task:
        return list(CATALOG)
    return [v for v in CATALOG if v.task == task]


# --------------------------------------------------------------------------- #
# Tải video (3 nguồn) — bền bỉ: đã có file thì bỏ qua; thử nhiều URL rồi mới bỏ.
# --------------------------------------------------------------------------- #
def _fetch(url: str, dest: str, min_bytes: int = 200_000) -> bool:
    """Tải 1 URL về dest; True nếu ra file > min_bytes (lọc trang lỗi/HTML nhỏ)."""
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as r, open(dest, "wb") as f:
            if getattr(r, "status", 200) != 200:
                return False
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        return False
    if os.path.getsize(dest) < min_bytes:
        os.remove(dest)
        return False
    return True


def download_video(vs: VideoScenario, dest_dir: str = "videos") -> str:
    """Tải video của 1 :class:`VideoScenario`, trả đường dẫn .mp4. Raise nếu thất bại."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, vs.filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

    # (1) supervision asset — chính thức, có hash-check.
    if vs.asset:
        try:
            from supervision.assets import VideoAssets, download_assets

            asset = getattr(VideoAssets, vs.asset, None)
            if asset is not None:
                cwd = os.getcwd()
                os.chdir(dest_dir)
                try:
                    path = download_assets(asset)
                finally:
                    os.chdir(cwd)
                full = os.path.join(dest_dir, os.path.basename(path))
                if os.path.exists(full):
                    return full
        except Exception as e:  # noqa: BLE001
            print(f"  ℹ️  supervision.assets lỗi ({e}); thử nguồn khác…")

    # (2) URL .mp4 đầy đủ.
    if vs.url and _fetch(vs.url, dest):
        return dest

    # (3) Pexels chỉ có ID → dò hậu tố chất lượng đến khi tải được.
    if vs.pexels_id:
        for q in _PEXELS_QUALITIES:
            url = f"{PEXELS_CDN}{vs.pexels_id}/{vs.pexels_id}-{q}.mp4"
            print(f"  ⬇️  thử {q} …")
            if _fetch(url, dest):
                return dest

    raise RuntimeError(
        f"Không tải được '{vs.name}'. Kiểm tra mạng/nguồn, hoặc tải tay đặt vào {dest!r}."
    )
