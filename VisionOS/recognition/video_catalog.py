"""Catalog VIDEO CÔNG KHAI (tải trực tiếp) để test đếm trên NHIỀU kịch bản/GÓC QUAY.

Nguyên tắc chọn nguồn (sau khi phát hiện nhãn của repo bên thứ 3 hay SAI):
  * ``asset``     — supervision video-examples (Roboflow) đặt tên THEO NỘI DUNG →
    tin cậy cao (vehicles.mp4 chắc chắn là xe, milk-bottling-plant là dây chuyền…).
  * ``pexels_id`` — CHỈ dùng ID mà tiêu đề trang Pexels tự mô tả đúng nội dung
    (vd "packages-moving-on-a-conveyor-belt-4156510"); downloader tự dò hậu tố.
  * ``url``       — link .mp4 đầy đủ đã xác minh.

Mỗi video kèm ``queries``: danh sách prompt từ DỄ → KHÓ để test khả năng mô tả
ngôn ngữ tự nhiên của LocateAnything (open-vocab). Chạy nhiều query trên 1 video
bằng ``run_scenarios.py --all-queries`` hoặc ``--queries "a,b,c"``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .base import MonitoringMode
from .scenarios import CountScenario

__all__ = [
    "VideoScenario", "CATALOG", "by_task", "download_video",
    "QUERY_SUITES", "suite_for", "RB_CDN", "PEXELS_CDN",
]

RB_CDN = "https://media.roboflow.com/supervision/video-examples/"
PEXELS_CDN = "https://videos.pexels.com/video-files/"
# Endpoint tải CHÍNH THỨC của Pexels: trả file res cao nhất, KHÔNG cần đoán hậu tố
# → cách tin cậy nhất cho MỌI video ID (dò hậu tố bên dưới chỉ là dự phòng).
PEXELS_DL = "https://www.pexels.com/download/video/{id}/"

_PEXELS_QUALITIES = (
    "hd_1920_1080_30fps", "hd_1920_1080_25fps", "hd_1920_1080_24fps", "hd_1920_1080_60fps",
    "hd_1280_720_30fps", "hd_1280_720_25fps", "hd_1280_720_24fps", "hd_1280_720_60fps",
    "sd_960_540_30fps", "sd_960_540_25fps",
    "sd_640_360_30fps", "sd_640_360_25fps", "sd_640_360_24fps",
    "uhd_2560_1440_30fps", "uhd_2560_1440_25fps", "uhd_2560_1440_24fps",
    "uhd_3840_2160_30fps", "uhd_3840_2160_25fps", "uhd_3840_2160_24fps",
    # dạng dọc (một số clip công nghiệp quay dọc)
    "hd_1080_1920_30fps", "uhd_1440_2560_30fps",
)


@dataclass(frozen=True)
class VideoScenario:
    name: str
    task: str                          # "vehicles" | "conveyor" | "people"
    scenario: CountScenario
    source: str
    filename: str
    asset: Optional[str] = None
    url: Optional[str] = None
    pexels_id: Optional[str] = None
    queries: Tuple[str, ...] = ()      # prompt dễ→khó (rỗng = dùng scenario.prompt)
    tips: str = ""


def _sc(key, title, prompt, model, ls, le, in_lbl, out_lbl, anchor, res, note):
    return CountScenario(
        key=key, title=title, usecase_id=f"uc-{key}",
        mode=(MonitoringMode.SMART if not model.startswith("YOLO") else MonitoringMode.STANDARD),
        model=model, prompt=prompt, counting_type="line",
        line_start_pct=ls, line_end_pct=le, in_label=in_lbl, out_label=out_lbl,
        zone_anchor=anchor, resolution=res, notes=note,
    )


def _veh(key, title, prompt="car", model="YOLO-NAS-S", res=(1280, 720), y=50.0):
    return _sc(key, title, prompt, model, (0.0, y), (100.0, y),
               "Chiều A", "Chiều B", "CENTER", res, "Xe chạy dọc → vạch NGANG.")


def _ppl(key, title, res=(1280, 720), y=50.0):
    return _sc(key, title, "person", "YOLO-NAS-S", (0.0, y), (100.0, y),
               "Vào", "Ra", "BOTTOM_CENTER", res, "Người → vạch ngang, neo chân.")


def _conv(key, title, prompt, model, res=(1280, 720)):
    return _sc(key, title, prompt, model, (50.0, 0.0), (50.0, 100.0),
               "Qua vạch", "Ngược", "CENTER", res, "Hàng chạy ngang → vạch DỌC.")


def _ppl_zone(key, title, points, res=(1280, 720)):
    """Bài đếm NGƯỜI TRONG VÙNG (occupancy) — hợp cảnh người đi lại lộn xộn."""
    return CountScenario(
        key=key, title=title, usecase_id=f"uc-{key}",
        mode=MonitoringMode.STANDARD, model="YOLO-NAS-S", prompt="person",
        counting_type="zone", zone_points_pct=points, zone_anchor="BOTTOM_CENTER",
        resolution=res, expect_min=0,
        notes="Đếm số người ĐANG ở trong vùng (không cần luồng vào/ra rõ).",
    )


def _zone_multi(key, title, zones, res=(1280, 720), prompt="person", anchor="BOTTOM_CENTER"):
    """Bài đếm TRONG NHIỀU VÙNG (gộp 1 bài) — đếm vật đang ở BẤT KỲ vùng nào."""
    return CountScenario(
        key=key, title=title, usecase_id=f"uc-{key}",
        mode=MonitoringMode.STANDARD, model="YOLO-NAS-S", prompt=prompt,
        counting_type="zone", zones_pct=zones, zone_anchor=anchor,
        resolution=res, expect_min=0,
        notes="Đếm số vật ĐANG trong bất kỳ vùng nào (gộp nhiều vùng, 1 số tổng).",
    )


# --------------------------------------------------------------------------- #
# 🚗 PHƯƠNG TIỆN — chỉ nguồn tin cậy (supervision + Pexels tiêu đề đúng/đã xác minh)
# --------------------------------------------------------------------------- #
_VEHICLES = [
    # Vạch xe do user vẽ tay trên frame thật (ngang gần đáy, hơi nghiêng theo đường).
    VideoScenario("Cao tốc top-down (supervision)", "vehicles",
        _sc("veh_hw", "Xe cao tốc", "car", "YOLO-NAS-S",
            (3.4, 90.3), (98.4, 82.9), "Chiều A", "Chiều B", "CENTER", (1280, 720),
            "Vạch ngang gần đáy (user vẽ)."),
        "Roboflow supervision · cao tốc quay dọc — TIN CẬY",
        "vehicles.mp4", asset="VEHICLES",
        queries=("car", "white car", "truck", "a vehicle changing lane"),
        tips="Kinh điển đếm xe; đổi prompt để test 'truck'/'bus'."),
    VideoScenario("Giao lộ nhiều làn (supervision)", "vehicles",
        _sc("veh_junc", "Xe giao lộ", "car", "YOLO-NAS-S",
            (3.5, 93.6), (93.2, 88.8), "Chiều A", "Chiều B", "CENTER", (1280, 720),
            "Vạch ngang gần đáy (user vẽ)."),
        "Roboflow supervision · nhiều làn — TIN CẬY",
        "vehicles-2.mp4", asset="VEHICLES_2",
        queries=("car", "bus", "truck", "a car turning")),
    VideoScenario("Cao tốc 1080p (Pexels 2103099)", "vehicles",
        _sc("veh_px1", "Xe cao tốc 1080p", "car", "YOLO-NAS-S",
            (0.0, 75.0), (99.4, 78.5), "Chiều A", "Chiều B", "CENTER", (1920, 1080),
            "Vạch ngang (user vẽ)."),
        "Pexels · highway traffic (tiêu đề Pexels: highway)", "traffic_pexels_2103099.mp4",
        pexels_id="2103099", queries=("car", "truck")),
    # veh_px2: 5 VÙNG user vẽ → GỘP 1 BÀI (đếm xe trong BẤT KỲ vùng nào, ra 1 số tổng).
    VideoScenario("Giao thông (Pexels 3121459) — đếm xe trong VÙNG", "vehicles",
        _zone_multi("veh_px2_zone", "Đếm xe trong các vùng", (
            ((21.6, 41.1), (37.7, 40.6), (36.7, 62.5), (20.5, 62.2)),
            ((41.4, 0.6), (60.8, 1.4), (60.6, 38.9), (40.5, 37.8)),
            ((63.3, 38.1), (85.3, 39.7), (86.1, 60.8), (63.9, 60.3)),
            ((41.6, 68.6), (58.4, 69.2), (58.3, 96.9), (42.3, 98.1)),
            ((41.3, 43.1), (59.5, 44.2), (59.2, 60.6), (41.9, 63.1)),
        ), res=(640, 360), prompt="car", anchor="CENTER"),
        "Pexels 3121459 · giao thông phố — 5 VÙNG đếm CHUNG (user vẽ)",
        "traffic_pexels_3121459.mp4", pexels_id="3121459", queries=("car",),
        tips="Đếm xe trong 5 vùng — ra 1 số tổng (không tách từng vùng)."),
]

# --------------------------------------------------------------------------- #
# 📦 DÂY CHUYỀN SẢN XUẤT — nhiều video + NHIỀU QUERY KHÓ (open-vocab)
#    Nguồn Pexels lấy từ TIÊU ĐỀ TRANG (Pexels tự mô tả nội dung) → tin cậy hơn.
# --------------------------------------------------------------------------- #
_CONVEYOR = [
    VideoScenario("Nhà máy chiết chai (supervision)", "conveyor",
        _sc("conv_milk", "Đếm chai", "bottle", "YOLO-NAS-S",
            (14.5, 0.7), (15.3, 99.6), "Qua vạch", "Ngược", "CENTER", (1280, 720),
            "Vạch DỌC lệch trái theo chỗ chai chạy (user vẽ)."),
        "Roboflow supervision · dây chuyền chiết sữa — TIN CẬY", "milk-bottling-plant.mp4",
        asset="MILK_BOTTLING_PLANT",
        queries=("bottle", "plastic bottle", "milk bottle", "bottle cap",
                 "a bottle without a cap", "a fallen bottle"),
        tips="'bottle' chạy YOLO nhanh; query khó cần --model locate."),
    # ƯU TIÊN video NHIỀU SẢN PHẨM chạy trên chuyền (theo yêu cầu):
    VideoScenario("Kiện hàng chạy trên chuyền (Pexels 4156510)", "conveyor",
        _sc("conv_pkg", "Đếm kiện hàng", "cardboard box on a conveyor belt", "LocateAnything-3B",
            (39.1, 41.4), (41.0, 47.5), "Qua vạch", "Ngược", "CENTER", (1280, 720),
            "Vạch theo chỗ thùng chạy (user vẽ)."),
        "Pexels · 'packages moving on a conveyor belt' — NHIỀU thùng chạy liên tục",
        "conveyor_packages_4156510.mp4", pexels_id="4156510",
        queries=("cardboard box", "package", "a sealed box", "a brown box",
                 "a damaged package", "the largest box"),
        tips="Bài 'đếm sản phẩm' điển hình — thùng carton KHÔNG thuộc COCO."),
    # (đã bỏ conv_action / Pexels 35069357 — user xác nhận KHÔNG phải băng chuyền.)
    VideoScenario("Dây chuyền nhà máy rộng (Pexels 30715848)", "conveyor",
        _sc("conv_line", "Đếm sản phẩm dây chuyền", "product on the production line", "LocateAnything-3B",
            (80.2, 66.8), (94.4, 63.2), "Qua vạch", "Ngược", "CENTER", (1280, 720),
            "Vạch theo chỗ sản phẩm chạy (user vẽ)."),
        "Pexels · 'wide view of modern factory production line' — góc rộng nhiều sản phẩm",
        "conveyor_line_30715848.mp4", pexels_id="30715848",
        queries=("product on the production line", "finished product",
                 "an item being assembled", "a bottle", "a box")),
    VideoScenario("Băng chuyền nhà máy (Pexels 4473250)", "conveyor",
        _sc("conv_fac", "Đếm vật trên chuyền", "item on the conveyor belt", "LocateAnything-3B",
            (14.8, 64.0), (84.2, 76.8), "Qua vạch", "Ngược", "CENTER", (1280, 720),
            "Vạch chéo theo chuyền (user vẽ)."),
        "Pexels · 'factory conveyor belt'", "conveyor_factory_4473250.mp4",
        pexels_id="4473250",
        queries=("item on the conveyor belt", "product", "a metal part", "a small component")),
    VideoScenario("Băng chuyền cận cảnh (Pexels 4473187)", "conveyor",
        _sc("conv_close", "Đếm vật băng chuyền", "product on the conveyor belt", "LocateAnything-3B",
            (37.9, 47.5), (73.1, 34.9), "Qua vạch", "Ngược", "CENTER", (1280, 720),
            "Vạch chéo cận cảnh (user vẽ)."),
        "Pexels · 'black conveyor belt' cận cảnh", "conveyor_black_4473187.mp4",
        pexels_id="4473187",
        queries=("object on the belt", "product", "a dark colored item")),
]

# --------------------------------------------------------------------------- #
# 🚶 NGƯỜI — chỉ supervision (đặt tên theo nội dung, tin cậy)
# --------------------------------------------------------------------------- #
_PEOPLE = [
    VideoScenario("Người đi bộ — vào/ra (supervision)", "people",
        _sc("ppl_walk", "Đếm người vào/ra", "person", "YOLO-NAS-S",
            (2.3, 97.8), (99.3, 94.3), "Vào", "Ra", "BOTTOM_CENTER", (1280, 720),
            "Vạch ngang gần đáy theo lối đi (user vẽ)."),
        "Roboflow supervision · lối đi bộ — TIN CẬY",
        "people-walking.mp4", asset="PEOPLE_WALKING",
        queries=("person", "a person wearing a backpack", "a person in white",
                 "a child", "a person carrying a bag"),
        tips="Vạch ngang gần đáy (user vẽ). Đếm lệch thì vẽ lại gửi tôi."),
    # subway: vạch CHÉO theo luồng người qua sảnh (toạ độ user tự vẽ trên frame thật).
    VideoScenario("Ga tàu — vào/ra (supervision)", "people",
        _sc("ppl_subway", "Đếm người vào/ra ga", "person", "YOLO-NAS-S",
            (59.6, 16.8), (31.6, 90.0), "Vào", "Ra", "BOTTOM_CENTER", (1280, 720),
            "Vạch chéo theo luồng người qua sảnh (user vẽ tay)."),
        "Roboflow supervision · ga tàu, luồng 2 chiều — TIN CẬY",
        "subway.mp4", asset="SUBWAY",
        queries=("person", "a person with luggage", "a person wearing a hat"),
        tips="Vạch CHÉO (user vẽ). Xem lại bằng --preview; đếm lệch thì vẽ lại gửi tôi."),
    VideoScenario("Ga tàu — đếm trong VÙNG (supervision)", "people",
        _ppl_zone("ppl_subway_zone", "Đếm người trong vùng ga",
                  ((59.6, 16.8), (75.2, 17.8), (85.7, 92.4), (49.5, 87.5))),
        "Roboflow supervision · ga tàu — bài ĐẾM VÙNG (occupancy, user vẽ)",
        "subway.mp4", asset="SUBWAY",
        queries=("person", "a person standing"),
        tips="Đếm số người ĐANG trong vùng bạn khoanh (không cần luồng vào/ra)."),
    VideoScenario("Siêu thị — vào/ra (supervision)", "people",
        _ppl("ppl_store", "Đếm người vào/ra siêu thị"),
        "Roboflow supervision · lối đi siêu thị — TIN CẬY", "grocery-store.mp4",
        asset="GROCERY_STORE",
        queries=("person", "a shopper pushing a cart", "a person holding a basket")),
    # Siêu thị: 3 VÙNG user vẽ → GỘP 1 BÀI (đếm người trong BẤT KỲ vùng nào, 1 số tổng).
    VideoScenario("Siêu thị — đếm người trong VÙNG", "people",
        _zone_multi("ppl_store_zone", "Đếm người trong các vùng siêu thị", (
            ((0.7, 31.8), (27.0, 34.0), (27.6, 96.1), (0.7, 99.2)),
            ((62.0, 40.8), (97.7, 9.4), (96.6, 99.6), (62.3, 55.8)),
            ((31.0, 14.9), (49.5, 41.3), (49.8, 60.7), (34.1, 97.2)),
        ), prompt="person", anchor="BOTTOM_CENTER"),
        "Roboflow supervision · siêu thị — 3 VÙNG đếm CHUNG (user vẽ)", "grocery-store.mp4",
        asset="GROCERY_STORE", queries=("person",),
        tips="Đếm người trong 3 vùng — ra 1 số tổng (không tách)."),
    # market-square: NGƯỜI ĐI LẠI nhiều hướng → tách 2 bài như user yêu cầu.
    VideoScenario("Quảng trường — vào/ra (supervision)", "people",
        _sc("ppl_square_line", "Đếm người vào/ra quảng trường", "person", "YOLO-NAS-S",
            (4.2, 94.4), (94.8, 92.1), "Vào", "Ra", "BOTTOM_CENTER", (1280, 720),
            "Vạch ngang gần đáy quảng trường (user vẽ)."),
        "Roboflow supervision · quảng trường — bài CẮT VẠCH (vào/ra)",
        "market-square.mp4", asset="MARKET_SQUARE",
        queries=("person", "a group of people"),
        tips="Vạch ngang gần đáy (user vẽ); xem thêm bài ZONE bên dưới."),
    VideoScenario("Quảng trường — đếm trong VÙNG (supervision)", "people",
        _ppl_zone("ppl_square_zone", "Đếm người trong vùng",
                  ((28.0, 32.0), (72.0, 32.0), (72.0, 85.0), (28.0, 85.0))),
        "Roboflow supervision · quảng trường — bài ĐẾM VÙNG (occupancy)",
        "market-square.mp4", asset="MARKET_SQUARE",
        queries=("person", "a person standing"),
        tips="Đếm số người đang trong vùng trung tâm; hợp cảnh đi lại lộn xộn."),
]

CATALOG: List[VideoScenario] = [*_VEHICLES, *_CONVEYOR, *_PEOPLE]


# --------------------------------------------------------------------------- #
# BỘ QUERY SUITE — nhiều TRƯỜNG HỢP test phân theo nhóm (như bảng Excel của user):
# cơ bản, màu/trang phục, phụ kiện, hành động/quan hệ, đếm/nhóm, khó/phủ định,
# và TIẾNG VIỆT (để kiểm tra khả năng đa ngữ — thường LA hiểu kém, kết quả cho
# thấy giới hạn). Chạy: run_scenarios.py --task <t> --suite [--only <video>].
# --------------------------------------------------------------------------- #
QUERY_SUITES = {
    "people": {
        "cơ bản": ["person", "a man", "a woman", "a child", "an elderly person"],
        "màu/trang phục": ["a person in a red shirt", "a person in a white shirt",
                           "a person wearing black", "a person in a jacket",
                           "a person wearing shorts"],
        "phụ kiện": ["a person wearing a backpack", "a person carrying a handbag",
                    "a person wearing a hat", "a person wearing a mask",
                    "a person wearing sunglasses"],
        "hành động/quan hệ": ["a person walking a dog", "a person riding a bicycle",
                             "a person talking on the phone", "a person pushing a stroller",
                             "a person running"],
        "đếm/nhóm": ["two people walking together", "a group of people",
                    "a person standing alone"],
        "khó/phủ định": ["a person without a backpack", "a person not wearing a hat",
                        "the tallest person", "a person facing the camera"],
        "tiếng Việt": ["người đi bộ", "người đeo ba lô", "người mặc áo trắng", "người đội mũ"],
    },
    "vehicles": {
        "loại xe": ["car", "truck", "bus", "motorcycle", "van", "bicycle"],
        "màu": ["a red car", "a white car", "a black car", "a silver car"],
        "đặc điểm": ["a large truck", "a small car", "a delivery truck", "a taxi"],
        "hành động": ["a car turning", "a vehicle changing lanes", "a moving car",
                     "a parked car"],
        "khó": ["a car with its headlights on", "the vehicle closest to the camera",
               "a vehicle carrying cargo"],
        "tiếng Việt": ["xe ô tô", "xe tải", "xe máy", "xe buýt"],
    },
    "conveyor": {
        "chai": ["bottle", "plastic bottle", "glass bottle", "milk bottle", "water bottle"],
        "trạng thái chai": ["a bottle with a cap", "a bottle without a cap",
                           "an empty bottle", "a full bottle", "a fallen bottle"],
        "hộp/kiện": ["cardboard box", "a sealed box", "an open box", "a damaged box",
                    "a brown box"],
        "sản phẩm": ["product on the conveyor", "a packaged product", "a defective product",
                    "an item being assembled"],
        "đặc điểm/đếm": ["a red product", "the largest item", "the smallest item",
                        "a shiny object"],
        "tiếng Việt": ["chai nước", "thùng carton", "sản phẩm lỗi", "chai nhựa"],
    },
}


def suite_for(task: str, lite: bool = False, per_group: int = 2):
    """Trả list (nhóm, query) của bộ suite cho một bài toán ('people'|'vehicles'|'conveyor').

    ``lite=True`` chỉ lấy ``per_group`` query ĐẦU mỗi nhóm (đại diện — vd màu = đỏ/trắng)
    để chạy NHANH trên Colab/Kaggle (session ngắn). Mặc định (lite=False) trả BỘ ĐẦY ĐỦ.
    """
    out = []
    for group, qs in QUERY_SUITES.get(task, {}).items():
        picked = qs[: max(1, per_group)] if lite else qs
        for q in picked:
            out.append((group, q))
    return out


def by_task(task: Optional[str] = None) -> List[VideoScenario]:
    if not task:
        return list(CATALOG)
    return [v for v in CATALOG if v.task == task]


def _fetch(url: str, dest: str, min_bytes: int = 200_000) -> bool:
    """Tải url → dest. True nếu ra file video hợp lệ (>min_bytes, không phải HTML)."""
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as r:
            if getattr(r, "status", 200) != 200:
                return False
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "text/html" in ctype or "application/json" in ctype:
                return False  # trang lỗi/redirect HTML, KHÔNG phải video
            with open(dest, "wb") as f:
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
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, vs.filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

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

    if vs.url and _fetch(vs.url, dest):
        return dest

    if vs.pexels_id:
        # (a) endpoint tải CHÍNH THỨC — không cần đoán hậu tố, hợp mọi độ phân giải.
        if _fetch(PEXELS_DL.format(id=vs.pexels_id), dest):
            print(f"  ✅ tải qua endpoint chính thức (id={vs.pexels_id})")
            return dest
        # (b) dò hậu tố trực tiếp (dự phòng).
        for q in _PEXELS_QUALITIES:
            if _fetch(f"{PEXELS_CDN}{vs.pexels_id}/{vs.pexels_id}-{q}.mp4", dest):
                print(f"  ✅ tải qua hậu tố {q} (id={vs.pexels_id})")
                return dest

    raise RuntimeError(
        f"Không tải được '{vs.name}' (id={vs.pexels_id or vs.url}). "
        f"Kiểm tra Internet, hoặc tải TAY đặt vào {dest!r}."
    )
