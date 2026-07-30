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

# Thư mục code (chứa sample_videos/) — để giải đường dẫn video LOCAL đã commit repo.
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    local: Optional[str] = None         # video ĐÃ commit repo (vd 'sample_videos/x.mp4')
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
        _sc("veh_hw", "Xe cao tốc", "vehicle", "YOLO-NAS-S",
            (3.4, 90.3), (98.4, 82.9), "Chiều A", "Chiều B", "CENTER", (1280, 720),
            "Vạch ngang gần đáy (user vẽ)."),
        "Roboflow supervision · cao tốc quay dọc — TIN CẬY",
        "vehicles.mp4", asset="VEHICLES",
        queries=("car", "white car", "truck", "a vehicle changing lane"),
        tips="Kinh điển đếm xe; đổi prompt để test 'truck'/'bus'."),
    VideoScenario("Giao lộ nhiều làn (supervision)", "vehicles",
        _sc("veh_junc", "Xe giao lộ", "vehicle", "YOLO-NAS-S",
            (3.5, 93.6), (93.2, 88.8), "Chiều A", "Chiều B", "CENTER", (1280, 720),
            "Vạch ngang gần đáy (user vẽ)."),
        "Roboflow supervision · nhiều làn — TIN CẬY",
        "vehicles-2.mp4", asset="VEHICLES_2",
        queries=("car", "bus", "truck", "a car turning")),
    VideoScenario("Cao tốc 1080p (Pexels 2103099)", "vehicles",
        _sc("veh_px1", "Xe cao tốc 1080p", "vehicle", "YOLO-NAS-S",
            (0.0, 75.0), (99.4, 78.5), "Chiều A", "Chiều B", "CENTER", (1920, 1080),
            "Vạch ngang (user vẽ)."),
        "Pexels · highway traffic (tiêu đề Pexels: highway)", "traffic_pexels_2103099.mp4",
        pexels_id="2103099", queries=("car", "truck")),
    # ĐÃ BỎ video top-down 3121459 (COCO YOLO không đọc nổi góc từ trên — det/frame≈0.3).
    # Thay bằng bài đếm xe TRONG VÙNG trên video RÕ NÉT của supervision (VEHICLES_2, giao
    # lộ nhiều làn) — chính là kiểu video repo họ dùng, detect rất tốt (283 xe/lượt chạy).
    VideoScenario("Giao lộ — đếm xe trong VÙNG (supervision)", "vehicles",
        _zone_multi("veh_junc_zone", "Đếm xe trong vùng giao lộ",
                    (((15.0, 35.0), (85.0, 35.0), (85.0, 92.0), (15.0, 92.0)),),
                    res=(1280, 720), prompt="vehicle", anchor="CENTER"),
        "Roboflow supervision · giao lộ (vehicles-2) — VÙNG, video RÕ như repo",
        "vehicles-2.mp4", asset="VEHICLES_2", queries=("car", "truck", "bus"),
        tips="Vùng mặc định phủ mặt đường; vẽ lại cho khớp bằng draw_gallery('giao lộ')."),
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
    # === Video THẬT của user (đã commit repo sample_videos/) — CÓ sản phẩm chạy rõ ===
    # (Đã BỎ 4 video Pexels lỗi: 4156510/30715848/4473250/4473187 — không có sản phẩm
    #  chạy trên chuyền.) Vật đi XUỐNG về phía camera → vạch NGANG; thùng/cà chua KHÔNG
    #  thuộc COCO → open-vocab LocateAnything-3B.
    VideoScenario("Kiện hàng — băng chuyền con lăn (user)", "conveyor",
        _sc("conv_rollers", "Đếm kiện hàng con lăn", "object", "LocateAnything-3B",
            (0.0, 65.0), (100.0, 65.0), "Qua vạch", "Ngược", "CENTER", (960, 540),
            "Thùng đi xuống → vạch NGANG y=65."),
        "User upload · kho hàng, thùng carton trên băng chuyền con lăn",
        "packages_rollers.mp4", local="sample_videos/packages_rollers.mp4",
        queries=("object", "carton box", "package", "box"),
        tips="Prompt 'object' (như notebook Kaggle) cho ra box TỪNG VẬT; 'carton box' đôi khi ra cả khung."),
    VideoScenario("Kiện hàng — băng chuyền có nhãn (user)", "conveyor",
        _sc("conv_belt", "Đếm kiện hàng belt", "object", "LocateAnything-3B",
            (0.0, 60.0), (100.0, 60.0), "Qua vạch", "Ngược", "CENTER", (960, 540),
            "Thùng trôi xuống belt → vạch NGANG y=60."),
        "User upload · công nhân phân loại, thùng carton có nhãn/mã vạch",
        "packages_belt.mp4", local="sample_videos/packages_belt.mp4",
        queries=("object", "package", "carton box")),
    VideoScenario("Cà chua — dây chuyền phân loại (user)", "conveyor",
        _sc("conv_tomato", "Đếm cà chua", "tomato", "LocateAnything-3B",
            (0.0, 72.0), (100.0, 72.0), "Qua vạch", "Ngược", "CENTER", (960, 540),
            "Cà chua trôi xuống làn → vạch NGANG y=72."),
        "User upload · nhà máy phân loại cà chua trên dây chuyền inox",
        "tomatoes_sorting.mp4", local="sample_videos/tomatoes_sorting.mp4",
        queries=("tomato", "object", "fruit")),
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
    # Video LOCAL (đã commit repo) → trả THẲNG đường dẫn, không tải mạng.
    if vs.local:
        p = vs.local if os.path.isabs(vs.local) else os.path.join(_CODE_DIR, vs.local)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
        raise RuntimeError(
            f"Không thấy video local: {p} — đã commit '{vs.local}' vào repo chưa?")

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
