"""Test catalog video test-đếm (thuần dữ liệu — không tải mạng, không GPU)."""

from recognition.scenarios import CountScenario
from recognition.video_catalog import (
    CATALOG,
    PEXELS_CDN,
    RB_CDN,
    VideoScenario,
    by_task,
)


def test_catalog_large_and_typed():
    # nhiều video (đã bỏ video sai nhãn) + rất nhiều query khó
    assert len(CATALOG) >= 13
    assert sum(len(v.queries) for v in CATALOG) >= 40   # tổng số ca test
    assert all(isinstance(v, VideoScenario) for v in CATALOG)


def test_many_angles_per_task():
    # mỗi bài phải có NHIỀU video (nhiều góc quay), không chỉ 1
    for task in ("vehicles", "conveyor", "people"):
        assert len(by_task(task)) >= 4, f"{task} có quá ít video"


def test_conveyor_and_vehicles_present():
    assert any(v.asset == "MILK_BOTTLING_PLANT" for v in CATALOG)   # dây chuyền
    assert any(v.task == "vehicles" for v in CATALOG)               # phương tiện
    # có video kiện hàng trên chuyền (đúng bài "đếm sản phẩm")
    assert any("4156510" == v.pexels_id for v in CATALOG)


def test_no_mislabeled_videos():
    # video nước chảy (1093662) đã bị BỎ; 3121459 (thực ra là xe) phải ở bài vehicles
    ids = {v.pexels_id for v in CATALOG}
    assert "1093662" not in ids                                     # video nước → đã bỏ
    v3121459 = [v for v in CATALOG if v.pexels_id == "3121459"]
    assert v3121459 and v3121459[0].task == "vehicles"             # xe, không phải người


def test_conveyor_has_hard_queries_for_products():
    # bài đếm sản phẩm phải có nhiều query khó (open-vocab) để test
    for v in by_task("conveyor"):
        assert len(v.queries) >= 2, f"{v.name} thiếu query để test"
    # tổng số query sản phẩm đủ phong phú
    total_q = sum(len(v.queries) for v in by_task("conveyor"))
    assert total_q >= 15


def test_conveyor_has_many_product_videos():
    assert len(by_task("conveyor")) >= 5      # nhiều video sản phẩm


def test_every_entry_has_exactly_one_source():
    # mỗi video phải có ĐÚNG 1 nguồn tải: asset | url | pexels_id
    for v in CATALOG:
        srcs = [bool(v.asset), bool(v.url), bool(v.pexels_id)]
        assert sum(srcs) == 1, f"{v.name} phải có đúng 1 nguồn, có {sum(srcs)}"


def test_urls_are_from_trusted_cdns():
    for v in CATALOG:
        if v.url:
            assert v.url.startswith(PEXELS_CDN) and v.url.endswith(".mp4")


def test_every_scenario_valid_and_line():
    for v in CATALOG:
        assert isinstance(v.scenario, CountScenario)
        v.scenario.validate()
        assert v.scenario.counting_type == "line"
        assert v.scenario.prompt.strip()


def test_scenario_keys_unique_and_identifier():
    keys = [v.scenario.key for v in CATALOG]
    assert len(keys) == len(set(keys))
    assert all(k.isidentifier() for k in keys)


def test_filenames_unique_and_mp4():
    files = [v.filename for v in CATALOG]
    assert len(files) == len(set(files))
    assert all(f.endswith(".mp4") for f in files)


def test_by_task_filter():
    assert by_task(None) == CATALOG
    assert by_task("khong-co") == []


def test_conveyor_line_vertical_vehicles_horizontal():
    conv = by_task("conveyor")[0].scenario
    assert conv.line_start_pct[0] == conv.line_end_pct[0]      # DỌC: cùng x
    veh = by_task("vehicles")[0].scenario
    assert veh.line_start_pct[1] == veh.line_end_pct[1]        # NGANG: cùng y


def test_supervision_and_pexels_both_used():
    assert any(v.asset for v in CATALOG)       # có nguồn supervision (hash-check)
    assert any(v.pexels_id or v.url for v in CATALOG)  # có nguồn Pexels
