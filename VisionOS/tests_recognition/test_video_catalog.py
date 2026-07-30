"""Test catalog video test-đếm (thuần dữ liệu — không tải mạng, không GPU)."""

from recognition.scenarios import CountScenario
from recognition.video_catalog import (
    CATALOG,
    PEXELS_CDN,
    QUERY_SUITES,
    RB_CDN,
    VideoScenario,
    by_task,
    suite_for,
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


def test_every_scenario_valid():
    for v in CATALOG:
        assert isinstance(v.scenario, CountScenario)
        v.scenario.validate()
        assert v.scenario.counting_type in ("line", "zone")
        assert v.scenario.prompt.strip()


def test_scenario_keys_unique_and_identifier():
    keys = [v.scenario.key for v in CATALOG]
    assert len(keys) == len(set(keys))
    assert all(k.isidentifier() for k in keys)


def test_filenames_mp4():
    # filename có thể TRÙNG (vd market-square.mp4 dùng cho cả bài vạch lẫn vùng —
    # tải 1 lần dùng lại); chỉ yêu cầu đuôi .mp4. Định danh duy nhất là scenario.key.
    assert all(v.filename.endswith(".mp4") for v in CATALOG)


def test_people_has_both_line_and_zone():
    # user yêu cầu: tách bài đếm NGƯỜI thành cắt VẠCH (vào/ra) + đếm VÙNG (occupancy)
    ppl = by_task("people")
    types = {v.scenario.counting_type for v in ppl}
    assert "line" in types and "zone" in types
    # market-square xuất hiện ở CẢ hai kiểu
    ms = [v for v in ppl if v.filename == "market-square.mp4"]
    assert {v.scenario.counting_type for v in ms} == {"line", "zone"}


def test_zone_scenarios_have_polygon():
    for v in CATALOG:
        if v.scenario.counting_type == "zone":
            assert len(v.scenario.zone_points_pct) >= 3   # đủ đỉnh để tạo vùng


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


# --------------------------------------------------------------------------- #
# QUERY SUITES — nhiều trường hợp test phân nhóm (như bảng Excel)
# --------------------------------------------------------------------------- #
def test_query_suites_cover_three_tasks_and_are_rich():
    assert {"people", "vehicles", "conveyor"} <= set(QUERY_SUITES)
    for task in ("people", "vehicles", "conveyor"):
        pairs = suite_for(task)
        assert len(pairs) >= 20, f"suite {task} quá ít trường hợp"       # nhiều ca test
        assert len({g for g, _ in pairs}) >= 5, f"suite {task} thiếu nhóm"  # nhiều nhóm


def test_suite_has_vietnamese_and_hard_cases():
    # bảng Excel test cả prompt tiếng Việt + trường hợp khó/phủ định
    ppl = QUERY_SUITES["people"]
    assert "tiếng Việt" in ppl and any("người" in q for q in ppl["tiếng Việt"])
    assert any("khó" in g or "phủ định" in g for g in ppl)


def test_suite_for_returns_group_query_pairs():
    pairs = suite_for("conveyor")
    assert all(isinstance(g, str) and isinstance(q, str) and q for g, q in pairs)
    assert suite_for("khong-co-task") == []


def test_suite_lite_is_small_and_representative():
    # LITE (Colab/session ngắn): mỗi nhóm chỉ 1-2 query, ít hơn HẲN bản đầy đủ.
    for task in ("people", "vehicles", "conveyor"):
        full = suite_for(task)
        lite2 = suite_for(task, lite=True, per_group=2)
        lite1 = suite_for(task, lite=True, per_group=1)
        assert len(lite2) < len(full)
        assert len(lite1) <= len(lite2)
        # mỗi nhóm ≤ per_group
        from collections import Counter
        assert all(c <= 2 for c in Counter(g for g, _ in lite2).values())
        assert all(c == 1 for c in Counter(g for g, _ in lite1).values())
    # màu = đỏ + trắng (đại diện) đứng đầu
    peo = suite_for("people", lite=True, per_group=2)
    colors = [q for g, q in peo if g == "màu/trang phục"]
    assert any("red" in q for q in colors) and any("white" in q for q in colors)


def test_total_test_cases_is_large():
    # tổng số ca test (suite + query per-video) đủ phong phú như yêu cầu
    total = sum(len(suite_for(t)) for t in QUERY_SUITES) + sum(len(v.queries) for v in CATALOG)
    assert total >= 100
