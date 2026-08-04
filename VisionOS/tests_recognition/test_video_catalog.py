"""Test catalog video test-đếm (thuần dữ liệu — không tải mạng, không GPU).

Hệ thống hiện CHỈ đếm NGƯỜI + PHƯƠNG TIỆN (YOLO + supervision). Phần đếm SẢN PHẨM
(dây chuyền/kiện hàng/cà chua, LocateAnything) đã BỎ khỏi catalog.
"""

from recognition.scenarios import CountScenario
from recognition.video_catalog import (
    CATALOG,
    PEXELS_CDN,
    QUERY_SUITES,
    VideoScenario,
    by_task,
    suite_for,
)

TASKS = ("vehicles", "people")


def test_catalog_large_and_typed():
    assert len(CATALOG) >= 10
    assert sum(len(v.queries) for v in CATALOG) >= 30    # tổng số ca test
    assert all(isinstance(v, VideoScenario) for v in CATALOG)


def test_only_people_and_vehicles():
    # CHỈ còn 2 bài toán; KHÔNG còn conveyor/sản phẩm.
    assert {v.task for v in CATALOG} == set(TASKS)
    assert by_task("conveyor") == []


def test_many_angles_per_task():
    for task in TASKS:
        assert len(by_task(task)) >= 3, f"{task} có quá ít video"


def test_vehicles_and_people_present():
    assert any(v.task == "vehicles" for v in CATALOG)
    assert any(v.task == "people" for v in CATALOG)


def test_no_mislabeled_videos():
    ids = {v.pexels_id for v in CATALOG}
    assert "1093662" not in ids                                     # video nước → đã bỏ
    assert "3121459" not in ids                                     # top-down khó → đã bỏ


def test_every_entry_has_exactly_one_source():
    for v in CATALOG:
        srcs = [bool(v.asset), bool(v.url), bool(v.pexels_id), bool(v.local)]
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
    assert all(v.filename.endswith(".mp4") for v in CATALOG)


def test_people_has_both_line_and_zone():
    ppl = by_task("people")
    types = {v.scenario.counting_type for v in ppl}
    assert "line" in types and "zone" in types
    ms = [v for v in ppl if v.filename == "market-square.mp4"]
    assert {v.scenario.counting_type for v in ms} == {"line", "zone"}


def test_zone_scenarios_have_polygon():
    for v in CATALOG:
        if v.scenario.counting_type == "zone":
            zones = v.scenario.build_zones()
            assert zones, f"{v.scenario.key} không có vùng nào"
            assert all(len(z.points_pct) >= 3 for z in zones), v.scenario.key


def test_multi_zone_scenarios_merged_not_split():
    byk = {v.scenario.key: v.scenario for v in CATALOG}
    assert "ppl_store_zone" in byk and len(byk["ppl_store_zone"].build_zones()) == 3
    assert "veh_junc_zone" in byk
    keys = set(byk)
    assert not any(k.endswith(("_z1", "_z2", "_z3", "_z4", "_z5")) for k in keys)


def test_no_conveyor_or_product_keys():
    keys = {v.scenario.key for v in CATALOG}
    for bad in ("conv_action", "conv_milk", "conv_rollers", "conv_belt", "conv_tomato",
                "conv_tomato_zone"):
        assert bad not in keys, f"scenario sản phẩm {bad} chưa bị xoá"


def test_by_task_filter():
    assert by_task(None) == CATALOG
    assert by_task("khong-co") == []


def test_vehicles_line_is_horizontal():
    veh = by_task("vehicles")[0].scenario
    vdx = abs(veh.line_start_pct[0] - veh.line_end_pct[0])
    vdy = abs(veh.line_start_pct[1] - veh.line_end_pct[1])
    assert vdy < vdx, "vạch xe phải NGANG-ish (xe chạy dọc)"


def test_supervision_and_pexels_both_used():
    assert any(v.asset for v in CATALOG)               # nguồn supervision
    assert any(v.pexels_id or v.url for v in CATALOG)  # nguồn Pexels


# --------------------------------------------------------------------------- #
# QUERY SUITES — nhiều trường hợp test phân nhóm
# --------------------------------------------------------------------------- #
def test_query_suites_cover_both_tasks_and_are_rich():
    assert set(TASKS) == set(QUERY_SUITES)             # chỉ people + vehicles
    for task in TASKS:
        pairs = suite_for(task)
        assert len(pairs) >= 20, f"suite {task} quá ít trường hợp"
        assert len({g for g, _ in pairs}) >= 5, f"suite {task} thiếu nhóm"


def test_suite_has_vietnamese_and_hard_cases():
    ppl = QUERY_SUITES["people"]
    assert "tiếng Việt" in ppl and any("người" in q for q in ppl["tiếng Việt"])
    assert any("khó" in g or "phủ định" in g for g in ppl)


def test_suite_for_returns_group_query_pairs():
    pairs = suite_for("people")
    assert all(isinstance(g, str) and isinstance(q, str) and q for g, q in pairs)
    assert suite_for("khong-co-task") == []


def test_suite_lite_is_small_and_representative():
    from collections import Counter
    for task in TASKS:
        full = suite_for(task)
        lite2 = suite_for(task, lite=True, per_group=2)
        lite1 = suite_for(task, lite=True, per_group=1)
        assert len(lite2) < len(full)
        assert len(lite1) <= len(lite2)
        assert all(c <= 2 for c in Counter(g for g, _ in lite2).values())
        assert all(c == 1 for c in Counter(g for g, _ in lite1).values())
    peo = suite_for("people", lite=True, per_group=2)
    colors = [q for g, q in peo if g == "màu/trang phục"]
    assert any("red" in q for q in colors) and any("white" in q for q in colors)


def test_total_test_cases_is_large():
    total = sum(len(suite_for(t)) for t in QUERY_SUITES) + sum(len(v.queries) for v in CATALOG)
    assert total >= 80


# --------------------------------------------------------------------------- #
# PHƯƠNG TIỆN đa lớp
# --------------------------------------------------------------------------- #
def test_vehicles_prompt_is_multiclass_vehicle():
    veh = by_task("vehicles")
    assert veh
    for v in veh:
        assert v.scenario.prompt == "vehicle", v.scenario.key


def test_vehicle_alias_maps_to_all_vehicle_classes():
    from recognition.detectors.yolo_nas import COCO_ALIASES
    assert set(COCO_ALIASES["vehicle"]) == {"car", "motorcycle", "truck", "bus"}
