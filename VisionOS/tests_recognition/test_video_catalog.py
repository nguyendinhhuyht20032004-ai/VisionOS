"""Test catalog video test-đếm (thuần dữ liệu — không tải mạng, không GPU)."""

import pytest

from recognition.scenarios import CountScenario
from recognition.video_catalog import CATALOG, RB_CDN, VideoScenario, by_task


def test_catalog_nonempty_and_typed():
    assert len(CATALOG) >= 4
    assert all(isinstance(v, VideoScenario) for v in CATALOG)


def test_covers_three_tasks():
    tasks = {v.task for v in CATALOG}
    assert {"vehicles", "conveyor", "people"} <= tasks


def test_conveyor_and_vehicles_present():
    # Đúng yêu cầu: có dây chuyền sản xuất + phương tiện
    assert any(v.filename == "milk-bottling-plant.mp4" for v in CATALOG)
    assert any(v.task == "vehicles" for v in CATALOG)


def test_every_scenario_valid_and_line():
    # Mỗi kịch bản phải validate được (vạch/prompt/độ phân giải hợp lệ)
    for v in CATALOG:
        assert isinstance(v.scenario, CountScenario)
        v.scenario.validate()          # không ném lỗi
        assert v.scenario.counting_type == "line"
        assert v.scenario.prompt.strip()


def test_urls_wellformed_on_roboflow_cdn():
    for v in CATALOG:
        assert v.url == RB_CDN + v.filename
        assert v.url.startswith("https://media.roboflow.com/")
        assert v.filename.endswith(".mp4")


def test_scenario_keys_unique_and_identifier():
    keys = [v.scenario.key for v in CATALOG]
    assert len(keys) == len(set(keys))          # không trùng key
    assert all(k.isidentifier() for k in keys)


def test_by_task_filter():
    assert by_task(None) == CATALOG
    veh = by_task("vehicles")
    assert veh and all(v.task == "vehicles" for v in veh)
    assert by_task("khong-co-task") == []


def test_conveyor_line_is_vertical():
    # Chai chạy ngang → vạch phải DỌC (x cố định, y chạy 0..100)
    conv = by_task("conveyor")[0].scenario
    assert conv.line_start_pct[0] == conv.line_end_pct[0]      # cùng x
    assert conv.line_start_pct[1] != conv.line_end_pct[1]      # khác y


def test_vehicles_line_is_horizontal():
    # Xe chạy dọc → vạch NGANG (y cố định, x chạy 0..100)
    veh = by_task("vehicles")[0].scenario
    assert veh.line_start_pct[1] == veh.line_end_pct[1]        # cùng y
    assert veh.line_start_pct[0] != veh.line_end_pct[0]        # khác x
