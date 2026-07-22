"""Unit test cho cấu hình 3 bài toán đếm và bộ dựng vạch."""

import pytest

from la_counting.scenarios import (
    CONVEYOR,
    PEOPLE_IN_OUT,
    SCENARIOS,
    VEHICLES,
    LineConfig,
    Scenario,
    build_line_zone,
)


def test_three_scenarios_registered():
    assert set(SCENARIOS) == {"people", "conveyor", "vehicles"}


@pytest.mark.parametrize("scn", list(SCENARIOS.values()), ids=lambda s: s.key)
def test_scenario_configs_valid(scn):
    scn.validate()  # không raise
    assert scn.prompt.strip()
    assert scn.in_label and scn.out_label


def test_scenario_semantics():
    # Đúng bài toán, đúng loại vạch
    assert PEOPLE_IN_OUT.prompt == "person"
    assert PEOPLE_IN_OUT.line.orientation == "horizontal"
    assert CONVEYOR.line.orientation == "vertical"  # chặn dòng chảy ngang
    assert VEHICLES.video_url and VEHICLES.video_url.endswith(".mp4")


def test_line_points_horizontal():
    lc = LineConfig("horizontal", 0.5, "BOTTOM_CENTER")
    assert lc.points(1280, 720) == ((0, 360), (1280, 360))


def test_line_points_vertical():
    lc = LineConfig("vertical", 0.25, "CENTER")
    assert lc.points(1280, 720) == ((320, 0), (320, 720))


# --------------------------------------------------------------------------- #
# validate() phải bắt các config sai
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(prompt="   "),  # prompt rỗng
        dict(line=LineConfig("diagonal", 0.5, "CENTER")),  # orientation sai
        dict(line=LineConfig("horizontal", 1.5, "CENTER")),  # position ngoài (0,1)
        dict(line=LineConfig("horizontal", 0.5, "MIDDLE")),  # anchor sai
        dict(resolution=(0, 720)),  # resolution sai
        dict(max_frames=0),  # max_frames sai
    ],
)
def test_validate_rejects_bad_config(kwargs):
    base = dict(
        key="tmp",
        title="t",
        prompt="person",
        line=LineConfig("horizontal", 0.5, "CENTER"),
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        Scenario(**base).validate()


def test_build_line_zone_for_all_scenarios():
    for scn in SCENARIOS.values():
        lz = build_line_zone(scn)
        assert lz.in_count == 0 and lz.out_count == 0  # khởi tạo sạch
