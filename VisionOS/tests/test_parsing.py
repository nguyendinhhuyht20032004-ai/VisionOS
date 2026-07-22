"""Unit test cho ``parse_boxes`` — giải mã text của model ra bbox.

Đây là phần logic thuần, chạy được không cần GPU/torch. Test cả 3 định dạng,
thứ tự ưu tiên giữa chúng, quy đổi toạ độ chuẩn hoá -> pixel, và các ca biên.
"""

import pytest

from la_counting.parsing import Detection, parse_boxes, rescale_box


# --------------------------------------------------------------------------- #
# Pattern 1: <box><x1><y1><x2><y2></box> (thang 0..1000)
# --------------------------------------------------------------------------- #
def test_pattern1_single_box_full_scale():
    # w=h=1000 -> toạ độ giữ nguyên
    dets = parse_boxes("<box><100><200><300><400></box>", "car", 1000, 1000)
    assert len(dets) == 1
    d = dets[0]
    assert d.bbox == (100, 200, 300, 400)
    assert d.class_name == "car"
    assert d.confidence == pytest.approx(0.85)


def test_pattern1_scales_to_frame_size():
    # 0..1000 -> pixel theo (w=1280, h=720)
    dets = parse_boxes("<box><0><0><500><500></box>", "person", 1280, 720)
    assert len(dets) == 1
    assert dets[0].bbox == (0, 0, 640, 360)  # 500/1000*1280=640 ; 500/1000*720=360


def test_pattern1_multiple_boxes():
    text = "<box><10><10><90><90></box> and <box><100><100><200><200></box>"
    dets = parse_boxes(text, "object", 1000, 1000)
    assert [d.bbox for d in dets] == [(10, 10, 90, 90), (100, 100, 200, 200)]


def test_pattern1_drops_degenerate_boxes():
    # x2<=x1 (300<300) và y2<=y1 (400<200) -> loại
    text = "<box><300><200><300><400></box><box><100><400><300><200></box>"
    assert parse_boxes(text, "x", 1000, 1000) == []


# --------------------------------------------------------------------------- #
# Pattern 2: [x1, y1, x2, y2] — tự đoán thang
# --------------------------------------------------------------------------- #
def test_pattern2_bracket_thousand_scale():
    # max>1.5 -> coi là thang 0..1000
    dets = parse_boxes("box at [100, 200, 300, 400]", "car", 1000, 1000)
    assert dets[0].bbox == (100, 200, 300, 400)


def test_pattern2_bracket_normalized_0_1():
    # max<=1.5 -> coi là chuẩn hoá 0..1
    dets = parse_boxes("[0.1, 0.2, 0.3, 0.4]", "car", 1000, 1000)
    assert dets[0].bbox == (100, 200, 300, 400)


def test_pattern2_only_when_pattern1_empty():
    # Có box-tag hợp lệ -> KHÔNG dùng tới bracket (đúng thứ tự ưu tiên notebook)
    text = "<box><10><10><90><90></box> ignored [500, 500, 900, 900]"
    dets = parse_boxes(text, "x", 1000, 1000)
    assert [d.bbox for d in dets] == [(10, 10, 90, 90)]


# --------------------------------------------------------------------------- #
# Pattern 3: <loc_n> theo bộ 4 (thang 0..1000)
# --------------------------------------------------------------------------- #
def test_pattern3_loc_tokens():
    text = "<loc_100><loc_200><loc_300><loc_400>"
    dets = parse_boxes(text, "car", 1000, 1000)
    assert dets[0].bbox == (100, 200, 300, 400)


def test_pattern3_two_groups_of_four():
    text = "".join(f"<loc_{v}>" for v in [10, 10, 90, 90, 100, 100, 200, 200])
    dets = parse_boxes(text, "x", 1000, 1000)
    assert [d.bbox for d in dets] == [(10, 10, 90, 90), (100, 100, 200, 200)]


def test_pattern3_ignores_incomplete_trailing_group():
    # 5 token -> chỉ dùng 4 đầu, bỏ token lẻ cuối (không crash)
    text = "".join(f"<loc_{v}>" for v in [10, 10, 90, 90, 55])
    dets = parse_boxes(text, "x", 1000, 1000)
    assert [d.bbox for d in dets] == [(10, 10, 90, 90)]


def test_pattern3_only_when_1_and_2_empty():
    text = "[100,100,200,200] then <loc_500><loc_500><loc_900><loc_900>"
    dets = parse_boxes(text, "x", 1000, 1000)
    assert [d.bbox for d in dets] == [(100, 100, 200, 200)]  # bracket thắng


# --------------------------------------------------------------------------- #
# Ca biên
# --------------------------------------------------------------------------- #
def test_empty_and_prose_text_returns_no_boxes():
    assert parse_boxes("", "x", 1000, 1000) == []
    assert parse_boxes("I cannot find any car in this image.", "car", 640, 480) == []


def test_default_conf_is_configurable():
    dets = parse_boxes("<box><1><1><9><9></box>", "x", 1000, 1000, default_conf=0.5)
    assert dets[0].confidence == pytest.approx(0.5)


def test_returns_detection_instances():
    dets = parse_boxes("<box><1><1><9><9></box>", "x", 1000, 1000)
    assert all(isinstance(d, Detection) for d in dets)


# --------------------------------------------------------------------------- #
# rescale_box — đổi hệ toạ độ (dùng ở bước đánh giá mAP)
# --------------------------------------------------------------------------- #
def test_rescale_box_down():
    # 1280x720 -> 640x360 : chia đôi
    assert rescale_box((100, 200, 300, 400), (1280, 720), (640, 360)) == (
        50.0, 100.0, 150.0, 200.0
    )


def test_rescale_box_identity():
    box = (12.0, 34.0, 56.0, 78.0)
    assert rescale_box(box, (1280, 720), (1280, 720)) == box
