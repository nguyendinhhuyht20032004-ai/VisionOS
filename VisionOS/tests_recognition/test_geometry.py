"""Test hình học nền tảng: point-in-polygon và cắt vạch có hướng."""

from recognition.geometry import (
    line_crossing,
    point_in_polygon,
    polygon_area,
    segments_intersect,
    side_of_line,
)


def square():
    return [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_inside_polygon():
    assert point_in_polygon((5, 5), square()) is True


def test_point_outside_polygon():
    assert point_in_polygon((15, 5), square()) is False
    assert point_in_polygon((-1, -1), square()) is False


def test_point_on_vertex_is_inside():
    assert point_in_polygon((0, 0), square()) is True


def test_polygon_too_few_points():
    assert point_in_polygon((0, 0), [(0, 0), (1, 1)]) is False


def test_concave_polygon():
    # Đa giác lõm hình chữ L; điểm trong phần lõm phải nằm NGOÀI.
    l_shape = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    assert point_in_polygon((2, 2), l_shape) is True
    assert point_in_polygon((7, 7), l_shape) is False


def test_polygon_area_square():
    assert polygon_area(square()) == 100.0


def test_side_of_line_sign():
    # Vạch dọc x=5 (từ dưới lên): điểm bên trái (x<5) và bên phải (x>5) khác dấu.
    a, b = (5, 0), (5, 10)
    assert (side_of_line((2, 5), a, b) > 0) != (side_of_line((8, 5), a, b) > 0)


def test_segments_intersect():
    assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0)) is True
    assert segments_intersect((0, 0), (1, 1), (5, 5), (6, 6)) is False


def test_line_crossing_detects_direction():
    line_s, line_e = (0, 5), (10, 5)  # vạch ngang y=5
    down = line_crossing((5, 2), (5, 8), line_s, line_e)   # đi xuống
    up = line_crossing((5, 8), (5, 2), line_s, line_e)     # đi lên
    assert down in ("in", "out")
    assert up in ("in", "out")
    assert down != up  # hai chiều phải cho kết quả khác nhau


def test_line_crossing_none_when_not_crossing():
    line_s, line_e = (0, 5), (10, 5)
    assert line_crossing((5, 1), (5, 2), line_s, line_e) is None


def test_line_crossing_touch_and_return_not_counted():
    # Chạm vạch rồi quay lại cùng phía -> không tính là cắt.
    line_s, line_e = (0, 5), (10, 5)
    assert line_crossing((5, 4), (5, 4), line_s, line_e) is None
