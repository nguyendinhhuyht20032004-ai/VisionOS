"""Test logic đếm cắt vạch ở tầng thấp: vạch của scenario + LineZone thật.

Ở đây ta *bỏ qua* ByteTrack và tự gán tracker_id để kiểm tra thuần logic cắt
vạch cho từng bố trí (ngang/dọc, các anchor), tách khỏi sự bất định của tracker.
Chiều đếm được xác minh bằng thực nghiệm với supervision 0.29:
  - vạch ngang: đi XUỐNG -> out_count, đi LÊN -> in_count
  - vạch dọc  : đi SANG PHẢI -> in_count
"""

import numpy as np
import supervision as sv

from la_counting.scenarios import CONVEYOR, PEOPLE_IN_OUT, build_line_zone


def _det(x1, y1, x2, y2, tid):
    return sv.Detections(
        xyxy=np.array([[x1, y1, x2, y2]], dtype=np.float32),
        confidence=np.array([0.85], dtype=np.float32),
        class_id=np.array([0]),
        tracker_id=np.array([tid]),
    )


def _sweep_vertical(lz, y_start, y_end, tid, step=20, box=(600, 680, 40)):
    """Cho 1 box (bottom-center trên trục x cố định) quét y_start->y_end."""
    x1, x2, hbox = box
    ys = range(y_start, y_end, step if y_end > y_start else -step)
    for y in ys:
        lz.trigger(_det(x1, y, x2, y + hbox, tid))


def _sweep_horizontal(lz, x_start, x_end, tid, step=30, box=(300, 380, 80)):
    y1, y2, wbox = box
    xs = range(x_start, x_end, step if x_end > x_start else -step)
    for x in xs:
        lz.trigger(_det(x, y1, x + wbox, y2, tid))


# --------------------------------------------------------------------------- #
# Người ra/vào — vạch ngang y=360, anchor BOTTOM_CENTER
# --------------------------------------------------------------------------- #
def test_people_cross_down_counts_out():
    lz = build_line_zone(PEOPLE_IN_OUT)  # y=360
    _sweep_vertical(lz, 200, 520, tid=1)  # bottom-center 240 -> 540 (đi xuống)
    assert (lz.in_count, lz.out_count) == (0, 1)


def test_people_cross_up_counts_in():
    lz = build_line_zone(PEOPLE_IN_OUT)
    _sweep_vertical(lz, 500, 180, tid=2)  # đi lên
    assert (lz.in_count, lz.out_count) == (1, 0)


def test_people_no_cross_stays_zero():
    lz = build_line_zone(PEOPLE_IN_OUT)
    _sweep_vertical(lz, 100, 260, tid=3)  # bottom-center 140->300, chưa tới 360
    assert (lz.in_count, lz.out_count) == (0, 0)


def test_two_people_two_crossings():
    lz = build_line_zone(PEOPLE_IN_OUT)
    _sweep_vertical(lz, 200, 520, tid=10)
    _sweep_vertical(lz, 220, 520, tid=11)
    assert lz.in_count + lz.out_count == 2


# --------------------------------------------------------------------------- #
# Băng chuyền — vạch dọc x=640, anchor CENTER
# --------------------------------------------------------------------------- #
def test_conveyor_cross_right_counts_in():
    lz = build_line_zone(CONVEYOR)  # x=640
    _sweep_horizontal(lz, 300, 1000, tid=1)  # center x 340 -> 1020 (sang phải)
    assert (lz.in_count, lz.out_count) == (1, 0)


def test_conveyor_no_cross_stays_zero():
    lz = build_line_zone(CONVEYOR)
    _sweep_horizontal(lz, 100, 500, tid=2)  # center 140->540, chưa tới 640
    assert (lz.in_count, lz.out_count) == (0, 0)
