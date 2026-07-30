"""Test file test riêng LocateAnything-3B đếm sản phẩm (run_la_conveyor.py).

Thuần logic — KHÔNG cần GPU/model. Kiểm tra: bộ query, lấy mẫu frame, NMS khử
box trùng, dựng scenario băng chuyền, và đường ống self-test đếm được cắt vạch.
"""

import numpy as np

import run_la_conveyor as R
from la_counting.counting import CountingPipeline
from la_counting.parsing import Detection


# --------------------------------------------------------------------------- #
# Bộ QUERY sản phẩm
# --------------------------------------------------------------------------- #
def test_suite_lite_smaller_than_full():
    full = R.suite(lite=False)
    lite1 = R.suite(lite=True, per_group=1)
    lite2 = R.suite(lite=True, per_group=2)
    assert len(lite1) < len(full)
    assert len(lite1) <= len(lite2) <= len(full)
    # lite=1 → đúng 1 query mỗi nhóm
    assert len(lite1) == len(R.PRODUCT_QUERIES)


def test_suite_has_openvocab_and_vietnamese():
    full_q = [q for _g, q in R.suite(lite=False)]
    assert "object" in full_q                      # thế mạnh open-vocab của LA
    assert any("bottle" in q for q in full_q)      # sản phẩm cụ thể (video chai)
    assert any("chai" in q for q in full_q)        # tiếng Việt
    # đại diện (lite) phải đứng đầu mỗi nhóm
    lite_q = [q for _g, q in R.suite(lite=True, per_group=1)]
    assert "object" in lite_q


def test_suite_returns_group_query_pairs():
    for g, q in R.suite():
        assert isinstance(g, str) and g
        assert isinstance(q, str) and q


# --------------------------------------------------------------------------- #
# Lấy mẫu frame thưa
# --------------------------------------------------------------------------- #
def test_strided_picks_every_n():
    src = list(range(10))
    assert list(R.strided(iter(src), 3)) == [0, 3, 6, 9]
    assert list(R.strided(iter(src), 1)) == src          # stride<=1 giữ hết
    assert list(R.strided(iter(src), 0)) == src


# --------------------------------------------------------------------------- #
# NMS khử box trùng (bbox tuple)
# --------------------------------------------------------------------------- #
def _det(x1, y1, x2, y2):
    return Detection((float(x1), float(y1), float(x2), float(y2)), "object", 0.9)


def test_nms_collapses_overlapping_boxes():
    a = _det(0, 0, 100, 100)
    b = _det(2, 2, 101, 101)        # ~trùng a
    c = _det(500, 500, 560, 560)    # tách biệt
    kept = R.nms([a, b, c], iou_thr=0.5)
    assert len(kept) == 2                       # a/b gộp 1, c riêng
    # giữ box LỚN HƠN khi trùng (tất định)
    assert _det(0, 0, 100, 100).bbox in [k.bbox for k in kept]


def test_nms_keeps_distinct_boxes():
    ds = [_det(i * 200, 0, i * 200 + 50, 50) for i in range(5)]
    assert len(R.nms(ds, iou_thr=0.5)) == 5


def test_nms_caps_max_boxes():
    ds = [_det(i * 200, 0, i * 200 + 50, 50) for i in range(80)]
    assert len(R.nms(ds, iou_thr=0.5, max_boxes=60)) == 60


def test_nms_trivial_sizes():
    assert R.nms([]) == []
    one = [_det(0, 0, 10, 10)]
    assert R.nms(one) == one


# --------------------------------------------------------------------------- #
# Scenario băng chuyền
# --------------------------------------------------------------------------- #
def test_build_scenario_is_vertical_and_valid():
    scn = R.build_scenario(640, 360, 0.5, "CENTER", "object", 30)
    scn.validate()                              # không raise
    assert scn.line.orientation == "vertical"   # sản phẩm chạy ngang → vạch dọc
    assert scn.resolution == (640, 360)
    assert scn.prompt == "object"
    assert scn.max_frames == 30


# --------------------------------------------------------------------------- #
# Detector giả + đường ống đếm (không GPU)
# --------------------------------------------------------------------------- #
def test_fakedet_box_moves_left_to_right():
    fd = R._FakeDet(10)
    frame = np.zeros((360, 640, 3), dtype="uint8")
    xs = [fd.detect_frame(frame, "object")[0][0].bbox[0] for _ in range(10)]
    assert xs == sorted(xs)          # x tăng dần
    assert xs[0] < xs[-1]            # có di chuyển


def test_selftest_pipeline_counts_a_crossing():
    n = 26
    scn = R.build_scenario(640, 360, 0.5, "CENTER", "object", n)
    pipe = CountingPipeline(R._FakeDet(n), scn, resize=False)
    res = pipe.run(R._blank_frames(n, 640, 360), max_frames=n)
    assert res.frames == n
    assert res.total_crossings >= 1              # vật đã cắt vạch dọc


def test_main_selftest_returns_zero():
    assert R.main(["--selftest"]) == 0


# --------------------------------------------------------------------------- #
# Arg parsing (mặc định tối ưu tốc độ)
# --------------------------------------------------------------------------- #
def test_defaults_are_fast():
    args = R.build_argparser().parse_args([])
    assert args.proc_width == 640
    assert args.max_new_tokens == 256
    assert args.max_frames == 30
    assert args.stride == 3
    assert not args.no_milk
