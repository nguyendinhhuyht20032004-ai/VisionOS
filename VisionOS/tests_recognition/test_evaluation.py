"""Test bộ chỉ số đánh giá độ chính xác (recognition.evaluation)."""

import math

import pytest

from recognition.base import BoundingBox
from recognition.evaluation import (
    DetectionMetrics,
    evaluate,
    greedy_match,
    nms_dedup,
    verdict,
)


def bb(x1, y1, x2, y2):
    return BoundingBox(x1, y1, x2, y2)


A = bb(0, 0, 10, 10)
B = bb(5, 0, 15, 10)      # IoU(A,B) = 1/3
FAR = bb(100, 100, 110, 110)


# --------------------------------------------------------------------------- #
# greedy_match
# --------------------------------------------------------------------------- #
def test_match_perfect():
    assert greedy_match([A], [A], 0.5)[:3] == (1, 0, 0)


def test_match_duplicate_pred_is_fp():
    assert greedy_match([A, A], [A], 0.5)[:3] == (1, 1, 0)


def test_match_missing_and_extra():
    assert greedy_match([], [A], 0.5)[:3] == (0, 0, 1)
    assert greedy_match([A], [], 0.5)[:3] == (0, 1, 0)


def test_match_below_threshold():
    assert greedy_match([B], [A], 0.5)[:3] == (0, 1, 1)
    assert greedy_match([B], [A], 0.3)[:3] == (1, 0, 0)


# --------------------------------------------------------------------------- #
# DetectionMetrics gộp
# --------------------------------------------------------------------------- #
def test_metrics_prf1_and_counting():
    m = evaluate([[A], [A, A]], [[A], [A]], iou_thr=0.5)
    assert (m.tp, m.fp, m.fn) == (2, 1, 0)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(0.8)
    assert m.count_mae == pytest.approx(0.5)
    assert m.count_rmse == pytest.approx(math.sqrt(0.5))
    assert m.count_bias == pytest.approx(0.5)


def test_metrics_all_wrong():
    m = evaluate([[FAR]], [[A]], 0.5)
    assert (m.precision, m.recall, m.f1) == (0.0, 0.0, 0.0)


def test_as_row_keys():
    row = evaluate([[A]], [[A]], 0.5, label="person").as_row()
    for k in ("lớp", "P", "R", "F1", "MAE", "kết luận"):
        assert k in row


# --------------------------------------------------------------------------- #
# nms_dedup
# --------------------------------------------------------------------------- #
def test_nms_removes_identical_by_score():
    assert nms_dedup([A, A], scores=[0.9, 0.8], iou_thr=0.5) == [0]


def test_nms_keeps_distinct():
    assert nms_dedup([A, FAR], scores=[0.9, 0.8], iou_thr=0.5) == [0, 1]


def test_nms_no_score_keeps_larger():
    small, big = bb(0, 0, 15, 15), bb(0, 0, 20, 20)   # IoU 0.5625
    assert nms_dedup([small, big], scores=None, iou_thr=0.5) == [1]


def test_nms_equal_scores_treated_as_no_score():
    # confidence bằng nhau (model không phát điểm) -> theo diện tích, khử trùng
    assert nms_dedup([A, A], scores=[0.85, 0.85], iou_thr=0.5) == [0]


def test_nms_empty():
    assert nms_dedup([], None) == []


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def test_verdict_bands():
    labels = [verdict(x) for x in (0.9, 0.7, 0.5, 0.3, 0.1)]
    assert len(set(labels)) == 5
    assert "Rất tốt" in labels[0] and "Kém" in labels[-1]
