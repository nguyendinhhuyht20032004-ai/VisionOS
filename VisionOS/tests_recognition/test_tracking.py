"""Test tracker: giữ đúng danh tính qua frame — điều kiện cần để đếm chính xác."""

from recognition.base import Detection
from recognition.tracking import CentroidTracker


def det(cx, cy, w=40, h=80):
    return Detection.from_xyxy(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, "person")


def test_single_object_keeps_same_id():
    tr = CentroidTracker(max_distance=60)
    ids = []
    for cy in range(100, 400, 20):  # một vật đi thẳng, bước nhỏ
        tracked = tr.update([det(200, cy)])
        ids.append(tracked[0].track_id)
    assert len(set(ids)) == 1  # cùng một track_id suốt


def test_two_objects_get_distinct_ids():
    tr = CentroidTracker(max_distance=60)
    tracked = tr.update([det(100, 200), det(600, 200)])
    assert tracked[0].track_id != tracked[1].track_id
    assert len(tr.tracks) == 2


def test_far_jump_creates_new_id():
    tr = CentroidTracker(max_distance=50)
    a = tr.update([det(100, 100)])[0].track_id
    b = tr.update([det(900, 600)])[0].track_id  # nhảy quá xa -> track mới
    assert a != b


def test_track_removed_after_missed():
    tr = CentroidTracker(max_distance=50, max_missed=2)
    tr.update([det(100, 100)])
    assert len(tr.tracks) == 1
    for _ in range(4):  # vắng mặt nhiều frame
        tr.update([])
    assert len(tr.tracks) == 0


def test_dwell_frames_counts_presence():
    tr = CentroidTracker(max_distance=60)
    for cy in range(100, 200, 20):
        tr.update([det(200, cy)])
    track = next(iter(tr.tracks.values()))
    assert track.dwell_frames() == 5
