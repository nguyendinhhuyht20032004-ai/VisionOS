"""Test toàn luồng detect -> track -> count với detector giả (không cần GPU).

Dùng ByteTrack + LineZone THẬT của supervision, chỉ thay mô hình bằng
``ScriptedDetector``. Giá trị kỳ vọng đã được xác minh thực nghiệm.
"""

from la_counting import CONVEYOR, PEOPLE_IN_OUT, VEHICLES, CountingPipeline
from fakes import ScriptedDetector, blank_frames, linear_track, merge_tracks

N = 26


def _run(scenario, script):
    det = ScriptedDetector(script)
    pipe = CountingPipeline(det, scenario)
    return pipe.run(blank_frames(len(script))), det


# --------------------------------------------------------------------------- #
# Người ra/vào
# --------------------------------------------------------------------------- #
def test_people_walk_in_counts_one_crossing():
    track = linear_track((600, 150, 680, 190), (600, 470, 680, 510), N)  # đi xuống
    result, det = _run(PEOPLE_IN_OUT, [[b] for b in track])
    assert result.frames == N
    assert result.total_detections == N  # 1 người mỗi frame
    assert result.total_crossings == 1
    assert (result.in_count, result.out_count) == (0, 1)
    assert det.prompts_seen[0] == "person"  # đúng prompt được đẩy vào model


def test_people_reverse_direction_flips_bucket():
    track = linear_track((600, 470, 680, 510), (600, 150, 680, 190), N)  # đi lên
    result, _ = _run(PEOPLE_IN_OUT, [[b] for b in track])
    assert (result.in_count, result.out_count) == (1, 0)


# --------------------------------------------------------------------------- #
# Băng chuyền
# --------------------------------------------------------------------------- #
def test_conveyor_product_crosses_once():
    track = linear_track((300, 300, 380, 380), (1000, 300, 1080, 380), N)  # sang phải
    result, det = _run(CONVEYOR, [[b] for b in track])
    assert result.total_crossings == 1
    assert det.prompts_seen[0] == "object"


# --------------------------------------------------------------------------- #
# Phương tiện — đa vật thể
# --------------------------------------------------------------------------- #
def test_two_vehicles_counted_separately():
    t1 = linear_track((300, 150, 400, 220), (300, 470, 400, 540), N)
    t2 = linear_track((800, 150, 900, 220), (800, 470, 900, 540), N)
    result, _ = _run(VEHICLES, merge_tracks(t1, t2))
    assert result.total_crossings == 2
    assert result.total_detections == 2 * N


# --------------------------------------------------------------------------- #
# Ca biên & thuộc tính kết quả
# --------------------------------------------------------------------------- #
def test_no_detections_no_crossings():
    result, _ = _run(VEHICLES, [[] for _ in range(N)])
    assert result.frames == N
    assert result.total_detections == 0
    assert result.total_crossings == 0
    assert result.avg_detections == 0.0


def test_result_properties_and_row():
    track = linear_track((600, 150, 680, 190), (600, 470, 680, 510), N)
    result, _ = _run(PEOPLE_IN_OUT, [[b] for b in track])
    assert result.avg_detections == 1.0
    assert result.fps > 0
    row = result.as_row()
    # scorecard dùng nhãn tiếng Việt của scenario làm khoá cột
    assert row["Vào (IN)"] == 0 and row["Ra (OUT)"] == 1
    assert row["total"] == 1 and row["scenario"] == "people"


def test_on_frame_callback_invoked_per_frame():
    track = linear_track((600, 150, 680, 190), (600, 470, 680, 510), N)
    seen = []
    det = ScriptedDetector([[b] for b in track])
    pipe = CountingPipeline(det, PEOPLE_IN_OUT)
    pipe.run(blank_frames(N), on_frame=lambda f, d, p: seen.append(len(d)))
    assert len(seen) == N
