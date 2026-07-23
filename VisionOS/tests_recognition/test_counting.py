"""Test HIỆU QUẢ ĐẾM end-to-end (detect → track → đếm) với kịch bản tất định.

Mỗi test dựng chuyển động biết trước đáp số rồi khẳng định pipeline đếm đúng:
đúng số lần cắt vạch, đúng chiều, không đếm lặp, không đếm vật đứng yên xa vạch.
Chạy hoàn toàn trên CPU nhờ ``ScriptedDetector`` — không cần GPU.
"""

from recognition.counting import CountingPipeline
from recognition.detectors.fake import ScriptedDetector, box_moving, merge_tracks
from recognition.scenarios import PACKAGES, PEOPLE_IN_OUT, QUEUE, VEHICLES

N = 30  # số frame mỗi kịch bản


def run(scenario, frames_boxes):
    det = ScriptedDetector(frames_boxes)
    pipe = CountingPipeline(det, scenario)
    result = pipe.run(range(len(frames_boxes)), max_frames=len(frames_boxes))
    return pipe, result


# --------------------------------------------------------------------------- #
# 1) Đếm người vào/ra — vạch ngang 50%
# --------------------------------------------------------------------------- #

def test_people_one_person_crossing_down_counts_once():
    # Một người đi xuống xuyên vạch: đúng 1 lần cắt (không đếm mỗi frame).
    track = box_moving(300, 100, 300, 620, w=40, h=120, n_frames=N)
    frames = [[b] for b in track]
    pipe, res = run(PEOPLE_IN_OUT, frames)
    assert res.total_crossings == 1
    assert res.unique_tracks == 1


def test_people_two_directions_split_in_out():
    # Một người xuống, một người lên: 2 lần cắt, chia đúng 2 chiều.
    down = box_moving(300, 100, 300, 620, w=40, h=120, n_frames=N)
    up = box_moving(900, 620, 900, 100, w=40, h=120, n_frames=N)
    frames = merge_tracks(down, up)
    pipe, res = run(PEOPLE_IN_OUT, frames)
    assert res.total_crossings == 2
    assert res.in_count == 1 and res.out_count == 1
    assert res.unique_tracks == 2


def test_people_static_far_from_line_not_counted():
    # Người đứng yên xa vạch (phía trên): 0 lần cắt nhưng vẫn là 1 track.
    static = [[(280, 40, 320, 160)] for _ in range(N)]  # bottom-center y=160 < 360
    pipe, res = run(PEOPLE_IN_OUT, static)
    assert res.total_crossings == 0
    assert res.unique_tracks == 1


def test_people_no_detection_zero_count():
    pipe, res = run(PEOPLE_IN_OUT, [[] for _ in range(N)])
    assert res.total_crossings == 0
    assert res.unique_tracks == 0


# --------------------------------------------------------------------------- #
# 2) Đếm xe — vạch ngang 55%
# --------------------------------------------------------------------------- #

def test_vehicles_three_cars_counted():
    # 3 xe lần lượt đi xuống qua vạch (mỗi xe một làn riêng) -> 3 lần cắt.
    lanes = [200, 640, 1080]
    tracks = [box_moving(x, 100, x, 650, w=80, h=60, n_frames=N) for x in lanes]
    frames = merge_tracks(*tracks)
    pipe, res = run(VEHICLES, frames)
    assert res.total_crossings == 3
    assert res.unique_tracks == 3


# --------------------------------------------------------------------------- #
# 3) Đếm kiện hàng — vạch DỌC 50%, open-vocab (LocateAnything)
# --------------------------------------------------------------------------- #

def test_packages_cross_vertical_line():
    # Kiện hàng chạy ngang trái->phải qua vạch dọc giữa khung: 1 lần cắt.
    track = box_moving(100, 360, 1180, 360, w=60, h=60, n_frames=N)
    frames = [[b] for b in track]
    pipe, res = run(PACKAGES, frames)
    assert res.total_crossings == 1
    # detector nhận đúng prompt open-vocab của bài toán.
    assert "carton" in pipe.detector.prompts_seen[0]


# --------------------------------------------------------------------------- #
# 4) Phân tích hàng chờ — đếm chiếm VÙNG
# --------------------------------------------------------------------------- #

def test_queue_zone_occupancy_peak():
    # Hai người đứng trong vùng chờ suốt -> đỉnh chiếm vùng = 2.
    # Vùng: x 20..80%, y 30..90% của 1280x720 -> px x256..1024, y216..648.
    p1 = [[(500, 500, 540, 620)] for _ in range(N)]   # bottom-center (520,620) trong vùng
    p2 = [[(700, 480, 740, 600)] for _ in range(N)]   # bottom-center (720,600) trong vùng
    frames = [p1[i] + p2[i] for i in range(N)]
    pipe, res = run(QUEUE, frames)
    assert res.zone_peak == 2
    assert res.zone_current == 2


def test_queue_person_outside_zone_not_counted():
    # Người ở góc trên-trái ngoài vùng -> không tính vào occupancy.
    outside = [[(50, 20, 90, 100)] for _ in range(N)]  # bottom-center (70,100) ngoài vùng
    pipe, res = run(QUEUE, outside)
    assert res.zone_peak == 0
