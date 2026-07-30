"""Test engine đếm bằng supervision (ByteTrack + LineZone/PolygonZone).

Bỏ qua nếu chưa cài supervision (chỉ chạy khi có thư viện, vd trên Colab/Kaggle).
"""

import numpy as np
import pytest

from recognition.base import BoundingBox, Detection, DetectorResult, MonitoringMode
from recognition.scenarios import CountScenario
from recognition.sv_counting import sv_available

pytestmark = pytest.mark.skipif(not sv_available(), reason="cần cài supervision")

from recognition.sv_counting import sv_run  # noqa: E402


def _scenario(counting_type, **kw):
    return CountScenario(
        key="t", title="t", usecase_id="t", mode=MonitoringMode.STANDARD,
        model="YOLO", prompt="car", counting_type=counting_type, resolution=(200, 200),
        in_label="Vao", out_label="Ra", **kw,
    )


def test_sv_line_counts_one_crossing():
    class Mover:
        def __init__(self):
            self.i = 0

        def detect(self, frame, prompt):
            y = 20 + self.i * 4          # đi xuống chậm (khớp IoU của ByteTrack)
            self.i += 1
            return DetectorResult([Detection(BoundingBox(80, y, 120, y + 40), "car", 0.9)], raw="")

    sc = _scenario("line", line_start_pct=(0, 50), line_end_pct=(100, 50))
    frames = [np.zeros((200, 200, 3), np.uint8) for _ in range(40)]
    res = sv_run(iter(frames), sc, Mover(), (200, 200), max_frames=40)
    assert res.total_crossings == 1                 # 1 vật qua vạch = đúng 1 lần
    assert res.unique_tracks == 1


def test_sv_multi_zone_counts_union():
    class TwoStatic:
        def detect(self, frame, prompt):
            return DetectorResult([
                Detection(BoundingBox(20, 20, 40, 40), "car", 0.9),
                Detection(BoundingBox(160, 160, 180, 180), "car", 0.9),
            ], raw="")

    sc = _scenario("zone", zone_anchor="CENTER",
                   zones_pct=(((5, 5), (25, 5), (25, 25), (5, 25)),
                              ((75, 75), (95, 75), (95, 95), (75, 95))))
    frames = [np.zeros((200, 200, 3), np.uint8) for _ in range(6)]
    res = sv_run(iter(frames), sc, TwoStatic(), (200, 200), max_frames=6)
    assert res.zone_peak == 2                        # 2 vật trong 2 vùng → đỉnh 2


def test_sv_no_detection_counts_zero():
    class Empty:
        def detect(self, frame, prompt):
            return DetectorResult([], raw="")

    sc = _scenario("line", line_start_pct=(0, 50), line_end_pct=(100, 50))
    res = sv_run(iter([np.zeros((200, 200, 3), np.uint8) for _ in range(5)]), sc, Empty(), (200, 200))
    assert res.total_crossings == 0 and res.frames == 5
