"""Test detector ultralytics: quy đổi kết quả + factory tự chọn backend.

Không cần cài ``ultralytics`` — chỉ dựng object 'boxes' giả giống output của nó.
"""

from recognition.base import BoundingBox, Detection
from recognition.detectors.ultralytics_yolo import UltralyticsYoloDetector


class _Arr(list):
    """Giả tensor: hỗ trợ .tolist() như ultralytics trả về."""

    def tolist(self):
        return list(self)


class FakeBox:
    """Giả một box của ultralytics: .cls[0], .conf[0], .xyxy[0].tolist()."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [_Arr(xyxy)]


NAMES = {0: "person", 2: "car", 3: "motorcycle", 7: "truck"}


def test_boxes_conversion_and_filter():
    boxes = [
        FakeBox(2, 0.9, [10, 20, 110, 220]),   # car — giữ
        FakeBox(0, 0.8, [0, 0, 50, 50]),       # person — loại (chỉ muốn xe)
        FakeBox(7, 0.7, [30, 40, 130, 240]),   # truck — giữ
    ]
    dets = UltralyticsYoloDetector._boxes_to_detections(
        boxes, NAMES, want={"car", "truck", "bus", "motorcycle"}
    )
    assert len(dets) == 2
    assert {d.label for d in dets} == {"car", "truck"}
    car = next(d for d in dets if d.label == "car")
    assert car.bbox.as_xyxy() == (10.0, 20.0, 110.0, 220.0)
    assert car.confidence == 0.9


def test_empty_want_keeps_all():
    boxes = [FakeBox(0, 0.8, [0, 0, 10, 10]), FakeBox(2, 0.9, [1, 1, 5, 5])]
    dets = UltralyticsYoloDetector._boxes_to_detections(boxes, NAMES, want=set())
    assert len(dets) == 2


def test_wanted_classes_from_prompt():
    det = UltralyticsYoloDetector()
    assert det._wanted_classes("person") == {"person"}
    assert det._wanted_classes("đếm ô tô") == {"car"}
    # "xe" chung → gồm nhiều loại phương tiện (COCO: car/motorcycle/truck/bus)
    assert det._wanted_classes("đếm xe qua trạm") == {"car", "motorcycle", "truck", "bus"}


def test_explicit_want_overrides_prompt():
    det = UltralyticsYoloDetector(want={"truck"})
    assert det._wanted_classes("person") == {"truck"}


def test_dedup_cross_class_removes_truck_bus_on_same_vehicle():
    # Cùng 1 xe: YOLO ra 'truck' (0.6) + 'bus' (0.5) box CHỒNG KHÍT → giữ 1 (conf cao nhất).
    dets = [
        Detection(BoundingBox(100, 100, 200, 180), "truck", 0.6),
        Detection(BoundingBox(103, 98, 198, 182), "bus", 0.5),   # ~trùng khít với box trên
    ]
    out = UltralyticsYoloDetector._dedup_cross_class(dets, iou_thr=0.8)
    assert len(out) == 1
    assert out[0].label == "truck"                                # giữ box conf cao hơn


def test_dedup_keeps_distinct_nearby_objects():
    # 2 vật KHÁC nhau đứng cạnh (chồng ít) → GIỮ cả hai (không gộp nhầm).
    dets = [
        Detection(BoundingBox(0, 0, 50, 100), "person", 0.9),
        Detection(BoundingBox(60, 0, 110, 100), "person", 0.8),   # cạnh nhau, IoU=0
    ]
    out = UltralyticsYoloDetector._dedup_cross_class(dets, iou_thr=0.8)
    assert len(out) == 2


def test_iou_basic():
    assert UltralyticsYoloDetector._iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert UltralyticsYoloDetector._iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_load_standard_detector_falls_back_to_ultralytics():
    # Môi trường test không cài super-gradients → auto phải trả về ultralytics.
    from recognition.detectors import load_standard_detector

    det = load_standard_detector(backend="auto", confidence=0.4)
    assert isinstance(det, UltralyticsYoloDetector)
    assert det.confidence == 0.4
