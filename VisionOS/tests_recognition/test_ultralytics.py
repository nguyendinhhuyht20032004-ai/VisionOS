"""Test detector ultralytics: quy đổi kết quả + factory tự chọn backend.

Không cần cài ``ultralytics`` — chỉ dựng object 'boxes' giả giống output của nó.
"""

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
    # "xe" chung → gồm nhiều loại phương tiện
    assert det._wanted_classes("đếm xe qua trạm") == {"car", "motorcycle", "truck", "bus"}


def test_explicit_want_overrides_prompt():
    det = UltralyticsYoloDetector(want={"truck"})
    assert det._wanted_classes("person") == {"truck"}


def test_load_standard_detector_falls_back_to_ultralytics():
    # Môi trường test không cài super-gradients → auto phải trả về ultralytics.
    from recognition.detectors import load_standard_detector

    det = load_standard_detector(backend="auto", confidence=0.4)
    assert isinstance(det, UltralyticsYoloDetector)
    assert det.confidence == 0.4
