import cv2
from recognition.service.engine import StreamingCounter
from recognition.service.stream_manager import get_detector

class DummyScenario:
    key = "test"
    counting_type = "fullscreen"
    resolution = (1920, 1080)
    prompt = None

cap = cv2.VideoCapture('/data/street.mp4')
ret, frame = cap.read()
if not ret: print("Cannot read video"); exit()

scenario = DummyScenario()
detector = get_detector("yolov8", 0.3)
counter = StreamingCounter(scenario, detector, resolution=(1920, 1080))
out = counter.process(frame)
det = counter.last_det
print("det:", det)
if det:
    print("data:", getattr(det, "data", None))
    print("tracker_id:", getattr(det, "tracker_id", None))
    print("bool(data):", bool(getattr(det, "data", None)))

