import cv2
import time
from recognition.service.engine import StreamingCounter
from recognition.service.builder import get_detector, make_scenario

def run():
    print("Loading video...")
    cap = cv2.VideoCapture('/data/street.mp4')
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video resolution: {width}x{height} @ {fps}fps")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/data/annotated.mp4', fourcc, fps, (width, height))

    print("Initializing AI Engine (YOLO + ByteTrack)...")
    scenario, _ = make_scenario(prompt="person", counting_type="fullscreen", resolution=(width, height))
    detector = get_detector("yolov8", confidence=0.3)
    counter = StreamingCounter(scenario, detector, resolution=(width, height))

    count = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process and get annotated frame
        annotated = counter.process(frame)
        out.write(annotated)
        
        count += 1
        if count % 30 == 0:
            print(f"Processed {count} frames...")

    cap.release()
    out.release()
    print(f"Done processing {count} frames in {time.time() - start:.2f} seconds.")
    print("Saved annotated video to /data/annotated.mp4")

if __name__ == "__main__":
    run()
