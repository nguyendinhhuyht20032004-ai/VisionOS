import cv2
import time
from recognition.service.engine import StreamingCounter
from recognition.service.builder import get_detector, make_scenario

def run():
    print("Loading video...")
    cap = cv2.VideoCapture('data/people-walking.mp4')
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video resolution: {width}x{height} @ {fps}fps")

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('/Users/builder4/.gemini/antigravity-ide/brain/4615d71e-3bdd-4ed6-9b85-c1e5f8cab4fb/annotated.mp4', fourcc, fps, (width, height))

    print("Initializing AI Engine (YOLO + ByteTrack)...")
    scenario, _ = make_scenario(prompt="person", counting_type="fullscreen", resolution=(width, height))
    detector = get_detector("yolov8", confidence=0.2) # Giảm confidence để giống Web
    counter = StreamingCounter(scenario, detector, resolution=(width, height))

    count = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process and get annotated frame
        annotated = counter.process(frame)
        
        # Hiển thị trực tiếp lên màn hình
        cv2.imshow("Local Visual Test (YOLOv8m)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        count += 1
        if count % 30 == 0:
            print(f"Processed {count} frames...")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done processing {count} frames in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    run()
