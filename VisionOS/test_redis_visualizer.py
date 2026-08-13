import cv2
import json
import time
import redis
import requests
import sys

API_URL = "http://localhost:8000/streams/cam-test-redis"
VIDEO_PATH = "data/people-walking.mp4"

# 1. Gọi API bật AI
payload = {
  "camera_id": "cam-test-redis",
  "rtsp_url": "/data/people-walking.mp4", # Path bên trong Docker Container
  "params": {
    "classes": ["person"],
    "publish_fps": 30.0,   # Ép AI nhả toạ độ cực nhanh
    "detect_every": 3
  }
}
print("Bật AI Service...")
requests.post(API_URL, json=payload)

# 2. Kết nối Redis
print("Kết nối Redis...")
try:
    r_cli = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    r_cli.ping()
except Exception:
    print("❌ Không kết nối được Redis ở localhost:6379. Nhớ bật Docker lên nhé!")
    sys.exit(1)

# 3. Đọc Video bằng OpenCV
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Không mở được video {VIDEO_PATH}")
    sys.exit(1)

last_id = '$'
import supervision as sv
import numpy as np

box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
trace_annotator = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK, trace_length=20)

print("Đang phát Video (đồng bộ với API) và bọc Toạ độ từ Redis... (Bấm 'q' để thoát)")

last_time = time.time()
frame_count = 0
fps = 0

while True:
    # --- CHỜ TOẠ ĐỘ TỪ REDIS ĐỂ ĐỒNG BỘ HOÀN HẢO ---
    # Block vô hạn cho đến khi API nhả frame mới
    messages = r_cli.xread({'VISIONOS_RESULTS': last_id}, count=100, block=5000)
    if not messages:
        continue # Chờ tiếp
        
    for stream_name, msgs in messages:
        for msg_id, msg_data in msgs:
            last_id = msg_id
            data = json.loads(msg_data['data'])
            
            if data['type'] == 'frame' and data['stream_id'] == 'cam-test-redis':
                latest_boxes = data['boxes']
                
                # CHỈ ĐỌC 1 FRAME VIDEO KHI CÓ 1 FRAME TỪ REDIS (Giúp xoá bỏ độ trễ)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # --- TÍNH FPS ---
                frame_count += 1
                if time.time() - last_time >= 1.0:
                    fps = frame_count / (time.time() - last_time)
                    frame_count = 0
                    last_time = time.time()
                
                # --- CHUYỂN ĐỔI SANG SUPERVISION (Đẹp & Mượt) ---
                frame_h, frame_w = frame.shape[:2]
                scale_x = frame_w / 960.0
                scale_y = frame_h / 540.0
                
                xyxy = []
                confidence = []
                tracker_id = []
                
                for box in latest_boxes:
                    x, y, w, h = box['bbox']
                    x1, y1, x2, y2 = x*scale_x, y*scale_y, (x+w)*scale_x, (y+h)*scale_y
                    xyxy.append([x1, y1, x2, y2])
                    confidence.append(box['confidence'])
                    tracker_id.append(int(box['track_id']))
                
                if len(xyxy) > 0:
                    detections = sv.Detections(
                        xyxy=np.array(xyxy),
                        confidence=np.array(confidence),
                        class_id=np.zeros(len(xyxy), dtype=int),
                        tracker_id=np.array(tracker_id)
                    )
                    
                    labels = [f"#{t_id} person {conf:.2f}" for t_id, conf in zip(detections.tracker_id, detections.confidence)]
                    
                    frame = box_annotator.annotate(scene=frame, detections=detections)
                    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                    frame = trace_annotator.annotate(scene=frame, detections=detections)
                
                # Hiển thị FPS
                cv2.putText(frame, f"API Processing: {fps:.1f} FPS", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # Hiển thị
                frame_resized = cv2.resize(frame, (960, 540))
                cv2.imshow("Supervision Redis Visualizer", frame_resized)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            continue
        break
    else:
        continue
    break

cap.release()
cv2.destroyAllWindows()

# 4. Tắt AI
print("Tắt AI Service...")
requests.delete(API_URL)
