import cv2
import json
import redis
import time
import threading
import supervision as sv
import numpy as np
import requests

# 1. Bật AI theo tên stream
import cv2
import json
import redis
import time
import threading
import supervision as sv
import numpy as np
import os

# Ép OpenCV dùng TCP thay vì UDP (vì Docker trên Mac hay lỗi drop gói tin UDP qua port mapping)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# 1. Mở luồng RTSP (video live đang phát qua MediaMTX)
rtsp_url = "rtsp://127.0.0.1:8554/test-people"
print(f"Đang kết nối luồng RTSP: {rtsp_url}")
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print(f"LỖI: Không thể mở luồng RTSP. Vui lòng chạy 'python3 test_people.py' trước!")
    exit(1)

# 2. Kết nối Redis
print("Kết nối Redis...")
r_cli = redis.Redis(host='localhost', port=6379, decode_responses=True)

# 3. Thread nhận dữ liệu từ Redis (không làm lag video)
latest_boxes = []
last_redis_time = time.time()
running = True

def redis_listener():
    global latest_boxes, last_redis_time
    last_id = '$'
    while running:
        messages = r_cli.xread({'VISIONOS_RESULTS': last_id}, count=10, block=100)
        if not messages:
            continue
        for stream_name, msgs in messages:
            for msg_id, msg_data in msgs:
                last_id = msg_id
                data = json.loads(msg_data['data'])
                # Lắng nghe stream-people từ test_people.py
                if data['type'] == 'frame' and data['stream_id'] == 'stream-people':
                    latest_boxes = data['boxes']
                    last_redis_time = time.time() # Cập nhật thời điểm nhận gói tin

threading.Thread(target=redis_listener, daemon=True).start()

# 4. Khởi tạo Supervision
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
trace_annotator = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK, trace_length=20)

print("Đang phát Video RTSP Live và Nội suy Toạ độ từ Redis... (Bấm 'q' để thoát)")

last_time = time.time()
frame_count = 0
fps_display = 0
unique_trackers = set()
current_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Mất kết nối RTSP, đang thử kết nối lại...")
        time.sleep(1)
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)
        continue

    # --- TÍNH FPS ---
    frame_count += 1
    if time.time() - last_time >= 1.0:
        fps_display = frame_count / (time.time() - last_time)
        frame_count = 0
        last_time = time.time()

    # --- NỘI SUY TOẠ ĐỘ (INTERPOLATION) ---
    delta_time = time.time() - last_redis_time
    # Không nội suy nếu gói tin quá cũ (> 0.5s)
    if delta_time > 0.5:
        delta_time = 0.0

    frame_h, frame_w = frame.shape[:2]
    scale_x = frame_w / 960.0
    scale_y = frame_h / 540.0

    xyxy_smooth = []
    xyxy_raw = []
    confidence = []
    tracker_id = []

    for box in latest_boxes:
        x, y, w, h = box['bbox']
        vx, vy = box.get('velocity', [0.0, 0.0])
        
        # Toạ độ gốc chưa nội suy (để vẽ Trace chính xác)
        rx1, ry1 = x * scale_x, y * scale_y
        rx2, ry2 = (x + w) * scale_x, (y + h) * scale_y
        xyxy_raw.append([rx1, ry1, rx2, ry2])
        
        # Toạ độ nội suy vận tốc (để vẽ Box mượt mà, bám dính lấy người)
        pred_x = x + (vx * delta_time)
        pred_y = y + (vy * delta_time)
        x1 = pred_x * scale_x
        y1 = pred_y * scale_y
        x2 = (pred_x + w) * scale_x
        y2 = (pred_y + h) * scale_y
        xyxy_smooth.append([x1, y1, x2, y2])
        
        confidence.append(box['confidence'])
        tracker_id.append(int(box['track_id']))

    if len(xyxy_smooth) > 0:
        # Detections mượt (Nội suy) dành cho Box và Label
        detections_smooth = sv.Detections(
            xyxy=np.array(xyxy_smooth),
            confidence=np.array(confidence),
            class_id=np.zeros(len(xyxy_smooth), dtype=int),
            tracker_id=np.array(tracker_id)
        )
        # Detections thật (Raw) dành cho Trace quỹ đạo (không bị zigzag)
        detections_raw = sv.Detections(
            xyxy=np.array(xyxy_raw),
            confidence=np.array(confidence),
            class_id=np.zeros(len(xyxy_raw), dtype=int),
            tracker_id=np.array(tracker_id)
        )
        
        labels = [f"#{t_id} {conf:.2f}" for t_id, conf in zip(detections_smooth.tracker_id, detections_smooth.confidence)]
        frame = box_annotator.annotate(scene=frame, detections=detections_smooth)
        frame = label_annotator.annotate(scene=frame, detections=detections_smooth, labels=labels)
        frame = trace_annotator.annotate(scene=frame, detections=detections_raw)
        
        for t_id in tracker_id:
            unique_trackers.add(t_id)
        current_count = len(xyxy_smooth)
    else:
        current_count = 0

    # Hiển thị FPS và Đếm
    cv2.putText(frame, f"Frontend Render: {fps_display:.1f} FPS", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
    cv2.putText(frame, f"Current Count: {current_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    cv2.putText(frame, f"Total Unique People: {len(unique_trackers)}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    frame_resized = cv2.resize(frame, (960, 540))
    cv2.imshow("Supervision + Velocity Interpolation", frame_resized)

    # RTSP stream tự pacing nên chỉ cần waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

running = False
cap.release()
cv2.destroyAllWindows()
