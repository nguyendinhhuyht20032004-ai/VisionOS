import cv2
import json
import time
import redis
import requests
import sys

API_URL = "http://localhost:8000/streams/cam-test-redis"
VIDEO_PATH = "data/vecteezy_traffic.mov"

# 1. Gọi API bật AI
payload = {
  "camera_id": "cam-test-redis",
  "rtsp_url": "/data/vecteezy_traffic.mov", # Path bên trong Docker Container
  "params": {
    "classes": ["car", "vehicle"],
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
latest_boxes = []

print("Đang phát Video và bọc Toạ độ từ Redis... (Bấm 'q' để thoát)")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ép video chạy đúng tốc độ thực tế (~30fps)
    time.sleep(0.033) 
    
    # --- KÉO TOẠ ĐỘ TỪ REDIS VỀ ---
    # block=1 (1ms) để không bị nghẽn video
    messages = r_cli.xread({'VISIONOS_RESULTS': last_id}, count=100, block=1)
    if messages:
        for stream_name, msgs in messages:
            for msg_id, msg_data in msgs:
                last_id = msg_id
                data = json.loads(msg_data['data'])
                print("\n=== NHẬN ĐƯỢC TỪ REDIS ===")
                print(json.dumps(data, indent=2, ensure_ascii=False))
                if data['type'] == 'frame' and data['stream_id'] == 'cam-test-redis':
                    latest_boxes = data['boxes']

    # --- VẼ HÌNH VUÔNG LÊN VIDEO GỐC ---
    # Phải scale toạ độ vì AI trả về trên gốc 960x540
    frame_h, frame_w = frame.shape[:2]
    scale_x = frame_w / 960.0
    scale_y = frame_h / 540.0

    for box in latest_boxes:
        x1, y1, w, h = box['bbox']
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int((x1+w) * scale_x), int((y1+h) * scale_y)
        
        # Vẽ hộp
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Vẽ ID & Class
        label = f"#{box['track_id']} {box['class']}"
        cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hiển thị
    frame_resized = cv2.resize(frame, (960, 540)) # Thu nhỏ lại xem cho vừa màn hình
    cv2.imshow("Redis Visualizer Test", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 4. Tắt AI
print("Tắt AI Service...")
requests.delete(API_URL)
