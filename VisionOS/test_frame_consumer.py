import cv2
import redis
import json
import time
import uuid

REDIS_URL = "redis://localhost:6379"
r = redis.from_url(REDIS_URL, decode_responses=False)

JOB_ID = f"test_{uuid.uuid4().hex[:6]}"
VIDEO_PATH = "https://media.roboflow.com/supervision/video-examples/people-walking.mp4"

# 1. Config for the AI
config = {
    "prompt": "person",
    "counting_type": "line",
    "line": [0, 50, 100, 50],
    "in_label": "IN",
    "out_label": "OUT",
    "model": "yolo",
    "confidence": 0.25,
    "detect_every": 1
}

print(f"=== 1. Chuẩn bị Video & Redis ===")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Không mở được video {VIDEO_PATH}")
    exit(1)

print(f"✅ Đang đẩy frame vào Redis với JOB_ID: {JOB_ID}")

# 2. Push frames
frame_count = 0
while frame_count < 150: # Test 150 frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Encode to JPEG
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        continue
    
    # Prepare message
    msg = {
        "job_id": JOB_ID,
        "frame_id": str(frame_count),
        "ts": str(time.time()),
        "image": buffer.tobytes()
    }
    
    # Send config on the first frame to auto-register
    if frame_count == 0:
        msg["config"] = json.dumps(config)
        
    r.xadd("visionos:frames:in", msg)
    frame_count += 1
    time.sleep(0.05) # Giả lập delay camera ~20fps

cap.release()
print(f"✅ Đã đẩy xong {frame_count} frames vào 'visionos:frames:in'")

print("\n=== 2. Đợi & Nhận Kết Quả từ AI ===")
time.sleep(15) # Chờ AI xử lý (lần đầu load YOLO model mất vài giây)

# Đọc kết quả từ VISIONOS_RESULTS
# Lấy các tin nhắn mới nhất
results = r.xrevrange("VISIONOS_RESULTS", max="+", min="-", count=20)
for msg_id, fields in reversed(results):
    job_id_out = fields.get(b"job_id", b"").decode("utf-8")
    if job_id_out == JOB_ID:
        event_type = fields.get(b"event_type", b"").decode("utf-8")
        payload = fields.get(b"payload", b"").decode("utf-8")
        print(f"[{event_type}] {payload}")

print("\n✅ Hoàn thành test!")
