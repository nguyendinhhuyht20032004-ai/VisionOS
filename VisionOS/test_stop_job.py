import time
import json
import redis
import threading
import requests

REDIS_URL = "redis://localhost:6379"
r = redis.from_url(REDIS_URL, decode_responses=True)
r_bytes = redis.from_url(REDIS_URL, decode_responses=False)

JOB_ID = "test_stop_001"
STREAM_IN = "visionos:frames:in"
STREAM_OUT = "VISIONOS_RESULTS"

# Dummy 1x1 black JPEG for testing
dummy_jpeg = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t\n\x0c\x0f\x0c\n\x0b\x0e\x0b\t\t\r\x11\r\x0e\x0f\x10\x10\x11\x10\n\x0c\x12\x13\x12\x10\x13\x0f\x10\x10\x10\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x12\x7f\xff\xd9'

config = {
    "prompt": "person",
    "counting_type": "line",
    "line": [0, 50, 100, 50],
    "resolution": [640, 480],
    "detect_every": 1
}

def push_frames():
    print(f"[Pusher] Đang đẩy frame cho job {JOB_ID}...")
    for i in range(1, 20):
        fields = {
            b"job_id": JOB_ID.encode(),
            b"frame_id": str(i).encode(),
            b"image": dummy_jpeg,
            b"ts": str(time.time()).encode()
        }
        if i == 1:
            fields[b"config"] = json.dumps(config).encode()
            
        r_bytes.xadd(STREAM_IN, fields)
        print(f"[Pusher] Đã gửi frame {i}")
        time.sleep(0.5)

threading.Thread(target=push_frames, daemon=True).start()

print("[Main] Chờ 3 giây cho AI xử lý vài frame đầu...")
time.sleep(3)

print("\n===========================================")
print(f"[Main] 🛑 Bắn API TẮT luồng {JOB_ID} !!!")
print("===========================================\n")
res = requests.delete(f"http://localhost:8000/streams/{JOB_ID}")
print("[Main] Phản hồi từ API:", res.json())

print("\n[Main] Pusher vẫn đang tiếp tục gửi frame (giống như camera vẫn chạy).")
print("[Main] Bạn sẽ thấy AI *không* xử lý các frame sau này nữa.\n")

time.sleep(7)
print("✅ Test xong!")
