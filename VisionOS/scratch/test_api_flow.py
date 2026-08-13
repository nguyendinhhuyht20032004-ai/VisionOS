import requests
import time
import json

BASE_URL = "http://localhost:8000"
STREAM_ID = "stream-api-flow-test"
URL = f"{BASE_URL}/streams/{STREAM_ID}"

def print_json(title, data):
    print(f"\n[LOG] === {title} ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("=" * 40)

# Dữ liệu luồng ban đầu
payload = {
    "camera_id": "cam-101",
    "rtsp_url": "rtsp://mediamtx:8554/test-people",
    "params": {
        "detect_every": 2,
        "publish_fps": 12.0
    }
}

print("🚀 BẮT ĐẦU TEST KỊCH BẢN LUỒNG API...")

# 1. Start Stream
print_json("1. Khởi tạo luồng (POST)", payload)
r1 = requests.post(URL, json=payload)
print_json("Kết quả Tạo mới", r1.json())
time.sleep(2) # Chờ API boot luồng

# 2. Get Streams (Check danh sách luồng)
print("\n[LOG] 2. Lấy danh sách luồng hiện tại (GET /streams)")
r2 = requests.get(f"{BASE_URL}/streams")
print_json("Danh sách luồng", r2.json())
time.sleep(1)

# 3. Dynamic Update Stream (Cập nhật thông số)
payload["params"]["detect_every"] = 10 # Thay đổi AI chạy thưa hơn
payload["params"]["publish_fps"] = 30.0 # Tăng tốc độ mượt của video
print_json("3. Cập nhật động cấu hình (POST)", payload)
r3 = requests.post(URL, json=payload)
print_json("Kết quả Cập nhật", r3.json())
time.sleep(2) # Chờ cập nhật

# 4. Get Streams Again (Check xem thông số đã đổi chưa)
print("\n[LOG] 4. Lấy lại danh sách luồng để kiểm tra thông số (GET /streams)")
r4 = requests.get(f"{BASE_URL}/streams")
print_json("Danh sách luồng sau khi Cập nhật", r4.json())
time.sleep(1)

# 5. Xoá luồng dọn dẹp
print("\n[LOG] 5. Xoá luồng để dọn dẹp (DELETE)")
r5 = requests.delete(URL)
print_json("Kết quả Xoá", r5.json())

print("\n✅ HOÀN TẤT BÀI TEST TOÀN DIỆN!")
