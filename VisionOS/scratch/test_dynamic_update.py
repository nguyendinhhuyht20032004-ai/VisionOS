import requests
import time

url = "http://localhost:8000/streams/stream-test"
payload = {
    "camera_id": "cam-test",
    "rtsp_url": "rtsp://mediamtx:8554/test-people",
    "params": {
        "detect_every": 2,
        "publish_fps": 12.0
    }
}

print("==================================================")
print("1. TẠO MỚI LUỒNG LẦN ĐẦU")
print("==================================================")
r1 = requests.post(url, json=payload)
print(f"Status Code: {r1.status_code}")
print(f"Response: {r1.json()}")

print("\n(Đợi 3 giây để hệ thống nhận diện AI...)")
time.sleep(3)

print("\n==================================================")
print("2. CẬP NHẬT ĐỘNG (DYNAMIC UPDATE)")
print("==================================================")
# Thay đổi cấu hình (nhưng GIỮ NGUYÊN rtsp_url để kích hoạt cập nhật động)
payload["params"]["detect_every"] = 5  # Bắt AI tính toán thưa hơn
payload["params"]["publish_fps"] = 24.0 # Gửi toạ độ nhiều hơn (công nghệ nội suy)
print(f"-> Gửi lại Payload: {payload['params']}")

r2 = requests.post(url, json=payload)
print(f"Status Code: {r2.status_code}")
print(f"Response: {r2.json()}")

print("\n(Đợi 3 giây xem kết quả...)")
time.sleep(3)

print("\n==================================================")
print("3. (BỎ QUA XOÁ LUỒNG) - Luồng vẫn đang chạy ngầm!")
print("==================================================")
# r3 = requests.delete(url)
# print(f"Status Code: {r3.status_code}")
# print(f"Response: {r3.json()}")
print("Hoàn tất! Bây giờ bạn hãy gõ lệnh curl để kiểm tra danh sách luồng nhé!")
