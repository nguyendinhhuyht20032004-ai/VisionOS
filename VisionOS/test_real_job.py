import requests
import time
import json
import urllib.request
import os
import subprocess

# 1. Prepare video and ffmpeg to send to mediamtx
os.makedirs("data", exist_ok=True)
video_path = "data/street.mp4"
if not os.path.exists(video_path):
    print("Downloading sample video...")
    urllib.request.urlretrieve("https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4", video_path)

print("Starting FFmpeg to stream to MediaMTX...")
ffmpeg_proc = subprocess.Popen([
    "docker", "exec", "visionos-api-1", "bash", "-c",
    "ffmpeg -re -stream_loop -1 -i /data/street.mp4 -c copy -f rtsp rtsp://mediamtx:8554/test-stream"
])
time.sleep(3)

# 2. Call real API
print("=== 1. Gọi API tạo Job thật (POST /api/jobs) ===")
payload = {
    "source": "rtsp://mediamtx:8554/test-stream",
    "prompt": "person",
    "counting_type": "line",
    "line": [0, 50, 100, 50],
    "record_events": False
}
resp = requests.post("http://localhost:8000/api/jobs", json=payload)
if resp.status_code != 200:
    print("Lỗi tạo job:", resp.text)
    exit(1)

job_data = resp.json()
job_id = job_data["id"]
print(f"✅ Đã tạo Job ID: {job_id}")

print("\n=== 2. Gọi API lấy dữ liệu Realtime (GET /api/jobs/{job_id}) sau 3 giây ===")
time.sleep(3)
resp = requests.get(f"http://localhost:8000/api/jobs/{job_id}")
real_stats = resp.json()

print(f"✅ Dữ liệu đếm thật sự của hệ thống AI:\n{json.dumps(real_stats, indent=2, ensure_ascii=False)}")

print("\n=== 3. Dừng Job ===")
requests.post(f"http://localhost:8000/api/jobs/{job_id}/stop")
ffmpeg_proc.kill()
subprocess.run(["docker", "exec", "visionos-api-1", "pkill", "ffmpeg"], capture_output=True)
print("✅ Đã dừng job.")
