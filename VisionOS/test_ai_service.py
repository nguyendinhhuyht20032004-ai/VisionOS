import os
import time
import urllib.request
import subprocess
import requests
import redis
import json

# 1. Download sample video if not exists
os.makedirs("data", exist_ok=True)
video_path = "data/street.mp4"
if not os.path.exists(video_path):
    print("Downloading sample video...")
    urllib.request.urlretrieve("https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4", video_path)
    print("Downloaded.")

# 2. Push video to MediaMTX via RTSP using ffmpeg in the visionos-api container
print("Starting FFmpeg to stream to MediaMTX...")
ffmpeg_proc = subprocess.Popen([
    "docker", "exec", "visionos-api-1", "bash", "-c",
    "ffmpeg -re -stream_loop -1 -i /data/street.mp4 -c copy -f rtsp rtsp://mediamtx:8554/test-stream"
], )
time.sleep(3) # Wait for stream to be ready

# 3. Call API to start tracking
print("Calling API to start AI tracking...")
url = "http://localhost:8000/streams/stream-123"
payload = {
    "camera_id": "cam-1",
    "rtsp_url": "rtsp://mediamtx:8554/test-stream",
    "params": {
        "conf": 0.3
    }
}
resp = requests.post(url, json=payload)
print("API Response:", resp.text)

# 4. Read from Redis Stream to verify output
print("\nReading from Redis Stream (VISIONOS_RESULTS)...")
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
try:
    last_id = '0'
    start_time = time.time()
    found_boxes = False

    while time.time() - start_time < 30:
        streams = r.xread({'VISIONOS_RESULTS': last_id}, count=100, block=2000)
        if streams:
            for stream_name, messages in streams:
                for msg_id, msg in messages:
                    last_id = msg_id
                    data_str = msg.get('data')
                    if data_str:
                        data = json.loads(data_str)
                        boxes = data.get('boxes', [])
                        if len(boxes) > 0:
                            print(f"\n--- Redis Message {msg_id} ---")
                            print(f"data: {data}")
                            found_boxes = True
                            break
                if found_boxes:
                    break
        if found_boxes:
            break

    if not found_boxes:
        print("No messages with detections found in Redis stream.")
except Exception as e:
    print("Error reading from redis:", e)

# Clean up
print("\nStopping stream in API...")
requests.delete("http://localhost:8000/streams/stream-123")
ffmpeg_proc.kill()
subprocess.run(["docker", "exec", "visionos-api-1", "pkill", "ffmpeg"], capture_output=True)
print("Done.")
