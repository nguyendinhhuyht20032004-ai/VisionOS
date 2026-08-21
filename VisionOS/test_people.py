import os
import time
import urllib.request
import subprocess
import requests
import redis
import json

# Lấy đường dẫn FFmpeg Native
try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    print("Vui lòng cài đặt: pip3 install imageio-ffmpeg")
    exit(1)

# 1. Download sample video if not exists
os.makedirs("data", exist_ok=True)
video_path = "data/people-walking.mp4"
video_url = "https://media.roboflow.com/supervision/video-examples/people-walking.mp4"
if not os.path.exists(video_path):
    print(f"Downloading {video_url}...")
    urllib.request.urlretrieve(video_url, video_path)
    print("Downloaded.")

# 2. Push video to MediaMTX via RTSP using Native FFmpeg
print("Starting FFmpeg NATIVE to stream to MediaMTX...")
ffmpeg_proc = subprocess.Popen([
    FFMPEG_BIN, "-re", "-stream_loop", "-1", "-i", video_path, "-c", "copy", "-f", "rtsp", "rtsp://localhost:8554/test-people"
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3) # Wait for stream to be ready

# 3. Call API to start tracking (NATIVE API chạy port 8001)
print("Calling Native API (8001) to start AI tracking...")
url = "http://localhost:8001/streams/stream-people"
payload = {
    "camera_id": "cam-people",
    "rtsp_url": "rtsp://localhost:8554/test-people",
    "params": {
        "conf": 0.3,
        "classes": ["person"],
        "roi": [[50.0, 0.0], [50.0, 100.0]] # Đặt vạch dọc ở giữa màn hình (x=50%)
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
    found_frame = False
    found_track = False

    print("\nReading from Redis Stream (VISIONOS_RESULTS)... (Giữ Terminal này mở để duy trì RTSP, nhấn Ctrl+C để thoát)")
    while time.time() - start_time < 600:
        streams = r.xread({'VISIONOS_RESULTS': last_id}, count=100, block=2000)
        if streams:
            for stream_name, messages in streams:
                for msg_id, msg in messages:
                    last_id = msg_id
                    data_str = msg.get('data')
                    if data_str:
                        data = json.loads(data_str)
                        if data.get("stream_id") == "stream-people":
                            msg_type = data.get("type")
                            
                            # Print only the first frame event to avoid spam
                            if msg_type == "frame" and not found_frame:
                                boxes = data.get('boxes', [])
                                if len(boxes) > 0:
                                    print(f"\n--- [FRAME EVENT] Redis Message {msg_id} ---")
                                    print(json.dumps(data, indent=2))
                                    found_frame = True
                            
                            # Print the track event when a track starts/ends
                            if msg_type == "track_event" and not found_track:
                                print(f"\n--- [TRACK EVENT] Redis Message {msg_id} ---")
                                print(json.dumps(data, indent=2))
                                found_track = True

    if not found_frame:
        print("No 'frame' messages with detections found in Redis stream.")
    if not found_track:
        print("No 'track_event' messages found in Redis stream.")
except Exception as e:
    print("Error reading from redis:", e)

# Clean up
print("\nStopping stream in API...")
try:
    requests.delete("http://localhost:8001/streams/stream-people")
except:
    pass
ffmpeg_proc.kill()
print("Done.")
