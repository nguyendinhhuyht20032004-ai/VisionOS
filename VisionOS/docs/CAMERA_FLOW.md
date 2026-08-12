# Luồng Camera: MediaMTX → AI Service → Redis → Backend

## Tổng quan

```
MediaMTX          AI Service                        Redis          Backend
   │                   │                               │               │
   │  rtsp://...        │                               │               │
   │◄──────────────────│ FrameSource.read()            │               │
   │  frame BGR        │                               │               │
   │──────────────────►│                               │               │
   │                   │  YOLO → ByteTrack → Đếm       │               │
   │                   │  _publish_result()            │               │
   │                   │──────────────────────────────►│               │
   │                   │  XADD VISIONOS_RESULTS        │               │
   │                   │                               │  XREAD        │
   │                   │                               │◄──────────────│
   │                   │                               │  {frame/event}│
   │                   │                               │──────────────►│
```

---

## Từng bước trong luồng

### 1. MediaMTX cung cấp RTSP

MediaMTX nhận video từ camera thật (hoặc file) và phát lại qua giao thức RTSP:

```
rtsp://mediamtx:8554/live/{tên-luồng}
```

AI Service kết nối vào địa chỉ này như một RTSP client bình thường.

---

### 2. AI Service đọc frame (FrameSource)

`FrameSource` chạy trong thread riêng, liên tục kéo frame từ MediaMTX:

```
FrameSource (thread riêng)
  │
  ├─ cv2.VideoCapture("rtsp://mediamtx:8554/live/cam-01")
  ├─ cap.read() → frame BGR (numpy array H×W×3)
  ├─ Lưu frame mới nhất vào bộ nhớ (có lock)
  └─ Tự kết nối lại nếu rớt (sau RTSP_RECONNECT_INTERVAL_SEC giây)
```

`StreamWorker` gọi `fs.read()` để lấy frame mới nhất bất cứ lúc nào cần, không bị chặn.

---

### 3. AI Service xử lý frame (Pipeline)

Với mỗi frame lấy được, `StreamWorker` chạy pipeline:

```
frame BGR
  │
  ├─ YOLO detect          → boxes [x1,y1,x2,y2], confidence, class_id
  ├─ ByteTrack            → gán tracker_id cho từng box (theo dõi liên frame)
  ├─ TrackStitcher (ReID) → ghép lại track_id bị đứt (vật đi khuất rồi quay lại)
  ├─ Smoother             → làm mượt vị trí box (rolling average)
  └─ Đếm (tùy chế độ)
       ├─ line       → LineZone.trigger()   → in_count / out_count
       ├─ zone       → PolygonZone.trigger() → số vật đang trong vùng
       └─ fullscreen → len(detections)      → số vật trong frame
```

---

### 4. AI Service publish lên Redis (XADD)

Sau khi xử lý, `_publish_result()` đẩy kết quả lên Redis Stream — **chỉ JSON, không có ảnh**:

**Mỗi frame** (tần suất giới hạn bởi `OVERLAY_PUBLISH_FPS`):

```json
{
  "type":            "frame",
  "camera_id":       "cam-01",
  "stream_id":       "hall-entrance",
  "frame_timestamp": "2026-08-12T09:15:32.450Z",
  "boxes": [
    { "track_id": "trk-3", "class": "person", "confidence": 0.87, "bbox": [120, 80, 60, 180] }
  ]
}
```

**Khi track xuất hiện / biến mất**:

```json
{
  "type":      "track_event",
  "camera_id": "cam-01",
  "stream_id": "hall-entrance",
  "track_id":  "trk-3",
  "class":     "person",
  "event":     "start",
  "timestamp": "2026-08-12T09:15:30.210Z"
}
```

> `event` chỉ có hai giá trị: `"start"` (track mới) hoặc `"end"` (track mất >2 giây).

---

### 5. Backend đọc từ Redis (XREAD)

Backend lắng nghe Redis Stream và xử lý kết quả theo nhu cầu:

```python
import redis, json

r = redis.from_url("redis://redis:6379")
last_id = "$"

while True:
    msgs = r.xread({"VISIONOS_RESULTS": last_id}, block=5000, count=20)
    for _, entries in (msgs or []):
        for entry_id, fields in entries:
            last_id = entry_id
            msg = json.loads(fields[b"data"])

            if msg["type"] == "frame":
                # Xử lý danh sách vật thể hiện tại
                for box in msg["boxes"]:
                    print(box["track_id"], box["class"], box["bbox"])

            elif msg["type"] == "track_event":
                # Xử lý sự kiện vào/ra
                print(msg["event"], msg["track_id"], msg["class"])
```

---

## Tóm tắt một luồng camera

```
POST /streams/hall-entrance
  body: { camera_id, rtsp_url, params: { classes, conf, roi } }
  → AI Service spawn 1 thread cho luồng này

Thread chạy mãi:
  FrameSource.read() → frame BGR
  → YOLO + ByteTrack + ReID + Smooth + Đếm
  → XADD VISIONOS_RESULTS {"data": "{...json...}"}

DELETE /streams/hall-entrance
  → Thread dừng, FrameSource đóng
```

Mỗi camera tương ứng đúng một `stream_id`, một thread, một chuỗi message trên Redis.
