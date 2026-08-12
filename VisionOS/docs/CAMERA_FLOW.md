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

## Tham số và kết nối

Luồng có **3 nhóm tham số** tương ứng với **3 điểm kết nối**:

```
┌───────────────────────────────────────────────────────────────────────┐
│  NHÓM 1: Backend → AI Service (Control API)                          │
│                                                                       │
│  POST http://<AI_SERVICE_HOST>:CONTROL_API_PORT/streams/{stream_id}  │
│  body: {                                                             │
│    "camera_id": "cam-01",          ← tên camera (xuất hiện trong    │
│                                       mọi message Redis)            │
│    "rtsp_url":  "rtsp://...",      ─────────────────────────────┐   │
│    "params": {                                                   │   │
│      "classes": ["person","car"],  ← lọc lớp vật thể (YOLO)    │   │
│      "conf":    0.3,               ← ngưỡng confidence (YOLO)  │   │
│      "roi":     [[x,y],[x,y]]     ← vùng/đường đếm             │   │
│    }                                                             │   │
│  }                                                               │   │
└──────────────────────────────────────────────────────────────────│───┘
                                                                   │
                              rtsp_url dùng để kết nối tới         │
                                                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  NHÓM 2: AI Service → MediaMTX (RTSP)                                │
│                                                                       │
│  ENV: RTSP_TRANSPORT=tcp           ← giao thức (tcp/udp)            │
│  ENV: RTSP_RECONNECT_INTERVAL_SEC=5 ← chờ bao lâu trước khi retry  │
│                                                                       │
│  cv2.VideoCapture("rtsp://mediamtx:8554/live/cam-01")               │
│       hostname "mediamtx" ────────► dịch vụ MediaMTX trong Compose  │
│       port     "8554"     ────────► cổng RTSP mặc định              │
│       path     "/live/cam-01" ────► tên luồng trên MediaMTX         │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│  NHÓM 3: AI Service → Redis (output) / Backend → Redis (đọc)         │
│                                                                       │
│  ENV: REDIS_URL=redis://redis:6379  ← AI Service kết nối tới Redis  │
│  ENV: REDIS_STREAM_KEY=VISIONOS_RESULTS ← tên stream để XADD        │
│  ENV: REDIS_STREAM_MAXLEN=1000      ← giới hạn số bản ghi           │
│  ENV: OVERLAY_PUBLISH_FPS=10        ← tần suất publish (msg/giây)   │
│                                                                       │
│  Backend đọc:                                                        │
│    XREAD STREAMS VISIONOS_RESULTS $ ← phải khớp với REDIS_STREAM_KEY│
│    kết nối: redis://redis:6379      ← phải khớp với REDIS_URL       │
└───────────────────────────────────────────────────────────────────────┘
```

### Bảng tham số đầy đủ theo điểm kết nối

| Tham số | Loại | Truyền ở đâu | Kết nối tới | Ý nghĩa |
|---|---|---|---|---|
| `stream_id` | URL path | `POST /streams/{stream_id}` | AI Service (nội bộ) | Khoá định danh luồng, dùng để PATCH/DELETE sau |
| `camera_id` | body | `POST /streams/…` | Redis output | Xuất hiện trong mọi message, backend dùng để phân biệt camera |
| `rtsp_url` | body | `POST /streams/…` | **MediaMTX** | Địa chỉ RTSP AI Service sẽ kết nối vào để đọc video |
| `params.classes` | body | `POST /streams/…` | YOLO (nội bộ) | Lọc chỉ phát hiện các lớp này. Bỏ qua = tất cả 80 lớp COCO |
| `params.conf` | body | `POST /streams/…` | YOLO (nội bộ) | Ngưỡng độ tin cậy tối thiểu (0.0–1.0) |
| `params.roi` | body | `POST /streams/…` | Counting zone | 2 điểm=line, 3+điểm=zone, bỏ qua=fullscreen. % frame (0–100) |
| `CONTROL_API_PORT` | ENV docker-compose | AI Service | Backend (inbound) | Cổng backend gọi Control API vào |
| `REDIS_URL` | ENV docker-compose | AI Service | **Redis** | Địa chỉ Redis AI Service kết nối để XADD kết quả |
| `REDIS_STREAM_KEY` | ENV docker-compose | AI Service + Backend | **Redis Stream** | Tên stream — AI Service XADD vào đây, backend XREAD từ đây |
| `RTSP_TRANSPORT` | ENV docker-compose | AI Service | **MediaMTX** | Giao thức RTSP (`tcp` hoặc `udp`) |
| `RTSP_RECONNECT_INTERVAL_SEC` | ENV docker-compose | AI Service | MediaMTX | Giây chờ trước khi thử kết nối lại khi RTSP rớt |
| `OVERLAY_PUBLISH_FPS` | ENV docker-compose | AI Service | Redis | Giới hạn tần suất XADD frame message (msg/giây) |
| `REDIS_STREAM_MAXLEN` | ENV docker-compose | AI Service | Redis | Số bản ghi tối đa trong stream — cũ tự xoá |

### Ví dụ giá trị cụ thể

```
Backend gọi:
  POST http://localhost:8000/streams/entrance-north
  {
    "camera_id": "cam-entrance",
    "rtsp_url":  "rtsp://mediamtx:8554/live/entrance",   ← MediaMTX hostname nội bộ
    "params":    { "classes": ["person"], "conf": 0.3, "roi": [[0,50],[100,50]] }
  }

AI Service kết nối:
  RTSP  → rtsp://mediamtx:8554/live/entrance   (từ rtsp_url trong request)
  Redis → redis://redis:6379                   (từ ENV REDIS_URL)
  Ghi   → XADD VISIONOS_RESULTS ...           (từ ENV REDIS_STREAM_KEY)

Backend đọc:
  XREAD STREAMS VISIONOS_RESULTS $             (cùng key với REDIS_STREAM_KEY)
  kết nối redis://localhost:6379
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
