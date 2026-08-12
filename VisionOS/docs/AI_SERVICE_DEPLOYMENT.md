# AI Service — Hướng Dẫn Triển Khai & Tích Hợp

AI Service chạy như **một tiến trình dài hạn duy nhất** (không phải 1 container/1 luồng), tự quản
lý nhiều luồng camera cùng lúc. Backend gán/gỡ camera qua Control API; kết quả phát hiện được đẩy
lên Redis Stream dạng JSON thuần — không có ảnh hay video trong output.

---

## Mục lục

1. [Kiến trúc tổng quan](#1-kiến-trúc-tổng-quan)
2. [Stack dịch vụ](#2-stack-dịch-vụ)
3. [Triển khai](#3-triển-khai)
4. [Cấu hình tham số ENV](#4-cấu-hình-tham-số-env)
5. [Luồng hoạt động chi tiết](#5-luồng-hoạt-động-chi-tiết)
6. [Control API](#6-control-api)
7. [Output — Redis Stream](#7-output--redis-stream)
8. [Ví dụ end-to-end](#8-ví-dụ-end-to-end)

---

## 1. Kiến trúc tổng quan

```
┌─────────────┐      rtsp://      ┌──────────────────────────────────────┐     XADD JSON    ┌───────────┐
│   MediaMTX  │ ────────────────► │            AI Service                │ ───────────────► │   Redis   │
│  :8554 RTSP │                   │  ┌──────────┐   ┌──────────────────┐ │                  │  Stream   │
└─────────────┘                   │  │FrameSource│   │  StreamWorker    │ │                  └───────────┘
                                  │  │  (thread) │──►│ YOLO → Track     │ │                       │
┌─────────────┐   POST/PATCH/DEL  │  └──────────┘   │ → Đếm → Publish  │ │                       │
│   Backend   │ ────────────────► │   /streams/{id}  └──────────────────┘ │                  XREAD / group
└─────────────┘   Control API     └──────────────────────────────────────┘                       │
                                                                                           ┌───────────────┐
                                                                                           │ Backend / App │
                                                                                           └───────────────┘
```

**Luồng dữ liệu:**

- **Backend → AI Service**: gọi `POST /streams/{id}` để giao việc; AI Service spawn một `StreamWorker`
  thread mới.
- **AI Service → MediaMTX**: `FrameSource` mở kết nối RTSP, đọc frame liên tục trong thread riêng.
- **AI Service → Redis**: mỗi lần xử lý frame xong, `StreamWorker` publish JSON vào Redis Stream.
- **Backend → Redis**: backend đọc kết quả bằng `XREAD` hoặc consumer group.

Qdrant được dùng song song để lưu embedding ngoại hình vật (ReID) — độc lập với luồng đếm chính.

---

## 2. Stack dịch vụ

| Dịch vụ | Image | Cổng | Vai trò |
|---|---|---|---|
| **api** (AI Service) | build: . | `8000` | Control API + pipeline AI |
| **redis** | `redis:7-alpine` | `6379` | Output stream (frame + track_event) |
| **mediamtx** | `bluenviron/mediamtx` | `8554` | RTSP server — nguồn video |
| **qdrant** | `qdrant/qdrant` | `6333` | Vector DB — lưu embedding ReID |

Tất cả dịch vụ nằm trong cùng Docker network (mặc định của Compose) để AI Service có thể kết nối
`rtsp://mediamtx:8554/...` bằng hostname nội bộ.

---

## 3. Triển khai

### 3.1 Build và khởi chạy

```bash
# Clone repo
git clone https://github.com/nguyendinhhuyht20032004-ai/VisionOS
cd VisionOS/VisionOS

# Build image AI Service
docker compose build

# Khởi chạy toàn bộ stack
docker compose up -d

# Kiểm tra log
docker compose logs -f api
```

Khi khởi động thành công, log sẽ in:

```
✅ Bật Stream Control API (Redis=redis://redis:6379)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3.2 Kiểm tra health

```bash
curl http://localhost:8000/healthz
# → {"status": "ok"}
```

### 3.3 Dừng stack

```bash
docker compose down          # giữ dữ liệu Qdrant
docker compose down -v       # xoá cả volume (reset hoàn toàn)
```

---

## 4. Cấu hình tham số ENV

Tất cả tham số được khai báo trong `docker-compose.yml` → mục `environment` của service `api`.
Thay đổi tại đó rồi chạy `docker compose up -d` để áp dụng — **không cần rebuild image**.

### 4.1 Tham số tích hợp (Control API & Redis)

| Biến | Bắt buộc | Mặc định | Ý nghĩa |
|---|:---:|---|---|
| `CONTROL_API_PORT` | ✅ | `8000` | Cổng expose Control API. Uvicorn dùng biến này làm port mặc định (CLI `--port` vẫn override được). |
| `REDIS_URL` | ✅ | `redis://redis:6379` | Địa chỉ Redis để publish kết quả. |
| `REDIS_STREAM_KEY` | ✅ | `VISIONOS_RESULTS` | Tên Redis Stream cho lệnh `XADD`. Backend đọc đúng key này. |
| `RTSP_TRANSPORT` | — | `tcp` | Giao thức truyền RTSP (`tcp` hoặc `udp`). TCP ổn định hơn qua NAT/firewall. |
| `RTSP_RECONNECT_INTERVAL_SEC` | — | `5` | Giây chờ giữa hai lần thử kết nối lại khi luồng RTSP bị rớt. |
| `OVERLAY_PUBLISH_FPS` | — | `10` | Số lần publish `frame` message tối đa mỗi giây. Giảm để tiết kiệm băng thông Redis. |
| `REDIS_STREAM_MAXLEN` | — | `1000` | Số bản ghi tối đa trong Redis Stream (XADD MAXLEN ~). Bản ghi cũ tự bị xoá. |

### 4.2 Tham số AI cơ bản (YOLO)

| Biến | Bắt buộc | Mặc định | Ý nghĩa |
|---|:---:|---|---|
| `YOLO_WEIGHTS` | — | `yolov8n.pt` | File trọng số YOLO. `yolov8n` (nano) cho CPU realtime; `yolov8s/m/x` cho GPU độ chính xác cao hơn. |
| `YOLO_IMGSZ` | — | `640` | Độ phân giải ảnh nạp vào YOLO (pixel). `640` cho CPU, `1280` cho GPU. Ảnh to → chính xác hơn nhưng chậm hơn. |
| `YOLO_CONF` | — | `0.3` | Ngưỡng tin cậy (0.0–1.0). Thấp → bắt được vật mờ/xa; cao → ít nhầm nhưng bỏ sót nhiều hơn. |
| `QDRANT_URL` | — | _(không set)_ | Địa chỉ Qdrant vector DB (`http://qdrant:6333`). Nếu không set, tính năng ReID/tìm kiếm vẫn tắt nhưng đếm vẫn chạy bình thường. |

### 4.3 Tham số AI nâng cao (tuning)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `YOLO_MAX_DET` | `1000` | Số vật thể tối đa YOLO được phép phát hiện trên 1 frame. |
| `YOLO_AUGMENT` | `0` | Đặt `1` để bật TTA (Test Time Augmentation) — lật/xoay ảnh để tăng chính xác, nhưng chậm hơn đáng kể. |
| `SMOOTHER_LEN` | `2` | Số frame dùng để làm mượt bounding box (rolling average). Tăng lên nếu box bị giật nhiều. |
| `REID_SIM` | `0.5` | Ngưỡng cosine similarity để ghép track (ReID). Tăng → ít ghép nhầm, giảm → ghép được nhiều hơn. |
| `REID_GAP` | `60` | Số frame tối đa được phép cách giữa hai lần xuất hiện của cùng một vật khi ReID. |
| `REID_DIST` | `0.3` | Khoảng cách tối đa (tỉ lệ chiều rộng frame) giữa vị trí cũ và mới để ReID chấp nhận ghép. |

### 4.4 Tham số SAHI (phát hiện vật nhỏ)

SAHI cắt frame thành nhiều ô nhỏ, mỗi ô soi riêng — giúp phát hiện người/xe ở rất xa. Chỉ nên bật
khi có GPU.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `YOLO_TILE` | `0` | Đặt `1` để bật SAHI. |
| `YOLO_TILE_WH` | `640` | Kích thước mỗi ô cắt (pixel). |
| `YOLO_TILE_OVERLAP` | `128` | Độ chồng lấp giữa các ô (pixel) — giúp vật thể sát mép ô vẫn được nhận diện. |
| `YOLO_DEDUP_IOU` | `0.8` | Ngưỡng IoU để loại bỏ box trùng lặp khi ghép kết quả các ô lại. |

---

## 5. Luồng hoạt động chi tiết

### 5.1 Khi nhận `POST /streams/{stream_id}`

```
Backend gọi POST /streams/cam-01
    │
    ▼
app.py:start_stream()
    │  Validate request (StreamControlRequest)
    │  Kiểm tra stream_id chưa tồn tại
    ▼
StreamManager.start()
    │  Tạo StreamWorker(stream_id, req, redis_client)
    │   ├─ Đọc ENV: RTSP_RECONNECT_INTERVAL_SEC, OVERLAY_PUBLISH_FPS,
    │   │           REDIS_STREAM_MAXLEN, REDIS_STREAM_KEY, SMOOTHER_LEN
    │   ├─ _build_counter(): xác định counting_type từ roi
    │   │    ├─ roi 2 điểm  → "line"  → LineZone
    │   │    ├─ roi >2 điểm → "zone"  → PolygonZone
    │   │    └─ không có roi → "fullscreen"
    │   └─ Khởi tạo StreamingCounter (ByteTrack + TrackStitcher + Smoother)
    │
    ▼
worker.start()  → Thread daemon bắt đầu chạy
    │
    ▼
Trả về: {"status": "success", "message": "Stream cam-01 started"}
```

### 5.2 Vòng lặp xử lý frame (StreamWorker.run)

```
StreamWorker.run()
    │
    ├─ _open_source()
    │    ├─ Đặt OPENCV_FFMPEG_CAPTURE_OPTIONS = rtsp_transport;<RTSP_TRANSPORT>
    │    └─ FrameSource(rtsp_url).start()
    │         └─ Thread đọc frame liên tục (cv2.VideoCapture)
    │              Tự kết nối lại sau RTSP_RECONNECT_INTERVAL_SEC nếu rớt
    │
    └─ LOOP (while self.running):
          │
          ├─ [Trường hợp fs=None] → sleep → _open_source() → tiếp tục
          │
          ├─ frame = fs.read()           ← frame BGR (H×W×3, uint8)
          │    ├─ [None, thread còn sống] → sleep 0.1s → tiếp tục
          │    └─ [None, thread chết]    → fs.stop() → sleep → _open_source()
          │
          ├─ counter.process(frame)      ← pipeline AI (xem §5.3)
          │
          └─ [nếu đã đủ interval = 1/OVERLAY_PUBLISH_FPS giây]
               └─ _publish_result()      ← đẩy lên Redis (xem §5.4)
```

### 5.3 Pipeline AI — `StreamingCounter.process(frame)`

```
frame BGR (H×W×3)
    │
    ├─ 1. YOLO detect
    │       model(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ)
    │       → raw boxes [N×4 xyxy], confidence[N], class_id[N]
    │
    ├─ 2. Chuyển sang supervision.Detections
    │       xyxy[N,4] · confidence[N] · class_id[N] · class_name[N]
    │
    ├─ 3. ByteTrack
    │       tracker.update_with_detections(det)
    │       → gán tracker_id[N] (Kalman filter + IoU matching)
    │          giữ track qua tối đa lost_track_buffer=120 frame
    │
    ├─ 4. TrackStitcher (ReID)
    │       crop từng vật → embed_crop() → histogram HSV 256-d
    │       → cosine similarity với gallery
    │       Ghép track_id cũ/mới nếu vượt ngưỡng REID_SIM
    │       (4 cổng an toàn: chưa gặp / cùng nhóm lớp / gần vị trí cũ / trong khoảng thời gian)
    │
    ├─ 5. DetectionsSmoother
    │       Rolling average vị trí box qua SMOOTHER_LEN frame → giảm giật
    │
    ├─ 6. Đếm theo chế độ
    │       line       → LineZone.trigger()  → in_count / out_count (luỹ kế)
    │       zone       → PolygonZone.trigger() → mặt nạ bool → đếm vật trong vùng (snapshot)
    │       fullscreen → len(det)            → số vật hiện tại trong frame (snapshot)
    │
    └─ 7. Annotate frame (không publish ảnh, chỉ lưu last_jpg cho /frame.jpg endpoint)
          Ghi last_det để _publish_result() đọc
```

### 5.4 Publish lên Redis — `_publish_result()`

```
_publish_result()
    │
    ├─ Duyệt last_det.tracker_id[]
    │    ├─ Mỗi track: tính bbox [x, y, w, h] từ xyxy
    │    ├─ [track mới] → _publish_track_event(tid, class, "start")
    │    └─ [track mất >2s] → _publish_track_event(tid, class, "end")
    │                          xoá khỏi _track_last_seen
    │
    ├─ Tạo frame_msg:
    │    {type, camera_id, stream_id, frame_timestamp (ms), boxes[]}
    │
    └─ redis.xadd(REDIS_STREAM_KEY,
                  {"data": json.dumps(frame_msg)},
                  maxlen=REDIS_STREAM_MAXLEN,
                  approximate=True)
```

---

## 6. Control API

Base URL: `http://<host>:CONTROL_API_PORT`

### `POST /streams/{stream_id}` — Bắt đầu theo dõi luồng

Tạo `StreamWorker` mới cho `stream_id`. Trả lỗi 400 nếu `stream_id` đã tồn tại.

**Request body:**

```json
{
  "camera_id": "cam-123",
  "rtsp_url":  "rtsp://mediamtx:8554/live/cam-123",
  "params": {
    "classes": ["person", "car"],
    "conf":    0.3,
    "roi":     [[10, 40], [90, 40]]
  }
}
```

| Trường | Kiểu | Bắt buộc | Mô tả |
|---|---|:---:|---|
| `camera_id` | string | ✅ | Định danh camera (dùng trong output Redis để phân biệt nguồn). |
| `rtsp_url` | string | ✅ | URL RTSP đầy đủ trỏ tới MediaMTX (phải trong cùng Docker network). |
| `params.classes` | string[] | — | Danh sách lớp cần theo dõi, VD `["person","car"]`. Nếu bỏ qua: theo dõi tất cả 80 lớp COCO. |
| `params.conf` | float | — | Ngưỡng confidence cho luồng này (override `YOLO_CONF`). |
| `params.roi` | number[][] | — | Toạ độ vùng quan tâm (% frame, 0–100). 2 điểm → **line**, >2 điểm → **zone**, bỏ qua → **fullscreen**. |

**Response:**

```json
{"status": "success", "message": "Stream cam-123 started"}
```

---

### `PATCH /streams/{stream_id}` — Cập nhật tham số

Áp dụng tham số mới cho luồng đang chạy, xây lại `StreamingCounter` — **không cần dừng/khởi động
lại**. Trả lỗi 404 nếu `stream_id` không tồn tại.

**Request body** (chỉ truyền những trường cần thay đổi):

```json
{
  "classes": ["vehicle"],
  "conf":    0.4,
  "roi":     [[5, 20], [60, 80], [95, 20]]
}
```

**Response:**

```json
{"status": "success", "message": "Stream cam-123 updated"}
```

---

### `DELETE /streams/{stream_id}` — Dừng luồng

Dừng `StreamWorker`, đóng `FrameSource`, giải phóng thread. Trả lỗi 404 nếu `stream_id` không tồn
tại.

**Response:**

```json
{"status": "success", "message": "Stream cam-123 stopped"}
```

---

## 7. Output — Redis Stream

AI Service publish vào stream `VISIONOS_RESULTS` (hoặc giá trị `REDIS_STREAM_KEY`).

Mỗi bản ghi Redis Stream có field `data` chứa chuỗi JSON. Đọc bằng:

```bash
# Đọc thô
redis-cli XREAD COUNT 10 STREAMS VISIONOS_RESULTS 0

# Đọc bản ghi mới nhất liên tục (block 5s)
redis-cli XREAD BLOCK 5000 COUNT 10 STREAMS VISIONOS_RESULTS $
```

### Message type: `frame`

Phát ra mỗi `1/OVERLAY_PUBLISH_FPS` giây cho mỗi luồng đang chạy.

```json
{
  "type":            "frame",
  "camera_id":       "cam-123",
  "stream_id":       "stream-456",
  "frame_timestamp": "2026-08-12T09:15:32.450Z",
  "boxes": [
    {
      "track_id":   "trk-7",
      "class":      "person",
      "confidence": 0.87,
      "bbox":       [142.5, 78.0, 64.0, 182.0]
    },
    {
      "track_id":   "trk-12",
      "class":      "car",
      "confidence": 0.91,
      "bbox":       [520.0, 300.0, 180.0, 110.0]
    }
  ]
}
```

| Trường | Kiểu | Mô tả |
|---|---|---|
| `camera_id` | string | Giống `camera_id` trong `POST /streams/…`. |
| `stream_id` | string | Giống `stream_id` trong `POST /streams/{stream_id}`. |
| `frame_timestamp` | string | Thời điểm publish (ISO 8601 UTC, độ chính xác mili giây). |
| `boxes[].track_id` | string | ID track duy nhất trong phiên làm việc của luồng này. |
| `boxes[].class` | string | Tên lớp vật thể (`"person"`, `"car"`, …). |
| `boxes[].confidence` | float | Độ tin cậy phát hiện (0.0–1.0). |
| `boxes[].bbox` | float[4] | Hộp bao `[x, y, w, h]` theo pixel của frame 960×540. |

### Message type: `track_event`

Phát ra khi track mới xuất hiện (`start`) hoặc biến mất quá 2 giây (`end`).

```json
{
  "type":      "track_event",
  "camera_id": "cam-123",
  "stream_id": "stream-456",
  "track_id":  "trk-7",
  "class":     "person",
  "event":     "start",
  "timestamp": "2026-08-12T09:15:30.210Z"
}
```

| Trường | Kiểu | Mô tả |
|---|---|---|
| `event` | string | `"start"` khi track mới xuất hiện · `"end"` khi track mất >2 giây. |
| `timestamp` | string | Thời điểm phát sự kiện (ISO 8601 UTC, độ chính xác mili giây). |

---

## 8. Ví dụ end-to-end

### Bước 1 — Khởi chạy stack

```bash
docker compose up -d
```

### Bước 2 — Gán camera vào AI Service

```bash
# Luồng fullscreen (không roi) — đếm tất cả người trong frame
curl -X POST http://localhost:8000/streams/hall-cam \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "hall-cam",
    "rtsp_url":  "rtsp://mediamtx:8554/live/hall",
    "params":    {"classes": ["person"], "conf": 0.3}
  }'

# Luồng line counting — đếm người qua vạch ngang ở giữa
curl -X POST http://localhost:8000/streams/gate-cam \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "gate-cam",
    "rtsp_url":  "rtsp://mediamtx:8554/live/gate",
    "params":    {"classes": ["person"], "roi": [[0, 50], [100, 50]]}
  }'
```

### Bước 3 — Đọc kết quả (Python)

```python
import redis, json

r = redis.from_url("redis://localhost:6379")
last_id = "$"   # chỉ đọc bản ghi MỚI từ thời điểm này

while True:
    msgs = r.xread({"VISIONOS_RESULTS": last_id}, block=5000, count=20)
    for _, entries in (msgs or []):
        for entry_id, fields in entries:
            last_id = entry_id
            msg = json.loads(fields[b"data"])

            if msg["type"] == "frame":
                cam = msg["camera_id"]
                n   = len(msg["boxes"])
                ts  = msg["frame_timestamp"]
                print(f"[{ts}] {cam}: {n} đối tượng")

            elif msg["type"] == "track_event":
                print(
                    f"  → Track {msg['track_id']} ({msg['class']}) "
                    f"[{msg['event'].upper()}] trên {msg['camera_id']}"
                )
```

### Bước 4 — Cập nhật tham số không cần dừng

```bash
# Đổi ngưỡng confidence và thêm lớp "car"
curl -X PATCH http://localhost:8000/streams/hall-cam \
  -H 'Content-Type: application/json' \
  -d '{"classes": ["person", "car"], "conf": 0.4}'
```

### Bước 5 — Dừng luồng

```bash
curl -X DELETE http://localhost:8000/streams/hall-cam
curl -X DELETE http://localhost:8000/streams/gate-cam
```
