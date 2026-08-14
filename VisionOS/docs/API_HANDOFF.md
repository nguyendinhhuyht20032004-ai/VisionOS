# Huong dan Tich hop AI Service (Danh cho Dev Frontend/Backend)

Tai lieu nay mo ta cach khoi chay va giao tiep voi **VisionOS AI Service**.
Tich hop chinh qua **Control API** (`/streams/`) + **Redis Stream**.

---

## 1. Khoi chay He thong

### Docker (production)

```bash
docker compose up --build
```

Ket qua:
- **AI Service**: `http://localhost:8000`
- **Redis**: `redis://localhost:6379`
- **MediaMTX (RTSP relay)**: `rtsp://localhost:8554`

### Native macOS (Apple Silicon)

Yeu cau: Redis chay local, model CoreML da export.

```bash
# Cai Redis
brew install redis && brew services start redis

# Export model CoreML (chi can 1 lan)
python export_coreml.py --weights yolov8m.pt --half

# Chay service
./run_native.sh
```

---

## 2. API Endpoints

### 2.1 Health check

```
GET /healthz
```
```json
{"status": "ok", "jobs": 0, "vectordb": "qdrant"}
```

### 2.2 Bat dau xu ly camera (Control API)

```
POST /streams/{stream_id}
Content-Type: application/json
```

```json
{
  "camera_id": "cam-01",
  "rtsp_url": "rtsp://admin:pass@192.168.1.10:554/stream",
  "params": {
    "classes": ["person"],
    "conf": 0.3,
    "detect_every": 3,
    "track_timeout": 2.0,
    "publish_fps": 12.0,
    "roi": null
  }
}
```

Response:
```json
{"status": "success", "message": "Stream entrance started"}
```

**`stream_id`**: chuoi bat ky do backend tu dat (VD: `"entrance"`, `"cam-01"`).

**`params.roi`**: 
- `null` hoac bo qua → fullscreen (dem tat ca trong khung)
- `[[x1,y1], [x2,y2]]` (2 diem) → line counting (dem qua vach)
- `[[x1,y1], [x2,y2], [x3,y3], ...]` (>2 diem) → zone counting (dem trong vung)

**`params.classes`**: danh sach lop can detect. VD: `["person"]`, `["car","truck","bus"]`, `["person","car"]`.

### 2.3 Cap nhat tham so (khong can dung camera)

```
PATCH /streams/{stream_id}
Content-Type: application/json
```

```json
{
  "conf": 0.25,
  "detect_every": 5,
  "publish_fps": 8.0
}
```

### 2.4 Xem danh sach stream dang chay

```
GET /streams
```

```json
{
  "status": "success",
  "count": 2,
  "streams": [
    {
      "stream_id": "entrance",
      "camera_id": "cam-01",
      "rtsp_url": "rtsp://...",
      "fps": 11.5,
      "params": {"classes": ["person"], "conf": 0.3, "detect_every": 3, "track_timeout": 2.0, "publish_fps": 12.0, "roi": null}
    }
  ]
}
```

### 2.5 Dung camera

```
DELETE /streams/{stream_id}
```

```json
{"status": "success", "message": "Stream entrance stopped"}
```

---

## 3. Doc ket qua tu Redis Stream

Ket qua AI duoc day len Redis Stream `VISIONOS_RESULTS` (cau hinh qua env `REDIS_STREAM_KEY`).

### Doc bang XREAD (blocking)

```python
import redis, json

r = redis.from_url("redis://localhost:6379", decode_responses=True)
last_id = "0-0"

while True:
    result = r.xread({stream_key: last_id}, block=5000, count=10)
    if not result:
        continue
    for stream_name, messages in result:
        for msg_id, fields in messages:
            last_id = msg_id
            data = json.loads(fields["data"])

            if data["type"] == "frame":
                boxes = data["boxes"]
                print(f"Camera {data['camera_id']}: {len(boxes)} objects")

            elif data["type"] == "track_event":
                print(f"Event: track {data['track_id']} {data['event']}")
```

### Cau truc message `type: "frame"`

```json
{
  "type": "frame",
  "camera_id": "cam-01",
  "stream_id": "entrance",
  "frame_timestamp": "2026-08-14T10:30:00.123Z",
  "resolution": {"width": 960, "height": 540},
  "boxes": [
    {
      "track_id": "5",
      "class": "person",
      "confidence": 0.87,
      "bbox": [120.5, 80.3, 60.0, 140.0],
      "velocity": [12.3, -5.1]
    }
  ]
}
```

- `bbox`: `[x, y, width, height]` (pixel, goc tren-trai)
- `velocity`: `[vx, vy]` (pixel/giay)

### Cau truc message `type: "track_event"`

```json
{
  "type": "track_event",
  "camera_id": "cam-01",
  "stream_id": "entrance",
  "track_id": "5",
  "class": "person",
  "event": "start",
  "timestamp": "2026-08-14T10:30:00.123Z"
}
```

Cac gia tri `event`:
| Event | Y nghia |
|---|---|
| `start` | Vat the moi xuat hien (track moi) |
| `end` | Mat dau qua `track_timeout` giay |
| `IN` | Di vao qua vach hoac vao vung |
| `OUT` | Di ra qua vach hoac ra khoi vung |

---

## 4. Legacy Jobs API (Web UI)

He thong van giu cac endpoint `/api/jobs/*` phuc vu cho trang web UI tich hop san (truy cap `GET /`).
Trang nay cho phep ve vach/vung tren canvas va xem MJPEG truc tiep.

| Endpoint | Mo ta |
|---|---|
| `GET /api/snapshot?source=...` | Lay 1 frame JPEG de ve |
| `POST /api/jobs` | Tao job dem (co MJPEG stream) |
| `GET /api/jobs/{id}` | Ket qua dem |
| `GET /api/jobs/{id}/mjpeg` | Luong video annotated |
| `POST /api/jobs/{id}/stop` | Dung job |

> **Luu y:** Doi voi tich hop backend, su dung Control API (`/streams/`) + Redis Stream thay vi Jobs API.
> Jobs API khong day ket qua len Redis.

---

## 5. Ghi chu quan trong

1. **Toa do ROI la pixel tuyet doi** (khong phai %). Lay frame tu camera de xac dinh toa do chinh xac.
2. **stream_id la unique.** Goi POST voi cung stream_id se cap nhat params neu URL khong doi, hoac restart neu URL thay doi.
3. **Redis Stream co gioi han.** Mac dinh `maxlen=1000` (env `REDIS_STREAM_MAXLEN`). Du lieu cu tu dong bi xoa.
4. **Backend can goi DELETE** khi nguoi dung tat camera de giai phong tai nguyen.
5. **Nhieu camera dong thoi:** Moi stream la 1 thread. Server 4 core xu ly duoc 4-8 camera (tuy model YOLO).
