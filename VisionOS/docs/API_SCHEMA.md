# API Schema — VisionOS AI Service

---

## 1. Control API — `/streams/`

### POST /streams/{stream_id} — Bat dau / cap nhat stream

**Request body:**
```json
{
  "camera_id": "string",
  "rtsp_url": "string",
  "params": {
    "roi": null | [[x1,y1],[x2,y2]] | [[x1,y1],[x2,y2],[x3,y3],...],
    "conf": 0.3,
    "classes": ["person"],
    "detect_every": 3,
    "track_timeout": 2.0,
    "publish_fps": 12.0
  }
}
```

| Field | Type | Required | Default | Mo ta |
|---|---|---|---|---|
| `camera_id` | string | yes | — | ID camera (dinh danh tu backend) |
| `rtsp_url` | string | yes | — | URL RTSP toi camera / MediaMTX |
| `params.roi` | list | no | null | null = fullscreen, 2 diem = line, >2 diem = zone |
| `params.conf` | float | no | 0.3 | YOLO confidence threshold |
| `params.classes` | list | no | ["person"] | Cac lop can detect |
| `params.detect_every` | int | no | 3 | Chay YOLO moi N frame |
| `params.track_timeout` | float | no | 2.0 | Giay mat dau truoc khi ban event `end` |
| `params.publish_fps` | float | no | 12.0 | Tan so gui ket qua len Redis |

**Response:**
```json
{"status": "success", "message": "Stream {stream_id} started"}
```

### PATCH /streams/{stream_id} — Cap nhat tham so

**Request body:** Bat ky field nao cua `params` (VD: `{"conf": 0.25, "detect_every": 5}`)

**Response:**
```json
{"status": "success", "message": "Stream {stream_id} updated"}
```

### GET /streams — Danh sach stream

**Response:**
```json
{
  "status": "success",
  "count": 1,
  "streams": [
    {
      "stream_id": "string",
      "camera_id": "string",
      "rtsp_url": "string",
      "fps": 11.5,
      "params": {
        "roi": null,
        "conf": 0.3,
        "classes": ["person"],
        "detect_every": 3,
        "track_timeout": 2.0,
        "publish_fps": 12.0
      }
    }
  ]
}
```

### DELETE /streams/{stream_id} — Dung stream

**Response:**
```json
{"status": "success", "message": "Stream {stream_id} stopped"}
```

**Error 404:**
```json
{"detail": "Stream not found"}
```

---

## 2. Redis Stream output

Stream key: `VISIONOS_RESULTS` (env `REDIS_STREAM_KEY`).
Moi message co 1 field `data` chua JSON string.

### type: "frame"

```json
{
  "type": "frame",
  "camera_id": "string",
  "stream_id": "string",
  "frame_timestamp": "2026-08-14T10:30:00.123Z",
  "resolution": {
    "width": 960,
    "height": 540
  },
  "boxes": [
    {
      "track_id": "string",
      "class": "string",
      "confidence": 0.87,
      "bbox": [120.5, 80.3, 60.0, 140.0],
      "velocity": [12.3, -5.1]
    }
  ]
}
```

| Field | Type | Mo ta |
|---|---|---|
| `camera_id` | string | ID camera tu request |
| `stream_id` | string | ID stream tu URL path |
| `frame_timestamp` | string | ISO 8601 UTC |
| `resolution` | object | Kich thuoc frame xu ly (pixel) |
| `boxes[].track_id` | string | ID theo doi (ByteTrack) |
| `boxes[].class` | string | Ten lop (person, car, truck, bus, ...) |
| `boxes[].confidence` | float | Do tin cay YOLO (0-1) |
| `boxes[].bbox` | [x, y, w, h] | Vi tri va kich thuoc (pixel, goc tren-trai) |
| `boxes[].velocity` | [vx, vy] | Van toc (pixel/giay) |

### type: "track_event"

```json
{
  "type": "track_event",
  "camera_id": "string",
  "stream_id": "string",
  "track_id": "string",
  "class": "string",
  "event": "start | end | IN | OUT",
  "timestamp": "2026-08-14T10:30:00.123Z"
}
```

| Event | Khi nao |
|---|---|
| `start` | Track moi xuat hien lan dau |
| `end` | Mat dau qua `track_timeout` giay |
| `IN` | Di vao qua vach (line) hoac vao trong vung (zone) |
| `OUT` | Di ra qua vach (line) hoac ra khoi vung (zone) |

---

## 3. Legacy Jobs API

### POST /api/jobs — Tao job dem

```json
{
  "source": "string",
  "prompt": "person",
  "counting_type": "line | zone | fullscreen",
  "line": [x1, y1, x2, y2],
  "zone": [[x, y], ...],
  "model": "yolo",
  "resolution": [960, 540],
  "max_fps": 12.0,
  "confidence": 0.25,
  "detect_every": 1,
  "group_label": false,
  "in_label": "IN",
  "out_label": "OUT",
  "record_events": true
}
```

> Toa do `line` va `zone` tinh theo % (0-100), doc lap do phan giai.

### GET /api/jobs/{id} — Ket qua dem

```json
{
  "id": "string",
  "running": true,
  "status": "dang chay",
  "error": null,
  "source": "string",
  "kind": "yolo",
  "source_ok": true,
  "in": 12,
  "out": 8,
  "total": 20,
  "in_zone": 0,
  "zone_peak": 0,
  "in_frame": 0,
  "peak": 0,
  "tracks": 15,
  "det_per_frame": 3.5,
  "fps": 10.2
}
```

Cac field tra ve phu thuoc `counting_type`:
- **line**: `in`, `out`, `total`
- **zone**: `in_zone`, `zone_peak`
- **fullscreen**: `in_frame`, `peak`, `total`

---

## 4. Utility endpoints

| Endpoint | Method | Mo ta |
|---|---|---|
| `/healthz` | GET | Health check |
| `/api/snapshot?source=...` | GET | Lay 1 frame JPEG tu camera |
| `/api/events?limit=20` | GET | Su kien gan nhat (vector DB) |
| `/api/search/similar` | POST | Upload anh → tim vat giong (ReID) |
| `/api/vectordb` | GET | Trang thai vector DB |
| `/api/v1/counting-result` | GET | Mock response (demo) |

---

## 5. Environment variables

| Variable | Default | Mo ta |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379` | Dia chi Redis |
| `REDIS_STREAM_KEY` | `VISIONOS_RESULTS` | Ten Redis Stream |
| `REDIS_STREAM_MAXLEN` | `1000` | Gioi han so message trong stream |
| `YOLO_WEIGHTS` | `yolov8s.pt` | Duong dan model YOLO |
| `YOLO_IMGSZ` | `640` | Kich thuoc anh dau vao YOLO |
| `YOLO_CONF` | `0.2` | Confidence threshold mac dinh |
| `DETECT_EVERY` | `3` | Chay YOLO moi N frame |
| `OVERLAY_PUBLISH_FPS` | `12` | Tan so publish mac dinh |
| `SMOOTHER_LEN` | `2` | So frame smooth (1 = tat) |
| `RTSP_TRANSPORT` | `tcp` | Transport cho RTSP (tcp/udp) |
| `RTSP_RECONNECT_INTERVAL_SEC` | `5` | Giay cho giua cac lan reconnect |
| `CONTROL_API_PORT` | `8000` | Port FastAPI |
