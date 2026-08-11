# Kế hoạch Schema Input/Output — VisionOS AI Service (Task 6)

Tài liệu này định nghĩa **chính xác kiểu dữ liệu, tên trường, ràng buộc** cho mọi endpoint
của VisionOS AI Service. Mục tiêu kép:

1. **Tài liệu tham chiếu** — dev frontend/backend tích hợp không cần đọc code Python.
2. **Bản kế hoạch** — làm nền cho file `docs/mock_responses.json` + endpoint mock
   `/api/v1/counting-result` (hardcode, không cần camera thật để test).

---

## 1. Tổng Quan Luồng Dữ Liệu

```
Camera / File video
      │
      ▼
FrameSource  ─── đọc frame, pace đúng FPS gốc
      │
      ▼
StreamingCounter.process(frame_bgr)
      │
      ├─► YOLO detect  →  supervision Detections
      ├─► ByteTrack    →  tracker_id thô
      ├─► TrackStitcher→  tracker_id ổn định (ReID)
      ├─► DetectionsSmoother → box mượt (anti-flicker)
      ├─► LineZone / PolygonZone → in / out / zone counts
      └─► annotate     →  frame BGR đã vẽ
            │
            ▼
      Job._record_events()
            ├─► embed_crop (HSV histogram 256-d, L2-normalize)
            ├─► VectorStore.add_event  (Qdrant / in-memory fallback)
            └─► _save_event_video     (H.264 clip ~2 giây, ffmpeg)

API Response ◄─── Job.stats() + CountResult
```

---

## 2. Kiểu Dữ Liệu Dùng Chung

### 2.1 `JobRequest` — body của `POST /api/jobs`

| Trường | Kiểu | Bắt buộc | Mặc định | Mô tả |
|---|---|:---:|---|---|
| `source` | `string` | ✅ | — | URL camera (`rtsp://...`, `http://...`) / đường dẫn file / `"0"` (webcam) |
| `prompt` | `string` | | `"person"` | Đối tượng cần đếm. Hỗ trợ: `"person"`, `"car"`, `"truck"`, `"bus"`, `"motorcycle"`, `"bicycle"`, `"vehicle"` (nhóm xe) |
| `counting_type` | `"line"\|"zone"\|"fullscreen"` | | `"line"` | Kiểu đếm |
| `line` | `[x1,y1,x2,y2]: number[4]` | Khi line | `[0,50,100,50]` | Toạ độ % (0–100) của vạch cắt |
| `zone` | `[[x,y],...]: number[][]` | Khi zone | `null` | Toạ độ % đa giác vùng đếm (≥3 điểm) |
| `model` | `"yolo"\|"locate"\|"auto"` | | `"yolo"` | Detector. `"yolo"` cho COCO; `"auto"` suy từ prompt |
| `resolution` | `[width,height]: number[2]` | | `[960,540]` | Độ phân giải xử lý (px) |
| `max_fps` | `number` | | `12.0` | Trần FPS xử lý (giảm tải CPU) |
| `confidence` | `number` (0–1) | | `0.25` | Ngưỡng tin cậy detect |
| `detect_every` | `integer ≥1` | | `1` | Chạy YOLO mỗi N frame. `1` = mỗi frame (mượt nhất) |
| `group_label` | `boolean` | | `false` | `true` = gộp mọi xe → `"vehicle"` (1 màu); `false` = giữ `car/truck/bus` riêng |
| `in_label` | `string` | | `"IN"` | Nhãn hiển thị chiều VÀO (chỉ dùng ở line mode) |
| `out_label` | `string` | | `"OUT"` | Nhãn hiển thị chiều RA (chỉ dùng ở line mode) |
| `anchor` | `string\|null` | | `null` | Điểm neo kiểm tra vạch: `"CENTER"`, `"BOTTOM_CENTER"`, `"TOP_CENTER"`. `null` = tự chọn theo kiểu đếm |
| `record_events` | `boolean` | | `true` | `true` = lưu ngoại hình vật vào vector DB |

**Ràng buộc quan trọng:**

- Toạ độ `line` và `zone` đều theo **% (0–100)**, độc lập độ phân giải camera thực tế.
- `counting_type = "line"` → `line` phải có đúng 4 số `[x1,y1,x2,y2]`.
- `counting_type = "zone"` → `zone` phải là mảng ≥3 điểm `[[x,y], ...]`.
- `counting_type = "fullscreen"` → không cần `line` hay `zone`.

---

### 2.2 `JobStats` — response của `GET/POST /api/jobs[/{id}]`

#### Trường chung (luôn có)

| Trường | Kiểu | Mô tả |
|---|---|---|
| `id` | `string` (8 ký tự hex) | Job ID |
| `running` | `boolean` | Job đang chạy không |
| `status` | `string` | `"đang nạp model (yolo)"` → `"đang kết nối camera"` → `"đang chạy"` → `"đã dừng"` \| `"lỗi"` |
| `error` | `string\|null` | Thông báo lỗi khi `status="lỗi"`, `null` nếu không có lỗi |
| `source` | `string` | Nguồn camera |
| `kind` | `"yolo"\|"locate"` | Detector đang dùng |
| `source_ok` | `boolean` | Camera kết nối OK không |
| `scenario` | `string` | Key scenario (mặc định `"stream"`) |
| `prompt` | `string` | Đối tượng đang đếm |
| `counting_type` | `"line"\|"zone"\|"fullscreen"` | Kiểu đếm |
| `frames` | `integer ≥0` | Tổng số frame đã xử lý |
| `tracks` | `integer ≥0` | Số vật khác nhau đã thấy (unique track ID, kể từ khi bắt đầu job) |
| `det_per_frame` | `number` | Số detect trung bình mỗi frame |
| `fps` | `number` | FPS thực tế hệ thống |
| `latency_ms` | `number` | Thời gian chạy YOLO mỗi frame (ms) |

#### Trường riêng theo `counting_type`

**Khi `counting_type = "line"`:**

| Trường | Kiểu | Mô tả |
|---|---|---|
| `in` | `integer ≥0` | Số lượt qua vạch chiều VÀO |
| `out` | `integer ≥0` | Số lượt qua vạch chiều RA |
| `total` | `integer ≥0` | Tổng = `in + out` |
| `in_label` | `string` | Nhãn chiều vào (VD: `"IN"`, `"Vào"`) |
| `out_label` | `string` | Nhãn chiều ra (VD: `"OUT"`, `"Ra"`) |

**Khi `counting_type = "zone"`:**

| Trường | Kiểu | Mô tả |
|---|---|---|
| `in_zone` | `integer ≥0` | Số vật đang trong vùng tại thời điểm query |
| `zone_peak` | `integer ≥0` | Số vật đông nhất từng có trong vùng |

**Khi `counting_type = "fullscreen"`:**

| Trường | Kiểu | Mô tả |
|---|---|---|
| `in_frame` | `integer ≥0` | Số vật đang trong khung tại thời điểm query |
| `peak` | `integer ≥0` | Số vật đông nhất từng có trong khung |
| `total` | `integer ≥0` | Tổng vật khác nhau đã thấy (= `tracks`) |

---

### 2.3 `EventPayload` — dữ liệu của 1 sự kiện trong vector DB

| Trường | Kiểu | Mô tả |
|---|---|---|
| `track_id` | `integer` | ID vật trong phiên (ByteTrack + ReID) |
| `class_name` | `string` | Lớp: `"person"`, `"car"`, `"truck"`, `"bus"`, `"motorcycle"`, ... |
| `source` | `string` | Nguồn camera ghi sự kiện |
| `counting_type` | `string` | Kiểu đếm của job (`"line"`, `"zone"`, `"fullscreen"`) |
| `ts` | `number` | Unix timestamp (float, giây, VD `1754470800.123`) |
| `image_base64` | `string` | Data URI ảnh crop vật (`"data:image/jpeg;base64,..."`) — chuỗi rỗng nếu lỗi crop |
| `full_frame_base64` | `string` | Data URI ảnh toàn cảnh thu nhỏ (max 640px wide) |
| `video_url` | `string` | Đường dẫn clip H.264 (`"/api/videos/event_1234.mp4"`) — rỗng nếu chưa ghi xong |

---

### 2.4 `EventRecord` — 1 mục trong response `/api/events`

| Trường | Kiểu | Mô tả |
|---|---|---|
| `id` | `integer` | ID nội bộ trong vector DB |
| `payload` | `EventPayload` | Thông tin vật (xem §2.3) |

---

### 2.5 `SearchResult` — 1 kết quả của `/api/search/similar`

| Trường | Kiểu | Mô tả |
|---|---|---|
| `id` | `integer` | ID điểm trong vector DB |
| `score` | `number` (0–1) | Độ tương đồng cosine (1.0 = giống hoàn toàn) |
| `payload` | `EventPayload` | Thông tin vật tìm được |

---

### 2.6 `VectorDBStatus` — response `/api/vectordb`

| Trường | Kiểu | Mô tả |
|---|---|---|
| `backend` | `"qdrant"\|"memory"` | Backend đang dùng |
| `url` | `string\|null` | Qdrant URL (null nếu không cấu hình `QDRANT_URL`) |
| `collection` | `string` | Tên collection Qdrant (mặc định `"tracks"`) |
| `events` | `integer ≥0` | Tổng số sự kiện đã lưu |

---

## 3. Đặc Tả Từng Endpoint

### 3.1 `GET /healthz`

**Mục đích:** Health check — Docker/K8s liveness probe, load balancer.

**Input:** Không có.

**Output 200:**
```json
{
  "status": "ok",
  "jobs": 2,
  "vectordb": "qdrant"
}
```

---

### 3.2 `GET /api/snapshot`

**Mục đích:** Lấy 1 frame JPEG từ nguồn để người dùng vẽ vạch/vùng lên canvas.

**Input (query params):**

| Param | Kiểu | Bắt buộc | Mô tả |
|---|---|:---:|---|
| `source` | `string` | ✅ | URL camera / file |
| `w` | `integer` | | Chiều rộng resize (px). Mặc định: `960` |

**Output 200:** Binary JPEG (`Content-Type: image/jpeg`).

**Lỗi:**
- `502 Bad Gateway` — `{"detail": "Không lấy được frame từ nguồn: \"...\""}`
- `500 Internal Server Error` — lỗi mã hoá JPEG.

---

### 3.3 `POST /api/jobs`

**Mục đích:** Tạo job đếm mới (chạy ngầm trên luồng riêng).

**Input:** JSON body kiểu `JobRequest` (§2.1).

**Output 200:** `JobStats` — thường ngay sau tạo nên `status = "đang nạp model (yolo)"`.

**Lỗi:**
- `400 Bad Request` — `{"detail": "Cấu hình sai: <lý do>"}`

---

### 3.4 `GET /api/jobs`

**Mục đích:** Liệt kê tất cả jobs.

**Output 200:** `JobStats[]` (mảng, có thể rỗng).

---

### 3.5 `GET /api/jobs/{job_id}`

**Mục đích:** Lấy số đếm realtime (frontend polling 1 giây / lần).

**Path param:** `job_id` — string 8 ký tự hex.

**Output 200:** `JobStats` đầy đủ.

**Lỗi:** `404 Not Found`.

---

### 3.6 `POST /api/jobs/{job_id}/stop`

**Mục đích:** Dừng job, giải phóng tài nguyên.

**Output 200:**
```json
{ "id": "e2d847fa", "stopped": true }
```

---

### 3.7 `GET /api/jobs/{job_id}/frame.jpg`

**Mục đích:** 1 frame JPEG đã annotate (bounding box + nhãn + số đếm).

**Output 200:** Binary JPEG.

**Lỗi:** `503` khi model chưa nạp xong hoặc chưa có frame nào.

---

### 3.8 `GET /api/jobs/{job_id}/mjpeg`

**Mục đích:** Luồng MJPEG để nhúng thẳng vào `<img>` HTML.

**Output:** `Content-Type: multipart/x-mixed-replace; boundary=frame`
- Mỗi phần là JPEG frame.

**Cách dùng:**
```html
<img src="http://localhost:8000/api/jobs/e2d847fa/mjpeg" />
```

---

### 3.9 `GET /api/vectordb`

**Mục đích:** Trạng thái vector DB (debug, monitoring).

**Output 200:** `VectorDBStatus` (§2.6).

---

### 3.10 `GET /api/events`

**Mục đích:** Lấy sự kiện gần nhất (vật đã qua camera).

**Input (query params):**

| Param | Kiểu | Mô tả |
|---|---|---|
| `source` | `string` | Lọc theo nguồn camera. Bỏ trống = tất cả |
| `limit` | `integer` | Số mục trả về. Mặc định: `20` |

**Output 200:**
```json
{
  "recent_events": [ <EventRecord>, ... ],
  "backend": "qdrant",
  "url": "http://qdrant:6333",
  "collection": "tracks",
  "events": 47
}
```

---

### 3.11 `POST /api/search/similar`

**Mục đích:** Upload ảnh → tìm vật trông giống nhất trong lịch sử (ReID).

**Input:** `multipart/form-data`
- `file` (UploadFile, bắt buộc): Ảnh JPEG/PNG bất kỳ.
- `limit` (query integer): Số kết quả. Mặc định: `5`.

**Output 200:**
```json
{
  "results": [ <SearchResult>, ... ]
}
```

**Lỗi:** `400 Bad Request` — ảnh không decode được.

---

### 3.12 `GET /api/v1/counting-result` ← MOCK ENDPOINT

**Mục đích:** Response hardcode cho dev frontend tích hợp mà không cần camera thật.

**Input:** Không có.

**Output 200:** Nội dung key `"success"` của `docs/mock_responses.json` — cấu trúc `JobStats`
đầy đủ cho `counting_type = "line"` (kịch bản mặc định).

**Lỗi (nếu file mock thiếu):**
```json
{ "status": "error", "message": "Không tìm thấy file mock_responses.json" }
```

---

## 4. Đặc Tả `docs/mock_responses.json`

File cung cấp dữ liệu mẫu tĩnh cho endpoint `/api/v1/counting-result`. Cấu trúc ngoài
cùng là object JSON với nhiều key kịch bản; endpoint đọc key `"success"` mặc định.

### Cấu trúc file

```json
{
  "success":    { <JobStats cho line mode> },
  "zone":       { <JobStats cho zone mode> },
  "fullscreen": { <JobStats cho fullscreen mode> }
}
```

Endpoint hiện chỉ dùng key `"success"`. Các key còn lại dùng để mở rộng sau (query param
`?scenario=zone`).

### Spec chi tiết của key `"success"` (line mode)

```json
{
  "id": "demo0001",
  "running": true,
  "status": "đang chạy",
  "error": null,
  "source": "rtsp://demo-camera.local/stream1",
  "kind": "yolo",
  "source_ok": true,
  "scenario": "stream",
  "prompt": "person",
  "counting_type": "line",
  "in_label": "IN",
  "out_label": "OUT",
  "in": 34,
  "out": 21,
  "total": 55,
  "tracks": 58,
  "frames": 1440,
  "det_per_frame": 3.2,
  "fps": 11.8,
  "latency_ms": 45.3
}
```

### Spec chi tiết của key `"zone"` (zone mode)

```json
{
  "id": "demo0002",
  "running": true,
  "status": "đang chạy",
  "error": null,
  "source": "rtsp://demo-camera.local/stream2",
  "kind": "yolo",
  "source_ok": true,
  "scenario": "stream",
  "prompt": "car",
  "counting_type": "zone",
  "in_zone": 7,
  "zone_peak": 12,
  "tracks": 43,
  "frames": 900,
  "det_per_frame": 5.1,
  "fps": 10.5,
  "latency_ms": 52.0
}
```

### Spec chi tiết của key `"fullscreen"` (fullscreen mode)

```json
{
  "id": "demo0003",
  "running": true,
  "status": "đang chạy",
  "error": null,
  "source": "rtsp://demo-camera.local/stream3",
  "kind": "yolo",
  "source_ok": true,
  "scenario": "stream",
  "prompt": "vehicle",
  "counting_type": "fullscreen",
  "in_frame": 15,
  "peak": 28,
  "total": 112,
  "tracks": 112,
  "frames": 2100,
  "det_per_frame": 8.3,
  "fps": 12.1,
  "latency_ms": 38.7
}
```

---

## 5. Kế Hoạch Triển Khai (Các Bước Cần Làm)

### Bước 1 — Tạo `docs/mock_responses.json`

Tạo file với đúng 3 kịch bản (line / zone / fullscreen) theo spec §4.
Endpoint `/api/v1/counting-result` đã có sẵn — tự đọc file này.

**Kết quả kiểm tra:** `curl localhost:8000/api/v1/counting-result` trả JSON đúng spec.

### Bước 2 — Mở rộng endpoint mock (tùy chọn)

Thêm query param `?scenario=zone|fullscreen|success` để dev test từng kịch bản:
```python
@app.get("/api/v1/counting-result")
def get_mock_counting_result(scenario: str = "success"):
    data = json.load(open("docs/mock_responses.json"))
    return data.get(scenario) or data.get("success", {})
```

### Bước 3 — Cập nhật `API_HANDOFF.md`

Bổ sung endpoint `/api/v1/counting-result` + giải thích mock cho dev.

### Bước 4 — (Tùy chọn) Pydantic Response Models

Thêm Pydantic `response_model` cho FastAPI để tự sinh OpenAPI/Swagger (`/docs`):
```python
class JobStatsLine(BaseModel):
    id: str
    running: bool
    status: str
    in_: int = Field(alias="in")
    out: int
    total: int
    # ...
```
Ưu điểm: FastAPI sinh OpenAPI spec tự động → dev dùng `/docs` thay vì đọc tài liệu.

---

## 6. Bảng Tóm Tắt Endpoint

| Endpoint | Method | Input | Output kiểu |
|---|:---:|---|---|
| `/healthz` | GET | — | `{status, jobs, vectordb}` |
| `/api/snapshot` | GET | `?source&w` | Binary JPEG |
| `/api/jobs` | POST | `JobRequest` JSON | `JobStats` |
| `/api/jobs` | GET | — | `JobStats[]` |
| `/api/jobs/{id}` | GET | path `job_id` | `JobStats` |
| `/api/jobs/{id}/stop` | POST | path `job_id` | `{id, stopped}` |
| `/api/jobs/{id}/frame.jpg` | GET | path `job_id` | Binary JPEG |
| `/api/jobs/{id}/mjpeg` | GET | path `job_id` | MJPEG stream |
| `/api/vectordb` | GET | — | `VectorDBStatus` |
| `/api/events` | GET | `?source&limit` | `{recent_events[], ...VectorDBStatus}` |
| `/api/search/similar` | POST | multipart `file` + `?limit` | `{results: SearchResult[]}` |
| `/api/v1/counting-result` | GET | `?scenario` | `JobStats` (mock hardcode) |

---

## 7. Ghi Chú Cho Dev Tích Hợp

1. **Toạ độ % (Percentage):** Vạch/vùng truyền dưới dạng % (0–100), không phải pixel. Mọi
   client phải quy đổi: `x_pct = x_pixel / canvas_width * 100`.

2. **Quản lý Job:** Mỗi job tiêu thụ ~1 CPU core. Gọi `POST /api/jobs/{id}/stop` khi user
   tắt camera hoặc chuyển trang để tránh tích lũy job chạy vô thời hạn.

3. **Polling vs. Streaming:** Số đếm lấy bằng polling `GET /api/jobs/{id}` mỗi 1 giây.
   Luồng video xem bằng `<img src=".../mjpeg">`. Không cần WebSocket.

4. **Sự kiện (Events):** `GET /api/events` trả tối đa `limit` sự kiện gần nhất. Mỗi sự
   kiện chứa ảnh crop + ảnh toàn cảnh (Data URI) + URL clip H.264.

5. **Mock Dev:** Dùng `/api/v1/counting-result` để test render UI mà không cần camera. Sau
   khi tích hợp xong thì chuyển sang poll `GET /api/jobs/{id}` với job thật.
