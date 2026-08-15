# Tài Liệu Bàn Giao AI Service → Backend (Electron)

> **Phiên bản:** 1.0.0  
> **Ngày cập nhật:** 2026-08-15  
> **Người viết:** AI Team  

---

## 1. Tổng Quan Kiến Trúc

```
Electron (parent) ─spawn─> MediaMTX, Backend, AI Service

Backend    ──REST────>  AI Service   Control API   (AI Service listen :8000)
AI Service ──gRPC────>  Backend      OverlayIngest (Backend listen :9090)
MediaMTX   ──RTSP────>  AI Service   AI Service pull từ MediaMTX
```

- Tất cả bind `127.0.0.1`, không broker, không xác thực.
- AI Service khởi động với **zero job**, không persist. Backend bật lại toàn bộ ở vòng reconcile (≤15s).

---

## 2. Cổng Mạng

| Kênh | Cổng ưu tiên | Ai lắng nghe | Biến môi trường |
|---|---|---|---|
| REST Control API | `8000` | AI Service | `CONTROL_API_PORT` |
| gRPC OverlayIngest | `9090` | **Backend** | `OVERLAY_GRPC_TARGET` |

---

## 3. REST Control API (Backend → AI Service)

### 3.1 `GET /health` — Kiểm tra trạng thái

**Response:**
```json
{
  "ready": true,
  "version": "1.0.0"
}
```
- `200` chỉ khi model đã nạp xong.
- Electron poll endpoint này để xác định startup hoàn tất. Timeout 120s.

---

### 3.2 `POST /streams/{jobKey}` — Tạo mới hoặc Cập nhật nóng luồng

`jobKey` = `camera_stream.id` (số nguyên dạng chuỗi).

**Request Body:**
```json
{
  "camera_id": "7",
  "rtsp_url": "rtsp://127.0.0.1:8554/camera-7-main",
  "params": {
    "classes": ["person", "car"],
    "conf": 0.3,
    "roi": [[12.5, 80.0], [87.5, 80.0]],
    "publish_fps": 30.0,
    "detect_every": 3,
    "track_timeout": 2.0
  }
}
```

**Chi tiết từng trường INPUT:**

| Trường | Kiểu | Bắt buộc | Mặc định | Mô tả |
|---|---|---|---|---|
| `camera_id` | string | ✅ | — | ID camera (số nguyên dạng chuỗi). Echo lại trong `OverlayFrame.camera_id` |
| `rtsp_url` | string | ✅ | — | URL RTSP tới MediaMTX hoặc đường dẫn file video |
| `params.classes` | string[] | ❌ | `["person"]` | Tên lớp COCO cần theo dõi, ≥1 phần tử. VD: `["person"]`, `["person", "car", "truck"]` |
| `params.conf` | float | ❌ | `0.3` | Ngưỡng confidence (0.0 → 1.0). Càng cao càng khắt khe |
| `params.roi` | `[[x,y],...]` \| `null` | ❌ | `null` | Vùng quan tâm. **Phần trăm 0..100**, không nhận pixel. `null` = toàn khung hình. 2 điểm = đếm vạch. >2 điểm = đa giác kín, đếm vùng |
| `params.publish_fps` | float | ❌ | `30.0` | Trần FPS đẩy OverlayFrame qua gRPC |
| `params.detect_every` | int | ❌ | `3` | Chạy AI detector mỗi N frame. Frame còn lại giữ box cũ (tiết kiệm CPU) |
| `params.track_timeout` | float | ❌ | `2.0` | Số giây giữ track khi vật bị che khuất trước khi xoá |

**Bảng map cấu hình từ Frontend:**

| Cấu hình | Giá trị | `conf` | `publish_fps` | `detect_every` | `track_timeout` |
|---|---|---|---|---|---|
| `sensitivity_level` | `sensitive` | 0.1 | — | — | 3.0 |
| | `balanced` | 0.3 | — | — | 2.0 |
| | `precise` | 0.8 | — | — | 1.0 |
| `performance_level` | `eco` | — | 15 | 5 | — |
| | `balanced` | — | 30 | 3 | — |
| | `max` | — | 60 | 1 | — |

**Response:**
```json
// Tạo mới:
{ "status": "started", "stream_id": "test-1" }

// Cập nhật nóng (cùng URL):
{ "status": "updated", "stream_id": "test-1" }
```

**Hành vi cập nhật:**
- Nếu `rtsp_url` **không đổi** → cập nhật nóng params mà không khởi động lại camera.
- Nếu `rtsp_url` **thay đổi** → tự động ngắt luồng cũ, khởi động luồng mới.
- Backend **không cần gọi DELETE trước** khi đổi params.

---

### 3.3 `DELETE /streams/{jobKey}` — Dừng luồng

Dừng job, giải phóng RTSP session. `404` được Backend coi như đã dừng.

**Response:**
```json
{ "status": "stopped", "stream_id": "test-1" }
```

---

### 3.4 `GET /streams` — Danh sách luồng đang chạy

Backend dùng để đối soát và chẩn đoán.

**Response:**
```json
{
  "count": 1,
  "streams": [
    {
      "stream_id": "12",
      "camera_id": "7",
      "rtsp_url": "rtsp://127.0.0.1:8554/camera-7-main",
      "fps": 15.4,
      "params": {
        "classes": ["person", "car"],
        "conf": 0.3,
        "roi": null,
        "publish_fps": 30.0,
        "detect_every": 3,
        "track_timeout": 2.0
      }
    }
  ]
}
```

**Chi tiết từng trường OUTPUT:**

| Trường | Kiểu | Mô tả |
|---|---|---|
| `count` | int | Tổng số luồng đang chạy |
| `streams[].stream_id` | string | ID luồng (= `jobKey` khi tạo) |
| `streams[].camera_id` | string | ID camera gốc |
| `streams[].rtsp_url` | string | URL RTSP đang kết nối |
| `streams[].fps` | float | FPS xử lý thực tế (trung bình) |
| `streams[].params` | object | Bản sao cấu hình đang áp dụng (cùng cấu trúc như khi POST) |

---

## 4. gRPC OverlayIngest (AI Service → Backend)

Backend implement gRPC server, AI Service gọi vào.

### 4.1 Proto Schema đầy đủ

```protobuf
syntax = "proto3";
package overlay;

service OverlayIngest {
  rpc Health(HealthRequest) returns (HealthResponse);
  rpc PublishFrames(stream OverlayFrame) returns (PublishSummary);
  rpc PublishEvent(DetectionEvent) returns (EventAck);
}

message HealthRequest {}
message HealthResponse { bool ready = 1; }

message OverlayFrame {
  string camera_id = 1;
  string job_key = 2;
  int64 captured_at_ms = 3;       // epoch millis UTC
  Resolution resolution = 4;
  repeated BoundingBox boxes = 5;
}

message Resolution {
  int32 width = 1;
  int32 height = 2;
}

message BoundingBox {
  string track_id = 1;
  string class_name = 2;
  double confidence = 3;
  double x = 4;                   // pixel, góc trên-trái
  double y = 5;
  double w = 6;
  double h = 7;
  double velocity_x = 8;         // pixel/giây
  double velocity_y = 9;         // pixel/giây
}

message PublishSummary {
  int64 received = 1;
  int64 dropped = 2;
}

message DetectionEvent {
  string event_id = 1;           // UUID, Backend dedup theo field này
  string camera_id = 2;
  string job_key = 3;
  string track_id = 4;
  string class_name = 5;
  Kind kind = 6;
  int64 occurred_at_ms = 7;

  enum Kind {
    START = 0;                   // track mới xuất hiện
    IN = 1;                      // cắt qua vạch / vào vùng
    OUT = 2;                     // cắt ngược / rời vùng
  }
}

message EventAck { bool accepted = 1; }
```

---

### 4.2 Chi tiết `OverlayFrame` — Dữ liệu Bounding Box mỗi frame

Backend nhận liên tục qua stream `PublishFrames`.

| Trường | Kiểu | Mô tả |
|---|---|---|
| `camera_id` | string | ID camera gốc (echo từ REST API) |
| `job_key` | string | ID luồng (= `jobKey` khi tạo) |
| `captured_at_ms` | int64 | Thời điểm ĐỌC ĐƯỢC frame (epoch millis UTC). Client delay video ~700ms rồi ghép bbox theo mốc này |
| `resolution.width` | int32 | Chiều rộng thật của frame (pixel) |
| `resolution.height` | int32 | Chiều cao thật của frame (pixel) |
| `boxes[]` | repeated | Danh sách vật thể được phát hiện trong frame này |

**Chi tiết từng trường trong `BoundingBox`:**

| Trường | Kiểu | Đơn vị | Mô tả |
|---|---|---|---|
| `track_id` | string | — | ID theo dõi vật thể. Ổn định suốt vòng đời một vật. Client dùng để nội suy box giữa 2 frame |
| `class_name` | string | — | Tên lớp COCO. VD: `person`, `car`, `truck` |
| `confidence` | double | 0..1 | Độ tự tin của AI |
| `x` | double | pixel | Toạ độ X góc **trên-trái** của box |
| `y` | double | pixel | Toạ độ Y góc **trên-trái** của box |
| `w` | double | pixel | Chiều rộng box |
| `h` | double | pixel | Chiều cao box |
| `velocity_x` | double | pixel/giây | Vận tốc theo trục X. Dùng để nội suy (extrapolate) vị trí box mượt mà ở 30fps dù AI chỉ chạy 5-10fps |
| `velocity_y` | double | pixel/giây | Vận tốc theo trục Y. Tương tự velocity_x |

> **Lưu ý quan trọng cho Frontend:**
> - Toạ độ là **pixel thô** (không chuẩn hoá). Dùng `resolution` để quy đổi sang tỷ lệ màn hình.
> - Format là `x, y, w, h` (góc trên-trái + kích thước). **KHÔNG PHẢI** `x1, y1, x2, y2`.
> - Frame rỗng (`boxes = []`) vẫn được gửi → client phải xoá box cũ khi nhận frame rỗng.
> - `velocity_x/y` cho phép Frontend tính: `new_x = x + velocity_x * dt` để box trượt mượt giữa 2 frame AI.

---

### 4.3 Chi tiết `DetectionEvent` — Sự kiện đếm

Gửi riêng lẻ qua unary RPC `PublishEvent`. **Không được mất**, có ack và retry.

| Trường | Kiểu | Mô tả |
|---|---|---|
| `event_id` | string | UUID do AI Service sinh. Backend **dedup** theo field này (retry vô hại) |
| `camera_id` | string | ID camera gốc |
| `job_key` | string | ID luồng |
| `track_id` | string | ID vật thể gây ra sự kiện |
| `class_name` | string | Tên lớp COCO |
| `kind` | enum | Loại sự kiện (xem bảng dưới) |
| `occurred_at_ms` | int64 | Thời điểm xảy ra (epoch millis UTC) |

**Enum `Kind`:**

| Giá trị | Tên | Ý nghĩa |
|---|---|---|
| `0` | `START` | Track mới xuất hiện lần đầu trong khung hình |
| `1` | `IN` | Vật cắt qua vạch / đi vào vùng ROI |
| `2` | `OUT` | Vật cắt ngược vạch / rời khỏi vùng ROI |

---

### 4.4 Quy tắc kênh gRPC

**Kênh `PublishFrames` (bbox):**

| Mục | Yêu cầu |
|---|---|
| Số stream | **1 stream duy nhất** cho toàn tiến trình, dùng chung mọi job |
| Backpressure | Backend queue 100 frame, drop **oldest**. `PublishSummary.dropped` > 0 kéo dài → giảm `publish_fps` |
| Cho phép mất | ✅ Có. Đây là kênh bbox, mất vài frame không ảnh hưởng |
| Reconnect | Backoff tối đa 5s, giữ nguyên job đang chạy khi stream đứt |

**Kênh `PublishEvent` (sự kiện đếm):**

| Mục | Yêu cầu |
|---|---|
| Giao thức | Unary, mỗi sự kiện một RPC, có ack |
| Cho phép mất | ❌ Không. Retry backoff tối đa 5s |
| Idempotent | Backend dedup theo `event_id` → retry vô hại |
| Thứ tự | Không bắt buộc — Backend sắp lại theo `occurred_at_ms` |

---

## 5. Biến Môi Trường Đầy Đủ

### 5.1 Biến cấu hình chính (Electron truyền khi spawn)

| Biến | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `CONTROL_API_PORT` | int | `8000` | Cổng REST Control API |
| `OVERLAY_GRPC_TARGET` | `host:port` | `127.0.0.1:9090` | Địa chỉ Backend gRPC lắng nghe |
| `YOLO_WEIGHTS` | string | `yolov8m_openvino_model` | Thư mục/file trọng số YOLO |
| `YOLO_IMGSZ` | int | `640` | Kích thước ảnh đầu vào cho YOLO |
| `YOLO_CONF` | float | `0.3` | Confidence mặc định (bị override bởi `params.conf`) |

### 5.2 Biến hiệu năng

| Biến | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `DETECT_EVERY` | int | `1` | Chạy detector mỗi N frame. Tăng để tiết kiệm CPU |
| `OVERLAY_PUBLISH_FPS` | float | `30` | Trần FPS đẩy OverlayFrame qua gRPC |
| `SMOOTHER_LEN` | int | `2` | Độ dài bộ lọc chống nhấp nháy box |
| `REID_STITCH` | `0`/`1` | `0` | Bật/tắt ReID nối track đứt |

### 5.3 Biến kết nối RTSP

| Biến | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `RTSP_TRANSPORT` | `tcp`/`udp` | `tcp` | Giao thức RTSP |
| `RTSP_RECONNECT_INTERVAL_SEC` | float | `5` | Giây chờ trước khi reconnect RTSP |

### 5.4 Biến debug

| Biến | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `DEBUG_SHOW` | `0`/`1` | `1` | Bật cửa sổ OpenCV xem video (tắt khi deploy) |

---

## 6. Ví Dụ Gọi API (PowerShell)

### Tạo luồng
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/streams/cam-1 -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"camera_id": "1", "rtsp_url": "rtsp://127.0.0.1:8554/camera-1-main", "params": {"classes": ["person"], "conf": 0.3, "publish_fps": 30, "detect_every": 3, "track_timeout": 2.0}}'
```

### Cập nhật nóng (đổi confidence)
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/streams/cam-1 -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"camera_id": "1", "rtsp_url": "rtsp://127.0.0.1:8554/camera-1-main", "params": {"classes": ["person"], "conf": 0.8}}'
```

### Lấy danh sách luồng
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/streams
```

### Dừng luồng
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/streams/cam-1 -Method DELETE
```

---

## 7. Luồng Vận Hành

### Khởi động
```
Electron start tuần tự: MediaMTX → Backend → AI Service
  └─ Chờ GET /health trả 200 (timeout 120s)
  └─ AI Service khởi động với ZERO job
```

### Tạo / cập nhật job
```
Backend reconcile (mỗi 15s, hoặc tức thì khi user sửa luồng)
  └─ Tập mong muốn = camera_stream active + mode=standard
  └─ MediaMtxRelayManager.ensureAsync(camera)
  └─ POST /streams/{jobKey}
```

### Xoá job
```
jobKey rời tập mong muốn (luồng tắt / camera inactive)
  └─ DELETE /streams/{jobKey}
```

### Publish dữ liệu
```
AI Service
  └─ Health() → chờ Backend gRPC sẵn sàng
  └─ Mở PublishFrames stream (giữ suốt vòng đời)
       ├─ Mỗi job: đọc frame → detect → track → throttle → OverlayFrame
       └─ Khi track cắt vạch / vào-ra vùng → PublishEvent
```

### Tắt
```
SIGTERM (Windows: taskkill)
  └─ Đóng gRPC stream, thả camera, exit trong 5s
  └─ Quá hạn → SIGKILL
```

---

## 8. File Quan Trọng

| File | Mô tả |
|---|---|
| `proto/overlay.proto` | Schema gRPC chính thức |
| `AI_SERVICE_CONTRACT.md` | Đặc tả kỹ thuật chi tiết |
| `run_native.bat` | Script chạy AI Service trên Windows |
| `mock_backend.py` | Mock gRPC Backend để test |
| `test_people.py` | Script test toàn vòng đời API |
| `recognition/service/grpc_publisher.py` | Client gRPC (AI Service gửi đi) |
| `recognition/service/stream_manager.py` | Quản lý luồng RTSP + AI pipeline |
