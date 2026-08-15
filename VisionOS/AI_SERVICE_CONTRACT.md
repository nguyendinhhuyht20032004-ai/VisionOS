# AI Service — Đặc tả tích hợp

Đặc tả kỹ thuật cho tiến trình AI Service chạy kèm bản desktop.
Schema gRPC: [`apps/backend/src/main/proto/overlay.proto`](apps/backend/src/main/proto/overlay.proto).

---

## 1. Kiến trúc

```
Electron (parent) ─spawn─> MediaMTX, Backend, AI Service

Backend    ──REST────>  AI Service   Control API   (AI Service listen)
AI Service ──gRPC────>  Backend      OverlayIngest (Backend listen)
MediaMTX   ──RTSP────>  AI Service   AI Service pull
```

Tất cả bind `127.0.0.1`, không broker, không xác thực giữa các tiến trình.

Cổng cấp động lúc khởi động (dò cổng trống tăng dần từ giá trị ưu tiên). Hai cổng AI Service quan tâm:

| Kênh | Ưu tiên | Listener | Nhận qua |
|---|---|---|---|
| Control API | 8000 | AI Service | `CONTROL_API_PORT` |
| Backend gRPC | 9090 | Backend | `OVERLAY_GRPC_TARGET` |

RTSP không cần env — URL đầy đủ nằm trong body `POST /streams/{jobKey}`.

---

## 2. Biến môi trường

Electron truyền lúc spawn. Mặc định chỉ dùng khi chạy tay lúc phát triển.

### Biến cấu hình chính (Electron truyền khi spawn)

| Biến | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `CONTROL_API_PORT` | int | `8000` | Cổng REST Control API |
| `OVERLAY_GRPC_TARGET` | `host:port` | `127.0.0.1:9090` | Địa chỉ Backend gRPC lắng nghe |
| `YOLO_WEIGHTS` | string | `yolov8m_openvino_model` | Thư mục/file trọng số YOLO |
| `YOLO_IMGSZ` | int | `640` | Kích thước ảnh đầu vào cho YOLO |
| `YOLO_CONF` | float | `0.3` | Confidence mặc định (bị override bởi `params.conf` từ API) |

### Biến hiệu năng (tuỳ chỉnh khi chạy tay)

| Biến | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `DETECT_EVERY` | int | `1` | Chạy detector mỗi N frame (frame còn lại giữ box cũ). Tăng để tiết kiệm CPU |
| `OVERLAY_PUBLISH_FPS` | float | `30` | Trần FPS đẩy OverlayFrame qua gRPC |
| `SMOOTHER_LEN` | int | `2` | Độ dài bộ lọc DetectionsSmoother (chống nhấp nháy box) |
| `REID_STITCH` | `0` \| `1` | `0` | Bật/tắt ReID nối track đứt (tốn thêm CPU) |

### Biến kết nối RTSP

| Biến | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `RTSP_TRANSPORT` | `tcp` \| `udp` | `tcp` | Giao thức kết nối RTSP (tcp ổn định hơn, udp nhanh hơn) |
| `RTSP_RECONNECT_INTERVAL_SEC` | float | `5` | Giây chờ trước khi kết nối lại RTSP khi luồng đứt |

### Biến debug (chỉ dùng khi phát triển)

| Biến | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `DEBUG_SHOW` | `0` \| `1` | `1` | Bật cửa sổ OpenCV hiển thị video trực quan (tắt khi deploy) |

---

## 3. Control API — AI Service implement

HTTP/1.1, JSON, không auth. Backend timeout **3s** cho cả connect và read → handler trả ngay, việc
nặng đẩy sang worker.

### `GET /health`

```json
{ "ready": true, "version": "1.0.0" }
```

`200` chỉ khi model đã nạp xong. Electron poll endpoint này để xác định startup hoàn tất.

### `POST /streams/{jobKey}`

`jobKey` = `camera_stream.id` (số nguyên dạng chuỗi). Tạo mới hoặc **cập nhật nóng** job đang chạy —
Backend không gọi `DELETE` trước khi đổi params.

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

| Trường | Kiểu | Ràng buộc |
|---|---|---|
| `camera_id` | string | Số nguyên dạng chuỗi. Echo lại trong `OverlayFrame.camera_id` |
| `rtsp_url` | string | Path dạng `camera-{id}-main`. Kéo thẳng, không token |
| `params.classes` | string[] | Tên lớp COCO, ≥1 phần tử |
| `params.conf` | double | 0..1 |
| `params.roi` | `[[x,y],...]` \| `null` | **Phần trăm 0..100**, không nhận pixel. 2 điểm = đếm vạch · >2 điểm = đa giác kín, đếm vùng · `null` = toàn khung hình |
| `params.publish_fps` | double | Trần fps đẩy qua gRPC |
| `params.detect_every` | int | Chạy detector mỗi N frame, còn lại nội suy |
| `params.track_timeout` | double | Giây giữ track khi vật bị che khuất |

Bảng map từ cấu hình luồng phía Backend:

| Cấu hình | Giá trị | `conf` | `publish_fps` | `detect_every` | `track_timeout` |
|---|---|---|---|---|---|
| `sensitivity_level` | `sensitive` | 0.1 | — | — | 3.0 |
| | `balanced` | 0.3 | — | — | 2.0 |
| | `precise` | 0.8 | — | — | 1.0 |
| `performance_level` | `eco` | — | 15 | 5 | — |
| | `balanced` | — | 30 | 3 | — |
| | `max` | — | 60 | 1 | — |

Response `2xx` = đã nhận job, không chờ tới bbox đầu tiên.

### `DELETE /streams/{jobKey}`

Dừng job, giải phóng RTSP session. `404` được Backend coi như đã dừng.

### `GET /streams`

Danh sách job đang chạy kèm `fps` thực tế — Backend dùng để đối soát và chẩn đoán.

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

---

## 4. gRPC OverlayIngest — Backend implement, AI Service gọi

```protobuf
service OverlayIngest {
  rpc Health(HealthRequest) returns (HealthResponse);
  rpc PublishFrames(stream OverlayFrame) returns (PublishSummary);  // bbox, cho phép mất
  rpc PublishEvent(DetectionEvent) returns (EventAck);              // sự kiện đếm, không được mất
}

message OverlayFrame {
  string camera_id = 1;
  string job_key = 2;
  int64 captured_at_ms = 3;          // epoch millis UTC, mốc ĐỌC ĐƯỢC frame
  Resolution resolution = 4;
  repeated BoundingBox boxes = 5;
}

message Resolution { int32 width = 1; int32 height = 2; }

message BoundingBox {
  string track_id = 1;
  string class_name = 2;
  double confidence = 3;
  double x = 4;                      // pixel, góc trên-trái
  double y = 5;
  double w = 6;
  double h = 7;
  double velocity_x = 8;             // pixel/giây, vận tốc theo trục X
  double velocity_y = 9;             // pixel/giây, vận tốc theo trục Y
}

message PublishSummary { int64 received = 1; int64 dropped = 2; }

message DetectionEvent {
  string event_id = 1;               // UUID do AI Service sinh, Backend dedup theo field này
  string camera_id = 2;
  string job_key = 3;
  string track_id = 4;
  string class_name = 5;
  Kind kind = 6;
  int64 occurred_at_ms = 7;

  enum Kind {
    START = 0;                       // track mới xuất hiện trong khung
    IN = 1;                          // cắt qua vạch / vào vùng
    OUT = 2;                         // cắt ngược / rời vùng
  }
}

message EventAck { bool accepted = 1; }
```

Sinh stub:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. overlay.proto
```

### `PublishFrames` — kênh bbox

| Mục | Yêu cầu |
|---|---|
| Số stream | **1 stream duy nhất** cho toàn tiến trình, dùng chung mọi job. Không mở stream mỗi job, không unary mỗi frame |
| `captured_at_ms` | Mốc đọc được frame, không phải sau inference. Client delay video ~700ms rồi ghép bbox theo mốc này |
| `resolution` | Kích thước thật của frame mà toạ độ box dựa vào |
| Toạ độ box | Pixel, `x,y` góc trên-trái, `w,h` rộng/cao. Không phải `x1,y1,x2,y2`, không chuẩn hoá |
| `track_id` | Ổn định suốt vòng đời một vật thể — client nội suy box giữa 2 frame theo id này |
| `velocity_x`, `velocity_y` | Pixel/giây. Client dùng để nội suy (extrapolate) vị trí box mượt mà giữa 2 frame AI, bù đắp độ trễ inference trên CPU |
| Frame rỗng | Vẫn gửi `boxes = []` để client xoá box cũ |
| Backpressure | Backend queue 100 frame, drop **oldest**. `PublishSummary.dropped` > 0 kéo dài → giảm `publish_fps` |
| Reconnect | Backoff tối đa 5s, giữ nguyên job đang chạy khi stream đứt |

Kênh này cho phép mất dữ liệu.

### `PublishEvent` — kênh sự kiện

| Mục | Yêu cầu |
|---|---|
| Giao thức | Unary, mỗi sự kiện một RPC, có ack |
| Idempotent | Backend dedup theo `event_id` → retry vô hại |
| Retry | Backoff tối đa 5s, giữ hàng đợi trong bộ nhớ tới khi có ack. Backend restart không được làm mất sự kiện |
| Thứ tự | Không bắt buộc theo thứ tự — Backend sắp lại theo `occurred_at_ms` |
| Tần suất | Chỉ khi có hành động thật. Không phát lặp mỗi frame |

---

## 5. Luồng vận hành

### Tạo / cập nhật job

```
Backend reconcile (mỗi 15s, hoặc tức thì khi user sửa luồng)
  └─ tập mong muốn = camera_stream active + mode=standard + problem="Đếm người/phương tiện qua khu vực"
  └─ MediaMtxRelayManager.ensureAsync(camera)     # đảm bảo path RTSP tồn tại
  └─ POST /streams/{jobKey}                        # params đổi → gọi lại đúng endpoint này
```

Backend giữ map `jobKey -> request` đã gửi thành công, mỗi vòng so với tập mong muốn.

### Xoá job

```
jobKey rời tập mong muốn (luồng tắt / camera inactive / xoá camera_stream)
  └─ DELETE /streams/{jobKey}
       └─ AI Service: dừng RTSP, huỷ worker, ngừng đẩy frame của jobKey
```

### Publish

```
AI Service khởi động
  └─ Health()                                      # chờ Backend sẵn sàng
  └─ mở PublishFrames stream, giữ suốt vòng đời
       ├─ mỗi job: đọc frame → detect mỗi detect_every frame → track → throttle publish_fps → OverlayFrame
       └─ khi track cắt vạch / vào-ra vùng → PublishEvent

Backend
  ├─ frame → queue 100 (drop oldest) → chuẩn hoá pixel về 0..1 → STOMP /topic/overlay/{cameraId}
  └─ event → dedup theo event_id → lưu DB → khớp stream_notifications theo severity
```

### Khởi động lại

AI Service lên với **zero job**, không persist, không tự đăng ký. Backend bật lại toàn bộ ở vòng
reconcile kế tiếp (≤15s).

---

## 6. Yêu cầu tiến trình

| Mục | Yêu cầu |
|---|---|
| Khởi động | Electron start tuần tự MediaMTX → Backend → AI Service, chờ `GET /health` `200`. Timeout 120s |
| Tắt | `SIGTERM` (Windows: `taskkill`) → đóng gRPC stream, thả camera, exit trong 5s. Quá hạn → `SIGKILL`. Tiến trình con (ffmpeg…) phải chết theo |
| Log | `stdout`/`stderr`, mỗi dòng một sự kiện, không progress bar / ANSI cursor |
| Trạng thái | Không persist job xuống đĩa. Thư mục cài chỉ đọc |
| Đóng gói | Binary tự chạy (PyInstaller/Nuitka) tại `apps/frontend/runtime/ai-service/`, build đúng OS/arch đích, không console window trên Windows |
| Model | Trọng số đi kèm bộ cài, không tải lúc chạy lần đầu |

---

## 7. Giới hạn hiện tại

- Chỉ bài toán `mode = standard` + "Đếm người/phương tiện qua khu vực" được bật tracking.
- Sự kiện không kèm ảnh minh chứng.
- Không xác thực trên Control API lẫn gRPC; chỉ hợp lệ khi mọi tiến trình cùng máy, bind loopback.
