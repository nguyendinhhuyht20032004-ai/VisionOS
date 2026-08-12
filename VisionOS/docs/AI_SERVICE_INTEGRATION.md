# AI Service — Input/Output & tham số tích hợp

AI service chạy như **1 tiến trình dài hạn** (không phải 1 container/1 luồng), tự quản lý nhiều
luồng camera cùng lúc theo lệnh từ backend qua Control API — bật/tắt/sửa luồng không cần khởi động
lại container.

## 1. Control API — backend gán/gỡ luồng động

AI service tự expose 1 HTTP API để backend gọi vào:

**`POST /streams/{stream_id}`** — bắt đầu theo dõi 1 luồng

```json
{
  "camera_id": "cam-1",
  "rtsp_url": "rtsp://mediamtx:8554/cam-1",
  "params": {
    "classes": ["person", "car"],
    "conf": 0.25,
    "detect_every": 3,
    "track_timeout": 2.0,
    "publish_fps": 10.0
  }
}
```

- `camera_id` (Bắt buộc): Chuỗi định danh camera do Backend quy định.
- `rtsp_url` (Bắt buộc): Link RTSP nội bộ cấp bởi MediaMTX.
- `params.classes` (Tùy chọn): Mảng tên tiếng Anh của vật thể cần đếm. Khuyên dùng: `["person", "car", "truck", "motorcycle", "bus"]`.
- `params.conf` (Tùy chọn): Ngưỡng tin cậy của AI (mặc định 0.25).
- `params.detect_every` (Tùy chọn): Chạy quét hình ảnh (YOLO) sau mỗi N khung hình. Mặc định là `3` (rất mượt). Nếu để `1`, AI sẽ quét liên tục gây nặng CPU.
- `params.track_timeout` (Tùy chọn): Số giây (thời gian) cho phép mất dấu vật thể trước khi AI quyết định khai tử và bắn sự kiện `end`. Mặc định `2.0`. Mở rộng thời gian này nếu vật thể hay bị khuất sau cây/vật cản.
- `params.publish_fps` (Tùy chọn): Tần số (khung hình / giây) xuất toạ độ lên Redis. Mặc định `10.0`. Tăng lên `30.0` nếu muốn video Frontend mượt tuyệt đối (tốn băng thông hơn).

**`PATCH /streams/{stream_id}`** — cập nhật `params` của luồng đang chạy (không cần dừng/khởi động lại)

**`DELETE /streams/{stream_id}`** — dừng theo dõi luồng

## 2. Input — AI service tự đọc RTSP theo lệnh đã nhận

```text
rtsp://mediamtx:8554/{stream-path}
```

- Cùng Docker network với `mediamtx` (không truy cập được từ ngoài network)

## 3. Output — Redis Stream (JSON, không gửi ảnh/video)

### 3.1 Frame result (Toạ độ Box theo từng khung hình)

```json
{
  "type": "frame",
  "camera_id": "cam-123",
  "stream_id": "stream-456",
  "frame_timestamp": "2026-08-10T09:15:32.120Z",
  "resolution": {"width": 960, "height": 540},
  "boxes": [
    {
      "track_id": "trk-abc",
      "class": "person",
      "confidence": 0.87,
      "bbox": [10.5, 20.0, 50.0, 100.5]
    }
  ]
}
```

**Mô tả chi tiết các trường dữ liệu (Fields):**
- `type`: Cố định là `"frame"`.
- `camera_id`: ID của camera (nhận từ API POST /streams).
- `stream_id`: ID của luồng đang xử lý (nhận từ API POST /streams).
- `frame_timestamp`: Thời điểm khung hình được AI xử lý (định dạng ISO-8601).
- `resolution`: Độ phân giải gốc mà AI dùng để đếm (thường là 960x540). Bạn dùng toạ độ này để scale lại hộp bao (bbox) khi vẽ lên UI.
- `boxes`: Mảng chứa thông tin tất cả vật thể đang xuất hiện trong khung hình này.
  - `track_id`: ID duy nhất của vật thể (do AI gán để theo dõi sự di chuyển). ID này không đổi khi vật thể di chuyển.
  - `class`: Tên lớp vật thể (ví dụ: `"person"`, `"car"`, `"motorcycle"`).
  - `confidence`: Độ tin cậy của nhận diện (0.0 đến 1.0).
  - `bbox`: Mảng 4 phần tử `[x, y, w, h]` thể hiện toạ độ góc trên bên trái (`x`, `y`) và chiều rộng `w`, chiều cao `h` của bounding box (đơn vị pixel).

### 3.2 Track event (Sự kiện Xuất hiện / Biến mất)

```json
{
  "type": "track_event",
  "camera_id": "cam-123",
  "stream_id": "stream-456",
  "track_id": "trk-abc",
  "class": "person",
  "event": "start",
  "timestamp": "2026-08-10T09:15:30.000Z"
}
```

**Mô tả chi tiết các trường dữ liệu (Fields):**
- `type`: Cố định là `"track_event"`.
- `track_id`: ID của vật thể tạo ra sự kiện này (tương ứng với `track_id` trong gói `frame`).
- `class`: Loại vật thể của track này (`"person"`, `"car"`...).
- `event`: Loại sự kiện. Có 2 loại:
  - `"start"`: Vật thể mới bước vào và được nhận diện trong khung hình.
  - `"end"`: Vật thể đã đi ra khỏi khung hình hoặc bị khuất tầm nhìn (mất dấu quá 2 giây).
- `timestamp`: Thời điểm sự kiện xảy ra (định dạng ISO-8601).

## 4. Tham số Docker (ENV)

| ENV | Bắt buộc | Ý nghĩa |
| --- | --- | --- |
| `CONTROL_API_PORT` | ✅ | cổng expose Control API (mục 1) |
| `REDIS_URL` | ✅ | VD `redis://redis:6379` |
| `REDIS_STREAM_KEY` | ✅ | tên Redis Stream để publish kết quả |
| `RTSP_TRANSPORT` | tuỳ chọn | mặc định `tcp` |
| `RTSP_RECONNECT_INTERVAL_SEC` | tuỳ chọn | thời gian chờ trước khi kết nối lại khi luồng rớt |
| `OVERLAY_PUBLISH_FPS` | tuỳ chọn | giới hạn tần suất publish frame result |
| `REDIS_STREAM_MAXLEN` | tuỳ chọn | trần độ dài stream |
