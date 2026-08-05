# Hướng dẫn Tích hợp AI Service (Dành cho Dev Frontend/Backend)

Tài liệu này mô tả chi tiết cách khởi chạy và giao tiếp với **VisionOS AI Service** (hệ thống đếm người/xe bằng camera sử dụng YOLO + Supervision + FastAPI + Qdrant).

---

## 1. Khởi chạy Hệ thống

AI Service được đóng gói hoàn toàn bằng Docker. Bạn không cần cài đặt Python hay thư viện AI nào trên máy host.

**Yêu cầu:** Máy tính cài sẵn Docker và Docker Compose.

**Cách chạy:**
Mở Terminal tại thư mục gốc của project (nơi chứa file `docker-compose.yml`) và chạy:

```bash
docker compose up --build
```
*(Nếu muốn chạy ngầm, thêm cờ `-d`: `docker compose up -d --build`)*

**Kết quả:**
- **AI API Service** sẽ chạy tại: `http://localhost:8000`
- **Qdrant (Vector DB)** dashboard sẽ chạy tại: `http://localhost:6333/dashboard`

---

## 2. Các Nguồn Camera (Input Source) Hỗ trợ

Khi gọi API, tham số `source` chấp nhận các định dạng sau:
- **Luồng RTSP:** `rtsp://admin:password@192.168.1.10:554/stream`
- **Luồng HTTP/HTTPS (MJPEG):** `http://domain.com/video.mjpg`
- **File video cục bộ:** Bằng cách mount volume vào Docker (ví dụ: tạo thư mục `videos` ngang hàng file docker-compose.yml, cấu hình volume mapping ` - ./videos:/app/videos` trong compose, rồi truyền source là `/app/videos/file.mp4`).
- **Webcam USB:** `0` hoặc `1` (Cần cấu hình device mapping trong Docker).

---

## 3. Danh sách API Endpoints

### 3.1. Kiểm tra trạng thái hệ thống
- **Endpoint:** `GET /healthz`
- **Mô tả:** Dùng để check xem service đã khởi động xong và sẵn sàng nhận request chưa (hữu ích cho k8s liveness probe hoặc load balancer).
- **Response:**
  ```json
  {"status": "ok"}
  ```

### 3.2. Lấy 1 Frame ảnh mẫu để người dùng vẽ (Snapshot)
- **Endpoint:** `GET /api/snapshot`
- **Mô tả:** Trả về một ảnh tĩnh (JPEG) trích xuất từ camera ngay tại thời điểm gọi. Dùng để làm ảnh nền trên frontend cho user vẽ các điểm toạ độ (vạch/vùng đếm).
- **Query Params:**
  - `source` (string, bắt buộc): Đường dẫn camera (VD: `?source=rtsp://...`)
- **Response:** Binary image (Content-Type: `image/jpeg`).

### 3.3. Tạo Job đếm mới
- **Endpoint:** `POST /api/jobs`
- **Mô tả:** Khởi tạo một tiến trình đếm chạy ngầm.
- **Request Body (JSON):**
  ```json
  {
    "source": "rtsp://192.168.1.10:554/stream",
    "prompt": "person",  // "person", "car", "truck", "motorcycle", "bus"
    "counting_type": "line", // "line" (cắt vạch) hoặc "zone" (trong vùng)
    "line": [10, 50, 90, 50], // [x1, y1, x2, y2] tính theo % chiều rộng/cao của ảnh. VD: 50 = giữa ảnh.
    "zone": null, // Nếu counting_type="zone", truyền mảng toạ độ đa giác: [[x1,y1], [x2,y2], [x3,y3], ...] (%)
    "model": "yolo", 
    "max_fps": 10.0, // Giới hạn số frame xử lý mỗi giây để tiết kiệm CPU/GPU
    "in_label": "Vào", // (Tuỳ chọn) Text hiển thị chiều IN
    "out_label": "Ra" // (Tuỳ chọn) Text hiển thị chiều OUT
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "id": "e2d847fa", // Lưu ID này lại để query kết quả hoặc xem stream
    "status": "running"
  }
  ```

### 3.4. Lấy kết quả đếm (Polling)
- **Endpoint:** `GET /api/jobs/{id}`
- **Mô tả:** Frontend gọi liên tục (VD: mỗi 1 giây) để lấy số liệu đếm mới nhất.
- **Response (200 OK):**
  ```json
  {
    "id": "e2d847fa",
    "status": "running",
    "source": "rtsp://...",
    "counting_type": "line",
    "in": 12,        // Số lượng đi vào
    "out": 8,        // Số lượng đi ra
    "total": 20,     // Tổng số lượng cắt vạch
    "tracks": 15,    // Số lượng object unique đã bắt được
    "zone_current": 0, // Số object HIỆN ĐANG CÓ trong vùng (nếu counting_type="zone")
    "zone_peak": 0,    // Kỷ lục số object nhiều nhất từng có trong vùng
    "det_per_frame": 3.5, 
    "fps": 10.2,     // FPS thực tế hệ thống đang xử lý
    "frames": 450,
    "elapsed_s": 44.1
  }
  ```
  *(Nếu ID không tồn tại, trả về `404 Not Found`)*

### 3.5. Xem luồng video trực tiếp (Livestream Annotated)
- **Endpoint:** `GET /api/jobs/{id}/mjpeg`
- **Mô tả:** Luồng video đã được AI vẽ đè bounding box, vạch đếm, id... Dùng thẻ `<img>` của HTML để hiển thị trực tiếp.
- **Cách dùng (Frontend):**
  ```html
  <img src="http://localhost:8000/api/jobs/e2d847fa/mjpeg" width="100%" />
  ```

### 3.6. Dừng một Job
- **Endpoint:** `POST /api/jobs/{id}/stop`
- **Mô tả:** Huỷ bỏ luồng xử lý AI. Giải phóng tài nguyên.
- **Response (200 OK):**
  ```json
  {"stopped": true}
  ```

### 3.7. Truy xuất Sự kiện (Lịch sử đếm)
- **Endpoint:** `GET /api/events`
- **Mô tả:** Lấy danh sách các đối tượng đã đi qua camera gần đây nhất (dữ liệu móc từ Qdrant Vector DB).
- **Query Params:**
  - `limit` (integer, tuỳ chọn): Số lượng sự kiện trả về. Mặc định: 10.
- **Response:**
  ```json
  [
    {
      "time": "2026-08-05T15:00:00Z",
      "class_name": "person",
      "track_id": 42,
      "score": 0.95,
      "crop_image_url": "..." 
    },
    ...
  ]
  ```

---

## 4. Ghi chú Quan trọng cho Dev

1. **Toạ độ là % (Percentage):** Khi user vẽ trên Frontend, bạn phải quy đổi toạ độ pixel trên màn hình của user thành **Phần trăm (0 - 100)** so với chiều rộng/cao của ảnh mẫu. Điều này giúp hệ thống hoạt động đúng bất kể độ phân giải của luồng camera thực tế là bao nhiêu (HD, Full HD, 4K).
2. **Quản lý Job ID:** Vì mỗi job ăn khá nhiều tài nguyên (CPU/GPU), Frontend cần đảm bảo gọi API `POST /api/jobs/{id}/stop` khi người dùng tắt camera hoặc chuyển trang, tránh việc server quá tải do chạy ngầm các camera không còn ai xem.
3. **Cấu hình Model:** Trong `docker-compose.yml`, mặc định đang dùng `yolov8m.pt` (Medium) và ảnh `960px`. Nếu deploy trên server không có GPU rời, hãy đổi thành `yolov8n.pt` (Nano) và `640px` để đảm bảo hệ thống không bị nghẽn (Xem chi tiết file docker-compose.yml).
