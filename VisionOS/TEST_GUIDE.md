# Hướng Dẫn Test Hệ Thống AI Service (Kiến trúc Contract mới)

Do hệ thống mới sử dụng gRPC để giao tiếp với Backend, nếu Backend thực tế (Electron) chưa sẵn sàng thì AI Service sẽ không thể gửi dữ liệu. Vì vậy, để test độc lập AI Service, mình đã tạo sẵn một file **Mock Backend** (`mock_backend.py`). File này sẽ đóng giả làm Backend để nhận dữ liệu từ AI Service và in ra màn hình.

Dưới đây là các bước để bạn test toàn bộ vòng đời của hệ thống.

---

### Bước 1: Chuẩn bị Model OpenVINO
Nếu bạn chưa tạo mô hình OpenVINO, hãy mở Terminal (Powershell) tại thư mục `VisionOS` và chạy:
```powershell
# Kích hoạt môi trường ảo (nếu bạn có dùng venv)
# .venv\Scripts\activate

python export_openvino.py --weights yolov8m.pt
```

---

### Bước 2: Chạy Mock Backend
Mở Terminal **thứ nhất**, chạy Mock Backend để giả lập máy chủ gRPC (lắng nghe ở cổng `9090`):
```powershell
python mock_backend.py
```
> [!NOTE]
> Bạn sẽ thấy thông báo: `🚀 Mock Backend đang chạy tại 127.0.0.1:9090`

---

### Bước 3: Khởi động AI Service
Mở Terminal **thứ hai**, chạy AI Service Native:
```powershell
./run_native.sh
```
*(Trên Windows, nếu không chạy được file `.sh`, bạn chạy thẳng lệnh này: `python run_service.py`)*

> [!TIP]
> **Dấu hiệu thành công:** 
> 1. Tại Terminal của AI Service, bạn sẽ thấy: `[gRPC] Backend sẵn sàng (Health OK)`
> 2. Tại Terminal của Mock Backend, bạn sẽ thấy: `[Mock Backend] Nhận request Health Check -> Phản hồi: ready=True`

---

### Bước 4: Gọi API Test để chạy Stream
Mở Terminal **thứ ba** và chạy file `test_people.py`. Mình đã viết lại file này để tự động gọi REST API tạo luồng xử lý và giám sát nó:

```powershell
# Chạy mặc định (Nó sẽ cố lấy luồng RTSP ở rtsp://127.0.0.1:8554/camera-1-main)
python test_people.py

# HOẶC BẠN CÓ THỂ TEST VỚI FILE VIDEO MP4 (KHUYÊN DÙNG ĐỂ TEST NHANH):
# (Sửa lại đường dẫn tới file video mp4 có sẵn trên máy bạn)
python test_people.py --source C:\path\to\your\video.mp4
```

> [!IMPORTANT]
> **Quan sát kết quả:**
> Ngay khi `test_people.py` gửi lệnh tạo luồng, AI Service sẽ bắt đầu đọc frame. Bạn hãy chuyển sang xem **Terminal thứ nhất (Mock Backend)**. Bạn sẽ thấy Mock Backend nhận được tấp nập các gRPC frames và các sự kiện cắt vạch (IN/OUT) như sau:
> ```
> [Mock Backend] Đã nhận 30 frames. Frame mới nhất: Stream 'test-1' có 5 objects.
> 
> [Mock Backend] 🔔 NHẬN SỰ KIỆN QUAN TRỌNG 🔔
>   - Event ID: abc123xyz
>   - Stream: test-1 | Camera: 1
>   - Track ID: 12 | Object: person
>   - Loại sự kiện: START
> ```

---

### Tóm tắt kiến trúc bạn đang test:
1. `test_people.py` đóng vai trò người dùng/Backend gọi **REST API** (POST `/streams/test-1`) để bảo AI Service phân tích video.
2. AI Service đọc video, chạy mô hình YOLO qua OpenVINO để tìm Bounding Box và track.
3. AI Service liên tục đẩy kết quả (Frame) và sự kiện (Events) sang `mock_backend.py` qua giao thức **gRPC** (cổng 9090).
4. `test_people.py` định kỳ gọi GET `/streams` để kiểm tra tiến độ, và sau 30 giây nó sẽ gọi lệnh `DELETE` để tắt luồng.
