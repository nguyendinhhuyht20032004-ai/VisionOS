# Hướng dẫn Tự Kiểm thử (Test) Toàn bộ Hệ thống CoreML Native

Để bạn có cảm nhận chân thực nhất, tôi đã **tắt toàn bộ các tiến trình chạy ngầm** của tôi. Bây giờ, máy của bạn là một tờ giấy trắng. Bạn hãy mở ứng dụng **Terminal** trên Mac và làm theo đúng 4 bước sau (mỗi bước mở một tab/cửa sổ Terminal mới).

> [!TIP]
> **Sẵn sàng môi trường Hạ tầng Local (Không dùng Docker)**
> Trước khi chạy các bước dưới, bạn hãy gõ lệnh này để bật nền tảng Redis & RTSP Native:
> ```bash
> cd /Users/builder4/Documents/VisionOS/VisionOS_CoreML_Native/VisionOS
> ./start_local_infrastructure.sh
> ```

---

### Terminal 1: Bật AI Service (CoreML)
Tab này sẽ chạy Trí tuệ nhân tạo. Nó sẽ load model và đợi lệnh.
```bash
cd /Users/builder4/Documents/VisionOS/VisionOS_CoreML_Native/VisionOS
export YOLO_WEIGHTS=yolov8m.mlpackage
export CONTROL_API_PORT=8001
python3 run_service.py
```
*(Hãy để tab này mở, bạn sẽ thấy log nó in ra "Service chạy tại http://0.0.0.0:8001")*

---

### Terminal 2: Phát luồng Video (Camera ảo)
Tab này đóng vai trò như một chiếc Camera giám sát. Nó sẽ đẩy file video vào máy chủ RTSP (MediaMTX) để stream live.
```bash
cd /Users/builder4/Documents/VisionOS/VisionOS_CoreML_Native/VisionOS
python3 test_people.py
```
*(Hãy để tab này chạy ngầm)*

---

### Terminal 3: Ra lệnh cho AI bắt đầu quét luồng (Trigger API)
Tab này đóng vai trò như phần mềm Backend của bạn. Nó ra lệnh cho Server AI (ở Terminal 1) phải bắt đầu đọc luồng Camera (ở Terminal 2).

```bash
curl -X POST "http://localhost:8001/streams/test-people" \
     -H "Content-Type: application/json" \
     -d '{"camera_id": "cam-101", "rtsp_url": "rtsp://localhost:8554/test-people", "params": {"detect_every": 2}}'
```
*(Ngay khi bạn gõ lệnh này xong, hãy nhìn sang Terminal 1, bạn sẽ thấy AI bắt đầu in ra log FPS liên tục!)*

---

### Terminal 4: Bật Giao diện Trực quan (Visualizer)
Bước cuối cùng, giống hệt hôm qua! Đọc dữ liệu từ Redis và vẽ lên màn hình.
```bash
cd /Users/builder4/Documents/VisionOS/VisionOS_CoreML_Native/VisionOS
python3 test_redis_visualizer.py
```

> [!SUCCESS]
> Một cửa sổ OpenCV sẽ hiện lên trên màn hình Mac của bạn với những khung hình tracking siêu mượt nhờ thuật toán dự đoán Velocity! Bạn có thể nhấn phím `q` ở cửa sổ đó để thoát bất cứ lúc nào.

---

### Mở rộng: Test các API Quản lý Luồng (GET / PUT / DELETE)

Giờ hệ thống đang chạy, bạn có thể mở thêm một Terminal mới và thử các tính năng quản lý luồng bằng API:

**1. Xem danh sách luồng đang chạy (GET)**
API này sẽ trả về toàn bộ các camera đang được phân tích và thông số `fps` siêu cao của CoreML:
```bash
curl -s http://localhost:8001/streams | json_pp
```

**2. Cập nhật cấu hình luồng trực tiếp (PATCH)**
Lệnh này cho phép thay đổi cấu hình luồng đang chạy (ví dụ tăng FPS hoặc thay vùng nhận diện) mà không cần tắt camera.
```bash
curl -X PATCH "http://localhost:8001/streams/test-people" \
     -H "Content-Type: application/json" \
     -d '{"detect_every": 1, "publish_fps": 5.0}'
```
*(Sau khi chạy, hãy dùng lệnh `GET` ở trên để xem thông số đã được update vào luồng chưa, và nhìn sang cửa sổ AI Server bạn sẽ thấy log cập nhật được in ra!)*

**3. Tắt luồng (DELETE)**
Khi bạn muốn dừng nhận diện camera này để tiết kiệm RAM:
```bash
curl -X DELETE "http://localhost:8001/streams/test-people"
```
*(Cửa sổ Video OpenCV sẽ lập tức dừng nhận Bounding Box, và AI Server sẽ báo ngắt kết nối Camera).*
