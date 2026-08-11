# Tài liệu Cấu hình Tham số AI Service (Environment Variables)

Hệ thống AI Service đã được đóng gói sẵn trong Docker Image (`huyhungds/visionos-api:latest`). Bên trong Image đã có **sẵn các giá trị mặc định** (Defaults), nên nếu chạy chay thì hệ thống vẫn hoạt động bình thường. 

Tuy nhiên, để tối ưu hoá hiệu năng (cho máy chạy CPU yếu hoặc máy có card GPU xịn), hệ thống cung cấp các "Tham số Ẩn" (Biến môi trường - Environment Variables) để lập trình viên tự do "vặn núm" tinh chỉnh từ bên ngoài mà không cần sửa code.

Dưới đây là danh sách toàn bộ các tham số cấu hình:

## 1. Cấu hình Cơ bản (Basic Settings)

| Biến môi trường | Giá trị mặc định | Giải thích chi tiết |
| :--- | :--- | :--- |
| `QDRANT_URL` | `None` | Đường dẫn API của cơ sở dữ liệu Vector (Qdrant). Ví dụ: `http://qdrant:6333`. Nếu không truyền, tính năng lưu sự kiện (ReID) sẽ bị tắt. |
| `YOLO_WEIGHTS` | `yolov8x.pt` | File trọng số của AI. Dùng `yolov8n.pt` (nano) để chạy mượt realtime trên CPU. Nếu có GPU, dùng `yolov8s.pt`, `yolov8m.pt` hoặc `yolov8x.pt` để tăng độ chính xác lên mức tối đa. |
| `YOLO_CONF` | `0.15` | Ngưỡng tin cậy (Từ 0.0 đến 1.0). AI chỉ khoanh vùng đối tượng nếu độ chắc chắn vượt qua ngưỡng này. Cài cao (`0.5`) thì ít bắt nhầm, cài thấp (`0.15`) thì bắt được vật ở xa/mờ. |
| `YOLO_IMGSZ` | `1280` | Kích thước (độ phân giải) ảnh khi nạp vào AI. Ảnh càng to AI nhìn càng rõ đồ vật nhỏ, nhưng chạy càng chậm. Khuyến nghị: `640` cho CPU, `1280` cho GPU. |
| `YOLO_CLASSES` | `None` | Lọc chỉ bắt các loại vật thể nhất định. Ví dụ: `person,car,motorcycle`. Nếu để trống, AI sẽ nhận diện toàn bộ 80 lớp vật thể của bộ dữ liệu COCO. |

## 2. Cấu hình Nâng cao (Advanced AI / Tuning)

Các thông số này dành cho máy chủ có năng lực xử lý mạnh (Có GPU) để kích hoạt các tính năng dò tìm vật thể cực nhỏ (SAHI) hoặc làm mượt tracking.

| Biến môi trường | Giá trị mặc định | Giải thích chi tiết |
| :--- | :--- | :--- |
| `YOLO_MAX_DET` | `1000` | Số lượng vật thể tối đa mà AI được phép phát hiện trên 1 khung hình. |
| `YOLO_AUGMENT` | `0` | Truyền `1` để bật Test Time Augmentation (TTA). Khi bật, AI sẽ tự lật úp, xoay ảnh để phân tích nhiều góc độ trước khi ra kết quả cuối, giúp tăng cực độ sự chính xác (nhưng sẽ chậm máy). |
| `SMOOTHER_LEN` | `2` | Khung trượt (số lượng frame) dùng để làm mượt hộp bao (Bounding Box Smoothing). Số càng lớn hộp bao bám càng mượt, ít bị giật (jitter). |

## 3. Cấu hình Slicing Aided Hyper Inference (SAHI)

SAHI là kỹ thuật cắt nhỏ khung hình (như camera 4K) thành nhiều ô vuông nhỏ rồi soi bằng AI, giúp phát hiện kiến, chim, hoặc người ở cách xa hàng trăm mét mà mạng YOLO bình thường không thấy được.

| Biến môi trường | Giá trị mặc định | Giải thích chi tiết |
| :--- | :--- | :--- |
| `YOLO_TILE` | `0` | Truyền `1` để bật tính năng cắt lưới (SAHI). Chỉ nên bật khi có GPU. |
| `YOLO_TILE_WH` | `640` | Kích thước chiều ngang/dọc của từng ô cắt nhỏ (Tile Width/Height). |
| `YOLO_TILE_OVERLAP` | `128` | Độ chồng lấp (Overlap) giữa các ô cắt. Giúp vật thể lỡ bị cắt ngang mép vẫn được nhận diện nguyên vẹn. |
| `YOLO_DEDUP_IOU` | `0.8` | Ngưỡng IoU (Intersection over Union) để loại bỏ các hộp bao bị trùng lặp sau khi ghép các ô cắt lại với nhau. |

---
**Hướng dẫn sử dụng (Ví dụ):**
Khi chạy Image, truyền tham số thông qua cờ `-e`:
```bash
docker run -d --name ai-service \
  -p 8000:8000 \
  -e YOLO_WEIGHTS="yolov8n.pt" \
  -e YOLO_IMGSZ="640" \
  -e YOLO_TILE="1" \
  huyhungds/visionos-api:latest
```
