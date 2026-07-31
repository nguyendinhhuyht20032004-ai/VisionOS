# 🎥 Service AI đếm theo CAMERA

Đưa nguồn camera vào → **detect + track (ByteTrack) + làm mượt (DetectionsSmoother) +
đếm (LineZone/PolygonZone)** → xuất **luồng video annotate + số đếm realtime** qua REST API
và trang web. Dùng chung engine đã kiểm chứng: **YOLO** cho người/xe (nhanh), **LocateAnything-3B**
cho sản phẩm open-vocab (thùng/cà chua…).

## 🚀 TEST NHANH — KHÔNG cần camera thật (dùng file video như "camera")
Chưa có camera? Coi 1 file `.mp4` là nguồn — chạy đúng engine của service rồi xuất
video annotate + in số đếm. **Chạy được ngay trên Colab/Kaggle (có GPU):**
```bash
# đếm cà chua trên video mẫu (đã có sẵn trong repo)
python run_stream.py --source sample_videos/tomatoes_sorting.mp4 \
    --prompt tomato --orient horizontal --line-pos 0.72 --max-frames 60 \
    --out out.mp4

# đếm kiện hàng
python run_stream.py --source sample_videos/packages_belt.mp4 \
    --prompt object --orient horizontal --line-pos 0.60 --max-frames 60 --out out.mp4
```
In ra số đếm từng frame + lưu `out.mp4` (annotate). **Đây là cách xác minh hệ thống chạy
hiệu quả mà không cần camera.** Khi có camera thật, chỉ đổi `--source` thành RTSP/URL/`0`.

Xem `out.mp4` trên Colab/Kaggle (đổi H.264 cho trình duyệt):
```bash
ffmpeg -y -i out.mp4 -vcodec libx264 -pix_fmt yuv420p out_h264.mp4
```
```python
from IPython.display import Video; Video("out_h264.mp4", embed=True, width=720)
```

## Cài & chạy service web (khi cần giao diện / camera trực tiếp)
```bash
pip install -r requirements-service.txt          # fastapi + uvicorn
# (cần thêm torch/transformers/ultralytics/supervision như phần counting để chạy model thật)
python run_service.py --host 0.0.0.0 --port 8000
```
Mở trình duyệt: **http://<máy>:8000** → nhập nguồn camera, prompt, vạch/vùng → xem trực tiếp.
Chưa có camera vẫn test được: nhập `--source` = đường dẫn file `.mp4` (service sẽ **lặp lại**
file như luồng liên tục).

### Chạy service trên Colab/Kaggle (không có máy local)
Colab không mở cổng ra ngoài trực tiếp → dùng tunnel:
```python
!pip install -q fastapi uvicorn pyngrok
from pyngrok import ngrok
import threading, uvicorn
from recognition.service.app import app
threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000), daemon=True).start()
print("Mở:", ngrok.connect(8000).public_url)     # cần token ngrok miễn phí
```
(Đơn giản hơn cho việc test: dùng `run_stream.py` ở trên — không cần web/tunnel.)

## Nguồn camera hỗ trợ
- RTSP: `rtsp://user:pass@ip:554/stream`
- HTTP(S) stream / file: `http://.../cam.mjpg`, `/kaggle/working/video.mp4`
- Webcam: `0` (chỉ số thiết bị)

## API
| Method | Endpoint | Ý nghĩa |
|---|---|---|
| POST | `/api/jobs` | Tạo job đếm trên 1 camera → trả job (chạy nền) |
| GET | `/api/jobs` | Liệt kê job |
| GET | `/api/jobs/{id}` | Số đếm hiện tại (JSON) |
| GET | `/api/jobs/{id}/frame.jpg` | Frame annotate mới nhất |
| GET | `/api/jobs/{id}/mjpeg` | Luồng MJPEG annotate (dán vào `<img src>`) |
| POST | `/api/jobs/{id}/stop` | Dừng job |

### Tạo job (ví dụ)
```bash
curl -X POST http://localhost:8000/api/jobs -H 'Content-Type: application/json' -d '{
  "source": "rtsp://192.168.1.10:554/stream",
  "prompt": "person",
  "counting_type": "line",
  "line": [0, 50, 100, 50],
  "model": "auto",
  "max_fps": 4
}'
```
- `prompt`: đối tượng cần đếm. `person`/`car` → YOLO (nhanh); `object`/`carton box`/`tomato` → LocateAnything.
- `counting_type`: `line` (cắt vạch vào/ra) hoặc `zone` (đếm trong vùng).
- `line`: `[x1,y1,x2,y2]` theo % (0..100). `zone`: `[[x,y],…]` ≥3 đỉnh theo %.
- `model`: `auto` | `yolo` | `locate`. `max_fps`: giới hạn tốc độ (LocateAnything chậm → 2–4).

Kết quả (`GET /api/jobs/{id}`): `in`/`out`/`total` (cắt vạch) hoặc `in_zone`/`zone_peak` (vùng),
kèm `tracks` (số vật khác nhau), `det_per_frame`, `fps`, `status`.

## Kiến trúc (recognition/service/)
- `engine.py` — `StreamingCounter`: detect→track→smooth→count→annotate cho **từng frame** (có state).
- `camera.py` — `FrameSource`: đọc frame ở luồng nền, tự kết nối lại (RTSP/HTTP/file/webcam).
- `builder.py` — dựng `CountScenario` + chọn/nạp detector (cache dùng chung; LA nạp 1 lần).
- `app.py` — FastAPI: quản lý job (mỗi job 1 luồng), API + trang web.

## Lưu ý
- **LocateAnything chậm** (~vài giây/frame) → không realtime; đặt `max_fps` thấp, chấp nhận trễ.
  Người/xe dùng YOLO thì mượt (chục fps).
- Mỗi kiểu model nạp **một lần** rồi dùng lại cho mọi job (tiết kiệm VRAM). GPU T4 chạy được LA float16.
- Vạch/vùng theo **%** nên không phụ thuộc độ phân giải camera.
