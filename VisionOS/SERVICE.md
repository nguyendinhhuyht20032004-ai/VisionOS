# 🎥 Service AI đếm theo CAMERA

Đưa nguồn camera vào → **detect (YOLOv8) + track (ByteTrack) + làm mượt (DetectionsSmoother) +
đếm (LineZone/PolygonZone)** → xuất **luồng video annotate + số đếm realtime** qua REST API
và trang web. Đếm **người & phương tiện** (YOLO làm tốt). Xem thêm kế hoạch tổng thể:
[`docs/AI_SERVICE_PLAN.md`](docs/AI_SERVICE_PLAN.md).

## 🐳 Chạy bằng Docker (khuyên dùng)
```bash
docker compose up --build          # dựng api (FastAPI) + qdrant (vector DB)
# Mở http://localhost:8000  ·  GPU: sửa TORCH_CUDA=cu121 trong docker-compose.yml + bật khối deploy.devices
bash scripts/docker_smoke.sh       # kiểm thử nhanh: build→healthz→tạo job→đọc số đếm
```

## 💻 Chạy trên CPU (máy local, KHÔNG cần GPU)
**Chạy được** — YOLO/supervision/FastAPI/Qdrant đều chạy CPU; Docker mặc định là bản **CPU**
(`TORCH_CUDA=cpu`), compose đặt sẵn model nhẹ `yolov8m.pt` + `imgsz=960`. Chỉ **chậm hơn** GPU.

Mẹo cho CPU nhanh hơn (đặt trong `docker-compose.yml` hoặc export env):
```bash
YOLO_WEIGHTS=yolov8n.pt   # nhẹ & nhanh nhất (yolov8m = cân bằng, yolov8x = cần GPU)
YOLO_IMGSZ=640            # ảnh nhỏ → nhanh hơn (960/1280 chậm trên CPU)
# max_fps thấp (2–4) khi tạo job để đỡ nghẽn CPU
```

**Không cần Docker** (chạy thẳng):
```bash
# 1) Cài torch + torchvision KHỚP nhau (CPU) TRƯỚC — TRÁNH lỗi torchvision::nms
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# 2) Cài phần còn lại + chạy
pip install -r requirements-service.txt          # ultralytics + supervision + fastapi…
python run_service.py --host 0.0.0.0 --port 8000 # mở http://localhost:8000
# (Vector DB tự chạy in-memory nếu không có Qdrant — vẫn đếm + tìm kiếm bình thường.)
```
Tốc độ tham khảo trên CPU: yolov8n@640 ~ vài–chục fps · yolov8m@960 ~ 1–5 fps · yolov8x@1280 rất chậm.

### 🩺 Lỗi thường gặp
- **`RuntimeError: operator torchvision::nms does not exist`** — torch và **torchvision LỆCH
  phiên bản** (hay thiếu torchvision). Cài lại KHỚP nhau:
  ```bash
  pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
  (GPU: đổi `cpu` → `cu121`.) Sau đó chạy lại service. Docker: build lại (`docker compose build --no-cache`).
- **"Không lấy được frame" (nút Lấy frame)** — container KHÔNG truy cập được nguồn:
  - **File trên máy** → copy video vào thư mục **`./data`** (cạnh `docker-compose.yml`, đã mount),
    rồi nhập nguồn **`/data/<tên>.mp4`**. (Container không thấy ổ đĩa Mac/Windows.)
  - **Webcam `0`** → KHÔNG dùng được trong Docker (nhất là Mac). Chạy bằng **pip** (không Docker)
    để dùng webcam, hoặc dùng app IP-camera trên điện thoại (RTSP/HTTP).
  - **RTSP** → mở bằng mặc định của OpenCV/FFmpeg (không tự ép transport). RTSP công khai trên
    mạng (vd `test.rtsp.stream`) hay **chập chờn/hết hạn** — nếu lỗi, thử URL mới hoặc 1 file
    `.mp4` trong `./data` để xác nhận service OK. Nếu camera/Docker cần ép transport (NAT hay
    rớt UDP), tự đặt env `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` (hoặc `;udp`).
  - **HTTP** → dùng URL trỏ THẲNG tới file/luồng video (mp4/mjpeg). Thông báo lỗi trên web đã ghi RÕ nguyên nhân.

## ⚡ Chạy REALTIME + hết nhầm car/truck
Video chậm trên CPU (do model nặng) khiến track chập chờn → **cùng 1 xe lúc gán car lúc truck**.
Đã xử lý + cách tăng tốc:
- **Ổn định LỚP theo track** (tự động): nhãn của mỗi xe = lớp **xuất hiện nhiều nhất** của track
  đó (bình chọn đa số) → hết nhấp nháy car↔truck, màu ổn định. *(Số đếm tổng không đổi.)*
- **Model nhẹ + ảnh nhỏ** (mặc định service): `YOLO_WEIGHTS=yolov8n.pt`, `YOLO_IMGSZ=640`,
  `YOLO_CONF=0.3` (ngưỡng cao → ít box mờ gây nhầm). Đây là bộ realtime cho CPU.
- **`detect_every`** (ô "Detect mỗi N frame" trên web, hoặc field trong POST /api/jobs): chạy YOLO
  **mỗi N frame** (frame giữa vẽ lại box gần nhất) → nhanh gấp ~N lần. **Để = 1 cho MƯỢT NHẤT**
  (nhịp đều). Tăng N nhanh hơn nhưng nhịp **không đều** (frame có/không detect nhanh–chậm xen kẽ
  → hơi giật). Chỉ tăng khi cần throughput cao và chấp nhận hơi giật.
- Tăng `max_fps` (8–15) để hiển thị mượt hơn.

> Cần chính xác cao hơn (phân biệt loại xe tốt hơn): dùng `yolov8m/x` + `imgsz 960/1280` — nhưng
> nên có **GPU** để vẫn realtime.

### ▶️ Video FILE chạy TUA NHANH / GIẬT (nhanh–chậm) → đếm sai?
Kiến trúc đọc frame mặc định là kiểu **camera trực tiếp**: luồng nền chỉ giữ **frame mới nhất**
(bỏ frame cũ để realtime). Với **file** thì sai: OpenCV giải mã nhanh hết cỡ, model lúc nhanh
lúc chậm → reader nhảy vượt số frame không đều → video **tua nhanh + giật (nhanh–chậm)**, tracker
nhận nhầm nên **đếm sai**. Đã sửa — service **tự phân biệt file vs stream**:

- **FILE** (`/data/x.mp4` hoặc URL .mp4 — đọc được tổng số frame): đọc **ĐỦ frame, ĐÚNG THỨ TỰ**
  (hàng đợi có backpressure, **không bỏ frame**) và **phát đúng FPS gốc** → mượt, đúng tốc độ,
  đếm chính xác. Job xử lý MỌI frame theo thứ tự (không throttle theo `max_fps` nữa).
- **STREAM** trực tiếp (rtsp/webcam/mjpeg — `frame_count<=0`): giữ **frame mới nhất** + throttle
  `max_fps` như cũ (ưu tiên realtime, bỏ frame trễ). Không đổi hành vi.

**Muốn MƯỢT nhất:** để **Detect mỗi N frame = 1** (mặc định) → mỗi frame xử lý như nhau nên nhịp
**ĐỀU** (không giật). Đặt N=2,3… tuy nhanh hơn nhưng frame detect (chậm) xen frame vẽ-lại (nhanh)
→ nhịp **không đều = hơi giật**. Model quá nặng so với CPU thì video chạy hơi chậm hơn thực nhưng
vẫn **MƯỢT + đúng thứ tự** → muốn nhanh hơn dùng `yolov8n` + `imgsz 640` (mặc định) hoặc GPU.
Rebuild để nhận bản vá.

## 🚗 Phân loại xe (car/truck/bus) & vạch sót làn ngoài
- **Phân loại LOẠI XE:** service dùng **YOLO COCO**, phân biệt **car / truck / bus** (+ motorcycle),
  mỗi loại **một màu** riêng. **Mặc định GIỮ phân loại** (không gộp). Muốn gộp tất cả về 1 nhãn
  "vehicle" (1 màu) thì **tick ô** trên web hoặc gửi `"group_label": true`.
- **Cùng 1 xe bị gán TRÙNG 2 nhãn (vd vừa "truck" vừa "bus"):** NMS của YOLO mặc định theo TỪNG
  lớp nên 2 box chồng khít khác lớp đều được giữ. Đã thêm **NMS class-agnostic** (gộp box chồng
  > 0.8 IoU bất kể lớp, giữ box conf cao nhất) → **mỗi xe 1 box, 1 nhãn**. Chỉnh ngưỡng qua env
  `YOLO_DEDUP_IOU` (mặc định 0.8; đặt ≥1 để tắt).
- **Xe cao (cứu thương/thùng cao) đôi khi bị gọi truck/bus:** đây là giới hạn của COCO (chỉ có
  car/truck/bus/motorcycle, không có lớp riêng cho các xe đó) — chấp nhận, hoặc **gộp** về
  "vehicle" cho gọn nếu không cần tách loại. *(Đã ổn định nhãn theo track nên không còn nhấp
  nháy car↔truck.)*
- **Vạch chỉ đếm làn trong, sót làn ngoài:** (1) **vẽ vạch phủ HẾT các làn** (kéo dài qua cả làn
  ngoài); (2) đã sửa `LineZone` đếm theo **1 điểm neo (tâm)** thay vì cả 4 góc → xe TO ở làn ngoài
  (gần camera) cũng đếm được. Rebuild lại để nhận bản vá.

## ✏️ Vẽ vạch/vùng rồi ĐẾM (trên trang web)
Mở `http://localhost:8000` → nhập nguồn → **📷 Lấy frame** → chọn kiểu đếm → **VẼ** trực tiếp
lên khung (vạch = 2 điểm, vùng = đa giác) → **▶ Bắt đầu đếm**. Toạ độ lưu theo **%** nên khớp
mọi độ phân giải camera. Xem luồng annotate + số đếm realtime; mục "Sự kiện" tra vector DB.

## 🚀 TEST NHANH — KHÔNG cần camera thật (dùng file video như "camera")
Chưa có camera? Coi 1 file `.mp4` là nguồn — chạy đúng engine của service rồi xuất
video annotate + in số đếm. **Chạy được ngay trên Colab/Kaggle (có GPU):**
```bash
# đếm NGƯỜI qua vạch (video mẫu supervision, tự tải)
python run_stream.py --source people-walking.mp4 \
    --prompt person --orient horizontal --line-pos 0.5 --max-frames 200 \
    --out out.mp4

# đếm PHƯƠNG TIỆN (car/truck/bus)
python run_stream.py --source vehicles.mp4 \
    --prompt vehicle --orient horizontal --line-pos 0.6 --max-frames 300 --out out.mp4
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
| GET | `/healthz` | Health check (Docker/K8s) |
| GET | `/api/snapshot?source=…` | **Lấy 1 frame** để VẼ vạch/vùng |
| POST | `/api/jobs` | Tạo job đếm (kèm vạch/vùng người dùng vẽ) → chạy nền |
| GET | `/api/jobs` · `/api/jobs/{id}` | Liệt kê / số đếm hiện tại (JSON) |
| GET | `/api/jobs/{id}/frame.jpg` · `/mjpeg` | Frame annotate / luồng MJPEG |
| POST | `/api/jobs/{id}/stop` | Dừng job |
| GET | `/api/events?limit=` | Sự kiện gần nhất (vector DB) |
| POST | `/api/search/similar` | Upload ảnh → tìm vật GIỐNG (ReID) |
| GET | `/api/vectordb` | Trạng thái vector DB (qdrant / in-memory) |

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
