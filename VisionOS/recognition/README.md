# 🧠 recognition — tầng mô hình nhận diện của VisionOS (ưu tiên bài toán ĐẾM)

Package này **xây dựng lại (rebuild)** phần mô hình nhận diện mà sản phẩm
`CV_product` mô tả ở frontend, theo đúng định hướng: **làm các bài toán đếm trước,
kiểm thử hiệu quả trước**, rồi mới mở rộng sang cảnh báo / kiểm lỗi.

Kiến trúc một chiều, tách bạch từng khâu:

```
detector  →  tracking  →  counting
(mô hình)   (giữ danh tính)  (đếm cắt vạch / chiếm vùng)
```

Điểm mấu chốt: detector là **duck-typed** (`detect(frame, prompt) -> DetectorResult`),
nên toàn bộ luồng chạy và **test được ngay trên CPU** bằng `ScriptedDetector`
(kịch bản chuyển động biết trước đáp số), rồi thay bằng mô hình thật khi lên GPU
mà **không phải sửa pipeline**.

---

## 4 bài toán đếm (khớp `AI_USECASES` ở frontend)

| Bài toán | use-case | Mô hình | Kiểu đếm |
|----------|----------|---------|----------|
| 🚶 Đếm người vào/ra | `uc-people-count` | YOLO-NAS-S | cắt vạch (ngang) |
| 🚗 Đếm xe ra vào | `uc-vehicle` | YOLO-NAS-S | cắt vạch (ngang) |
| 📦 Đếm kiện hàng băng chuyền | `uc-package` | LocateAnything-3B | cắt vạch (dọc) |
| 🧍 Phân tích hàng chờ | `uc-queue` | YOLO-NAS-S | chiếm vùng |

Hai bài đầu dùng **YOLO-NAS** (từ vựng cố định: người/xe). Bài kiện hàng dùng
**LocateAnything-3B** (open-vocabulary — đếm theo mô tả ngôn ngữ tự nhiên). Bài
hàng chờ đếm số đối tượng đang đứng trong vùng.

---

## Cấu trúc

```
recognition/
├── base.py         · Detection / BoundingBox / DetectorResult / MonitoringMode (thuần)
├── geometry.py     · point-in-polygon, cắt vạch có hướng (thuần, test kỹ)
├── zones.py        · Zone (đa giác) + CountingLine (vạch hai chiều)
├── tracking.py     · CentroidTracker — giữ danh tính qua frame (thuần Python)
├── scenarios.py    · 4 bài toán đếm chuẩn
├── counting.py     · CountingPipeline + CountResult (scorecard)
├── router.py       · suy luận model từ mô tả NL (port từ PipelineBuilder.tsx)
├── registry.py     · sổ đăng ký mô hình ↔ nghiệp vụ
└── detectors/
    ├── fake.py            · ScriptedDetector + sinh kịch bản (test, không GPU)
    ├── yolo_nas.py        · YOLO-NAS-S (super-gradients, import lazy)
    └── locate_anything.py · adapter bọc la_counting.LocateAnythingDetector (GPU)
```

Chỉ `numpy` là bắt buộc để chạy phần đếm + test. `torch` / `super-gradients` /
`transformers` / `opencv` chỉ cần khi chạy **mô hình thật** và đều được import
*lazy* (chỉ khi gọi `load()`).

---

## Kiểm thử hiệu quả (không cần GPU)

```bash
cd VisionOS
pip install numpy pytest

# 1) Unit test toàn bộ luồng detect → track → đếm (kịch bản tất định)
pytest tests_recognition -q          # => 33 passed

# 2) Scorecard trực quan cho cả 4 bài toán
python run_counting.py --selftest
```

Scorecard mẫu (`--selftest`):

```
▶ Đếm cắt vạch (line):
scenario  frames  IN  OUT  chiều                  total  tracks  status
people    30      1   1    Vào/Ra                 2      2       ✅ OK
vehicles  30      3   0    Chiều tới/Chiều lui    3      3       ✅ OK
packages  30      0   1    Qua vạch/Ngược (loại)  1      1       ✅ OK

▶ Đếm chiếm vùng (zone):
scenario  frames  trong_vùng  đỉnh_vùng  tracks  status
queue     30      2           2          2       ✅ OK
```

Các bất biến được khẳng định (bằng chứng "đếm hiệu quả"):

- **Không đếm lặp**: một đối tượng đi xuyên vạch chỉ đếm **đúng 1 lần**, dù xuất
  hiện ở nhiều frame.
- **Đúng chiều**: đi xuống ≠ đi lên → vào/ra tách riêng.
- **Không đếm nhầm**: vật đứng yên xa vạch → 0; không phát hiện → 0.
- **Đa vật**: 3 xe qua vạch → đúng 3; giữ danh tính riêng (`tracks == 3`).
- **Chiếm vùng**: đúng số đối tượng trong vùng, bỏ đối tượng ngoài vùng.

---

## Chạy mô hình thật (GPU)

Bài người/xe dùng YOLO; chọn 1 backend:

```bash
# (khuyến nghị) YOLOv8 — cài ổn định trên Kaggle/Colab:
pip install ultralytics opencv-python-headless
python run_counting.py --vehicles-video cars.mp4 --max-frames 100   # tự dùng ultralytics

# hoặc ép YOLO-NAS (super-gradients) nếu đã cài được:
python run_counting.py --vehicles-video cars.mp4 --yolo-backend super_gradients
```

`--yolo-backend auto` (mặc định) tự chọn: có super-gradients thì dùng YOLO-NAS,
không thì rơi về YOLOv8 — nên **không cần cài super-gradients** vẫn chạy được.

Bài kiện hàng (open-vocab) dùng LocateAnything-3B — **cần GPU NVIDIA**:

```bash
pip install transformers==4.57.1 accelerate opencv-python-headless
python run_counting.py --packages-video belt.mp4
```

Bài nào thiếu video sẽ được bỏ qua. Detector thật được nạp *lazy* theo nhu cầu
từng bài.

---

## Ví dụ dùng như thư viện

```python
from recognition import CountingPipeline, PEOPLE_IN_OUT
from recognition.detectors import load_yolo_nas

pipe = CountingPipeline(load_yolo_nas(), PEOPLE_IN_OUT)
result = pipe.run(frames_bgr, max_frames=150)   # frames_bgr: iterable numpy BGR
print(result.as_row())     # {'scenario': 'people', 'IN': .., 'OUT': .., ...}
```

Định tuyến model từ mô tả ngôn ngữ tự nhiên (giống frontend tự chọn mô hình):

```python
from recognition import infer_monitoring_config

cfg = infer_monitoring_config("đếm số người đi qua cổng vào ra")
# cfg.mode = STANDARD, cfg.model = 'YOLO-NAS-S', cfg.counting_type = 'line'
```
