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

## Đánh giá độ chính xác zero-shot trên COCO (`run_eval.py`) — Kaggle T4

Đo **Precision / Recall / F1 @ IoU** và **MAE đếm** của LocateAnything-3B trên
ảnh COCO val2017 có nhãn (person / car / bottle). Đây là "điểm số thật".

> ⚠️ **GHIM `transformers==4.57.1`** — đúng bản NVIDIA test. Bản mới hơn đổi loạt
> API nội bộ (`rope_theta`, tied-weights, `DynamicCache.to_legacy_cache`…) khiến
> code bundled của model vỡ. Ghim bản này là cách bền vững nhất.

**Cell cài đặt cho Kaggle (chạy 1 lần, rồi RESTART kernel):**

```bash
# Python 3.12 của Kaggle KHÔNG có wheel decord gốc → dùng eva-decord.
# KHÔNG ghim tokenizers (để pip tự giải, tránh xung đột với transformers 4.57.1).
!pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless -q
!pip install -q -U "numpy<2.0.0" "transformers==4.57.1" accelerate \
    "opencv-python-headless==4.11.0.86" "Pillow==11.1.0" eva-decord lmdb
```

Sau khi cài xong **Restart kernel** (Kaggle: *Run → Restart & Clear Cell Outputs*),
rồi chạy:

```bash
# Kiểm tra bộ đánh giá KHÔNG cần GPU/model:
python run_eval.py --selftest

# Đánh giá thật trên COCO (tự tải ảnh + nhãn), lưu ảnh dự đoán tô màu TP/FP/FN:
python run_eval.py --model locate --classes person car bottle \
    --n 50 --iou 0.5 --dedup-iou 0.9 --save-dir eval_out
```

Ảnh minh hoạ trong `eval_out/` tô: **xanh lá = TP**, **đỏ = FP (thừa)**,
**vàng = GT bị bỏ sót (FN)** — để thấy rõ model sai ở đâu.

---

## Test ĐẾM trên VIDEO thật, nhiều kịch bản (`run_scenarios.py`)

Ngoài đếm người (đã test kỹ), có sẵn **catalog 16 video công khai, nhiều góc quay**
cho **phương tiện vào/ra** và **dây chuyền sản xuất** — tải trực tiếp (không cần API
key), mỗi bài nhiều video, đã chỉnh sẵn vạch đếm.

| Bài toán | Số video | Ghi chú |
|----------|----------|---------|
| 🚗 Phương tiện | 4 | cao tốc, giao lộ, phố (nguồn tin cậy — đã bỏ video sai nhãn) |
| 📦 Dây chuyền | 6 | chiết chai (YOLO) + kiện hàng/đóng gói (open-vocab) |
| 🚶 Người | 4 | lối đi, ga tàu, siêu thị, quảng trường |

Mỗi video kèm **danh sách query DỄ→KHÓ** để test khả năng mô tả của LocateAnything.

```bash
python run_scenarios.py --list                       # video + query gợi ý
python run_scenarios.py --task vehicles              # đếm xe (YOLO, nhanh)
python run_scenarios.py --task conveyor --only milk  # đếm chai (YOLO, nhanh)

# ⭐ XEM TRƯỚC vạch/vùng trên MỌI video (không cần model) — ĐẶT VẠCH cho đúng rồi mới đếm:
python run_scenarios.py --preview prev               # vẽ lưới % + vạch/vùng lên frame CÓ vật
#   → prev/_ALL.jpg = ảnh TỔNG HỢP mọi trường hợp; prev/<task>_<key>.jpg = từng cảnh full-res
# Vạch/vùng đặt sai chỗ vật đi qua? Thử ngay toạ độ khác (theo % đọc trên lưới):
python run_scenarios.py --only milk --preview prev --line "35,0,35,100"   # vạch dọc lệch trái
python run_scenarios.py --only milk --save-dir out --line "35,0,35,100"   # ưng thì đếm luôn

# LƯU VIDEO OUTPUT (vẽ vạch/vùng + box + track-id + số đếm) để soi mắt thường:
python run_scenarios.py --task people --save-dir out_videos

# ĐẾM SẢN PHẨM với QUERY KHÓ (open-vocab) — test nhiều prompt trên mỗi video:
python run_scenarios.py --task conveyor --all-queries          # query gợi ý sẵn
python run_scenarios.py --task conveyor --queries "cardboard box,a damaged package"

# BỘ QUERY SUITE ĐẦY ĐỦ (nhiều trường hợp như bảng Excel) — phân theo nhóm:
python run_scenarios.py --task conveyor --only milk --suite    # ~27 query/6 nhóm
python run_scenarios.py --task people   --only walk --suite    # ~31 query/7 nhóm
```

`--preview [thư_mục]` là bước **QUAN TRỌNG trước khi đếm**: nó lấy 1 frame CÓ vật của
mỗi video (không nạp model, vài giây), kẻ **lưới % 0→100** rồi vẽ vạch (vàng) / vùng
(xanh) kèm **nhãn toạ độ**. Nhìn `_ALL.jpg` (tổng hợp mọi trường hợp) → nếu vạch đặt
sai chỗ vật đi qua, đọc toạ độ theo lưới rồi chỉnh bằng `--line`/`--zone` (áp cho mọi
video được lọc). Vạch đặt sai = đếm ra 0 (đây là lý do IN/OUT=0 trước đây, không phải model).

`--suite` chạy `QUERY_SUITES` (trong `recognition/video_catalog.py`) — mỗi bài toán
có **20–30+ query phân nhóm**: cơ bản · màu/trang phục · phụ kiện · hành động/quan hệ ·
đếm/nhóm · khó/phủ định · **tiếng Việt** (kiểm tra đa ngữ). Scorecard thêm cột **nhóm**.

Bài **NGƯỜI** tách 2 kiểu đếm (scorecard in riêng): **cắt VẠCH** (vào/ra) và
**đếm VÙNG** (occupancy — hợp cảnh người đi lại lộn xộn như quảng trường). Video
output tô: **vàng = vạch**, **xanh mờ = vùng**, **xanh dương = box + #track-id**.

Nhãn video được kiểm chứng theo nội dung thật (supervision đặt tên theo nội dung;
Pexels theo tiêu đề trang). Video sai nhãn từng gặp (nước chảy / giao thông bị gán
nhầm người) đã bị loại hoặc chuyển đúng bài.

Nguồn video: [supervision assets (Roboflow)](https://supervision.roboflow.com/assets/)
+ [Pexels video-files](https://www.pexels.com) (CDN ổn định, không cần key).
Downloader hỗ trợ 3 nguồn: `asset` (supervision, hash-check), `url` (.mp4 đầy đủ),
`pexels_id` (tự dò hậu tố chất lượng). Thêm/sửa kịch bản trong
`recognition/video_catalog.py` — chỉ cần thêm 1 `VideoScenario`.

Trên Kaggle: mở **`run_scenarios_kaggle.ipynb`** rồi Run All (GPU + Internet).

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
