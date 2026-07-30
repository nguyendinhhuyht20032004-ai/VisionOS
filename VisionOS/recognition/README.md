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

# QUERY SUITE — mặc định LITE (1-2 query/nhóm, NHANH, hợp Colab session ngắn):
python run_scenarios.py --task conveyor --only pkg  --suite                 # ~12 query, ít frame
python run_scenarios.py --task conveyor --only pkg  --suite --suite-per-group 1 --max-frames 40  # nhanh nhất
python run_scenarios.py --task conveyor --only pkg  --suite-full            # bảng ĐẦY ĐỦ (~27 query, chậm)
```

`--preview [thư_mục]` là bước **QUAN TRỌNG trước khi đếm**: nó lấy 1 frame CÓ vật của
mỗi video (không nạp model, vài giây), kẻ **lưới % 0→100** rồi vẽ vạch (vàng) / vùng
(xanh) kèm **nhãn toạ độ**. Nhìn `_ALL.jpg` (tổng hợp mọi trường hợp) → nếu vạch đặt
sai chỗ vật đi qua, đọc toạ độ theo lưới rồi chỉnh bằng `--line`/`--zone` (áp cho mọi
video được lọc). Vạch đặt sai = đếm ra 0 (đây là lý do IN/OUT=0 trước đây, không phải model).

**TỰ VẼ bằng chuột (khỏi đoán toạ độ)** — `recognition/draw_tool.py` cho canvas HTML+JS
(chạy trên Kaggle, không cần cài gì) để bấm chuột vẽ **nhiều vùng + nhiều vạch** ngay
trên frame thật; công cụ tự xuất chuỗi `--zone`/`--line` theo %. Trong notebook:

```python
from IPython.display import HTML
from recognition.draw_tool import draw_for
HTML(draw_for("subway"))       # walk/store/square/milk/conv_pkg/'giao lộ'…
```

Copy toạ độ ra rồi chạy `--zone/--line`, hoặc gửi lại để đưa vào catalog (mỗi vùng
thành 1 `VideoScenario` riêng — như `market-square` đang có cả bài VẠCH lẫn bài VÙNG
trên cùng 1 video, scorecard in mỗi vùng 1 dòng).

> ⏳ **Bài open-vocab tải model ~6GB, lần đầu 3–8 phút.** `run_scenarios.py` giờ nạp
> model NGAY tại một mốc in rõ ("🧠 Nạp LocateAnything-3B …") thay vì nạp lười giữa
> vòng đếm (trước đây trông như treo → dễ bị bấm Stop). `KeyboardInterrupt` trong lúc
> tải = **bị ngắt giữa chừng, KHÔNG phải lỗi code** — chạy lại và chờ. Trên Kaggle nên
> chạy cell "⏳ Tải model 3B TRƯỚC" một lần để kéo model về cache trước khi đếm.

`--suite` chạy `QUERY_SUITES` (trong `recognition/video_catalog.py`) — mỗi bài toán có
**20–30+ query phân nhóm**: cơ bản · màu/trang phục · phụ kiện · hành động/quan hệ ·
đếm/nhóm · khó/phủ định · **tiếng Việt**. Scorecard thêm cột **nhóm**.

**Tối ưu Colab/session ngắn:** `--suite` mặc định chạy bản **LITE** — chỉ **1-2 query
đại diện mỗi nhóm** (vd màu = *đỏ/trắng*) + tự giảm `--max-frames` (60) và bỏ bớt frame
(`--stride` 2) → nhanh hơn nhiều lần. Chỉnh: `--suite-per-group N` (1 = nhanh nhất),
`--max-frames`, `--stride`. Cần **bộ đầy đủ** thì dùng `--suite-full` (chậm). Nên chạy
**1 video** với `--only` khi test open-vocab trên Colab (model 3B chậm ~1-2s/frame).

**Engine đếm:** mặc định `--engine auto` dùng **supervision** (`ByteTrack` +
`LineZone`/`PolygonZone`) nếu đã cài — tracker mạnh hơn `CentroidTracker` tự viết nên
đếm **cắt vạch / chiếm vùng chính xác hơn**; overlay + số đếm vẽ bằng cv2. Ép bộ tự
viết (thuần Python, test CPU): `--engine builtin`. Cả hai trả cùng `CountResult`/scorecard.

**Nhận diện yếu (bỏ sót người/xe)?** Mặc định dùng **YOLOv8m** (mạnh hơn nano nhiều),
`--confidence 0.25`, `--imgsz 960`. Vật NHỎ (xe top-down) → thêm `--imgsz 1280`. Vẫn
sót → model mạnh hơn: `--yolo-weights yolov8l.pt` / `yolov8x.pt` (chậm hơn) hoặc hạ
`--confidence 0.2`. Chỉnh nhanh qua env: `YOLO_WEIGHTS`, `YOLO_IMGSZ`, `YOLO_CONF`.

**Ảnh AERIAL/top-down (xe nhìn từ trên rất nhỏ)?** Thêm **`--tile`** — YOLO chạy **toàn
ảnh + nhiều ô 640px** rồi gộp NMS (supervision `InferenceSlicer`): vật nhỏ to lên trong
từng ô, vật to ở gần vẫn bắt bằng lượt toàn ảnh. Chậm hơn ~số ô lần. Chỉnh ô qua
`YOLO_TILE_WH` / `YOLO_TILE_OVERLAP`.

**Giữ chi tiết (đừng hạ res):** `--proc-width 1920` xử lý ở độ phân giải cao (thay vì
hạ về scenario) → detect nét hơn; vạch/vùng theo %% nên vẫn khớp. Bản mạnh nhất cho cảnh
khó: `--yolo-weights yolo11x.pt --tile --proc-width 1920 --confidence 0.15` (chậm).

**RT-DETR (detector transformer — supervision demo hay dùng):** thử
`--yolo-weights rtdetr-x.pt` (hoặc `rtdetr-l.pt`). Detector này KHÁC YOLO, đôi khi bắt
tốt hơn ở cảnh đông. Lưu ý: **supervision KHÔNG tự detect** — nó chỉ track/vẽ/đếm trên
kết quả của model ngoài (YOLO/RT-DETR), đúng như package này đang làm.

**Ảnh TOP-DOWN/DRONE (từ trên xuống):** YOLO COCO học ảnh chụp NGANG nên bắt kém góc
bird's-eye. Cắm **model train trên ảnh aerial (VisDrone)**:
`--yolo-weights hf://<owner>/<repo>/<file>.pt --classes "car,van,truck,bus"` (tải model
từ HuggingFace; `--classes` để lọc theo tên lớp của model đó, vì tên khác COCO). Cũng
nhận http URL hoặc đường dẫn .pt cục bộ. Đây là cách supervision demo top-down detect được.

Video output ghi đúng **FPS nguồn** (chia cho `--stride`) nên không còn phát chậm như
slow-motion.

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

**Notebook chạy được cả Google Colab lẫn Kaggle** — mở `run_scenarios_colab.ipynb`
(Colab) hoặc `run_scenarios_kaggle.ipynb` (Kaggle); cả hai giống nhau, **cell 1 tự nhận
diện** môi trường và đặt thư mục làm việc (`/content` cho Colab, `/kaggle/working` cho
Kaggle). Nhớ **bật GPU** trước: Colab → *Runtime → Change runtime type → T4 GPU*;
Kaggle → *Settings → Accelerator → GPU*. Rồi Run All.

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
