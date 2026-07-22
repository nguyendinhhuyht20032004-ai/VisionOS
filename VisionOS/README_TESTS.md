# 🧪 Bộ test chuẩn cho LocateAnything-3B (3 bài toán đếm)

Bộ test kiểm thử mô hình `nvidia/LocateAnything-3B` như một **hệ thống đếm** trên
3 bài toán, dùng chung đường ống `detect → track (ByteTrack) → đếm cắt vạch (LineZone)`:

| # | Bài toán | prompt | Vạch |
|---|----------|--------|------|
| 1 | 🚶 Đếm người **ra / vào** toà nhà | `person` | ngang, tách 2 chiều Vào/Ra |
| 2 | 📦 Đếm **sản phẩm** băng chuyền | `object` | dọc, chặn dòng chảy ngang |
| 3 | 🚗 Đếm **phương tiện** qua lại | `car` | ngang, nửa dưới khung |

Bộ test có **2 tầng**, tách bạch phần *chạy được không cần GPU* và phần *cần GPU*:

```
VisionOS/
├── la_counting/                     ← THƯ VIỆN (đã tách khỏi notebook, import được)
│   ├── parsing.py                   ·  Detection + parse text model → bbox (thuần)
│   ├── scenarios.py                 ·  3 bài toán + dựng LineZone
│   ├── counting.py                  ·  CountingPipeline (ByteTrack + LineZone)
│   └── detector.py                  ·  LocateAnythingDetector (torch import lazy)
├── tests/                           ← 43 UNIT TEST pytest — KHÔNG cần GPU
│   ├── test_parsing.py              ·  16 test: 3 định dạng bbox + quy đổi toạ độ
│   ├── test_scenarios.py            ·  11 test: cấu hình + validate + dựng vạch
│   ├── test_counting.py             ·  6 test:  logic cắt vạch (LineZone thật)
│   ├── test_pipeline.py             ·  8 test:  full detect→track→count (detector giả)
│   ├── fakes.py                     ·  ScriptedDetector + sinh kịch bản chuyển động
│   └── conftest.py
├── run_benchmark.py                 ← CLI chạy 3 bài toán (có --selftest không cần GPU)
├── locate_anything_test_suite.ipynb ← NOTEBOOK Kaggle self-contained (chạy model thật)
├── requirements-test.txt
└── pytest.ini
```

---

## Tầng 1 — Unit test (KHÔNG cần GPU, chạy trong ~1.5s)

Kiểm chứng toàn bộ **logic tất định** của harness (phần dễ sai nhất): parse toạ
độ, quy đổi thang 0–1000 → pixel, thứ tự ưu tiên 3 định dạng đầu ra, và logic
đếm cắt vạch dùng **supervision thật** (không mock). Mô hình 6GB được thay bằng
một *detector giả* di chuyển vật thể theo kịch bản dựng sẵn.

```bash
cd VisionOS
pip install -r requirements-test.txt
pytest
# => 43 passed
```

Các bất biến được khẳng định (đã xác minh thực nghiệm với supervision 0.29):

- **Parse**: `<box><..></box>`, `[x1,y1,x2,y2]` (tự đoán thang 0–1 hay 0–1000),
  `<loc_n>` theo bộ 4; loại box suy biến (`x2≤x1`/`y2≤y1`); thứ tự ưu tiên đúng.
- **Đếm cắt vạch**: vạch ngang → đi *xuống* = OUT, đi *lên* = IN; vạch dọc → sang
  *phải* = IN. Người/sản phẩm/xe đi qua vạch được đếm **đúng 1 lần**; 2 vật = 2 lần;
  không có phát hiện = 0.
- **Cấu hình scenario**: `validate()` bắt các sai lệch (prompt rỗng, vạch sai
  hướng, vị trí ngoài (0,1), anchor sai, resolution/max_frames không hợp lệ).

## Tầng 2 — Benchmark mô hình thật (cần GPU T4)

### A. Trên Kaggle — `locate_anything_test_suite.ipynb`
Notebook **self-contained** (không cần các file `.py` ở trên):

1. Bật **GPU T4 + Internet**, mở notebook, **Run All**.
2. **Cell 6** chạy *smoke test* (không cần model) — xác nhận đường ống đếm đúng
   trước khi tốn GPU.
3. **Cell 9**: trỏ đường dẫn video *people* / *conveyor* tới file bạn upload
   (*vehicles* tự tải video mẫu của Roboflow). Bài thiếu video sẽ được bỏ qua.
4. **Cell 10** in **scorecard** cho cả 3 bài; **Cell 11** hiển thị video annotate.
5. **Cell 12** (tuỳ chọn): đo Precision / Recall / mAP@50 nếu có dataset ảnh YOLO.

### B. Dòng lệnh — `run_benchmark.py`
```bash
# Kiểm tra đường ống KHÔNG cần GPU (dùng detector giả) — chạy được ở mọi máy:
python run_benchmark.py --selftest

# Chạy model thật (Kaggle/GPU): tự tải video xe, trỏ video người & băng chuyền:
python run_benchmark.py \
    --people-video   /kaggle/input/.../people.mp4 \
    --conveyor-video /kaggle/input/.../belt.mp4 \
    --max-frames 150
```

Ví dụ scorecard (`--selftest`):
```
scenario   frames    IN   OUT  total  det/frame     fps  status
people         26     0     2      2       2.00  777.60  ✅ OK
conveyor       26     1     0      1       1.00 1058.76  ✅ OK
vehicles       26     0     2      2       2.00  893.25  ✅ OK
```

---

## Ghi chú về harness (phát hiện khi viết test)

- **Confidence cố định 0.85**: model chỉ phát toạ độ, không phát điểm tin cậy,
  nên mọi box bị gán cứng `0.85`. Vì vậy `mAP@50` (Cell 12) có đường PR *suy biến*
  — coi như tham khảo; chỉ tiêu chính là **số đếm cắt vạch**.
- **Chiều IN/OUT phụ thuộc bố trí camera**: quy ước down=OUT/up=IN chỉ là hình
  học. Nếu ngược thực tế, đổi `in_label`/`out_label` của scenario — logic không đổi.
- **`sv.ByteTrack` đã deprecate** ở supervision ≥0.28 (còn chạy tới 0.30). Giữ
  nguyên cho khớp notebook gốc; cảnh báo được lọc trong `pytest.ini`.
- **Vá cho T4**: ép `float16` (Turing không có bfloat16 kernel) và
  `attn_implementation="sdpa"`; ghim `transformers==4.57.1`. Giữ đúng như notebook.

## Đổi cấu hình nhanh
- Đổi đối tượng đếm: sửa `prompt` (`person`→`worker`, `car`→`truck`, `object`→`bottle`).
- Đổi vị trí vạch: `LineConfig(position=…)` (tỉ lệ 0..1), `orientation`, `anchor`.
- Thêm bài toán mới: khai báo một `Scenario` nữa trong `la_counting/scenarios.py`
  và (nếu muốn test) thêm case vào `tests/`.
