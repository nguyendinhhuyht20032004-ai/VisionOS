# Detection — AI phát hiện vật thể và gửi dữ liệu cho Backend

Tài liệu bàn giao cho hai hạng mục:

1. **AI trả ra dữ liệu detection cho vật thể**
2. **AI gửi dữ liệu detection cho backend**

Cả hai đã chạy được và đã kiểm chứng trên hệ thống thật.

---

## 1. AI phát hiện được gì

Model **YOLOv8** nhận diện 80 lớp vật thể của bộ dữ liệu COCO. Backend chọn lớp cần theo dõi
qua trường `classes` khi gán camera.

| Backend gửi | AI thực sự bắt |
|---|---|
| `["person"]` | person |
| `["person","car"]` | person, car |
| `["car","truck"]` | car, truck |
| `["vehicle"]` | car, motorcycle, truck, bus (nhóm gộp sẵn) |
| bỏ trống | **chỉ person** — không phải cả 80 lớp |

Tên lớp dùng tiếng Anh theo chuẩn COCO (`person`, `car`, `motorcycle`, `truck`, `bus`,
`bicycle`…). Bảng gộp nhóm nằm ở `recognition/coco.py`.

---

## 2. Mỗi vật thể trả ra dữ liệu gì

Với mỗi vật thể xuất hiện trong khung hình, AI trả về 4 thông tin:

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `track_id` | chuỗi | Mã theo dõi. **Giữ nguyên** suốt quá trình vật di chuyển trong khung hình |
| `class` | chuỗi | Loại vật thể (`"person"`, `"car"`…) |
| `confidence` | số thực 0–1 | Độ chắc chắn của model |
| `bbox` | mảng 4 số | Khung bao `[x, y, w, h]` — góc trên-trái, chiều rộng, chiều cao |

**Ví dụ thật** (đo trên video 28 người đi bộ):

```json
{
  "track_id":   "73",
  "class":      "person",
  "confidence": 0.90,
  "bbox":       [115, 145, 32, 74]
}
```

### Toạ độ khung bao

`bbox` tính theo **pixel của khung hình xử lý 960×540**, không phải độ phân giải camera gốc.
Frontend vẽ đè lên video 1920×1080 phải nhân đôi toạ độ:

```
x_thật = x × (rộng_video / 960)
y_thật = y × (cao_video  / 540)
```

### Ngưỡng tin cậy

Trường `conf` lọc bớt vật thể mà model không chắc chắn. **Chỉ siết lên được, không nới
xuống.** Model YOLO nạp một lần và dùng chung cho mọi camera (tiết kiệm VRAM), chạy ở ngưỡng
`YOLO_CONF`. Ngưỡng `conf` của từng luồng áp dụng bằng cách lọc lại sau khi nhận diện — nên
đặt cao hơn `YOLO_CONF` thì có tác dụng, đặt thấp hơn thì không lấy lại được vật đã bị model
loại từ đầu. Muốn bắt vật mờ/ở xa phải hạ `YOLO_CONF` trong `docker-compose.yml`.

### Vòng đời một vật thể

Ngoài toạ độ theo từng khung hình, AI còn báo hai mốc:

- **`start`** — vật thể mới xuất hiện lần đầu.
- **`end`** — vật thể rời khỏi khung hình hoặc bị che khuất quá 2 giây.

Nhờ đó backend biết chính xác một vật xuất hiện lúc nào và biến mất lúc nào, không phải tự
suy từ việc `track_id` ngừng xuất hiện.

---

## 3. AI gửi dữ liệu cho backend bằng cách nào

### Đường đi

```
MediaMTX  ──RTSP──▶  AI Service  ──XADD──▶  Redis  ──XREAD──▶  Backend
(nguồn video)        (YOLO+track)          (hàng đợi)         (đọc kết quả)
```

Backend **không gọi AI để lấy dữ liệu**. Backend chỉ giao việc một lần qua Control API
(`POST /streams/{id}`), sau đó AI tự đẩy kết quả vào Redis liên tục.

### Ba loại tin

| Loại | Nhịp | Nội dung | Mất có sao không |
|---|---|---|---|
| `frame` | ~6–10 tin/giây | Danh sách vật thể + khung bao trong khung hình đó | Không sao — tin sau đè tin trước |
| `track_event` | thưa | Vật thể `start` / `end` | **Có** — đây là dữ liệu nghiệp vụ, ghi vào DB |
| `stream_status` | rất thưa | Camera sống/chết: `started` · `source_ok` · `source_lost` · `reconnected` · `stopped` | **Có** |

Không có `stream_status`, backend **không phân biệt được** "camera hỏng" với "camera chạy
nhưng không có ai đi qua" — cả hai đều chỉ là tin ngừng chảy.

### Hai hàng đợi Redis

```
VISIONOS_RESULTS   10.000 ô   ← nhận TẤT CẢ: frame + track_event + stream_status
VISIONOS_EVENTS    50.000 ô   ← chỉ tin nghiệp vụ, KHÔNG có frame
```

Redis Stream là hàng đợi có số ô cố định — đầy rồi thì tin mới đẩy văng tin cũ nhất. Vì
`frame` bắn dày gấp hàng trăm lần nên nó chiếm hết chỗ: đo thực tế với hàng đợi 1.000 ô và
**một** camera, sau **2 phút 27 giây** trong hàng đợi chỉ còn `frame`, toàn bộ `track_event`
đã bị đẩy đi.

Hàng đợi thứ hai không có `frame` chen vào nên giữ được hàng giờ. Hàng đợi chính vẫn nhận đủ
mọi loại tin như cũ, backend đang chạy không phải sửa gì.

**Khuyến nghị:** đọc `VISIONOS_RESULTS` cho phần vẽ khung bao realtime, đọc `VISIONOS_EVENTS`
bằng **consumer group** cho dữ liệu ghi vào database.

> Cách đọc `XREAD` với `last_id = "$"` sẽ **bỏ qua toàn bộ thời gian backend chết**. Muốn
> không mất thì dùng consumer group để Redis nhớ hộ vị trí đã đọc. Chi tiết ở
> [`BACKEND_INTEGRATION.md`](BACKEND_INTEGRATION.md).

### Lọc theo camera

Mọi camera cùng ghi vào một hàng đợi. Backend **bắt buộc lọc** theo `camera_id` hoặc
`stream_id` trong mỗi tin, nếu không sẽ lẫn dữ liệu của camera khác.

---

> Xem thêm: [`BACKEND_INTEGRATION.md`](BACKEND_INTEGRATION.md) — hướng dẫn tích hợp từng bước
> kèm ví dụ Python/Node/Go · [`AI_SERVICE_INTEGRATION.md`](AI_SERVICE_INTEGRATION.md) — đặc tả
> Control API và định dạng tin · [`CAMERA_FLOW.md`](CAMERA_FLOW.md) — luồng dữ liệu chi tiết.
