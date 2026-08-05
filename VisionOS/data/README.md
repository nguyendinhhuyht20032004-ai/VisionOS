# 📁 data/ — bỏ VIDEO của bạn vào đây

Thư mục này được **mount sẵn** vào container (`./data` → `/data`, xem `docker-compose.yml`).
Container **KHÔNG thấy** ổ đĩa Mac/Windows của bạn, nên muốn dùng video tải về máy thì phải
đưa nó vào đúng thư mục này.

## Cách dùng (khi chạy bằng Docker)

1. **Copy video** bạn tải về vào thư mục này (`data/`, nằm cạnh `docker-compose.yml`).
   Ví dụ: `data/traffic.mp4`
2. Mở web `http://localhost:8000`, ở ô **Nguồn (source)** gõ đường dẫn **trong container**:

   ```
   /data/traffic.mp4
   ```

   (KHÔNG gõ đường dẫn kiểu `/Users/ban/Downloads/traffic.mp4` — container không thấy.)
3. Bấm **Lấy frame** → vẽ vạch/vùng → **Bắt đầu đếm**.

> Nếu đang chạy service mà mới copy video vào, **không cần** build lại — bind mount thấy file ngay.
> (Chỉ cần `docker compose up`; đã chạy sẵn thì file mới xuất hiện luôn trong `/data`.)

## Mẹo đặt tên file

- Nên đặt tên **không dấu, không khoảng trắng**: `traffic.mp4`, `nguoi_di_bo.mp4`.
- Có khoảng trắng thì gõ đủ trong ô nguồn: `/data/xe co.mp4` (service tự xử lý chuỗi, nhưng
  tên đơn giản đỡ rắc rối).

## Chạy bằng pip (KHÔNG Docker)

Nếu chạy trực tiếp bằng `uvicorn` trên máy (không qua Docker), thì gõ **đường dẫn thật** tới
file luôn — không cần `/data`:

```
/Users/ban/Downloads/traffic.mp4      # Mac/Linux
C:\Users\ban\Downloads\traffic.mp4    # Windows
```

Lúc này cũng dùng được **webcam** (gõ `0`) vì không bị rào chắn của Docker.

---
*File `.mp4` trong thư mục này không được commit lên git (`.gitignore`/`.dockerignore` loại `*.mp4`) —
chỉ nằm trên máy bạn.*
