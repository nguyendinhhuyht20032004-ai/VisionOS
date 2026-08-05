# Kế Hoạch Tích Hợp AI Service & Hardcode Dữ Liệu

Tài liệu này giải thích chi tiết **quy trình ghép nối** hệ thống AI vào dự án hiện tại, và giải thích khái niệm **"Trả về response hardcode"** (làm giả dữ liệu) ở mục 6 trên bảng công việc của bạn.

---

## 1. Bài toán đặt ra: Tại sao phải "Hardcode"?

**Vấn đề:**
- Hệ thống AI (VisionOS) rất nặng, yêu cầu máy tính mạnh (hoặc có Card rời) và phải cài đặt Docker phức tạp.
- Anh Dev Frontend (người làm giao diện Web/App) và anh Dev Backend (người làm server Database của dự án) máy tính của họ chỉ dùng để code, nếu bắt họ phải bật nguyên hệ thống AI lên chỉ để lấy số đếm làm giao diện thì máy sẽ rất lag và mất thời gian.

**Giải pháp: "Hardcode" (Làm giả dữ liệu AI)**
- Thay vì gọi hệ thống AI thật, chúng ta sẽ "viết chết" (hardcode) kết quả trả về. Nghĩa là lúc nào gọi API cũng trả về một con số giả cố định (Ví dụ: báo là có 10 người đi vào, 5 người đi ra).
- Nhờ vậy, team Dev có thể thoải mái code giao diện, vẽ biểu đồ... mà **hoàn toàn không cần bật hệ thống AI lên**.

---

## 2. Kế hoạch Tích hợp chi tiết (3 Bước)

### BƯỚC 1: Dev Backend làm giả (Hardcode) API
*Người thực hiện: Dev Backend*

- Dev Backend mở file **`docs/API_HANDOFF.md`** ra đọc xem AI trả về cái gì.
- Dev Backend tạo ra các đường link API trên hệ thống Backend chính, nhưng **KHÔNG KẾT NỐI VỚI AI**.
- Thay vào đó, Dev Backend code thẳng một cục dữ liệu giả (Hardcode). 
- **Ví dụ code của Backend lúc này:**
  ```javascript
  // API lấy kết quả đếm (Chỉ là giả)
  app.get('/api/get-counting-result', (req, res) => {
      // Đáng lẽ phải gọi sang server AI để hỏi, nhưng tạm thời hardcode:
      return res.json({
          "in": 12,       // Hardcode số 12
          "out": 8,       // Hardcode số 8
          "total": 20,
          "status": "running"
      });
  });
  ```

### BƯỚC 2: Dev Frontend xây dựng Giao diện
*Người thực hiện: Dev Frontend*

- Dev Frontend làm nút "Bắt đầu đếm".
- Frontend gọi API của Backend, nhận được số liệu giả là `Vào: 12, Ra: 8`.
- Frontend dùng số liệu này để vẽ lên màn hình, làm biểu đồ nhấp nháy, hiện con số thật đẹp.
- **Kết quả:** Giao diện của dự án hoàn thiện 100%, chạy mượt mà, bấm nút nào ăn nút nấy (dù dữ liệu đằng sau chỉ là đồ giả).

### BƯỚC 3: Tích hợp thật (Tháo bỏ Hardcode)
*Người thực hiện: Team AI & Dev Backend*

- Khi Giao diện (Frontend) đã làm xong và duyệt OK.
- Bạn (Team AI) sẽ bật Docker AI Service thật lên một máy chủ mạnh (có địa chỉ IP, ví dụ `http://192.168.1.100:8000`).
- Dev Backend vào lại file code, xoá cái đoạn Hardcode giả mạo đi, thay bằng lệnh kết nối đến IP của AI thật.
- **Ví dụ code Backend sau khi tích hợp:**
  ```javascript
  app.get('/api/get-counting-result', async (req, res) => {
      // ĐÃ THÁO HARDCODE - Gọi sang server AI thật
      const ai_data = await fetch('http://192.168.1.100:8000/api/jobs/xxx');
      return res.json(ai_data); 
  });
  ```
- **Hoàn thành:** Bây giờ mọi thứ đã kết nối. Người dùng bấm trên Web -> Web gọi Backend -> Backend gọi AI -> AI phân tích Camera thật -> Trả số liệu về Web.

---

## 3. Tổng kết Lợi ích của Kế hoạch này

1. **Làm việc song song:** Bạn cứ làm việc của AI, anh Dev cứ làm việc của App/Web. Không ai phải chờ ai.
2. **Không lỗi vặt:** Giao diện test với dữ liệu giả sẽ không bao giờ bị lỗi do AI chạy chậm hay AI sập. 
3. **Tích hợp cực nhanh:** Ở bước 3, chỉ mất 15 phút đổi link API là hệ thống chạy thật luôn, vì giao diện đã làm xong hết từ trước rồi.
