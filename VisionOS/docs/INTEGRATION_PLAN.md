# Kế Hoạch Tích Hợp AI Service & Hardcode Dữ Liệu

Tài liệu này giải thích chi tiết **quy trình ghép nối** hệ thống AI vào dự án hiện tại, bao gồm việc thống nhất **tài liệu định dạng schema** và **4 công việc cụ thể** để thực hiện **"Tích hợp vào AI service, trả về response (hardcode)"**.

---

## 1. Bài toán đặt ra: Tại sao phải "Hardcode"?

**Vấn đề:**
- Hệ thống AI (VisionOS) rất nặng, yêu cầu phần cứng máy tính mạnh (thường cần Card đồ hoạ rời GPU) và phải thiết lập môi trường Docker phức tạp.
- Team Dev Frontend (người làm giao diện) và Dev Backend (người làm server quản lý chính) thường sử dụng máy tính làm việc tiêu chuẩn. Nếu bắt họ phải chạy nguyên hệ thống AI trên máy chỉ để kiểm thử giao diện thì hệ thống sẽ rất lag, tốn tài nguyên và mất thời gian setup.

**Giải pháp: "Hardcode" (Làm giả dữ liệu AI)**
- Thay vì gọi hệ thống AI thật để tính toán, chúng ta sẽ "viết chết" (hardcode) kết quả trả về. Nghĩa là lúc nào gọi API, hệ thống cũng trả về một định dạng giả cố định theo đúng chuẩn schema.
- Nhờ vậy, team Dev có thể thoải mái code logic giao diện, vẽ biểu đồ... mà **hoàn toàn không cần khởi động hệ thống AI lên**.

---

## 2. Tài liệu định dạng Schema & Kiểu dữ liệu trả về

Trước khi hardcode hay code thật, các bên phải thống nhất chuẩn giao tiếp chung (Schema) để việc trao đổi dữ liệu không bị lỗi. 

**Kiểu dữ liệu trả về (Response Data Type):** Cấu trúc chuẩn **JSON**.

**Ví dụ Schema trả về khi gọi API đếm (Counting Result):**
```json
{
  "status": "success",          // Kiểu String: Trạng thái (success, processing, failed)
  "job_id": "job_123456789",    // Kiểu String: Mã ID định danh của phiên chạy
  "data": {
    "total_in": 15,             // Kiểu Integer: Tổng số lượng đi vào
    "total_out": 5,             // Kiểu Integer: Tổng số lượng đi ra
    "current_total": 10,        // Kiểu Integer: Tổng số lượng hiện tại bên trong
    "timestamp": "2026-08-05T10:00:00Z" // Kiểu String (ISO 8601): Thời điểm ghi nhận
  },
  "message": "Counting data retrieved successfully" // Kiểu String: Thông báo kết quả
}
```
*Lưu ý:* Cả dữ liệu Hardcode lẫn dữ liệu AI thật đều **bắt buộc** phải tuân thủ Schema này để Frontend/Backend parse (đọc hiểu) dữ liệu một cách đồng nhất.

---

## 3. Tích hợp vào AI Service (Hardcode): 4 Công Việc Phải Làm

Để thực hiện quy trình "Tích hợp vào AI service, trả về response hardcode" một cách suôn sẻ, chúng ta cần hoàn thành 4 công việc chính:

### Công việc 1: Thống nhất và định nghĩa Data Schema
*Người thực hiện: Team AI & Team Backend*
- Lập tài liệu API (API Documentation / Swagger) rõ ràng.
- Thống nhất cấu trúc JSON, tên các trường (như `total_in`, `total_out`) và kiểu dữ liệu chuẩn (String, Integer, Object, Array).
- Đảm bảo các hệ thống đều có chung một "ngôn ngữ" giao tiếp.

### Công việc 2: Xây dựng Mock Service (Hardcode API)
*Người thực hiện: Dev Backend (Hoặc Team AI dựng Mock server)*
- Tạo ra các đường link API (endpoints) trên hệ thống, nhưng **bỏ qua phần kết nối với core AI**.
- Viết thẳng dữ liệu giả mạo (hardcode) trả về theo đúng định dạng Schema đã chốt ở Công việc 1.
  ```javascript
  // Ví dụ Hardcode API trên Backend (NodeJS)
  app.get('/api/ai/counting-result', (req, res) => {
      // Hardcode dữ liệu trả về ngay lập tức
      return res.json({
          "status": "success",
          "data": { 
              "total_in": 15, 
              "total_out": 5, 
              "current_total": 10 
          }
      });
  });
  ```

### Công việc 3: Tích hợp Giao diện (Frontend) với Mock API
*Người thực hiện: Dev Frontend*
- Frontend tiến hành gọi API giả (Mock API) đã làm ở Công việc 2.
- Sử dụng số liệu giả (Vào: 15, Ra: 5) để dàn trang giao diện: hiện số trên Dashboard, cập nhật biểu đồ thống kê, xử lý hiệu ứng nhấp nháy, v.v.
- Hoàn thiện 100% flow của người dùng (User Flow) mà không bị phụ thuộc vào tiến độ làm core AI.

### Công việc 4: Tích hợp AI thật (Tháo gỡ Hardcode)
*Người thực hiện: Team AI & Dev Backend*
- Giai đoạn cuối, khi AI model đã làm xong và được deploy lên một server GPU mạnh mẽ.
- Dev Backend vào lại file API, **xoá đoạn dữ liệu hardcode giả mạo đi**.
- Cập nhật hàm gọi API (fetch/axios) để trỏ thẳng tới địa chỉ IP của Server AI thật. 
- Mọi thứ hoàn tất: 
  **Web (Frontend) -> gọi Backend -> Backend gọi AI Thật -> Trả số liệu Thật về Web theo đúng Schema.**

---

## 4. Tổng kết Lợi ích của Quy Trình

1. **Làm việc song song:** Team AI cứ tối ưu model, Team Web cứ làm giao diện. Nhờ có cái neo là **Schema**, không bên nào bị kẹt vì bên kia.
2. **Không lỗi vặt, không sập nguồn:** Giao diện test với API giả sẽ luôn phản hồi siêu nhanh, không bị lỗi do AI chạy ngốn RAM hay sập giữa chừng.
3. **Switch sang hệ thống thật siêu nhanh:** Ở bước 4, chỉ mất vài phút đổi link API là hệ thống chạy thật luôn, mọi logic hiển thị đã được kiểm chứng bằng hardcode từ trước đó.
