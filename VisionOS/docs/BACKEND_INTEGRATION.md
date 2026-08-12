# Hướng dẫn tích hợp Backend ↔ AI Service

Tài liệu này dành cho backend developer cần tích hợp với AI Service.
Backend làm đúng **2 việc**: gọi Control API để gán/gỡ camera, và đọc kết quả từ Redis Stream.

---

## Mục lục

1. [Điều kiện kết nối](#1-điều-kiện-kết-nối)
2. [Bước 1 — Gán camera vào AI Service](#2-bước-1--gán-camera-vào-ai-service)
3. [Bước 2 — Đọc kết quả từ Redis Stream](#3-bước-2--đọc-kết-quả-từ-redis-stream)
4. [Bước 3 — Cập nhật tham số không cần dừng](#4-bước-3--cập-nhật-tham-số-không-cần-dừng)
5. [Bước 4 — Dừng camera](#5-bước-4--dừng-camera)
6. [Xử lý lỗi & kết nối lại](#6-xử-lý-lỗi--kết-nối-lại)
7. [Ví dụ hoàn chỉnh theo ngôn ngữ](#7-ví-dụ-hoàn-chỉnh-theo-ngôn-ngữ)

---

## 1. Điều kiện kết nối

Backend cần truy cập được 2 địa chỉ:

| Dịch vụ | Địa chỉ (trong Docker network) | Địa chỉ (từ ngoài) | Dùng để |
|---|---|---|---|
| AI Service Control API | `http://api:8000` | `http://localhost:8000` | Gán/gỡ/sửa camera |
| Redis | `redis://redis:6379` | `redis://localhost:6379` | Đọc kết quả |

> Nếu backend chạy **trong cùng Docker Compose**, dùng hostname nội bộ (`api`, `redis`).
> Nếu backend chạy **ngoài Docker**, dùng `localhost` hoặc IP máy chủ.

---

## 2. Bước 1 — Gán camera vào AI Service

Gọi `POST /streams/{stream_id}` để AI Service bắt đầu xử lý một camera.

- `stream_id`: chuỗi bất kỳ do backend tự đặt, dùng để quản lý sau này (VD: `"cam-01"`, `"entrance-north"`).
- Mỗi `stream_id` chỉ được dùng một lần — gọi lại sẽ trả lỗi `400`.

### Request

```
POST http://api:8000/streams/{stream_id}
Content-Type: application/json
```

```json
{
  "camera_id": "cam-01",
  "rtsp_url":  "rtsp://mediamtx:8554/live/cam-01",
  "params": {
    "classes": ["person", "car"],
    "conf":    0.3,
    "roi":     [[0, 50], [100, 50]]
  }
}
```

### Các trường trong `params`

| Trường | Kiểu | Mô tả |
|---|---|---|
| `classes` | `string[]` | Danh sách lớp cần theo dõi, VD `["person","car"]`. **Bỏ qua = chỉ theo dõi `person`** (không phải cả 80 lớp COCO). |
| `conf` | `float` | Ngưỡng confidence (0.0–1.0). Bỏ qua = lấy theo env `YOLO_CONF` (mặc định `0.3`). Xem lưu ý bên dưới. |
| `roi` | `number[][]` | Toạ độ vùng quan tâm (% frame 0–100):<br>• **2 điểm** → theo dõi qua **đường kẻ** (line crossing)<br>• **3+ điểm** → theo dõi trong **vùng** (polygon zone)<br>• **bỏ qua** → theo dõi toàn **màn hình** (fullscreen) |

> **`conf` chỉ siết lên được, không nới xuống.** Model YOLO được nạp **một lần và dùng chung
> cho mọi camera** (để tiết kiệm VRAM), chạy ở ngưỡng `YOLO_CONF`. Ngưỡng `conf` của từng
> luồng được áp dụng bằng cách lọc lại kết quả sau khi nhận diện — nên đặt `conf` **cao hơn**
> `YOLO_CONF` thì có tác dụng, đặt **thấp hơn** thì không lấy lại được các vật đã bị model
> loại từ đầu. Cần bắt vật mờ/ở xa thì phải hạ `YOLO_CONF` trong `docker-compose.yml`.

### Ví dụ các chế độ ROI

```jsonc
// Đếm qua đường kẻ ngang giữa màn hình
"roi": [[0, 50], [100, 50]]

// Đếm trong vùng tam giác (3 điểm)
"roi": [[10, 90], [50, 10], [90, 90]]

// Đếm tất cả vật trong frame (bỏ roi)
// không truyền trường "roi"
```

### Response thành công

```json
{"status": "success", "message": "Stream cam-01 started"}
```

### Response lỗi

| HTTP | Nguyên nhân |
|---|---|
| `400` | `stream_id` đã tồn tại |
| `500` | Redis chưa khởi động hoặc không kết nối được |

---

## 3. Bước 2 — Đọc kết quả từ Redis Stream

Sau khi gán camera, AI Service tự động publish kết quả vào Redis Stream `VISIONOS_RESULTS`.
Backend đọc bằng lệnh `XREAD`.

### Cấu trúc message

Mỗi bản ghi trong stream có field `data` chứa chuỗi JSON. Có 3 loại message:

#### `frame` — kết quả phát hiện theo frame

Phát ra liên tục (tần suất giới hạn bởi `OVERLAY_PUBLISH_FPS`, mặc định 10 msg/giây/luồng).

```json
{
  "type":            "frame",
  "camera_id":       "cam-01",
  "stream_id":       "entrance-north",
  "frame_timestamp": "2026-08-12T09:15:32.450Z",
  "boxes": [
    {
      "track_id":   "trk-7",
      "class":      "person",
      "confidence": 0.87,
      "bbox":       [142.5, 78.0, 64.0, 182.0]
    }
  ]
}
```

> `bbox` = `[x, y, w, h]` theo pixel của frame 960×540.

#### `track_event` — vòng đời của một đối tượng

Phát ra khi một đối tượng xuất hiện lần đầu (`start`) hoặc biến mất quá 2 giây (`end`).

```json
{
  "type":      "track_event",
  "camera_id": "cam-01",
  "stream_id": "entrance-north",
  "track_id":  "trk-7",
  "class":     "person",
  "event":     "start",
  "timestamp": "2026-08-12T09:15:30.210Z"
}
```

> `event` chỉ có: `"start"` hoặc `"end"`.

#### `stream_status` — vòng đời của luồng camera

```json
{
  "type":      "stream_status",
  "camera_id": "cam-01",
  "stream_id": "entrance-north",
  "status":    "source_lost",
  "detail":    "mất kết nối RTSP",
  "timestamp": "2026-08-12T09:20:11.004Z"
}
```

| `status` | Nghĩa |
|---|---|
| `started` | AI Service đã nhận việc, đang mở kết nối tới camera |
| `source_ok` | Đã lấy được frame đầu tiên — camera thực sự chạy |
| `source_lost` | Mất kết nối RTSP, AI đang tự thử lại |
| `reconnected` | Kết nối lại được, frame chảy tiếp |
| `stopped` | Luồng đã dừng |

Không có loại tin này thì backend **không phân biệt được** "camera hỏng" với "đang không có
ai đi qua" — cả hai đều chỉ là tin ngừng chảy.

**Khi camera chết, AI ngừng hẳn việc bắn tin `frame`.** Trước đây bộ đọc RTSP giữ lại ảnh
cuối trong bộ nhớ nên AI vẫn nhận diện trên ảnh đông cứng và bắn tin đều đặn — đo thực tế:
camera tắt lúc 04:30:10 mà tin vẫn chảy tới 04:30:55, lặp đi lặp lại y hệt 31 khung bao ở
nguyên vị trí cũ. Backend sẽ hiển thị "31 người" vĩnh viễn trên một camera đã chết. Giờ AI coi là mất nguồn
khi quá `SOURCE_STALE_SEC` giây (mặc định 5) không có ảnh mới, và im lặng cho tới khi có ảnh
thật trở lại.

> Mọi camera cùng ghi vào một stream Redis → backend **phải lọc theo `camera_id` hoặc
> `stream_id`**, nếu không sẽ lẫn tin của camera khác.

### Hai Redis Stream

| Stream | Chứa gì | Giữ được bao lâu |
|---|---|---|
| `VISIONOS_RESULTS` | **Tất cả** — `frame` + `track_event` + `stream_status` | ngắn: `frame` bắn 10 tin/giây nên đẩy tin cũ đi rất nhanh |
| `VISIONOS_EVENTS` | **Chỉ tin nghiệp vụ** — `track_event` + `stream_status` | dài: mặc định 50 000 bản ghi |

Đo thực tế với `REDIS_STREAM_MAXLEN=1000` và **một** camera: `VISIONOS_RESULTS` chỉ giữ được
**2 phút 27 giây**, và toàn bộ `track_event` đã bị đẩy khỏi cửa sổ — chỉ còn `frame`. Với 4
camera thì còn khoảng 37 giây.

**Khuyến nghị:** đọc `VISIONOS_RESULTS` để vẽ khung bao realtime, và đọc `VISIONOS_EVENTS`
để ghi dữ liệu nghiệp vụ vào DB. Tin nghiệp vụ có ở cả hai stream nên nếu chỉ cần một luồng
đơn giản thì đọc `VISIONOS_RESULTS` như cũ vẫn chạy.

### Cách đọc Redis Stream

Dùng `XREAD` với `block` để không cần polling:

```
XREAD BLOCK 5000 COUNT 20 STREAMS VISIONOS_RESULTS <last_id>
```

- `last_id = "$"` → chỉ đọc message MỚI từ thời điểm gọi trở đi.
- `last_id = "0"` → đọc TẤT CẢ từ đầu (kể cả message cũ còn trong stream).
- Lưu `entry_id` của bản ghi cuối cùng đọc được để dùng làm `last_id` cho lần sau.

> ⚠️ **`$` sẽ làm mất dữ liệu khi backend khởi động lại.** `$` nghĩa là "bỏ qua mọi thứ
> trước thời điểm này" — deploy lại backend 1 phút là mất trắng 1 phút sự kiện. Muốn không
> mất thì backend phải tự lưu `entry_id` cuối cùng vào DB và đọc tiếp từ đó, hoặc dùng
> **consumer group** để Redis nhớ hộ:
>
> ```
> XGROUP CREATE VISIONOS_EVENTS backend-main 0 MKSTREAM
> XREADGROUP GROUP backend-main worker-1 BLOCK 5000 COUNT 20 STREAMS VISIONOS_EVENTS >
> XACK VISIONOS_EVENTS backend-main <entry_id>     # sau khi ghi DB xong
> ```
>
> Redis giữ danh sách tin chưa `XACK`, backend chết giữa chừng bật lại vẫn nhận lại được.
> Đây là cách nên dùng cho dữ liệu nghiệp vụ.

---

## 4. Bước 3 — Cập nhật tham số không cần dừng

Khi cần thay đổi loại vật theo dõi, ngưỡng, hoặc vùng ROI mà không muốn gián đoạn luồng:

```
PATCH http://api:8000/streams/{stream_id}
Content-Type: application/json
```

```json
{
  "classes": ["vehicle"],
  "conf":    0.4,
  "roi":     [[5, 20], [60, 80], [95, 20]]
}
```

Chỉ cần truyền những trường muốn thay đổi — các trường không gửi vẫn giữ nguyên.

> ⚠️ **`PATCH` khởi động lại việc theo dõi.** Tracker được dựng lại nên `track_id` đánh lại từ
> đầu. Trước khi dựng lại, AI bắn `track_event` kiểu `end` cho toàn bộ track đang mở — không
> đóng thì backend treo track cũ vĩnh viễn rồi lại nhận đúng những id đó từ tracker mới.

---

## 5. Bước 4 — Dừng camera

```
DELETE http://api:8000/streams/{stream_id}
```

AI Service dừng thread, đóng kết nối RTSP. Sau đó có thể gán lại cùng `stream_id`.

---

## 6. Xử lý lỗi & kết nối lại

### AI Service không phản hồi

```
POST /streams/cam-01 → timeout hoặc Connection refused
```

→ Kiểm tra `docker compose ps` và `docker compose logs api`.
→ Thử lại sau khoảng 2–5 giây (AI Service thường khởi động trong 10–15 giây).

### Redis mất kết nối giữa chừng

→ Redis tự khởi động lại nếu dùng `restart: unless-stopped` trong Compose.
→ Backend nên thử kết nối lại với backoff:

```
Lần 1: chờ 1s → thử lại
Lần 2: chờ 2s → thử lại
Lần 3: chờ 4s → thử lại
...
```

### Stream bị hết hoặc camera rớt

→ AI Service tự kết nối lại sau `RTSP_RECONNECT_INTERVAL_SEC` (mặc định 5s).
→ Backend không cần làm gì — `frame` message sẽ tự chạy lại khi camera phục hồi.

### Phân biệt message của từng camera

Dùng `camera_id` hoặc `stream_id` trong message để lọc:

```python
if msg["camera_id"] == "cam-entrance":
    # xử lý camera lối vào
```

---

## 7. Ví dụ hoàn chỉnh theo ngôn ngữ

### Python

```python
import requests
import redis
import json
import threading

AI_URL   = "http://localhost:8000"
REDIS_URL = "redis://localhost:6379"

# ── Gán camera ──────────────────────────────────────────────────────────────
def start_stream(stream_id: str, camera_id: str, rtsp_url: str, **params):
    resp = requests.post(f"{AI_URL}/streams/{stream_id}", json={
        "camera_id": camera_id,
        "rtsp_url":  rtsp_url,
        "params":    params,
    })
    resp.raise_for_status()
    return resp.json()

# ── Đọc Redis Stream ─────────────────────────────────────────────────────────
def consume(on_frame=None, on_event=None):
    r = redis.from_url(REDIS_URL)
    last_id = "$"
    while True:
        msgs = r.xread({"VISIONOS_RESULTS": last_id}, block=5000, count=20)
        for _, entries in (msgs or []):
            for entry_id, fields in entries:
                last_id = entry_id
                msg = json.loads(fields[b"data"])
                if msg["type"] == "frame" and on_frame:
                    on_frame(msg)
                elif msg["type"] == "track_event" and on_event:
                    on_event(msg)

# ── Sử dụng ──────────────────────────────────────────────────────────────────
start_stream(
    stream_id="hall-entrance",
    camera_id="cam-01",
    rtsp_url="rtsp://mediamtx:8554/live/cam-01",
    classes=["person"],
    conf=0.3,
    roi=[[0, 50], [100, 50]],
)

def on_frame(msg):
    print(f"[{msg['camera_id']}] {len(msg['boxes'])} đối tượng")

def on_event(msg):
    print(f"  Track {msg['track_id']} ({msg['class']}) → {msg['event']}")

# Chạy consumer trong thread riêng
t = threading.Thread(target=consume, kwargs={"on_frame": on_frame, "on_event": on_event}, daemon=True)
t.start()
t.join()
```

---

### Node.js (TypeScript)

```typescript
import axios from "axios";
import { createClient } from "redis";

const AI_URL    = "http://localhost:8000";
const REDIS_URL = "redis://localhost:6379";

// ── Gán camera ──────────────────────────────────────────────────────────────
async function startStream(streamId: string, cameraId: string, rtspUrl: string) {
  const { data } = await axios.post(`${AI_URL}/streams/${streamId}`, {
    camera_id: cameraId,
    rtsp_url:  rtspUrl,
    params: {
      classes: ["person", "car"],
      conf:    0.3,
      roi:     [[0, 50], [100, 50]],
    },
  });
  return data;
}

// ── Dừng camera ──────────────────────────────────────────────────────────────
async function stopStream(streamId: string) {
  await axios.delete(`${AI_URL}/streams/${streamId}`);
}

// ── Đọc Redis Stream ─────────────────────────────────────────────────────────
async function consume() {
  const client = createClient({ url: REDIS_URL });
  await client.connect();

  let lastId = "$";

  while (true) {
    const results = await client.xRead(
      { key: "VISIONOS_RESULTS", id: lastId },
      { BLOCK: 5000, COUNT: 20 }
    );

    for (const { messages } of results ?? []) {
      for (const { id, message } of messages) {
        lastId = id;
        const msg = JSON.parse(message.data);

        if (msg.type === "frame") {
          console.log(`[${msg.camera_id}] ${msg.boxes.length} đối tượng`);
        } else if (msg.type === "track_event") {
          console.log(`Track ${msg.track_id} (${msg.class}) → ${msg.event}`);
        }
      }
    }
  }
}

// ── Chạy ─────────────────────────────────────────────────────────────────────
(async () => {
  await startStream("hall-entrance", "cam-01", "rtsp://mediamtx:8554/live/cam-01");
  console.log("Stream started. Listening...");
  await consume();
})();
```

---

### Go

```go
package main

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"

    "github.com/redis/go-redis/v9"
)

const (
    aiURL    = "http://localhost:8000"
    redisURL = "redis://localhost:6379"
    streamKey = "VISIONOS_RESULTS"
)

// ── Gán camera ───────────────────────────────────────────────────────────────
func startStream(streamID, cameraID, rtspURL string) error {
    body, _ := json.Marshal(map[string]any{
        "camera_id": cameraID,
        "rtsp_url":  rtspURL,
        "params": map[string]any{
            "classes": []string{"person", "car"},
            "conf":    0.3,
            "roi":     [][]float64{{0, 50}, {100, 50}},
        },
    })
    resp, err := http.Post(aiURL+"/streams/"+streamID, "application/json", bytes.NewReader(body))
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("start stream failed: %s", resp.Status)
    }
    return nil
}

// ── Đọc Redis Stream ──────────────────────────────────────────────────────────
func consume(ctx context.Context) {
    opt, _ := redis.ParseURL(redisURL)
    rdb := redis.NewClient(opt)
    lastID := "$"

    for {
        results, err := rdb.XRead(ctx, &redis.XReadArgs{
            Streams: []string{streamKey, lastID},
            Count:   20,
            Block:   5000,
        }).Result()
        if err != nil {
            log.Println("redis xread error:", err)
            continue
        }
        for _, stream := range results {
            for _, msg := range stream.Messages {
                lastID = msg.ID
                var payload map[string]any
                json.Unmarshal([]byte(msg.Values["data"].(string)), &payload)

                switch payload["type"] {
                case "frame":
                    boxes := payload["boxes"].([]any)
                    fmt.Printf("[%s] %d đối tượng\n", payload["camera_id"], len(boxes))
                case "track_event":
                    fmt.Printf("Track %s (%s) → %s\n",
                        payload["track_id"], payload["class"], payload["event"])
                }
            }
        }
    }
}

func main() {
    if err := startStream("hall-entrance", "cam-01", "rtsp://mediamtx:8554/live/cam-01"); err != nil {
        log.Fatal(err)
    }
    fmt.Println("Stream started. Listening...")
    consume(context.Background())
}
```

---

## Tóm tắt nhanh

```
1. POST   /streams/{id}               body: {camera_id, rtsp_url, params}  → bắt đầu
2. XREAD  VISIONOS_RESULTS $                                                → đọc kết quả liên tục
3. PATCH  /streams/{id}               body: {classes, conf, roi}            → đổi tham số
4. DELETE /streams/{id}                                                     → dừng
```

Backend không cần biết gì về YOLO, ByteTrack, hay MediaMTX — chỉ cần HTTP + Redis.
