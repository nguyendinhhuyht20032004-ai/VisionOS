# Kien truc he thong — VisionOS AI Service

Tai lieu kien truc cho he thong **dem nguoi & phuong tien** realtime.
Detect bang **YOLOv8**, track/dem bang **supervision** (ByteTrack), ket qua day len **Redis Stream**,
dieu khien qua **Control API** (`/streams/`). Ho tro chay **Docker** hoac **native macOS** (CoreML).

---

## 1. Tong quan cong nghe

| Lop | Cong nghe | Vai tro |
|---|---|---|
| **Phat hien** | YOLOv8 (ultralytics) | Detect nguoi/xe moi frame. Ho tro PyTorch (.pt) va CoreML (.mlpackage) |
| **Theo doi + dem** | supervision (Roboflow) | ByteTrack + DetectionsSmoother + LineZone/PolygonZone |
| **API** | FastAPI + Uvicorn | Control API (`/streams/`) + Legacy Jobs API (`/api/jobs`) |
| **Output** | Redis Stream | Ket qua realtime (frame boxes + track events) |
| **Vector DB** | Qdrant (fallback in-memory) | Luu embedding ngoai hinh vat → ReID / tim kiem |
| **RTSP Relay** | MediaMTX (Docker) | Chuyen tiep RTSP tu camera vao AI Service |
| **Dong goi** | Docker Compose | 3 service: `api` + `redis` + `mediamtx` |

**3 kieu dem:** cat **VACH** (IN/OUT) · trong **VUNG** (IN/OUT) · **TOAN MAN HINH** (moi vat trong khung).

---

## 2. So do thanh phan

```
                    ┌─────────────────────────────────────┐
                    │         Backend / Frontend           │
                    │  (goi Control API, doc Redis Stream) │
                    └──────┬────────────────────┬──────────┘
                           │ HTTP               │ Redis
                           ▼                    ▼
┌──────────────────────────────────────────────────────────┐
│                   AI Service (FastAPI)                    │
│                                                          │
│  ┌──────────────┐    ┌─────────────────────────────────┐ │
│  │ Control API  │    │        StreamManager             │ │
│  │ /streams/*   │───▶│  (quan ly cac StreamWorker)      │ │
│  └──────────────┘    └──────────┬──────────────────────┘ │
│                                 │                        │
│                      ┌──────────▼──────────────┐         │
│                      │    StreamWorker (Thread) │ x N     │
│                      │  ┌────────────────────┐ │         │
│                      │  │ FrameSource (RTSP) │ │         │
│                      │  │        ▼           │ │         │
│                      │  │ StreamingCounter   │ │         │
│                      │  │  YOLO → ByteTrack  │ │         │
│                      │  │  → Line/PolygonZone│ │         │
│                      │  │        ▼           │ │         │
│                      │  │  Redis XADD        │ │         │
│                      │  └────────────────────┘ │         │
│                      └─────────────────────────┘         │
│                                                          │
│  ┌──────────────┐    ┌───────────────┐                   │
│  │ Legacy API   │    │  VectorStore  │                   │
│  │ /api/jobs/*  │    │  (Qdrant)     │                   │
│  └──────────────┘    └───────────────┘                   │
└──────────────────────────────────────────────────────────┘
         │                          │
    ┌────▼────┐              ┌──────▼──────┐
    │  Redis  │              │   Qdrant    │
    │ Stream  │              │ Vector DB   │
    └─────────┘              └─────────────┘
```

---

## 3. Luong du lieu chinh

### 3.1 Luong Control API (tich hop chinh)

```
Backend                    AI Service                   Redis
  │                           │                           │
  │  POST /streams/cam-01     │                           │
  │  {camera_id, rtsp_url}    │                           │
  │──────────────────────────▶│                           │
  │                           │  Tao StreamWorker         │
  │                           │  Ket noi RTSP             │
  │                           │  Chay YOLO pipeline       │
  │                           │                           │
  │                           │  XADD type="frame"        │
  │                           │──────────────────────────▶│
  │                           │  XADD type="track_event"  │
  │                           │──────────────────────────▶│
  │                           │                           │
  │                     XREAD (blocking)                  │
  │◀──────────────────────────────────────────────────────│
  │                           │                           │
  │  DELETE /streams/cam-01   │                           │
  │──────────────────────────▶│  Dung StreamWorker        │
```

### 3.2 Pipeline 1 frame (trong StreamingCounter.process)

```
frame → YOLO.detect(prompt) → _to_sv (class_id on dinh)
      → ByteTrack.update → DetectionsSmoother.update
      → LineZone.trigger (dem vach IN/OUT)
      | PolygonZone.trigger (dem vung IN/OUT)
      | len(det) (toan khung)
      → annotate (RoundBox+Label+Trace, MAU THEO LOP) → frame BGR
```

---

## 4. Ban do module (file → vai tro)

| File | Vai tro |
|---|---|
| `recognition/service/app.py` | **FastAPI**: Control API `/streams/*` + Legacy `/api/jobs/*` + healthz + vector DB |
| `recognition/service/stream_manager.py` | **StreamManager** (singleton) + **StreamWorker** (thread/camera) + publish Redis |
| `recognition/service/engine.py` | **StreamingCounter** — dem tung frame (co trang thai), IN/OUT events |
| `recognition/service/camera.py` | `FrameSource` (doc nen, reconnect) + `grab_snapshot` + `encode_jpeg` |
| `recognition/service/builder.py` | `make_scenario` (dung tu request) + `get_detector` (cache 1 lan) |
| `recognition/service/vectordb.py` | `VectorStore` (Qdrant/in-memory) + `embed_crop` (hist mau HSV) |
| `recognition/detectors/ultralytics_yolo.py` | **YOLOv8** — detect nguoi/xe; ho tro PyTorch + CoreML |
| `recognition/scenarios.py` | `CountScenario` — cau hinh 1 bai dem (line/zone/fullscreen) |
| `recognition/sv_counting.py` | Engine supervision: `_to_sv`/annotate + class_id on dinh |
| `recognition/base.py` | Kieu nen: `BoundingBox`, `Detection`, `DetectorResult` |
| `run_service.py` | Entrypoint uvicorn |
| `export_coreml.py` | Export YOLOv8 → CoreML (.mlpackage) cho Apple Silicon |

---

## 5. Redis Stream output

Moi ket qua duoc day len Redis Stream (`VISIONOS_RESULTS` mac dinh) voi 2 loai message:

### type: "frame"
```json
{
  "type": "frame",
  "camera_id": "cam-01",
  "stream_id": "entrance",
  "frame_timestamp": "2026-08-14T10:30:00.123Z",
  "resolution": {"width": 960, "height": 540},
  "boxes": [
    {
      "track_id": "5",
      "class": "person",
      "confidence": 0.87,
      "bbox": [120.5, 80.3, 60.0, 140.0],
      "velocity": [12.3, -5.1]
    }
  ]
}
```

### type: "track_event"
```json
{
  "type": "track_event",
  "camera_id": "cam-01",
  "stream_id": "entrance",
  "track_id": "5",
  "class": "person",
  "event": "start",
  "timestamp": "2026-08-14T10:30:00.123Z"
}
```

Cac gia tri `event`: `start` (xuat hien), `end` (mat dau), `IN` (vao vach/vung), `OUT` (ra vach/vung).

---

## 6. 2 che do trien khai

### Docker (production)
```
docker-compose.yml
├── api        (AI Service — FastAPI + YOLO + Redis client)
├── redis      (Redis — luu tru ket qua Stream)
└── mediamtx   (RTSP relay — chuyen tiep camera)
```

### Native macOS (Apple Silicon)
```
./run_native.sh
├── Redis chay local (brew services start redis)
├── YOLOv8 CoreML (.mlpackage) — FP16 tren ANE
└── FastAPI chay truc tiep (python run_service.py)
```

Export CoreML:
```bash
python export_coreml.py --weights yolov8m.pt --half
# → yolov8m.mlpackage (FP16, ~3-5x nhanh hon PyTorch CPU)
```

---

## 7. Dac diem thiet ke

- **Dem chinh xac:** ByteTrack bam dai (`lost_track_buffer=120`), DetectionsSmoother chong nhap nhay.
- **IN/OUT events:** Line counting (supervision LineZone.trigger) + Zone counting (set comparison giua cac frame).
- **Event queue:** `_pending_cross_events` tich luy events giua cac lan publish de khong mat event.
- **Mau theo LOP:** `ColorLookup.CLASS` + class_id on dinh → car/truck/bus moi loai 1 mau.
- **Chiu loi:** Vector DB loi → tu ve in-memory; FrameSource tu reconnect; Redis loi → log warning.
- **CoreML auto-detect:** Tu dong phat hien .mlpackage/.mlmodel va bo cac param khong tuong thich (half, augment, max_det).
- **Toa do %:** Vach/vung doc lap do phan giai camera.

---

## 8. Diem mo rong

| Hang muc | Hien tai | Nang cap de xuat |
|---|---|---|
| Embedding ReID | Histogram mau HSV (nhe) | Model ReID that (OSNet) |
| Luu ket qua | Redis Stream (volatile) | TimescaleDB + dashboard |
| Bao mat | Khong | API token / OAuth, HTTPS |
| GPU | CPU / CoreML ANE | CUDA + TensorRT |
| Quan sat | log stdout | Prometheus metrics |

---

> Xem them: huong dan tich hop backend tai [`BACKEND_INTEGRATION.md`](BACKEND_INTEGRATION.md),
> tham so cau hinh tai [`AI_SERVICE_PARAMETERS.md`](AI_SERVICE_PARAMETERS.md),
> huong dan trien khai tai [`AI_SERVICE_DEPLOYMENT.md`](AI_SERVICE_DEPLOYMENT.md).
