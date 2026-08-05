# =============================================================================
# AI SERVICE — đếm người/xe theo camera (FastAPI + YOLOv8 + supervision)
# Build (CPU, chạy mọi nơi):
#     docker build -t visionos-ai .
# Build (GPU — torch CUDA 12.1, chạy kèm --gpus all):
#     docker build --build-arg TORCH_CUDA=cu121 -t visionos-ai:gpu .
# Run:
#     docker run --rm -p 8000:8000 visionos-ai
#     docker run --rm --gpus all -p 8000:8000 visionos-ai:gpu   # có GPU
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# Thư viện hệ thống cho OpenCV (libGL) + đọc video/RTSP (ffmpeg).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch: CPU mặc định; đặt --build-arg TORCH_CUDA=cu121 để lấy bản GPU.
ARG TORCH_CUDA=cpu
COPY requirements-service.txt .
# QUAN TRỌNG: cài torch + torchvision CÙNG LÚC, CÙNG NGUỒN (khớp phiên bản) — nếu không
# sẽ lỗi "operator torchvision::nms does not exist" khi YOLO chạy NMS.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA} torch torchvision && \
    pip install -r requirements-service.txt

# Nạp sẵn trọng số YOLO vào image (khỏi tải lại mỗi lần chạy container).
# Mặc định yolov8n (realtime trên CPU). GPU: --build-arg YOLO_WEIGHTS=yolov8x.pt.
ARG YOLO_WEIGHTS=yolov8n.pt
RUN python -c "from ultralytics import YOLO; YOLO('${YOLO_WEIGHTS}')" || true

COPY . .

EXPOSE 8000
# healthcheck: service sống khi GET / trả 200
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8000/ >/dev/null 2>&1 || exit 1

CMD ["python", "run_service.py", "--host", "0.0.0.0", "--port", "8000"]
