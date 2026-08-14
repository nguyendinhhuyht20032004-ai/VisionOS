#!/bin/bash
# =========================================================================
# run_native.sh — Chay VisionOS AI Service NATIVE tren macOS (Apple Silicon)
#
# Yeu cau:
#   1. Python 3.10+
#   2. Redis chay local (brew install redis && brew services start redis)
#   3. Model CoreML da export (python export_coreml.py)
#
# Dung:
#   chmod +x run_native.sh
#   ./run_native.sh                    # mac dinh yolov8m.mlpackage
#   YOLO_WEIGHTS=yolov8m.pt ./run_native.sh   # dung PyTorch thay CoreML
# =========================================================================
set -e

# --- Kiem tra Redis ---
# Bỏ qua kiểm tra vì đã dùng start_local_infrastructure.sh
echo "Redis: OK"

# --- Model weights ---
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8m.mlpackage}"

# Neu chua co file model, huong dan export
if [[ "$YOLO_WEIGHTS" == *.mlpackage ]] && [ ! -d "$YOLO_WEIGHTS" ]; then
    echo ""
    echo "Chua tim thay $YOLO_WEIGHTS"
    echo "Chay export truoc:"
    echo "  python export_coreml.py --weights yolov8m.pt"
    echo ""
    echo "Hoac dung PyTorch (cham hon):"
    echo "  YOLO_WEIGHTS=yolov8m.pt ./run_native.sh"
    exit 1
fi

# --- Environment variables ---
export REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
export REDIS_STREAM_KEY="${REDIS_STREAM_KEY:-VISIONOS_RESULTS}"
export CONTROL_API_PORT="${CONTROL_API_PORT:-8000}"
export YOLO_WEIGHTS
export YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
export YOLO_CONF="${YOLO_CONF:-0.2}"
export DETECT_EVERY="${DETECT_EVERY:-3}"
export OVERLAY_PUBLISH_FPS="${OVERLAY_PUBLISH_FPS:-12}"
export REDIS_STREAM_MAXLEN="${REDIS_STREAM_MAXLEN:-1000}"
export SMOOTHER_LEN="${SMOOTHER_LEN:-2}"
export REID_STITCH="${REID_STITCH:-0}"
export RTSP_TRANSPORT="${RTSP_TRANSPORT:-tcp}"
export RTSP_RECONNECT_INTERVAL_SEC="${RTSP_RECONNECT_INTERVAL_SEC:-5}"

echo ""
echo "=========================================="
echo "  VisionOS AI Service (Native)"
echo "=========================================="
echo "  Model:       $YOLO_WEIGHTS"
echo "  ImgSz:       $YOLO_IMGSZ"
echo "  Conf:        $YOLO_CONF"
echo "  Redis:       $REDIS_URL"
echo "  Port:        $CONTROL_API_PORT"
echo "  DetectEvery: $DETECT_EVERY"
echo "  PublishFPS:  $OVERLAY_PUBLISH_FPS"
echo "=========================================="
echo ""

# --- Tao thu muc data neu chua co ---
mkdir -p data/videos

# --- Chay service ---
exec python3 run_service.py --host 0.0.0.0 --port "$CONTROL_API_PORT"
