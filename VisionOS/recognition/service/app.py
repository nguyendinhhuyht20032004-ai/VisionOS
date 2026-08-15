# -*- coding: utf-8 -*-
"""AI Service -- FastAPI Control API theo AI_SERVICE_CONTRACT.md.

Endpoints:
  GET  /health              -- ready + version (200 chi khi model da nap xong)
  POST /streams/{jobKey}    -- tao moi hoac cap nhat nong job dang chay
  DELETE /streams/{jobKey}  -- dung job, giai phong RTSP session
  GET  /streams             -- danh sach job dang chay kem fps thuc te

Giao tiep voi Backend:
  * Control API (REST) -- Backend goi sang AI Service (listen tren CONTROL_API_PORT)
  * OverlayIngest (gRPC) -- AI Service goi sang Backend (target = OVERLAY_GRPC_TARGET)

Chay: python run_service.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .builder import get_detector
from .grpc_publisher import OverlayPublisher
from .stream_manager import StreamManager, StreamControlRequest, StreamParams

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Globals (khoi tao trong lifespan)
# ---------------------------------------------------------------------------
_model_ready = False
_stream_manager: Optional[StreamManager] = None
_grpc_publisher: Optional[OverlayPublisher] = None


def _preload_model():
    """Nap model YOLO truoc de /health tra ready=true dung thoi diem."""
    global _model_ready
    try:
        det = get_detector("yolo", confidence=float(os.getenv("YOLO_CONF", "0.3")))
        # get_detector da goi det.load() ben trong
        _model_ready = True
        print(f"[AI Service] Model loaded, ready=True", flush=True)
    except Exception as e:
        print(f"[AI Service] Model load FAILED: {e}", flush=True)
        _model_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global _stream_manager, _grpc_publisher, _model_ready

    # 1. Preload YOLO model (blocking, chay trong thread de khong block event loop)
    load_thread = threading.Thread(target=_preload_model, daemon=True)
    load_thread.start()

    # 2. Khoi tao gRPC Publisher
    _grpc_publisher = OverlayPublisher()
    _grpc_publisher.start()

    # 3. Khoi tao Stream Manager
    _stream_manager = StreamManager(grpc_publisher=_grpc_publisher)

    # Doi model load xong (timeout 120s theo Contract)
    load_thread.join(timeout=120.0)
    if not _model_ready:
        print("[AI Service] WARNING: Model chua load xong sau 120s", flush=True)

    print(f"{'='*60}", flush=True)
    print(f"  AI Service v{_VERSION} started", flush=True)
    print(f"  Control API:  :{os.getenv('CONTROL_API_PORT', '8000')}", flush=True)
    print(f"  gRPC Target:  {os.getenv('OVERLAY_GRPC_TARGET', '127.0.0.1:9090')}", flush=True)
    print(f"  YOLO Weights: {os.getenv('YOLO_WEIGHTS', 'yolov8s.pt')}", flush=True)
    print(f"  YOLO ImgSz:   {os.getenv('YOLO_IMGSZ', '640')}", flush=True)
    print(f"  YOLO Conf:    {os.getenv('YOLO_CONF', '0.3')}", flush=True)
    print(f"{'='*60}", flush=True)

    yield  # ----- app chay -----

    # Shutdown: dung tat ca stream + dong gRPC
    print("[AI Service] Shutting down...", flush=True)
    if _stream_manager:
        _stream_manager.stop_all()
    if _grpc_publisher:
        _grpc_publisher.stop()
    print("[AI Service] Shutdown complete", flush=True)


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="VisionOS AI Service", version=_VERSION, lifespan=lifespan)


# ---------------------------------------------------------------------------
# GET /health (Contract muc 3)
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """200 chi khi model da nap xong. Electron poll endpoint nay de xac dinh startup hoan tat."""
    if not _model_ready:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "version": _VERSION},
        )
    return {"ready": True, "version": _VERSION}


# ---------------------------------------------------------------------------
# POST /streams/{jobKey} (Contract muc 3)
# ---------------------------------------------------------------------------
@app.post("/streams/{job_key}")
def create_or_update_stream(job_key: str, req: StreamControlRequest):
    """Tao moi hoac cap nhat nong job dang chay.

    Backend khong goi DELETE truoc khi doi params -- endpoint nay tu xu ly.
    Response 2xx = da nhan job, khong cho toi bbox dau tien.
    """
    if not _stream_manager:
        raise HTTPException(status_code=500, detail="StreamManager not initialized")
    try:
        result = _stream_manager.start(job_key, req)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# DELETE /streams/{jobKey} (Contract muc 3)
# ---------------------------------------------------------------------------
@app.delete("/streams/{job_key}")
def delete_stream(job_key: str):
    """Dung job, giai phong RTSP session. 404 duoc Backend coi nhu da dung."""
    if not _stream_manager:
        raise HTTPException(status_code=500, detail="StreamManager not initialized")
    try:
        _stream_manager.stop(job_key)
        return {"status": "stopped", "stream_id": job_key}
    except KeyError:
        raise HTTPException(status_code=404, detail="Stream not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# GET /streams (Contract muc 3)
# ---------------------------------------------------------------------------
@app.get("/streams")
def list_streams():
    """Danh sach job dang chay kem fps thuc te -- Backend dung de doi soat va chan doan."""
    if not _stream_manager:
        raise HTTPException(status_code=500, detail="StreamManager not initialized")
    try:
        active = _stream_manager.get_active_streams()
        return {"count": len(active), "streams": active}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
