#!/usr/bin/env python3
"""Chạy SERVICE AI đếm theo camera (FastAPI + uvicorn).

Đưa nguồn camera vào → detect + track + đếm → xuất luồng annotate + số đếm realtime.

Cài (lần đầu):   pip install fastapi uvicorn
Chạy:            python run_service.py --host 0.0.0.0 --port 8000
Rồi mở trình duyệt: http://<máy>:8000  (nhập RTSP/URL/file/webcam → xem trực tiếp).

API nhanh (curl):
    curl -X POST http://localhost:8000/api/jobs -H 'Content-Type: application/json' \
      -d '{"source":"rtsp://...","prompt":"person","counting_type":"line","line":[0,50,100,50]}'
    curl http://localhost:8000/api/jobs/<id>          # số đếm
    # xem luồng: http://localhost:8000/api/jobs/<id>/mjpeg
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CONTROL_API_PORT", "8000")))
    ap.add_argument("--reload", action="store_true", help="tự nạp lại khi sửa code (dev)")
    args = ap.parse_args()

    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
    except ImportError:
        print("❌ Thiếu thư viện. Cài trước:  pip install fastapi uvicorn")
        return 1

    print(f"🎥 Service AI đếm-theo-camera chạy tại: http://{args.host}:{args.port}")
    print("   Mở URL đó trong trình duyệt để nhập camera & xem trực tiếp.")
    if args.reload:
        uvicorn.run("recognition.service.app:app", host=args.host, port=args.port, reload=True)
    else:
        from recognition.service.app import app
        uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
