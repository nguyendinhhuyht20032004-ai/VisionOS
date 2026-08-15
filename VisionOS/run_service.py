#!/usr/bin/env python3
"""Chay AI Service (FastAPI + uvicorn) -- Control API cho VisionOS.

Theo AI_SERVICE_CONTRACT.md:
  * Bind 127.0.0.1 (loopback, khong auth)
  * Cong doc tu CONTROL_API_PORT (mac dinh 8000)

Chay:  python run_service.py
       python run_service.py --port 8000
       python run_service.py --reload       # dev mode
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    # Ep kieu in ra console dung utf-8 de khong bi loi tren Windows
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address (default: 127.0.0.1 -- loopback theo Contract)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CONTROL_API_PORT", "8000")))
    ap.add_argument("--reload", action="store_true", help="tu nap lai khi sua code (dev)")
    args = ap.parse_args()

    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
    except ImportError:
        print("Thieu thu vien. Cai truoc:  pip install fastapi uvicorn")
        return 1

    print(f"AI Service Control API: http://{args.host}:{args.port}")
    print(f"  Health check:  GET /health")
    print(f"  Streams API:   POST/DELETE /streams/{{jobKey}}")
    if args.reload:
        uvicorn.run("recognition.service.app:app", host=args.host, port=args.port, reload=True)
    else:
        from recognition.service.app import app
        uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
