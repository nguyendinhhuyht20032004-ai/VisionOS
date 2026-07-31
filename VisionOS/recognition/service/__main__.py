"""``python -m recognition.service`` → chạy service (FastAPI + uvicorn)."""

from __future__ import annotations


def main():
    import argparse

    ap = argparse.ArgumentParser(description="VisionOS camera counting service")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("Cần cài: pip install fastapi uvicorn")
    from .app import app

    print(f"🎥 Service tại http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
