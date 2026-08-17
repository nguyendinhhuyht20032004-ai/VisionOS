#!/usr/bin/env python3
"""Test script -- tao stream va verify AI Service hoat dong dung.

Theo kien truc moi (AI_SERVICE_CONTRACT.md):
  * Tao stream qua POST /streams/{jobKey}
  * Doc danh sach stream qua GET /streams
  * Dung stream qua DELETE /streams/{jobKey}
  * Ket qua (bbox + events) di qua gRPC ve Backend (khong qua Redis)

Dung:
    python test_people.py                          # dung default
    python test_people.py --source rtsp://...      # RTSP stream
    python test_people.py --source video.mp4       # file video
    python test_people.py --port 8000              # port khac
"""

from __future__ import annotations

import argparse
import json
import time
import sys

try:
    import requests
except ImportError:
    print("Can cai requests:  pip install requests")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="AI Service host")
    ap.add_argument("--port", type=int, default=8000, help="AI Service port")
    ap.add_argument("--source", default="rtsp://127.0.0.1:8554/camera-1-main",
                    help="RTSP URL hoac duong dan file video")
    ap.add_argument("--camera-id", default="1", help="Camera ID")
    ap.add_argument("--job-key", default="test-1", help="Job key")
    ap.add_argument("--classes", default="person", help="Lop can detect (cach nhau boi dau phay)")
    ap.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    ap.add_argument("--duration", type=int, default=30, help="Thoi gian chay (giay)")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"

    # 1. Health check
    print(f"\n{'='*60}")
    print(f"  Test AI Service: {base}")
    print(f"{'='*60}")

    print("\n[1] Health check...")
    try:
        r = requests.get(f"{base}/health", timeout=5)
        data = r.json()
        print(f"    Status: {r.status_code}")
        print(f"    Response: {json.dumps(data, indent=2)}")
        if not data.get("ready"):
            print("    ⚠️ Model chua san sang. Doi them...")
            for _ in range(30):
                time.sleep(2)
                r = requests.get(f"{base}/health", timeout=5)
                if r.json().get("ready"):
                    print("    ✅ Model san sang!")
                    break
            else:
                print("    ❌ Model khong san sang sau 60s. Thoat.")
                return 1
    except requests.ConnectionError:
        print(f"    ❌ Khong ket noi duoc toi {base}. AI Service dang chay chua?")
        return 1

    # 2. Tao stream
    print(f"\n[2] Tao stream '{args.job_key}'...")
    classes = [c.strip() for c in args.classes.split(",")]
    body = {
        "camera_id": args.camera_id,
        "rtsp_url": args.source,
        "params": {
            "classes": classes,
            "conf": args.conf,
            "roi": None,
            "publish_fps": 30,
            "detect_every": 3,
            "track_timeout": 2.0,
        }
    }
    print(f"    Body: {json.dumps(body, indent=2)}")
    r = requests.post(f"{base}/streams/{args.job_key}", json=body, timeout=10)
    print(f"    Status: {r.status_code}")
    print(f"    Response: {json.dumps(r.json(), indent=2)}")
    if r.status_code >= 400:
        print("    ❌ Khong tao duoc stream!")
        return 1

    # 3. Theo doi stream
    print(f"\n[3] Theo doi stream trong 15s... (Dang tim '{args.classes}' voi conf={args.conf})")
    for i in range(15):
        time.sleep(1)
        r = requests.get(f"{base}/streams", timeout=5)
        streams = r.json()
        if streams.get("count", 0) == 0:
            print(f"    [{i+1:3d}s] Khong co stream nao dang chay!")
            continue
        for s in streams.get("streams", []):
            print(f"    [{i+1:3d}s] Stream={s.get('stream_id')} | Camera={s.get('camera_id')} | FPS={s.get('fps')}")

    # 4. Cap nhat luong (Update stream parameters)
    print(f"\n[4] Cap nhat thong so luong '{args.job_key}' (Doi Conf len 0.8 de loc chat hon)...")
    body["params"]["conf"] = 0.8
    r = requests.post(f"{base}/streams/{args.job_key}", json=body, timeout=10)
    print(f"    Status: {r.status_code}")
    print(f"    Response: {json.dumps(r.json(), indent=2)}")

    # 5. Theo doi tiep 15s sau khi cap nhat
    print(f"\n[5] Theo doi stream them 15s sau khi cap nhat...")
    for i in range(15):
        time.sleep(1)
        r = requests.get(f"{base}/streams", timeout=5)
        streams = r.json()
        if streams.get("count", 0) == 0:
            print(f"    [{i+1:3d}s] Khong co stream nao dang chay!")
            continue
        for s in streams.get("streams", []):
            print(f"    [{i+1:3d}s] Stream={s.get('stream_id')} | Conf hien tai={s.get('params',{}).get('conf')} | FPS={s.get('fps')}")

    # 6. Cap nhat luong: Them class moi (Tracking them 'car')
    print(f"\n[6] Cap nhat thong so luong '{args.job_key}' (Tracking them 'car' vao danh sach)...")
    body["params"]["classes"] = ["person", "car"]
    r = requests.post(f"{base}/streams/{args.job_key}", json=body, timeout=10)
    print(f"    Status: {r.status_code}")
    print(f"    Response: {json.dumps(r.json(), indent=2)}")

    # 7. Theo doi them 10s
    print(f"\n[7] Theo doi stream them 10s sau khi cap nhat class...")
    for i in range(10):
        time.sleep(1)
        r = requests.get(f"{base}/streams", timeout=5)
        streams = r.json()
        if streams.get("count", 0) == 0:
            print(f"    [{i+1:3d}s] Khong co stream nao dang chay!")
            continue
        for s in streams.get("streams", []):
            print(f"    [{i+1:3d}s] Stream={s.get('stream_id')} | Classes hien tai={s.get('params',{}).get('classes')} | FPS={s.get('fps')}")

    # 8. Cap nhat luong: Thay doi han URL (Test Hard Restart)
    # URL mac dinh la data/videos/people-walking.mp4. Ta se thu doi sang cai khac (neu co)
    # Vi test cuc bo, cu dung lai chinh url do hoac 1 url ao de xem log he thong co ngat luong cu di bat lai khong.
    print(f"\n[8] Cap nhat luong '{args.job_key}': Thay doi URL stream (De test chuc nang Restart luong)...")
    body["rtsp_url"] = "data/videos/people-walking.mp4?dummy=1" # Them dummy de gia lap URL moi
    r = requests.post(f"{base}/streams/{args.job_key}", json=body, timeout=10)
    print(f"    Status: {r.status_code}")
    print(f"    Response: {json.dumps(r.json(), indent=2)}")

    # 9. Theo doi 10s cuoi cung
    print(f"\n[9] Theo doi stream 10s sau khi Hard Restart...")
    for i in range(10):
        time.sleep(1)
        r = requests.get(f"{base}/streams", timeout=5)
        streams = r.json()
        if streams.get("count", 0) == 0:
            print(f"    [{i+1:3d}s] Khong co stream nao dang chay!")
            continue
        for s in streams.get("streams", []):
            print(f"    [{i+1:3d}s] Stream={s.get('stream_id')} | URL hien tai={s.get('rtsp_url')} | FPS={s.get('fps')}")

    # 10. List streams
    print(f"\n[10] Danh sach streams hien tai...")
    r = requests.get(f"{base}/streams", timeout=5)
    print(f"    {json.dumps(r.json(), indent=2)}")

    # 11. Dung stream
    print(f"\n[11] Dung stream '{args.job_key}'...")
    r = requests.delete(f"{base}/streams/{args.job_key}", timeout=10)
    print(f"    Status: {r.status_code}")
    print(f"    Response: {json.dumps(r.json(), indent=2)}")

    # 12. Verify dung
    print(f"\n[12] Verify stream da dung...")
    r = requests.get(f"{base}/streams", timeout=5)
    data = r.json()
    print(f"    Active streams: {data.get('count', 'unknown')}")
    if data.get("count", 0) == 0:
        print("    Tat ca stream da dung!")
    else:
        print(f"    Van con {data['count']} stream dang chay")

    print(f"\n{'='*60}")
    print(f"  Test hoan tat!")
    print(f"  Luu y: Ket qua bbox + events di qua gRPC ve Backend.")
    print(f"  Kiem tra phia Backend (Electron) de xac nhan nhan duoc du lieu.")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
