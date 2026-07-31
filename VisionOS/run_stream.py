#!/usr/bin/env python3
"""Chạy/TEST hệ thống đếm trên MỘT nguồn — KHÔNG cần camera thật (dùng file video).

Đây là cách test nhanh xem service chạy hiệu quả không: coi 1 file .mp4 như "camera",
chạy đúng engine của service (detect → track → smooth → đếm) rồi XUẤT video annotate
+ in số đếm mỗi frame.

VÍ DỤ (Colab/Kaggle — không cần camera):
    !python run_stream.py --source sample_videos/tomatoes_sorting.mp4 \
        --prompt tomato --orient horizontal --line-pos 0.72 \
        --max-frames 60 --out /kaggle/working/out.mp4

Camera THẬT sau này (chỉ đổi --source):
    python run_stream.py --source "rtsp://user:pass@ip:554/stream" --prompt person \
        --orient horizontal --line-pos 0.5
    python run_stream.py --source 0 --prompt person           # webcam
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recognition.service import StreamingCounter, make_scenario  # noqa: E402
from recognition.service.builder import get_detector  # noqa: E402


def iter_frames(source, max_frames):
    """Đọc frame. File → đọc TUẦN TỰ (để track/đếm đúng). Stream/webcam → frame mới nhất."""
    import cv2

    if isinstance(source, str) and os.path.isfile(source):
        cap = cv2.VideoCapture(source)
        i = 0
        while i < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            yield fr
            i += 1
        cap.release()
    else:
        from recognition.service import FrameSource

        fs = FrameSource(source).start()
        i = 0
        while i < max_frames:
            fr = fs.read()
            if fr is None:
                if not fs.alive:
                    break
                time.sleep(0.05)
                continue
            yield fr
            i += 1
        fs.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="file .mp4 | rtsp://… | http://… | 0 (webcam)")
    ap.add_argument("--prompt", default="person", help="đối tượng: person/car/object/carton box/tomato…")
    ap.add_argument("--counting-type", default="line", choices=["line", "zone"])
    ap.add_argument("--orient", default="horizontal", choices=["horizontal", "vertical"],
                    help="hướng vạch khi dùng --line-pos")
    ap.add_argument("--line-pos", type=float, default=0.5, help="vị trí vạch 0..1 (dùng với --orient)")
    ap.add_argument("--line", default=None, help="ép vạch 'x1,y1,x2,y2' %% (ưu tiên hơn --orient/--line-pos)")
    ap.add_argument("--zone", default=None, help="vùng 'x1,y1;x2,y2;…' %% (khi --counting-type zone)")
    ap.add_argument("--model", default="auto", choices=["auto", "yolo", "locate"])
    ap.add_argument("--proc-width", type=int, default=960)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--out", default="stream_out.mp4", help="video annotate xuất ra")
    args = ap.parse_args()

    # dựng vạch/vùng
    line = zone = None
    if args.counting_type == "line":
        if args.line:
            line = [float(x) for x in args.line.split(",")]
        else:
            p = args.line_pos * 100.0
            line = [0, p, 100, p] if args.orient == "horizontal" else [p, 0, p, 100]
    elif args.zone:
        zone = [[float(c) for c in pt.split(",")] for pt in args.zone.split(";")]

    w = args.proc_width
    h = max(2, round(w * 9 / 16))
    sc, kind = make_scenario(args.prompt, args.counting_type, line, zone, (w, h), model=args.model)
    print(f"▶ nguồn={args.source!r} · prompt={args.prompt!r} · model={kind} · {w}x{h} · "
          f"{args.counting_type} {line or zone}")
    print("  Nạp detector… (LocateAnything lần đầu ~6GB)")
    detector = get_detector(kind)
    counter = StreamingCounter(sc, detector, resolution=(w, h))

    import cv2

    writer = None
    n = 0
    t0 = time.time()
    for fr in iter_frames(args.source if not str(args.source).isdigit() else int(args.source),
                          args.max_frames):
        out = counter.process(fr)
        n += 1
        if writer is None:
            hh, ww = out.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 10, (ww, hh))
        writer.write(out)
        s = counter.stats()
        if s["counting_type"] == "line":
            prog = f"in={s.get('in')} out={s.get('out')} total={s.get('total')}"
        else:
            prog = f"in_zone={s.get('in_zone')} peak={s.get('zone_peak')}"
        print(f"  frame {n:3d}: {prog}  tracks={s['tracks']}  det/fr={s['det_per_frame']}", flush=True)
    if writer is not None:
        writer.release()

    st = counter.stats()
    print(f"\n✅ XONG {n} frame trong {time.time()-t0:.0f}s. Video annotate: {args.out}")
    print(f"   Kết quả: {st}")
    if n == 0:
        print("⚠️  Không đọc được frame nào — kiểm tra --source (đường dẫn/URL đúng chưa?).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
