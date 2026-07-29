#!/usr/bin/env python3
"""Chạy đếm trên NHIỀU video THẬT (catalog) + in scorecard — mở rộng bộ test.

Tải video công khai (Roboflow supervision assets, không cần API key) rồi chạy
detect → track → đếm cho từng kịch bản:

  * ``vehicles`` — phương tiện vào/ra (cao tốc, giao lộ)
  * ``conveyor`` — dây chuyền sản xuất (nhà máy chiết chai)
  * ``people``   — người vào/ra (đi bộ, quảng trường, ga tàu)

Ví dụ:
    python run_scenarios.py --list                          # xem catalog, KHÔNG tải
    python run_scenarios.py --task vehicles --max-frames 300
    python run_scenarios.py --task conveyor
    python run_scenarios.py --task all --max-frames 300
    python run_scenarios.py --task conveyor --model locate --prompt "chai nhựa"
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recognition.counting import CountingPipeline  # noqa: E402
from recognition.video_catalog import by_task, download_video  # noqa: E402


def print_scorecard(rows: list) -> None:
    if not rows:
        print("(không có kết quả)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    line = "  ".join(str(c).ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def _frames_of(path, resolution):
    """Đọc video, resize về resolution của scenario (để vạch %/toạ độ khớp)."""
    import cv2

    w, h = resolution
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {path}")
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if (fr.shape[1], fr.shape[0]) != (w, h):
                fr = cv2.resize(fr, (w, h))
            yield fr
    finally:
        cap.release()


def run(args) -> int:
    scenarios = by_task(None if args.task == "all" else args.task)
    if not scenarios:
        print(f"⚠️  Không có kịch bản cho task={args.task!r}. Dùng: vehicles|conveyor|people|all")
        return 1

    if args.list:
        print("📚 CATALOG VIDEO (tải trực tiếp, không cần API key)\n")
        for v in scenarios:
            s = v.scenario
            print(f"• [{v.task}] {v.name}")
            print(f"    nguồn : {v.source}")
            print(f"    url   : {v.url}")
            print(f"    vạch  : {s.line_start_pct}→{s.line_end_pct}  prompt={s.prompt!r}  model={s.model}")
            if v.tips:
                print(f"    mẹo   : {v.tips}")
            print()
        return 0

    from recognition.detectors import load_locate_anything, load_standard_detector

    _cache: dict = {}

    def get_detector(model_field: str):
        want_locate = args.model == "locate" or (args.model == "auto" and not model_field.startswith("YOLO"))
        kind = "locate" if want_locate else "yolo"
        if kind not in _cache:
            if kind == "locate":
                print("🧠 Nạp LocateAnything-3B …")
                _cache[kind] = load_locate_anything()
            else:
                print("🎯 Nạp YOLO …")
                _cache[kind] = load_standard_detector(backend=args.yolo_backend, confidence=args.confidence)
        return _cache[kind]

    rows = []
    for v in scenarios:
        print(f"\n{'='*70}\n▶ {v.name}  [{v.task}]\n  {v.source}")
        try:
            path = download_video(v)
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ tải video lỗi: {e}")
            continue
        sc = v.scenario
        if args.prompt:  # cho phép ép prompt (ví dụ hàng không thuộc COCO)
            sc = _with_prompt(sc, args.prompt)
        detector = get_detector(sc.model)
        pipe = CountingPipeline(detector, sc)
        pipe.run(_frames_of(path, sc.resolution), max_frames=args.max_frames)
        r = pipe.result.as_row()
        row = {"video": v.name, "task": v.task, **r}
        row["verdict"] = "✅" if (r.get("det/frame", 0) > 0 and r.get("total", 0) >= sc.expect_min) else "⚠️"
        rows.append(row)
        print(f"  → IN={r.get('IN')} OUT={r.get('OUT')} total={r.get('total')} "
              f"det/frame={r.get('det/frame')} fps={r.get('fps')}")

    print(f"\n{'='*70}\n📊 SCORECARD — đếm trên video thật (IoU tracking + cắt vạch)\n{'='*70}")
    print_scorecard(rows)
    return 0


def _with_prompt(scenario, prompt):
    """Tạo bản sao scenario với prompt khác (CountScenario là frozen)."""
    from dataclasses import replace

    return replace(scenario, prompt=prompt)


def main() -> int:
    ap = argparse.ArgumentParser(description="Chạy đếm trên nhiều video thật + scorecard")
    ap.add_argument("--task", default="all", choices=["all", "vehicles", "conveyor", "people"])
    ap.add_argument("--model", default="auto", choices=["auto", "yolo", "locate"],
                    help="auto = theo scenario (mặc định YOLO cho car/person/bottle)")
    ap.add_argument("--prompt", default=None, help="ép prompt (vd hàng không thuộc COCO: 'thùng carton')")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--confidence", type=float, default=0.35)
    ap.add_argument("--yolo-backend", choices=["auto", "ultralytics", "super_gradients"], default="auto")
    ap.add_argument("--list", action="store_true", help="chỉ liệt kê catalog, không tải/chạy")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
