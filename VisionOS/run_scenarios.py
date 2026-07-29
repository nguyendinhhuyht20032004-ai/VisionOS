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
    if args.only:
        kw = args.only.lower()
        scenarios = [v for v in scenarios
                     if kw in v.name.lower() or kw in v.filename.lower() or kw in v.scenario.key.lower()]
    if not scenarios:
        print(f"⚠️  Không có kịch bản khớp (task={args.task!r}, only={args.only!r}).")
        return 1

    if args.list:
        print(f"📚 CATALOG VIDEO ({len(scenarios)}) — tải trực tiếp, không cần API key\n")
        for v in scenarios:
            s = v.scenario
            if v.asset:
                src = f"supervision:{v.asset}"
            elif v.url:
                src = v.url
            else:
                src = f"pexels:{v.pexels_id} (tự dò chất lượng)"
            print(f"• [{v.task}] {v.name}")
            print(f"    nguồn : {v.source}")
            print(f"    tải   : {src}")
            print(f"    vạch  : {s.line_start_pct}→{s.line_end_pct}  prompt={s.prompt!r}  model={s.model}")
            if v.queries:
                print(f"    query : {' | '.join(v.queries)}")
            if v.tips:
                print(f"    mẹo   : {v.tips}")
            print()
        return 0

    from dataclasses import replace

    from recognition.detectors import load_locate_anything, load_standard_detector

    # Query khó → open-vocab (LocateAnything). Nếu test nhiều query mà chưa chỉ định
    # model thì tự chuyển sang locate (YOLO chỉ biết lớp COCO, không hiểu mô tả).
    manual_queries = [q.strip() for q in args.queries.split(",") if q.strip()] if args.queries else None
    testing_queries = bool(manual_queries) or args.all_queries
    if testing_queries and args.model == "auto":
        args.model = "locate"

    def _wants_locate(model_field: str) -> bool:
        return args.model == "locate" or (args.model == "auto" and not model_field.startswith("YOLO"))

    # Nếu SẼ dùng LocateAnything, đảm bảo transformers==4.57.1 TRƯỚC khi nạp (re-exec).
    if args.model == "locate" or any(_wants_locate(v.scenario.model) for v in scenarios):
        try:
            from run_eval import _autopin_transformers

            _autopin_transformers()
        except Exception as e:  # noqa: BLE001
            print(f"ℹ️  bỏ qua auto-pin transformers ({e})")

    _cache: dict = {}

    def get_detector(model_field: str):
        kind = "locate" if _wants_locate(model_field) else "yolo"
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

        # Chọn danh sách prompt để test trên video này (DỄ→KHÓ).
        if manual_queries:
            qlist = manual_queries
        elif args.all_queries:
            qlist = list(v.queries) or [v.scenario.prompt]
        elif args.prompt:
            qlist = [args.prompt]
        else:
            qlist = [v.scenario.prompt]

        for q in qlist:
            sc = replace(v.scenario, prompt=q)
            detector = get_detector(sc.model)
            pipe = CountingPipeline(detector, sc)
            pipe.run(_frames_of(path, sc.resolution), max_frames=args.max_frames)
            r = pipe.result.as_row()
            row = {"video": v.name, "task": v.task, "query": q, **r}
            row["verdict"] = "✅" if (r.get("det/frame", 0) > 0 and r.get("total", 0) >= sc.expect_min) else "⚠️"
            rows.append(row)
            print(f"  · query={q!r} → IN={r.get('IN')} OUT={r.get('OUT')} "
                  f"total={r.get('total')} det/frame={r.get('det/frame')} fps={r.get('fps')}")

    print(f"\n{'='*70}\n📊 SCORECARD — đếm trên video thật (IoU tracking + cắt vạch)\n{'='*70}")
    print_scorecard(rows)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Chạy đếm trên nhiều video thật + scorecard")
    ap.add_argument("--task", default="all", choices=["all", "vehicles", "conveyor", "people"])
    ap.add_argument("--model", default="auto", choices=["auto", "yolo", "locate"],
                    help="auto = theo scenario (mặc định YOLO cho car/person/bottle)")
    ap.add_argument("--prompt", default=None, help="ép 1 prompt cho mọi video")
    ap.add_argument("--queries", default=None,
                    help="danh sách prompt ngăn cách bởi dấu phẩy, test lần lượt trên mỗi video "
                         "(vd 'cardboard box,plastic bottle,damaged package') — tự bật open-vocab")
    ap.add_argument("--all-queries", action="store_true",
                    help="test TẤT CẢ query khó gợi ý sẵn của mỗi video (cột 'queries' trong --list)")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--confidence", type=float, default=0.35)
    ap.add_argument("--yolo-backend", choices=["auto", "ultralytics", "super_gradients"], default="auto")
    ap.add_argument("--only", default=None, help="lọc video theo từ khoá (tên/file/key), vd 'milk'")
    ap.add_argument("--list", action="store_true", help="chỉ liệt kê catalog, không tải/chạy")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
