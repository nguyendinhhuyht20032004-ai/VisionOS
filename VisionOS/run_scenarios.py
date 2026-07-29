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


def _draw_overlay(frame, tracked, pipe):
    """Vẽ VẠCH/VÙNG + box + track-id + số đếm lên frame (để lưu video kiểm tra)."""
    import cv2
    import numpy as np

    img = frame
    w, h = pipe.w, pipe.h
    GREEN, YELLOW, CYAN, WHITE = (0, 200, 0), (0, 255, 255), (255, 255, 0), (255, 255, 255)

    # Vạch cắt (line) — vàng, dày.
    if pipe.line is not None:
        (sx, sy), (ex, ey) = pipe.line.endpoints(w, h)
        cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)), YELLOW, 3)
    # Vùng (zone) — tô xanh mờ + viền.
    if pipe.zone is not None:
        pts = np.array([[int(x), int(y)] for x, y in pipe.zone.to_pixels(w, h)], dtype=np.int32)
        ov = img.copy()
        cv2.fillPoly(ov, [pts], (0, 170, 0))
        cv2.addWeighted(ov, 0.25, img, 0.75, 0, img)
        cv2.polylines(img, [pts], True, GREEN, 2)
    # Box + track id
    for d in tracked:
        x1, y1, x2, y2 = (int(v) for v in d.bbox.as_xyxy())
        cv2.rectangle(img, (x1, y1), (x2, y2), CYAN, 2)
        tid = d.track_id if d.track_id is not None else "?"
        cv2.putText(img, f"#{tid}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
    # Banner số đếm
    r = pipe.result
    if pipe.line is not None:
        txt = f"{pipe.scenario.in_label}:{r.in_count}  {pipe.scenario.out_label}:{r.out_count}  frame:{r.frames}"
    else:
        txt = f"trong vung:{r.zone_current}  dinh:{r.zone_peak}  frame:{r.frames}"
    cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    return img


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:60]


def run(args) -> int:
    scenarios = by_task(None if args.task == "all" else args.task)
    if args.only:
        kw = args.only.lower()
        scenarios = [v for v in scenarios
                     if kw in v.name.lower() or kw in v.filename.lower() or kw in v.scenario.key.lower()]
    if not scenarios:
        print(f"⚠️  Không có kịch bản khớp (task={args.task!r}, only={args.only!r}).")
        return 1

    if args.download_only:
        # CHỈ tải video + báo ✅/❌ (không nạp model) — kiểm tra nhanh nguồn nào chạy.
        ok = fail = 0
        seen = set()
        for v in scenarios:
            if v.filename in seen:
                continue
            seen.add(v.filename)
            try:
                path = download_video(v)
                sz = os.path.getsize(path) / 1e6
                print(f"  ✅ {v.filename:38} {sz:6.1f} MB  [{v.task}]")
                ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"  ❌ {v.filename:38} — {e}")
                fail += 1
        print(f"\nKết quả tải: ✅ {ok}  ❌ {fail}. (Video ❌ báo tôi ID để tôi đổi nguồn.)")
        return 0 if fail == 0 else 1

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
            if s.counting_type == "zone":
                geom = f"vùng  : {len(s.zone_points_pct)} đỉnh {s.zone_points_pct}"
            else:
                geom = f"vạch  : {s.line_start_pct}→{s.line_end_pct}"
            print(f"    {geom}  prompt={s.prompt!r}  model={s.model}")
            if v.queries:
                print(f"    query : {' | '.join(v.queries)}")
            if v.tips:
                print(f"    mẹo   : {v.tips}")
            print()
        return 0

    from dataclasses import replace

    from recognition.detectors import load_locate_anything, load_standard_detector
    from recognition.video_catalog import suite_for

    # Query khó → open-vocab (LocateAnything). Nếu test nhiều query mà chưa chỉ định
    # model thì tự chuyển sang locate (YOLO chỉ biết lớp COCO, không hiểu mô tả).
    manual_queries = [q.strip() for q in args.queries.split(",") if q.strip()] if args.queries else None
    testing_queries = bool(manual_queries) or args.all_queries or args.suite
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
        # Danh sách (nhóm, query) để test trên video này.
        if args.suite:
            qpairs = suite_for(v.task) or [("", v.scenario.prompt)]
        elif manual_queries:
            qpairs = [("", q) for q in manual_queries]
        elif args.all_queries:
            qpairs = [("", q) for q in (v.queries or (v.scenario.prompt,))]
        elif args.prompt:
            qpairs = [("", args.prompt)]
        else:
            qpairs = [("", v.scenario.prompt)]

        for group, q in qpairs:
            sc = replace(v.scenario, prompt=q)
            detector = get_detector(sc.model)
            pipe = CountingPipeline(detector, sc)

            # Lưu video output (vẽ vạch/vùng + box + số đếm) nếu có --save-dir.
            writer, out_path, on_frame = None, None, None
            if args.save_dir:
                import cv2

                task_dir = os.path.join(args.save_dir, v.task)
                os.makedirs(task_dir, exist_ok=True)
                out_path = os.path.join(task_dir, f"{sc.key}__{_safe_name(q)}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, 20.0, tuple(sc.resolution))

                def on_frame(frame, tracked, pipe, _w=writer):
                    _w.write(_draw_overlay(frame, tracked, pipe))

            pipe.run(_frames_of(path, sc.resolution), max_frames=args.max_frames, on_frame=on_frame)
            if writer is not None:
                writer.release()
                print(f"  🎥 lưu video: {out_path}")

            r = pipe.result.as_row()
            row = {"video": v.name, "task": v.task}
            if args.suite:
                row["nhóm"] = group
            row.update({"query": q, "type": sc.counting_type, **r})
            ok = (r.get("det/frame", 0) > 0
                  and (r.get("total", r.get("đỉnh_vùng", 0)) >= sc.expect_min))
            row["verdict"] = "✅" if ok else "⚠️"
            rows.append(row)
            tag = f"[{group}] " if group else ""
            if sc.counting_type == "line":
                print(f"  · {tag}query={q!r} → IN={r.get('IN')} OUT={r.get('OUT')} "
                      f"total={r.get('total')} det/frame={r.get('det/frame')} fps={r.get('fps')}")
            else:
                print(f"  · {tag}query={q!r} → trong_vùng={r.get('trong_vùng')} đỉnh={r.get('đỉnh_vùng')} "
                      f"det/frame={r.get('det/frame')} fps={r.get('fps')}")

    # Tách scorecard theo kiểu đếm (line/zone khác cột) cho gọn.
    line_rows = [r for r in rows if r.get("type") == "line"]
    zone_rows = [r for r in rows if r.get("type") == "zone"]
    print(f"\n{'='*70}\n📊 SCORECARD — đếm trên video thật\n{'='*70}")
    if line_rows:
        print("\n▶ Đếm cắt VẠCH (vào/ra):")
        print_scorecard([{k: v for k, v in r.items() if k != "type"} for r in line_rows])
    if zone_rows:
        print("\n▶ Đếm trong VÙNG (occupancy):")
        print_scorecard([{k: v for k, v in r.items() if k != "type"} for r in zone_rows])
    if args.save_dir:
        print(f"\n🎥 Video output đã lưu trong: {args.save_dir}/<task>/")
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
    ap.add_argument("--suite", action="store_true",
                    help="chạy BỘ QUERY SUITE đầy đủ theo NHÓM cho bài toán (cơ bản/màu/phụ kiện/"
                         "hành động/khó/tiếng Việt) — nhiều trường hợp như bảng test Excel")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--confidence", type=float, default=0.35)
    ap.add_argument("--yolo-backend", choices=["auto", "ultralytics", "super_gradients"], default="auto")
    ap.add_argument("--only", default=None, help="lọc video theo từ khoá (tên/file/key), vd 'milk'")
    ap.add_argument("--download-only", action="store_true",
                    help="CHỈ tải video + báo ✅/❌ (không nạp model) — kiểm tra nguồn nào chạy")
    ap.add_argument("--save-dir", default=None,
                    help="LƯU VIDEO OUTPUT (vẽ vạch/vùng + box + số đếm) vào thư mục này")
    ap.add_argument("--list", action="store_true", help="chỉ liệt kê catalog, không tải/chạy")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
