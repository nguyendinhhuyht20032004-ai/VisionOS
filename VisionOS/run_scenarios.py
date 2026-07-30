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
from dataclasses import replace as replace_sc

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


def _frames_of(path, resolution, stride=1):
    """Đọc video, resize về resolution của scenario (để vạch %/toạ độ khớp).

    ``stride>1`` bỏ bớt frame (chỉ lấy mỗi frame thứ ``stride``) → chạy NHANH hơn
    (ít lần gọi model). Dùng cho chế độ suite/thử nhanh; đếm chính xác nên để stride=1.
    """
    import cv2

    w, h = resolution
    stride = max(1, int(stride))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {path}")
    try:
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                if (fr.shape[1], fr.shape[0]) != (w, h):
                    fr = cv2.resize(fr, (w, h))
                yield fr
            i += 1
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
    # Vùng (zone) — tô xanh mờ + viền. Hỗ trợ NHIỀU vùng (đếm chung).
    for z in getattr(pipe, "zones", []) or ([pipe.zone] if pipe.zone else []):
        pts = np.array([[int(x), int(y)] for x, y in z.to_pixels(w, h)], dtype=np.int32)
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
    # Banner số đếm (chuyển ASCII vì cv2.putText KHÔNG vẽ được dấu tiếng Việt → "V??o")
    r = pipe.result
    if pipe.line is not None:
        txt = f"{pipe.scenario.in_label}:{r.in_count}  {pipe.scenario.out_label}:{r.out_count}  frame:{r.frames}"
    else:
        txt = f"trong vung:{r.zone_current}  dinh:{r.zone_peak}  frame:{r.frames}"
    cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(img, _ascii(txt), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    return img


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:60]


def _ascii(s: str) -> str:
    """Bỏ dấu tiếng Việt để cv2.putText hiển thị được (cv2 KHÔNG vẽ Unicode → 'V??o')."""
    import unicodedata

    s = s.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _draw_pct_grid(frame, step=10):
    """Kẻ LƯỚI % (0..100) + nhãn để BẠN đọc thẳng toạ độ đặt vạch/vùng trên ảnh."""
    import cv2

    h, w = frame.shape[:2]
    for p in range(step, 100, step):
        x, y = int(w * p / 100), int(h * p / 100)
        cv2.line(frame, (x, 0), (x, h), (55, 55, 55), 1)
        cv2.line(frame, (0, y), (w, y), (55, 55, 55), 1)
    for p in range(0, 101, 20):          # nhãn % dày hơn ở mốc 0/20/.../100
        x, y = int(w * p / 100), int(h * p / 100)
        cv2.putText(frame, str(p), (min(x + 2, w - 22), 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 215), 1)
        cv2.putText(frame, str(p), (2, min(max(y + 4, 13), h - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 215), 1)
    return frame


def _draw_scenario_geom(frame, sc):
    """Vẽ VẠCH (vàng)/VÙNG (xanh) + NHÃN TOẠ ĐỘ % — để bạn thấy đang đặt ở đâu mà chỉnh."""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    if sc.counting_type == "zone":
        for zi, z in enumerate(sc.build_zones()):          # hỗ trợ NHIỀU vùng
            pts = np.array([[int(x), int(y)] for x, y in z.to_pixels(w, h)], dtype=np.int32)
            ov = frame.copy()
            cv2.fillPoly(ov, [pts], (0, 170, 0))
            cv2.addWeighted(ov, 0.3, frame, 0.7, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            for (px, py), (xp, yp) in zip(pts, z.points_pct):   # chấm + toạ độ % mỗi đỉnh
                cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"({xp:.0f},{yp:.0f})", (int(px) + 5, int(py) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(frame, f"VUNG {zi + 1}", (pts[0][0] + 4, pts[0][1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        (x1, y1), (x2, y2) = sc.line_start_pct, sc.line_end_pct
        (sx, sy), (ex, ey) = sc.build_line().endpoints(w, h)
        cv2.line(frame, (int(sx), int(sy)), (int(ex), int(ey)), (0, 255, 255), 3)
        if abs(x1 - x2) < 1e-6:
            lbl = f"VACH doc x={x1:.0f}"
        elif abs(y1 - y2) < 1e-6:
            lbl = f"VACH ngang y={y1:.0f}"
        else:
            lbl = f"VACH ({x1:.0f},{y1:.0f})->({x2:.0f},{y2:.0f})"
        mx, my = int((sx + ex) / 2), int((sy + ey) / 2)
        cv2.putText(frame, lbl, (min(max(mx - 60, 4), w - 230), max(my - 8, 48)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


def run(args) -> int:
    scenarios = by_task(None if args.task == "all" else args.task)
    if args.only:
        kw = args.only.lower()
        scenarios = [v for v in scenarios
                     if kw in v.name.lower() or kw in v.filename.lower() or kw in v.scenario.key.lower()]
    if not scenarios:
        print(f"⚠️  Không có kịch bản khớp (task={args.task!r}, only={args.only!r}).")
        return 1

    # Parse ép vạch/vùng (theo %). --line "x1,y1,x2,y2" · --zone "x1,y1;x2,y2;..."
    _line_pts = _zone_pts = None
    if args.line:
        try:
            _line_pts = [float(t) for t in args.line.split(",")]
            assert len(_line_pts) == 4
        except Exception:
            print("⚠️  --line phải là 'x1,y1,x2,y2' theo % (0-100). Bỏ qua.")
            _line_pts = None
    if args.zone and not _line_pts:
        try:
            _zone_pts = tuple(tuple(float(c) for c in p.split(",")) for p in args.zone.split(";"))
            assert len(_zone_pts) >= 3 and all(len(p) == 2 for p in _zone_pts)
        except Exception:
            print("⚠️  --zone phải là 'x1,y1;x2,y2;...' (≥3 đỉnh, %). Bỏ qua.")
            _zone_pts = None

    def _apply_geom(sc):
        if _line_pts:
            return replace_sc(sc, counting_type="line",
                              line_start_pct=(_line_pts[0], _line_pts[1]),
                              line_end_pct=(_line_pts[2], _line_pts[3]))
        if _zone_pts:
            return replace_sc(sc, counting_type="zone", zone_points_pct=_zone_pts)
        return sc

    if args.preview is not None:
        # Vẽ LƯỚI % + vạch/vùng lên frame CÓ VẬT của MỌI video (KHÔNG cần model)
        # → bạn xem đặt đúng chưa rồi báo toạ độ mới, hoặc thử ngay --line/--zone.
        import cv2
        import numpy as np

        outdir = args.preview or "geom_preview"
        os.makedirs(outdir, exist_ok=True)
        seen, thumbs = set(), []
        for v in scenarios:
            sc0 = _apply_geom(v.scenario)
            if sc0.key in seen:        # dedup theo KỊCH BẢN (market-square có cả line lẫn zone)
                continue
            seen.add(sc0.key)
            try:
                path = download_video(v)
            except Exception as e:  # noqa: BLE001
                print(f"  ❌ {v.filename}: {e}")
                continue
            # Lấy frame ~giây thứ 2 (đọc tuần tự, tin cậy hơn seek) — frame đầu thường TRỐNG.
            cap = cv2.VideoCapture(path)
            fr = None
            for _ in range(50):
                ok, f = cap.read()
                if not ok:
                    break
                fr = f
            cap.release()
            if fr is None:
                print(f"  ❌ không đọc được frame: {v.filename}")
                continue
            fr = cv2.resize(fr, tuple(sc0.resolution))
            _draw_pct_grid(fr)
            _draw_scenario_geom(fr, sc0)
            out = os.path.join(outdir, f"{v.task}_{sc0.key}.jpg")
            cv2.imwrite(out, fr)
            # thumbnail + tên (ASCII) cho ảnh tổng hợp _ALL.jpg
            th = cv2.resize(fr, (480, 270))
            bar = np.zeros((26, 480, 3), np.uint8)
            cv2.putText(bar, _ascii(f"[{v.task}] {v.name}")[:56], (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            thumbs.append(np.vstack([bar, th]))
            geom = (f"vạch {sc0.line_start_pct}->{sc0.line_end_pct}" if sc0.counting_type == "line"
                    else f"vùng {sc0.zone_points_pct}")
            print(f"  🖼️  {out}   ({sc0.counting_type}: {geom})")
        # Ảnh TỔNG HỢP: xem HẾT các trường hợp trong 1 ảnh.
        if thumbs:
            cols = 2
            rows = (len(thumbs) + cols - 1) // cols
            ch, cw = thumbs[0].shape[:2]
            canvas = np.zeros((rows * ch, cols * cw, 3), np.uint8)
            for i, t in enumerate(thumbs):
                r, c = divmod(i, cols)
                canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = t
            all_path = os.path.join(outdir, "_ALL.jpg")
            cv2.imwrite(all_path, canvas)
            print(f"\n  🧩 ẢNH TỔNG HỢP (xem HẾT trong 1 ảnh): {all_path}")
        print(f"\nXem ảnh trong {outdir}/ (lưới % giúp đọc toạ độ). Muốn đổi vạch/vùng thì báo tôi")
        print("toạ độ %, hoặc thử ngay:  --only <video> --line 'x1,y1,x2,y2'  /  --zone 'x1,y1;x2,y2;...'")
        return 0

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

    from recognition.detectors import load_locate_anything, load_standard_detector
    from recognition.video_catalog import suite_for

    # Query khó → open-vocab (LocateAnything). Nếu test nhiều query mà chưa chỉ định
    # model thì tự chuyển sang locate (YOLO chỉ biết lớp COCO, không hiểu mô tả).
    run_suite = args.suite or args.suite_full         # --suite = LITE (nhanh), --suite-full = đầy đủ
    manual_queries = [q.strip() for q in args.queries.split(",") if q.strip()] if args.queries else None
    testing_queries = bool(manual_queries) or args.all_queries or run_suite
    if testing_queries and args.model == "auto":
        args.model = "locate"

    # TỐI ƯU TỐC ĐỘ (Colab/Kaggle session ngắn): chế độ suite mặc định ÍT frame + bỏ
    # bớt frame (stride) vì suite là "thử khả năng mô tả", không cần đếm cực chuẩn.
    max_frames = args.max_frames if args.max_frames is not None else (60 if run_suite else 300)
    stride = args.stride if args.stride is not None else (2 if run_suite else 1)

    # Engine đếm: supervision (ByteTrack + LineZone/PolygonZone — chuẩn hơn) hay bộ tự viết.
    if args.engine == "sv":
        use_sv = True
    elif args.engine == "builtin":
        use_sv = False
    else:  # auto: dùng supervision nếu cài được
        from recognition.sv_counting import sv_available

        use_sv = sv_available()
    print(f"🧮 Engine đếm: {'supervision (ByteTrack)' if use_sv else 'tự viết (CentroidTracker)'}")
    if run_suite:
        print(f"⚡ Suite {'ĐẦY ĐỦ' if args.suite_full else 'LITE'}: "
              f"{args.suite_per_group} query/nhóm · max_frames={max_frames} · stride={stride} "
              f"(đổi bằng --suite-per-group/--max-frames/--stride; nên chạy 1 video với --only)")

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
        import time as _time

        kind = "locate" if _wants_locate(model_field) else "yolo"
        if kind not in _cache:
            if kind == "locate":
                # NẠP NGAY tại đây (không để lười tới frame đầu). Load 3B ~6GB lần đầu
                # mất vài phút — nếu để nó chạy GIỮA vòng đếm sẽ TRÔNG như treo, dễ bị
                # bấm Stop → KeyboardInterrupt (KHÔNG phải lỗi code). In mốc rõ ràng.
                print("🧠 Nạp LocateAnything-3B (mô hình ~6GB) …")
                print("   ⏳ LẦN ĐẦU tải + nạp mất 3–8 phút (kéo ~6GB từ HuggingFace) — ĐỪNG bấm Stop.")
                print("   ℹ️  Nếu thấy 'KeyboardInterrupt' nghĩa là đã NGẮT giữa chừng, không phải bug.")
                det = load_locate_anything()
                t0 = _time.time()
                det.load()                       # tải + nạp NGAY, có mốc thời gian
                print(f"   ✅ Model sẵn sàng sau {_time.time() - t0:.0f}s — bắt đầu đếm.")
                _cache[kind] = det
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
        if run_suite:
            qpairs = suite_for(v.task, lite=not args.suite_full,
                               per_group=args.suite_per_group) or [("", v.scenario.prompt)]
        elif manual_queries:
            qpairs = [("", q) for q in manual_queries]
        elif args.all_queries:
            qpairs = [("", q) for q in (v.queries or (v.scenario.prompt,))]
        elif args.prompt:
            qpairs = [("", args.prompt)]
        else:
            qpairs = [("", v.scenario.prompt)]

        for group, q in qpairs:
            # ÉP vạch/vùng theo tay (bạn xem video rồi đặt cho khớp hướng vật chạy).
            sc = _apply_geom(replace_sc(v.scenario, prompt=q))
            detector = get_detector(sc.model)

            # Lưu video output (vẽ vạch/vùng + box + số đếm) nếu có --save-dir.
            writer, out_path = None, None
            if args.save_dir:
                import cv2

                task_dir = os.path.join(args.save_dir, v.task)
                os.makedirs(task_dir, exist_ok=True)
                out_path = os.path.join(task_dir, f"{sc.key}__{_safe_name(q)}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, 20.0, tuple(sc.resolution))

            result = None
            if use_sv:
                # ENGINE supervision (ByteTrack + LineZone/PolygonZone) — đếm chuẩn hơn.
                try:
                    from recognition.sv_counting import sv_run

                    result = sv_run(_frames_of(path, sc.resolution, stride=stride), sc, detector,
                                    sc.resolution, max_frames=max_frames, writer=writer)
                except Exception as e:  # noqa: BLE001 — lỗi API supervision → rơi về builtin
                    print(f"  ⚠️  engine supervision lỗi ({e}); dùng bộ đếm tự viết.")
                    result = None
            if result is None:
                pipe = CountingPipeline(detector, sc)
                on_frame = None
                if writer is not None:
                    def on_frame(frame, tracked, pipe, _w=writer):
                        _w.write(_draw_overlay(frame, tracked, pipe))
                pipe.run(_frames_of(path, sc.resolution, stride=stride),
                         max_frames=max_frames, on_frame=on_frame)
                result = pipe.result
            if writer is not None:
                writer.release()
                print(f"  🎥 lưu video: {out_path}")

            r = result.as_row()
            row = {"video": v.name, "task": v.task}
            if run_suite:
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
                    help="chạy QUERY SUITE theo NHÓM — bản LITE (1-2 query/nhóm, đại diện: "
                         "màu=đỏ/trắng…) + ÍT frame → NHANH, hợp Colab/Kaggle session ngắn")
    ap.add_argument("--suite-full", action="store_true",
                    help="chạy BỘ SUITE ĐẦY ĐỦ (20-30+ query/bài) — chậm, chỉ khi cần bảng test lớn")
    ap.add_argument("--suite-per-group", type=int, default=2,
                    help="số query mỗi nhóm ở suite LITE (mặc định 2 = vd đỏ/trắng; để 1 cho nhanh nhất)")
    ap.add_argument("--stride", type=int, default=None,
                    help="bỏ bớt frame: chỉ lấy mỗi frame thứ N (nhanh hơn). Mặc định suite=2, thường=1")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="số frame tối đa mỗi lần đếm. Mặc định suite=60 (nhanh), thường=300")
    ap.add_argument("--confidence", type=float, default=0.25,
                    help="ngưỡng tin cậy YOLO (thấp = bắt nhiều hơn, mặc định 0.25)")
    ap.add_argument("--yolo-backend", choices=["auto", "ultralytics", "super_gradients"], default="auto")
    ap.add_argument("--engine", choices=["auto", "sv", "builtin"], default="auto",
                    help="bộ đếm: sv = supervision (ByteTrack+LineZone, CHUẨN hơn), "
                         "builtin = tự viết (CentroidTracker). auto = sv nếu cài được")
    ap.add_argument("--yolo-weights", default=None,
                    help="model YOLO: yolov8m.pt (mặc định) / yolov8l.pt / yolov8x.pt (mạnh hơn, chậm hơn)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="cỡ ảnh suy luận YOLO (mặc định 960; 1280 bắt vật NHỎ/top-down tốt hơn)")
    ap.add_argument("--only", default=None, help="lọc video theo từ khoá (tên/file/key), vd 'milk'")
    ap.add_argument("--line", default=None,
                    help="ÉP vạch đếm: 'x1,y1,x2,y2' theo %% (0-100). VD dọc lệch trái: '35,0,35,100'")
    ap.add_argument("--zone", default=None,
                    help="ÉP vùng đếm: 'x1,y1;x2,y2;x3,y3;...' theo %% (≥3 đỉnh). Xem video rồi khoanh")
    ap.add_argument("--download-only", action="store_true",
                    help="CHỈ tải video + báo ✅/❌ (không nạp model) — kiểm tra nguồn nào chạy")
    ap.add_argument("--preview", nargs="?", const="geom_preview", default=None,
                    help="VẼ vạch/vùng lên FRAME ĐẦU của mỗi video (không cần model) → xem đặt "
                         "đúng chưa. Kèm --line/--zone để thử vị trí. VD: --preview /kaggle/working/prev")
    ap.add_argument("--save-dir", default=None,
                    help="LƯU VIDEO OUTPUT (vẽ vạch/vùng + box + số đếm) vào thư mục này")
    ap.add_argument("--list", action="store_true", help="chỉ liệt kê catalog, không tải/chạy")
    args = ap.parse_args()
    # Truyền lựa chọn YOLO qua env để detector (nạp lazy) đọc được.
    if args.yolo_weights:
        os.environ["YOLO_WEIGHTS"] = args.yolo_weights
    if args.imgsz:
        os.environ["YOLO_IMGSZ"] = str(args.imgsz)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
