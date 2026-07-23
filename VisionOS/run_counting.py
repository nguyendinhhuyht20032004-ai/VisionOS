#!/usr/bin/env python3
"""CLI chạy 4 bài toán đếm và in scorecard — chứng minh hiệu quả đếm.

Hai chế độ:

  * ``--selftest`` : chạy bằng ``ScriptedDetector`` (kịch bản tất định, CPU) — kiểm
    tra toàn bộ đường ống detect → track → đếm ở BẤT KỲ máy nào, không cần GPU.
  * chạy thật     : truyền video + chọn detector thật (YOLO-NAS / LocateAnything)
    khi có GPU. Bài nào thiếu video sẽ bị bỏ qua.

Ví dụ:
    python run_counting.py --selftest
    python run_counting.py --people-video people.mp4 --vehicles-video cars.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recognition.counting import CountingPipeline
from recognition.detectors.fake import ScriptedDetector, box_moving, merge_tracks
from recognition.scenarios import COUNT_SCENARIOS, PACKAGES, PEOPLE_IN_OUT, QUEUE, VEHICLES


# --------------------------------------------------------------------------- #
# Kịch bản giả lập cho --selftest (biết trước đáp số)
# --------------------------------------------------------------------------- #
def scripted_frames(scenario) -> list:
    n = 30
    if scenario.key == "people":
        down = box_moving(300, 100, 300, 620, 40, 120, n)
        up = box_moving(900, 620, 900, 100, 40, 120, n)
        return merge_tracks(down, up)               # kỳ vọng 2 lần cắt (1 vào, 1 ra)
    if scenario.key == "vehicles":
        lanes = [box_moving(x, 100, x, 650, 80, 60, n) for x in (200, 640, 1080)]
        return merge_tracks(*lanes)                 # kỳ vọng 3 lần cắt
    if scenario.key == "packages":
        return [[b] for b in box_moving(100, 360, 1180, 360, 60, 60, n)]  # 1 lần cắt
    if scenario.key == "queue":
        p1 = box_moving(520, 560, 520, 560, 40, 120, n)  # đứng yên trong vùng
        p2 = box_moving(720, 540, 720, 540, 40, 120, n)
        return merge_tracks(p1, p2)                 # kỳ vọng đỉnh vùng = 2
    return []


def print_scorecard(rows: list) -> None:
    if not rows:
        print("(không có kết quả)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(str(c).ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def run_selftest() -> int:
    print("=" * 70)
    print("SELF-TEST — 4 bài toán đếm (ScriptedDetector, không cần GPU)")
    print("=" * 70)
    rows = []
    all_ok = True
    for key, scenario in COUNT_SCENARIOS.items():
        frames = scripted_frames(scenario)
        det = ScriptedDetector(frames)
        pipe = CountingPipeline(det, scenario)
        pipe.run(range(len(frames)), max_frames=len(frames))
        row = pipe.result.as_row()
        ok = pipe.sanity_ok()
        row["status"] = "✅ OK" if ok else "❌ FAIL"
        all_ok = all_ok and ok
        rows.append(row)
    # In từng nhóm cột (line/zone khác cột) cho gọn.
    line_rows = [r for r, s in zip(rows, COUNT_SCENARIOS.values()) if s.counting_type == "line"]
    zone_rows = [r for r, s in zip(rows, COUNT_SCENARIOS.values()) if s.counting_type == "zone"]
    if line_rows:
        print("\n▶ Đếm cắt vạch (line):")
        print_scorecard(line_rows)
    if zone_rows:
        print("\n▶ Đếm chiếm vùng (zone):")
        print_scorecard(zone_rows)
    print("\nKết luận:", "TẤT CẢ ĐẠT ✅" if all_ok else "CÓ BÀI SAI ❌")
    return 0 if all_ok else 1


def run_real(args) -> int:
    from recognition.counting import CountingPipeline
    from recognition.detectors import load_locate_anything, load_standard_detector

    # Import lazy đọc video (cần opencv).
    def frames_of(path):
        import cv2

        cap = cv2.VideoCapture(path)
        try:
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                yield fr
        finally:
            cap.release()

    videos = {
        "people": args.people_video,
        "vehicles": args.vehicles_video,
        "packages": args.packages_video,
        "queue": args.queue_video,
    }
    yolo = None
    locate = None
    rows = []
    for key, scenario in COUNT_SCENARIOS.items():
        path = videos.get(key)
        if not path or not os.path.exists(path):
            print(f"⏭  bỏ qua [{key}] — không có video")
            continue
        if scenario.model.startswith("YOLO"):
            yolo = yolo or load_standard_detector(
                backend=args.yolo_backend, confidence=args.confidence
            )
            detector = yolo
        else:
            locate = locate or load_locate_anything()
            detector = locate
        pipe = CountingPipeline(detector, scenario)
        pipe.run(frames_of(path), max_frames=args.max_frames)
        rows.append(pipe.result.as_row())
    print_scorecard(rows)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Chạy các bài toán đếm của VisionOS")
    ap.add_argument("--selftest", action="store_true", help="chạy kịch bản giả lập (không GPU)")
    ap.add_argument("--people-video", dest="people_video")
    ap.add_argument("--vehicles-video", dest="vehicles_video")
    ap.add_argument("--packages-video", dest="packages_video")
    ap.add_argument("--queue-video", dest="queue_video")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--confidence", type=float, default=0.65)
    ap.add_argument(
        "--yolo-backend",
        choices=["auto", "ultralytics", "super_gradients"],
        default="auto",
        help="động cơ YOLO cho bài người/xe (auto = tự chọn; mặc định ultralytics nếu thiếu super-gradients)",
    )
    args = ap.parse_args()

    if args.selftest or not any(
        [args.people_video, args.vehicles_video, args.packages_video, args.queue_video]
    ):
        return run_selftest()
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
