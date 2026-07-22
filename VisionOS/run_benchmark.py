#!/usr/bin/env python3
"""Chạy benchmark đếm của LocateAnything-3B trên 3 bài toán và in scorecard.

Hai chế độ:

  * ``--selftest``  : KHÔNG cần GPU/model. Dùng detector giả (chuyển động dựng
                      sẵn) để kiểm tra toàn bộ đường ống + scorecard chạy đúng.
                      Rất hữu ích để test CI hoặc kiểm tra nhanh trước khi tốn
                      GPU. Kết quả kỳ vọng: mỗi bài đếm được đúng số lần cắt vạch.

  * mặc định (thật) : tải model nvidia/LocateAnything-3B, chạy trên video thật
                      của từng scenario, xuất video annotate + scorecard.

Ví dụ::

    python run_benchmark.py --selftest
    python run_benchmark.py --scenarios vehicles            # tự tải video mẫu
    python run_benchmark.py --scenarios people conveyor \\
        --people-video /kaggle/input/.../people.mp4 \\
        --conveyor-video /kaggle/input/.../belt.mp4 --max-frames 150
"""

from __future__ import annotations

import argparse
import os
import sys

# đảm bảo import được la_counting khi chạy trực tiếp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from la_counting import SCENARIOS, CountingPipeline  # noqa: E402
from la_counting.counting import iter_video_frames  # noqa: E402


# --------------------------------------------------------------------------- #
# Chuẩn bị model (chỉ dùng khi chạy thật)
# --------------------------------------------------------------------------- #
def prepare_model_dir(model_id: str = "nvidia/LocateAnything-3B") -> str:
    """Tải snapshot model + patch bfloat16->float16 cho GPU T4 (như notebook)."""
    import shutil

    from huggingface_hub import snapshot_download

    print(f"📥 Downloading {model_id} (lần đầu ~6GB)...")
    model_dir = snapshot_download(model_id)

    # Xoá module cache cũ để patch có hiệu lực
    mc = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    if os.path.exists(mc):
        shutil.rmtree(mc)

    # T4 (Turing) không có bfloat16 CUDA kernel -> ép float16
    f = os.path.join(model_dir, "modeling_locateanything.py")
    if os.path.exists(f):
        real_f = os.path.realpath(f)
        code = open(real_f).read()
        old = "pixel_values = pixel_values.to(self.language_model.dtype)"
        if old in code:
            code = code.replace(
                old, "pixel_values = pixel_values.to(torch.float16)  # T4 fix"
            )
            open(real_f, "w").write(code)
            print("✅ Patched bfloat16 -> float16")
    print(f"📁 {model_dir}")
    return model_dir


def build_real_detector():
    from la_counting.detector import LocateAnythingDetector

    model_dir = prepare_model_dir()
    detector = LocateAnythingDetector(model_dir=model_dir)
    detector.load()
    return detector


def resolve_video(scn, overrides: dict) -> str | None:
    """Tìm video cho scenario: ưu tiên override -> path_hint có sẵn -> tải URL."""
    import glob
    import urllib.request

    if scn.key in overrides and overrides[scn.key]:
        return overrides[scn.key]

    # thử path_hint (hỗ trợ wildcard cho /kaggle/input)
    if scn.video_path_hint:
        hits = glob.glob(scn.video_path_hint, recursive=True)
        if hits:
            return sorted(hits)[0]

    # tải URL công khai nếu có
    if scn.video_url:
        dst = scn.video_path_hint or f"/tmp/{scn.key}.mp4"
        if "*" in dst:  # path_hint là wildcard, không dùng làm đích tải
            dst = f"/tmp/{scn.key}.mp4"
        if not os.path.exists(dst):
            print(f"📥 Tải video mẫu cho '{scn.key}': {scn.video_url}")
            urllib.request.urlretrieve(scn.video_url, dst)
        return dst

    return None


# --------------------------------------------------------------------------- #
# Chế độ selftest (không GPU)
# --------------------------------------------------------------------------- #
def selftest_detector_and_frames(scn):
    """Tạo detector giả + frame cho scenario, mô phỏng 1-2 vật cắt vạch."""
    # import cục bộ để không phụ thuộc khi chạy thật
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))
    from fakes import ScriptedDetector, blank_frames, linear_track, merge_tracks

    n = 26
    if scn.line.orientation == "horizontal":
        # 2 vật đi xuống cắt vạch ngang
        t1 = linear_track((300, 120, 400, 190), (300, 500, 400, 570), n)
        t2 = linear_track((820, 120, 920, 190), (820, 500, 920, 570), n)
        script = merge_tracks(t1, t2)
    else:
        # 1 vật đi sang phải cắt vạch dọc
        script = [[b] for b in linear_track((260, 300, 340, 380), (1040, 300, 1120, 380), n)]
    return ScriptedDetector(script), blank_frames(n)


# --------------------------------------------------------------------------- #
# Scorecard
# --------------------------------------------------------------------------- #
def print_scorecard(results):
    print("\n" + "=" * 74)
    print("📊 SCORECARD — LocateAnything-3B counting benchmark")
    print("=" * 74)
    header = f"{'scenario':10} {'frames':>6} {'IN':>5} {'OUT':>5} {'total':>6} {'det/frame':>10} {'fps':>7}  status"
    print(header)
    print("-" * 74)
    for scn_key, r in results.items():
        scn = SCENARIOS[scn_key]
        ok = r.total_crossings >= scn.expect_min_crossings
        status = "✅ OK" if ok else "⚠️  KIỂM TRA"
        print(
            f"{scn_key:10} {r.frames:>6} {r.in_count:>5} {r.out_count:>5} "
            f"{r.total_crossings:>6} {r.avg_detections:>10.2f} {r.fps:>7.2f}  {status}"
        )
    print("=" * 74)
    print("IN/OUT = 2 chiều cắt vạch (đổi nghĩa tuỳ bố trí camera).")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                    choices=list(SCENARIOS), help="các bài toán cần chạy")
    ap.add_argument("--selftest", action="store_true",
                    help="chạy detector giả (không cần GPU) để kiểm tra đường ống")
    ap.add_argument("--max-frames", type=int, default=None, help="giới hạn số frame")
    ap.add_argument("--out-dir", default="/kaggle/working", help="thư mục xuất video")
    ap.add_argument("--people-video", default=None)
    ap.add_argument("--conveyor-video", default=None)
    ap.add_argument("--vehicles-video", default=None)
    ap.add_argument("--no-video-out", action="store_true", help="không ghi video annotate")
    args = ap.parse_args(argv)

    overrides = {
        "people": args.people_video,
        "conveyor": args.conveyor_video,
        "vehicles": args.vehicles_video,
    }

    detector = None
    results = {}

    for key in args.scenarios:
        scn = SCENARIOS[key]
        print(f"\n▶️  {scn.title}  (prompt={scn.prompt!r})")

        if args.selftest:
            det, frames = selftest_detector_and_frames(scn)
            pipe = CountingPipeline(det, scn)
            results[key] = pipe.run(frames, max_frames=args.max_frames)
            continue

        # ---- chế độ thật ----
        if detector is None:
            detector = build_real_detector()

        video = resolve_video(scn, overrides)
        if not video or not os.path.exists(video):
            print(f"   ⏭️  Bỏ qua '{key}': không tìm thấy video "
                  f"(cấp --{key}-video hoặc upload dataset).")
            continue

        writer = None
        out_path = os.path.join(args.out_dir, f"{key}_counting.mp4")

        def on_frame(frame, sv_d, pipe, _key=key, _out=out_path):
            nonlocal writer
            if args.no_video_out:
                return
            import cv2
            import supervision as sv
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    _out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
                )
            a = sv.BoxAnnotator(thickness=2).annotate(frame.copy(), sv_d)
            a = pipe.line_zone and sv.LineZoneAnnotator(thickness=3).annotate(a, pipe.line_zone) or a
            writer.write(a)

        pipe = CountingPipeline(detector, scn)
        results[key] = pipe.run(
            iter_video_frames(video), max_frames=args.max_frames, on_frame=on_frame
        )
        if writer is not None:
            writer.release()
            print(f"   💾 Video: {out_path}")

    if results:
        print_scorecard(results)
    else:
        print("Không có scenario nào chạy được.")
    return results


if __name__ == "__main__":
    main()
