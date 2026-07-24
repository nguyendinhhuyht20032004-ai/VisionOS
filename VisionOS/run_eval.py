#!/usr/bin/env python3
"""Đánh giá ĐỘ CHÍNH XÁC mô hình trên COCO val2017 — P/R/F1 + sai số đếm.

Đây là "điểm số thật": chạy detector trên ảnh COCO có nhãn, so khớp box dự đoán
với ground-truth theo IoU → Precision / Recall / F1 và MAE đếm → biết mô hình
**tốt hay không**, và **có cần fine-tune không**.

Hai chế độ:
  * ``--selftest`` : KHÔNG cần GPU/mạng. Dùng detector giả + dữ liệu tổng hợp có
    đáp án biết trước để xác minh bộ chấm điểm chạy đúng.
  * chạy thật      : tải COCO + chạy detector thật (cần GPU + Internet).

Ví dụ:
    python run_eval.py --selftest
    python run_eval.py --model locate --classes person car bottle --n 50
    python run_eval.py --model yolo   --classes person car        --n 100
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recognition.base import BoundingBox  # noqa: E402
from recognition.evaluation import DetectionMetrics, nms_dedup, verdict  # noqa: E402


# --------------------------------------------------------------------------- #
# Lõi dùng chung: cộng dồn chỉ số từ (frame, gt) qua một detector
# --------------------------------------------------------------------------- #
def accumulate(detector, frame_gt_pairs, prompt, iou_thr, dedup_iou, label):
    m = DetectionMetrics(iou_thr=iou_thr, label=label)
    for frame, gt in frame_gt_pairs:
        if frame is None:
            continue
        res = detector.detect(frame, prompt)
        boxes = [d.bbox for d in res.detections]
        scores = [d.confidence for d in res.detections]
        if dedup_iou is not None and boxes:
            keep = nms_dedup(boxes, scores, dedup_iou)
            boxes = [boxes[i] for i in keep]
            scores = [scores[i] for i in keep]
        m.add_image(boxes, [BoundingBox(*g) for g in gt], scores)
    return m


def print_scorecard(rows, iou_thr):
    if not rows:
        print("(không có kết quả)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("\n" + "=" * 96)
    print(f"📊 ĐỘ CHÍNH XÁC TRÊN COCO val2017 (IoU={iou_thr}) — P/R/F1 + MAE đếm")
    print("=" * 96)
    header = "  ".join(str(c).ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    print("=" * 96)
    f1s = [r["F1"] for r in rows]
    if f1s:
        avg = sum(f1s) / len(f1s)
        print(f"F1 trung bình (macro): {avg:.3f}  →  {verdict(avg)}")
    print("P = báo đúng không · R = có bỏ sót không · MAE = sai số đếm/ảnh · "
          "F1<0.45 → cân nhắc fine-tune.")


# --------------------------------------------------------------------------- #
# Self-test (không GPU, không mạng) — kiểm chứng bộ chấm điểm
# --------------------------------------------------------------------------- #
def run_selftest() -> int:
    import numpy as np

    from recognition.base import Detection, DetectorResult

    print("=" * 70)
    print("SELF-TEST run_eval — kiểm chứng bộ chấm điểm (không cần GPU)")
    print("=" * 70)

    A, B, C = (10, 10, 50, 90), (60, 20, 100, 95), (0, 0, 40, 80)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    # ảnh1: GT=[A,B]; pred=[A'(khớp A), A'(trùng→dedup bỏ), thừa] -> TP=1,FP=1,FN=1(B)
    # ảnh2: GT=[C];   pred=[C'(khớp C)]                          -> TP=1
    frames_gt = [(frame, [A, B]), (frame, [C])]
    script = [
        [(11, 11, 51, 91), (11, 11, 51, 91), (300, 300, 340, 360)],
        [(1, 1, 41, 79)],
    ]

    class Fake:
        def __init__(self, s):
            self.s, self.i = s, 0

        def detect(self, frame, prompt):
            boxes = self.s[self.i]
            self.i += 1
            dets = [Detection(BoundingBox(*b), prompt, 0.85) for b in boxes]
            return DetectorResult(dets, raw="fake", model_name="Fake")

    m = accumulate(Fake(script), frames_gt, prompt="person", iou_thr=0.5,
                   dedup_iou=0.9, label="person")
    print(f"TP={m.tp} FP={m.fp} FN={m.fn} | P={m.precision:.3f} R={m.recall:.3f} "
          f"F1={m.f1:.3f} | MAE đếm={m.count_mae:.2f}")
    ok = (m.tp, m.fp, m.fn) == (2, 1, 1)
    print("dedup khử box trùng:", "✅" if m.fp == 1 else "❌ (đáng lẽ 1 FP)")
    print("Kết luận:", "✅ BỘ CHẤM ĐIỂM ĐÚNG" if ok else "❌ SAI")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Chạy thật trên COCO
# --------------------------------------------------------------------------- #
def run_real(args) -> int:
    import cv2

    from recognition.coco import EVAL_CLASSES, load_eval_samples
    from recognition.detectors import load_locate_anything, load_standard_detector

    if args.model == "locate":
        print("🧠 Detector: LocateAnything-3B (open-vocab, zero-shot)")
        detector = load_locate_anything()
    else:
        print("🎯 Detector: YOLOv8/YOLO-NAS (STANDARD, đã học sẵn COCO)")
        detector = load_standard_detector(backend=args.yolo_backend, confidence=args.confidence)

    rows = []
    for ec in args.classes:
        if ec not in EVAL_CLASSES:
            print(f"⏭  bỏ qua lớp không hợp lệ: {ec}")
            continue
        prompt = EVAL_CLASSES[ec]["prompt"]
        print(f"\n🔎 Đánh giá '{ec}' (prompt={prompt!r}) — tải/đọc ảnh COCO...")
        samples = load_eval_samples(
            ec, limit=args.n, ann_file=args.ann_file, images_dir=args.images_dir
        )

        def pairs():
            for path, gt in samples:
                yield cv2.imread(path), gt

        m = accumulate(detector, pairs(), prompt, args.iou, args.dedup_iou, ec)
        print(f"   → P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} "
              f"(ảnh {m.n_images}, GT {m.n_gt})")
        rows.append(m.as_row())

    print_scorecard(rows, args.iou)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Đánh giá độ chính xác mô hình trên COCO")
    ap.add_argument("--selftest", action="store_true", help="kiểm chứng bộ chấm điểm (không GPU)")
    ap.add_argument("--model", choices=["locate", "yolo"], default="locate",
                    help="locate = LocateAnything-3B (zero-shot); yolo = YOLO (đã học COCO)")
    ap.add_argument("--classes", nargs="+", default=["person", "car", "bottle"],
                    help="lớp COCO cần đánh giá (person/car/bottle/vehicle)")
    ap.add_argument("--n", type=int, default=50, help="số ảnh mỗi lớp")
    ap.add_argument("--iou", type=float, default=0.5, help="ngưỡng IoU khi ghép")
    ap.add_argument("--dedup-iou", type=float, default=0.9,
                    help="ngưỡng NMS khử box trùng (đặt 1.0 để tắt)")
    ap.add_argument("--confidence", type=float, default=0.35, help="ngưỡng conf cho YOLO")
    ap.add_argument("--yolo-backend", choices=["auto", "ultralytics", "super_gradients"],
                    default="auto")
    ap.add_argument("--ann-file", default=None,
                    help="đường dẫn instances_val2017.json có sẵn (Kaggle 'Add Data')")
    ap.add_argument("--images-dir", default=None, help="thư mục ảnh val2017 có sẵn")
    args = ap.parse_args()

    if args.dedup_iou >= 1.0:
        args.dedup_iou = None
    if args.selftest:
        return run_selftest()
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
