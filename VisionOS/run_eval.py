#!/usr/bin/env python3
"""Đánh giá ĐỘ CHÍNH XÁC mô hình trên COCO val2017 + LƯU ẢNH & PHÂN TÍCH ĐẦU VÀO.

Trả lời "mô hình tốt/tệ ở đâu, vì sao": ngoài P/R/F1 + MAE đếm, tool còn
  * **lưu ảnh minh hoạ** (TP xanh / FP đỏ / GT bỏ sót vàng) để bạn tự soi, và
  * **phân tích dữ liệu đầu vào** (số ảnh, số GT/ảnh, kích thước vật, số box model
    dự đoán…) để hiểu vì sao điểm thấp.

Chế độ:
  * ``--selftest``     : kiểm chứng bộ chấm điểm (không GPU/mạng).
  * ``--analyze-only`` : CHỈ phân tích dữ liệu COCO (không chạy model, nhanh).
  * chạy thật          : tải COCO + chạy model + chấm điểm (+ ``--save-dir`` để lưu ảnh).

Ví dụ:
    python run_eval.py --analyze-only --classes person car bottle --n 50
    python run_eval.py --model locate --classes person --n 30 --save-dir /kaggle/working/viz
    python run_eval.py --model yolo   --classes person car --n 50 --save-dir /kaggle/working/viz
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recognition.base import BoundingBox  # noqa: E402
from recognition.evaluation import (  # noqa: E402
    DetectionMetrics,
    match_assign,
    nms_dedup,
    verdict,
)


# --------------------------------------------------------------------------- #
# Vẽ ảnh minh hoạ: GT vs prediction, tô màu theo TP/FP/miss
# --------------------------------------------------------------------------- #
def draw_eval(frame, pred_boxes, scores, gt_boxes, pred_status, gt_missed, prompt, iou_thr):
    """Vẽ GT + prediction lên ảnh. Xanh=TP, Đỏ=FP, Vàng=GT bị bỏ sót (FN)."""
    import cv2

    img = frame.copy()
    GREEN, RED, YELLOW, GT_C, WHITE = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 180, 0), (255, 255, 255)

    # GT: khung tham chiếu màu lam nhạt; nếu bị bỏ sót thì tô vàng đậm + "MISS"
    for g, missed in zip(gt_boxes, gt_missed):
        x1, y1, x2, y2 = (int(v) for v in g.as_xyxy())
        if missed:
            cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, 3)
            cv2.putText(img, "MISS-GT", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), GT_C, 1)

    # Prediction: TP xanh, FP đỏ
    for b, sc, st in zip(pred_boxes, scores, pred_status):
        x1, y1, x2, y2 = (int(v) for v in b.as_xyxy())
        color = GREEN if st == "TP" else RED
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{st} {sc:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    tp = pred_status.count("TP")
    fp = pred_status.count("FP")
    fn = sum(gt_missed)
    banner = f"'{prompt}' IoU>={iou_thr} | TP={tp} FP={fp} FN={fn} | GT lam, TP xanh, FP do, MISS vang"
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, banner, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    return img


# --------------------------------------------------------------------------- #
# Thống kê dữ liệu đầu vào theo lớp
# --------------------------------------------------------------------------- #
class ClassStats:
    def __init__(self, label):
        self.label = label
        self.img_sizes = []      # (w, h)
        self.gt_per_img = []     # số GT mỗi ảnh
        self.gt_area_frac = []   # diện tích box GT / diện tích ảnh (%)
        self.pred_per_img = []   # số box model dự đoán mỗi ảnh
        self.confs = []          # confidence các prediction

    def add(self, w, h, gt_boxes, n_pred, scores):
        self.img_sizes.append((w, h))
        self.gt_per_img.append(len(gt_boxes))
        area = float(w * h) or 1.0
        for g in gt_boxes:
            bw, bh = (g[2] - g[0]), (g[3] - g[1])
            self.gt_area_frac.append(100.0 * (bw * bh) / area)
        self.pred_per_img.append(n_pred)
        self.confs.extend(scores)

    def print_report(self):
        def avg(xs):
            return sum(xs) / len(xs) if xs else 0.0

        n = len(self.img_sizes)
        print(f"\n📥 DỮ LIỆU ĐẦU VÀO — lớp '{self.label}'")
        if n == 0:
            print("   ⚠️ 0 ảnh — COCO chưa tải được (xem --analyze-only để kiểm tra).")
            return
        ws = [w for w, _ in self.img_sizes]
        hs = [h for _, h in self.img_sizes]
        gt_total = sum(self.gt_per_img)
        print(f"   Ảnh: {n} | GT tổng: {gt_total} | GT/ảnh: {avg(self.gt_per_img):.1f} "
              f"(min {min(self.gt_per_img)}, max {max(self.gt_per_img)})")
        print(f"   Kích thước ảnh: {min(ws)}x{min(hs)} ~ {max(ws)}x{max(hs)} "
              f"(TB {avg(ws):.0f}x{avg(hs):.0f})")
        small = sum(1 for f in self.gt_area_frac if f < 1.0)
        print(f"   Box GT: TB {avg(self.gt_area_frac):.2f}% diện tích ảnh | "
              f"vật nhỏ (<1%): {small}/{len(self.gt_area_frac)} "
              f"({100*small/max(1,len(self.gt_area_frac)):.0f}%)  ← vật càng nhỏ càng khó")
        # chỉ in dòng model khi thực sự đã chạy model (analyze-only thì bỏ)
        if any(self.pred_per_img):
            line = f"   Model dự đoán: {avg(self.pred_per_img):.1f} box/ảnh"
            if self.confs:
                line += f" | conf TB {avg(self.confs):.2f}"
            print(line)


# --------------------------------------------------------------------------- #
# Lõi: chấm điểm + (tuỳ chọn) lưu ảnh + thống kê
# --------------------------------------------------------------------------- #
def accumulate(detector, frame_gt_pairs, prompt, iou_thr, dedup_iou, label,
               save_dir=None, stats=None):
    import cv2

    m = DetectionMetrics(iou_thr=iou_thr, label=label)
    saved = 0
    for idx, (frame, gt, fname) in enumerate(frame_gt_pairs):
        if frame is None:
            continue
        h, w = frame.shape[:2]
        res = detector.detect(frame, prompt)
        boxes = [d.bbox for d in res.detections]
        scores = [d.confidence for d in res.detections]
        if dedup_iou is not None and boxes:
            keep = nms_dedup(boxes, scores, dedup_iou)
            boxes = [boxes[i] for i in keep]
            scores = [scores[i] for i in keep]
        gt_bb = [BoundingBox(*g) for g in gt]
        m.add_image(boxes, gt_bb, scores)
        if stats is not None:
            stats.add(w, h, gt, len(boxes), scores)
        if save_dir:
            status, gt_missed = match_assign(boxes, gt_bb, iou_thr, scores)
            viz = draw_eval(frame, boxes, scores, gt_bb, status, gt_missed, prompt, iou_thr)
            out = os.path.join(save_dir, label)
            os.makedirs(out, exist_ok=True)
            cv2.imwrite(os.path.join(out, fname or f"{idx:04d}.jpg"), viz)
            saved += 1
    if save_dir and saved:
        print(f"   🖼️  đã lưu {saved} ảnh minh hoạ vào {os.path.join(save_dir, label)}/")
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
    print("  ".join(str(c).ljust(widths[c]) for c in cols))
    print("-" * sum(widths[c] + 2 for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    print("=" * 96)
    f1s = [r["F1"] for r in rows]
    if f1s:
        print(f"F1 trung bình (macro): {sum(f1s)/len(f1s):.3f}  →  {verdict(sum(f1s)/len(f1s))}")
    print("P=báo đúng không · R=có bỏ sót không · MAE=sai số đếm/ảnh · F1<0.45 → cân nhắc fine-tune.")


# --------------------------------------------------------------------------- #
# Nạp mẫu COCO (dùng cho analyze-only và chạy thật)
# --------------------------------------------------------------------------- #
def load_pairs(ec, args):
    """Trả list ``(frame_bgr, gt_xyxy, filename)`` đã đọc ảnh (bỏ ảnh lỗi)."""
    import cv2

    from recognition.coco import load_eval_samples

    samples = load_eval_samples(ec, limit=args.n, ann_file=args.ann_file, images_dir=args.images_dir)
    pairs = []
    for path, gt in samples:
        frame = cv2.imread(path)
        if frame is not None:
            pairs.append((frame, gt, os.path.basename(path)))
    return pairs


# --------------------------------------------------------------------------- #
# Các chế độ
# --------------------------------------------------------------------------- #
def run_selftest() -> int:
    import numpy as np

    from recognition.base import Detection, DetectorResult

    print("=" * 70)
    print("SELF-TEST run_eval — kiểm chứng bộ chấm điểm + vẽ ảnh (không GPU)")
    print("=" * 70)
    A, B, C = (10, 10, 50, 90), (60, 20, 100, 95), (0, 0, 40, 80)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frames_gt = [(frame, [A, B], "img1.jpg"), (frame, [C], "img2.jpg")]
    script = [[(11, 11, 51, 91), (11, 11, 51, 91), (300, 300, 340, 360)], [(1, 1, 41, 79)]]

    class Fake:
        def __init__(self, s):
            self.s, self.i = s, 0

        def detect(self, frame, prompt):
            boxes = self.s[self.i]
            self.i += 1
            return DetectorResult([Detection(BoundingBox(*b), prompt, 0.85) for b in boxes])

    import tempfile
    viz = os.path.join(tempfile.gettempdir(), "run_eval_selftest_viz")
    m = accumulate(Fake(script), frames_gt, "person", 0.5, 0.9, "person", save_dir=viz)
    ok = (m.tp, m.fp, m.fn) == (2, 1, 1)
    saved_ok = os.path.exists(os.path.join(viz, "person", "img1.jpg"))
    print(f"TP={m.tp} FP={m.fp} FN={m.fn} | P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}")
    print("vẽ+lưu ảnh minh hoạ:", "✅" if saved_ok else "❌")
    print("Kết luận:", "✅ ĐÚNG" if (ok and saved_ok) else "❌ SAI")
    return 0 if (ok and saved_ok) else 1


def run_analyze_only(args) -> int:
    print("🔍 PHÂN TÍCH DỮ LIỆU ĐẦU VÀO (không chạy model)")
    for ec in args.classes:
        pairs = load_pairs(ec, args)
        stats = ClassStats(ec)
        for frame, gt, _ in pairs:
            h, w = frame.shape[:2]
            stats.add(w, h, gt, 0, [])
        stats.print_report()
    return 0


def run_real(args) -> int:
    from recognition.coco import EVAL_CLASSES
    from recognition.detectors import load_locate_anything, load_standard_detector

    if args.model == "locate":
        print("🧠 Detector: LocateAnything-3B (open-vocab, zero-shot)")
        detector = load_locate_anything()
    else:
        print("🎯 Detector: YOLOv8/YOLO-NAS (đã học sẵn COCO)")
        detector = load_standard_detector(backend=args.yolo_backend, confidence=args.confidence)

    rows = []
    for ec in args.classes:
        if ec not in EVAL_CLASSES:
            print(f"⏭  bỏ qua lớp không hợp lệ: {ec}")
            continue
        prompt = EVAL_CLASSES[ec]["prompt"]
        print(f"\n🔎 Đánh giá '{ec}' (prompt={prompt!r})...")
        pairs = load_pairs(ec, args)
        stats = ClassStats(ec)
        m = accumulate(detector, pairs, prompt, args.iou, args.dedup_iou, ec,
                       save_dir=args.save_dir, stats=stats)
        stats.print_report()
        print(f"   → P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} "
              f"(ảnh {m.n_images}, GT {m.n_gt}, dự đoán {m.n_pred})")
        rows.append(m.as_row())

    print_scorecard(rows, args.iou)
    return 0


def _autopin_transformers(target: str = "4.57.1") -> None:
    """Bảo đảm ``transformers==target`` (bản NVIDIA test LocateAnything-3B).

    Kaggle mặc định **transformers 5.0.0** — bản *major* khác hẳn 4.57.1: cache,
    rotary, attention, position_ids đều viết lại → code bundled của model KHÔNG
    chạy được (kết cục là CUDA "device-side assert"). Vá từng dòng cho 5.x là vô
    vọng, nên bắt buộc về 4.57.1.

    Pin thủ công hay trượt (quên restart kernel / Kaggle giữ bản mới). Hàm này cài
    đúng bản rồi **RE-EXEC chính tiến trình** ``run_eval`` — vì đây là subprocess
    riêng của lệnh ``!python run_eval.py`` nên KHÔNG cần restart kernel notebook.
    Cờ env ``_LA_PIN_TRIED`` chặn lặp vô hạn nếu mạng/Kaggle chặn cài.
    """
    def _ver():
        try:
            import importlib.metadata as md
            return md.version("transformers")
        except Exception:
            return None

    if os.environ.get("_LA_PIN_TRIED") == "1":
        if _ver() != target:
            print("=" * 74)
            print(f"⚠️  ĐÃ thử cài transformers=={target} nhưng hiện vẫn là {_ver()!r}.")
            print("   Mạng bị chặn hoặc Kaggle bật 'Always use latest environment'.")
            print(f'   Hãy cài TAY 1 cell:  !pip install "transformers=={target}" accelerate')
            print("   rồi RESTART KERNEL và chạy lại.")
            print("=" * 74)
        return  # đã thử 1 lần rồi — không re-exec nữa (tránh vòng lặp)

    if _ver() == target:
        return  # đã đúng bản, không cần làm gì

    print("=" * 74)
    print(f"⚙️  transformers=={_ver()} KHÔNG khớp → cài {target} (bản NVIDIA test)…")
    print("   run_eval sẽ TỰ khởi động lại (không cần restart kernel).")
    print("=" * 74, flush=True)
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"transformers=={target}", "accelerate"],
        check=False,
    )
    os.environ["_LA_PIN_TRIED"] = "1"
    print(f"🔄 Khởi động lại với transformers=={_ver()} …", flush=True)
    prog = os.path.abspath(sys.argv[0])
    os.execv(sys.executable, [sys.executable, prog] + sys.argv[1:])  # thay tiến trình


def main() -> int:
    ap = argparse.ArgumentParser(description="Đánh giá độ chính xác mô hình trên COCO")
    ap.add_argument("--selftest", action="store_true", help="kiểm chứng bộ chấm điểm (không GPU)")
    ap.add_argument("--analyze-only", action="store_true", help="chỉ phân tích dữ liệu COCO (không model)")
    ap.add_argument("--model", choices=["locate", "yolo"], default="locate")
    ap.add_argument("--classes", nargs="+", default=["person", "car", "bottle"])
    ap.add_argument("--n", type=int, default=50, help="số ảnh mỗi lớp")
    ap.add_argument("--iou", type=float, default=0.5, help="ngưỡng IoU khi ghép")
    ap.add_argument("--dedup-iou", type=float, default=0.9, help="NMS khử box trùng (1.0=tắt)")
    ap.add_argument("--confidence", type=float, default=0.35, help="ngưỡng conf cho YOLO")
    ap.add_argument("--yolo-backend", choices=["auto", "ultralytics", "super_gradients"], default="auto")
    ap.add_argument("--save-dir", default=None, help="thư mục LƯU ẢNH minh hoạ TP/FP/miss")
    ap.add_argument("--ann-file", default=None, help="instances_val2017.json có sẵn")
    ap.add_argument("--images-dir", default=None, help="thư mục ảnh val2017 có sẵn")
    args = ap.parse_args()

    if args.dedup_iou >= 1.0:
        args.dedup_iou = None
    if args.selftest:
        return run_selftest()
    if args.analyze_only:
        return run_analyze_only(args)
    if args.model == "locate":
        _autopin_transformers()  # có thể cài 4.57.1 + re-exec tiến trình
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
