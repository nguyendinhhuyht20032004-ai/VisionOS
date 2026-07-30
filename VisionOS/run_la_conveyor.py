#!/usr/bin/env python3
"""Test RIÊNG cho **LocateAnything-3B** — bài ĐẾM SẢN PHẨM trên băng chuyền, chạy NHANH NHẤT.

Vì sao có file riêng: LocateAnything-3B open-vocab đếm được "bất kỳ vật gì mô tả
bằng lời" — đúng thế mạnh cho SẢN PHẨM trên chuyền (thứ YOLO COCO không có lớp).
Nhưng model RẤT CHẬM (~vài giây/khung). File này gom mọi "núm" tăng tốc vào một
chỗ để mỗi lượt test ngắn nhất có thể:

  * ẢNH NHỎ    (--proc-width 640): ít vision-token → prefill nhanh hơn nhiều.
  * ÍT TOKEN   (--max-new-tokens 256): LA sinh toạ độ box dưới dạng text; 256 token
                đủ cho vài chục box mà giải mã nhanh hơn HẲN mặc định 1024.
  * BỎ KHUNG   (--stride 3 + --max-frames 30): lấy mẫu thưa, đủ để vật cắt vạch.
  * KHỬ BOX TRÙNG (NMS): LA hay bung box lặp → gộp lại để tracker ổn định + nhanh.
  * fp16 + SDPA math-backend trên T4 (đã cài sẵn trong detector) để không crash.

NGUỒN VIDEO (quan trọng):
  Chỉ **milk-bottling-plant** của supervision là CHẮC CHẮN có sản phẩm (chai) chạy
  trên chuyền → dùng làm video MẶC ĐỊNH (tải tự động, không cần upload).
  Muốn thêm video khác: `--video <path|URL>`. Trước khi tốn GPU, hãy VET nhanh bằng
  `--preview` (chỉ tải + lưu vài frame JPG, KHÔNG nạp model) để mắt thường xác nhận
  đúng là băng chuyền có sản phẩm.

VÍ DỤ (Colab/Kaggle — chạy trong 1 cell):

    # 0) Vet nguồn video trong vài giây (không cần GPU):
    !python run_la_conveyor.py --preview --video https://.../my_belt.mp4

    # 1) Test nhanh trên video chai mặc định (2 query đại diện):
    !python run_la_conveyor.py

    # 2) Chạy full bộ query sản phẩm trên video của bạn:
    !python run_la_conveyor.py --video /kaggle/input/ds/belt.mp4 --suite

    # 3) Kiểm tra đường ống KHÔNG cần GPU:
    !python run_la_conveyor.py --selftest
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Sequence, Tuple

# cho phép `python run_la_conveyor.py` chạy từ mọi cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from la_counting.counting import CountingPipeline, iter_video_frames  # noqa: E402
from la_counting.parsing import Detection  # noqa: E402
from la_counting.scenarios import LineConfig, Scenario  # noqa: E402

# Video CHẮC CHẮN có sản phẩm (chai) chạy trên chuyền — supervision, tải tự động.
MILK_ASSET = "MILK_BOTTLING_PLANT"
MILK_FILE = "milk-bottling-plant.mp4"


# --------------------------------------------------------------------------- #
# Bộ QUERY test đếm sản phẩm (open-vocab) — phân nhóm dễ → khó, có tiếng Việt.
# --------------------------------------------------------------------------- #
PRODUCT_QUERIES = {
    "tổng quát": ["object", "product on the conveyor belt"],
    "chai/bottle": ["bottle", "plastic bottle", "a white bottle", "a bottle cap"],
    "trạng thái": ["a full bottle", "a fallen bottle", "the bottle closest to the camera"],
    "tiếng Việt": ["chai", "chai nhựa", "sản phẩm trên băng chuyền"],
}


def suite(lite: bool = True, per_group: int = 1) -> List[Tuple[str, str]]:
    """Trả list (nhóm, query). ``lite`` chỉ lấy ``per_group`` query đầu mỗi nhóm
    (đại diện, chạy nhanh); full = toàn bộ."""
    out: List[Tuple[str, str]] = []
    for group, qs in PRODUCT_QUERIES.items():
        picked = qs[: max(1, per_group)] if lite else qs
        for q in picked:
            out.append((group, q))
    return out


def strided(frames, stride: int):
    """Chỉ giữ mỗi ``stride`` frame (lấy mẫu thưa cho nhanh). stride<=1 = giữ hết."""
    for i, f in enumerate(frames):
        if stride <= 1 or i % stride == 0:
            yield f


# --------------------------------------------------------------------------- #
# NMS class-agnostic tự chứa (làm việc trên bbox tuple) — khử box LA lặp.
# --------------------------------------------------------------------------- #
def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def nms(dets: List[Detection], iou_thr: float = 0.5, max_boxes: int = 60) -> List[Detection]:
    """Gộp box chồng nhau (IoU≥``iou_thr``, giữ box lớn hơn) + chặn trần số box.

    LA không phát confidence riêng nên xếp theo DIỆN TÍCH (tất định).
    """
    if len(dets) <= 1:
        return dets
    order = sorted(range(len(dets)), key=lambda i: _area(dets[i].bbox), reverse=True)
    keep: List[int] = []
    for i in order:
        if all(_iou(dets[i].bbox, dets[j].bbox) < iou_thr for j in keep):
            keep.append(i)
            if len(keep) >= max_boxes:
                break
    return [dets[i] for i in keep]


# --------------------------------------------------------------------------- #
# Detector NHANH: LA-3B + max_new_tokens nhỏ + NMS.
# --------------------------------------------------------------------------- #
class FastLA:
    """Bọc ``la_counting`` LocateAnythingDetector: ít token + khử box trùng."""

    def __init__(self, model_dir: str, max_new_tokens: int = 256,
                 iou: float = 0.5, max_boxes: int = 60):
        from la_counting.detector import LocateAnythingDetector as _LA

        self._impl = _LA(model_dir=model_dir, max_new_tokens=max_new_tokens)
        self.iou = iou
        self.max_boxes = max_boxes

    def load(self):
        self._impl.load()
        return self

    def detect_frame(self, frame_bgr, prompt: str):
        dets, raw = self._impl.detect_frame(frame_bgr, prompt)
        return nms(dets, self.iou, self.max_boxes), raw


def build_fast_detector(model_id: str, max_new_tokens: int, iou: float, max_boxes: int):
    """Tải + VÁ snapshot model (dùng lại logic đã test ở recognition) rồi dựng FastLA."""
    from recognition.detectors.locate_anything import _ensure_locate_deps, _prepare_model_dir

    _ensure_locate_deps()
    local_dir = _prepare_model_dir(model_id)
    return FastLA(local_dir, max_new_tokens=max_new_tokens, iou=iou, max_boxes=max_boxes).load()


# --------------------------------------------------------------------------- #
# Scenario băng chuyền (vạch DỌC — sản phẩm chạy ngang cắt qua).
# --------------------------------------------------------------------------- #
def build_scenario(proc_w: int, proc_h: int, prompt: str, max_frames: int,
                   orient: str = "vertical", line_pos: float = 0.5,
                   anchor: str = "CENTER") -> Scenario:
    """Dựng scenario đếm.

    orient="vertical"  → vạch DỌC (hàng chạy NGANG trên chuyền cắt qua).
    orient="horizontal"→ vạch NGANG (vật đi XUỐNG/lên cắt qua — vd box/cà chua
                         trôi về phía camera).
    """
    scn = Scenario(
        key="la_conveyor",
        title="LA-3B · đếm sản phẩm",
        prompt=prompt,
        line=LineConfig(orientation=orient, position=line_pos, anchor=anchor),
        resolution=(proc_w, proc_h),
        in_label="Qua vạch",
        out_label="Ngược",
        max_frames=max_frames,
        expect_min_crossings=1,
    )
    scn.validate()
    return scn


def _annotate(frame_bgr, sv_d, pipe):
    """Vẽ vạch + box + bộ đếm lên 1 frame (BGR) — để xem model bắt gì."""
    import cv2

    img = frame_bgr.copy()
    h, w = img.shape[:2]
    (sx, sy), (ex, ey) = pipe.scenario.line.points(w, h)
    cv2.line(img, (sx, sy), (ex, ey), (0, 0, 255), 2)
    xyxy = getattr(sv_d, "xyxy", [])
    for box in xyxy:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    lz = pipe.line_zone
    cv2.putText(img, f"IN {int(lz.in_count)}  OUT {int(lz.out_count)}  box {len(xyxy)}",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img


def run_video(detector, video, query, *, orient="vertical", line_pos=0.5,
              anchor="CENTER", proc_width=640, max_frames=30, stride=3,
              save_annotated=None):
    """Chạy 1 (video, query) → ``CountResult``. Model nạp SẴN ở ngoài để DÙNG LẠI
    cho nhiều video/query (tránh nạp lại 6GB mỗi lần — đây là mấu chốt tốc độ).

    save_annotated: path .jpg → lưu frame NHIỀU box nhất (đã vẽ vạch + box) để xem.
    """
    proc_h = max(2, round(proc_width * 9 / 16))
    scn = build_scenario(proc_width, proc_h, query, max_frames,
                         orient=orient, line_pos=line_pos, anchor=anchor)
    pipe = CountingPipeline(detector, scn, resize=True)
    best = {"n": -1, "img": None}

    def cb(frame_bgr, sv_d, _pipe):
        if len(sv_d) > best["n"]:
            best["n"] = len(sv_d)
            best["img"] = _annotate(frame_bgr, sv_d, _pipe)

    res = pipe.run(strided(iter_video_frames(video), stride),
                   max_frames=max_frames, on_frame=cb if save_annotated else None)
    if save_annotated is not None and best["img"] is not None:
        import cv2

        d = os.path.dirname(os.path.abspath(save_annotated))
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(save_annotated, best["img"])
    return res


# --------------------------------------------------------------------------- #
# Tải / vet nguồn video
# --------------------------------------------------------------------------- #
def download_milk(dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dst = os.path.join(dest_dir, MILK_FILE)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    from supervision.assets import VideoAssets, download_assets

    cwd = os.getcwd()
    os.chdir(dest_dir)
    try:
        path = download_assets(getattr(VideoAssets, MILK_ASSET))
    finally:
        os.chdir(cwd)
    full = os.path.join(dest_dir, os.path.basename(path))
    return full if os.path.exists(full) else dst


def download_url(url: str, dest_dir: str) -> str:
    import urllib.request

    os.makedirs(dest_dir, exist_ok=True)
    name = url.rstrip("/").split("/")[-1] or "video.mp4"
    if not name.endswith(".mp4"):
        name += ".mp4"
    dst = os.path.join(dest_dir, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        f.write(r.read())
    return dst


def resolve_videos(args) -> List[str]:
    vids: List[str] = []
    if not args.no_milk:
        try:
            vids.append(download_milk(args.dest))
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  không tải được video chai (supervision): {e}")
    for v in (args.video or []):
        if os.path.exists(v):
            vids.append(v)
        elif v.startswith("http"):
            try:
                vids.append(download_url(v, args.dest))
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  tải lỗi {v}: {e}")
        else:
            print(f"⚠️  bỏ qua (không phải path/URL hợp lệ): {v}")
    seen, out = set(), []
    for v in vids:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def preview(video_path: str, out_dir: str, n: int = 3) -> List[str]:
    """Lưu ``n`` frame (10%/50%/85%) ra JPG để VET nội dung — KHÔNG nạp model."""
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total / fps if fps else 0
    stem = os.path.splitext(os.path.basename(video_path))[0]
    print(f"  {stem}: {w}x{h}, {fps:.0f}fps, {total} frame (~{dur:.0f}s), "
          f"{os.path.getsize(video_path) // 1024}KB")
    idxs = [max(0, int(total * p)) for p in (0.1, 0.5, 0.85)][:n] if total > 0 else [0]
    saved: List[str] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok:
            p = os.path.join(out_dir, f"{stem}_f{idx}.jpg")
            cv2.imwrite(p, fr)
            saved.append(p)
    cap.release()
    for p in saved:
        print(f"     🖼️  {p}")
    return saved


# --------------------------------------------------------------------------- #
# Self-test detector (không GPU): 1 box chạy trái→phải để cắt vạch dọc.
# --------------------------------------------------------------------------- #
class _FakeDet:
    def __init__(self, n_frames: int):
        self.i = 0
        self.n = n_frames

    def detect_frame(self, frame_bgr, prompt: str):
        h, w = frame_bgr.shape[:2]
        frac = self.i / max(self.n - 1, 1)
        x = int(frac * (w - 80))
        self.i += 1
        box = (float(x), float(h // 2 - 40), float(x + 80), float(h // 2 + 40))
        return [Detection(box, "object", 0.9)], "fake"


def _blank_frames(n: int, w: int, h: int):
    import numpy as np

    for _ in range(n):
        yield np.zeros((h, w, 3), dtype="uint8")


# --------------------------------------------------------------------------- #
# Scorecard
# --------------------------------------------------------------------------- #
def print_header():
    print("\n" + "=" * 92)
    print("📊 LocateAnything-3B · SCORECARD đếm sản phẩm băng chuyền")
    print("=" * 92)
    print(f"{'video':28} {'query':26} {'fr':>4} {'qua vạch':>9} {'det/fr':>7} {'fps':>6} {'giây':>6}")
    print("-" * 92)


def print_row(video: str, query: str, res, secs: float):
    vid = (os.path.basename(video))[:27]
    q = query[:25]
    print(f"{vid:28} {q:26} {res.frames:>4} "
          f"{res.total_crossings:>9} {res.avg_detections:>7.1f} {res.fps:>6.2f} {secs:>6.1f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", default=[],
                    help="thêm video (path hoặc URL); lặp lại để thêm nhiều")
    ap.add_argument("--no-milk", action="store_true",
                    help="KHÔNG dùng video chai mặc định (chỉ dùng --video)")
    ap.add_argument("--dest", default="videos", help="thư mục lưu video tải về")
    ap.add_argument("--preview", action="store_true",
                    help="chỉ tải + lưu vài frame JPG để VET nguồn (không nạp model)")
    ap.add_argument("--selftest", action="store_true",
                    help="chạy detector giả (không GPU) để kiểm tra đường ống")
    # query
    ap.add_argument("--queries", default=None,
                    help="danh sách query 'a,b,c' (mặc định: bộ đại diện)")
    ap.add_argument("--suite", action="store_true",
                    help="chạy TOÀN BỘ bộ query sản phẩm (nhiều ca, chậm hơn)")
    ap.add_argument("--per-group", type=int, default=1,
                    help="số query đầu mỗi nhóm khi KHÔNG --suite (mặc định 1)")
    # tốc độ
    ap.add_argument("--proc-width", type=int, default=640,
                    help="bề rộng xử lý (nhỏ = nhanh; mặc định 640)")
    ap.add_argument("--max-frames", type=int, default=30, help="số frame tối đa/lượt")
    ap.add_argument("--stride", type=int, default=3, help="lấy mỗi N frame (thưa=nhanh)")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="số token sinh tối đa (nhỏ = nhanh; mặc định 256)")
    ap.add_argument("--iou", type=float, default=0.5, help="ngưỡng IoU khử box trùng")
    ap.add_argument("--max-boxes", type=int, default=60, help="trần số box/frame sau NMS")
    # vạch
    ap.add_argument("--orient", choices=["vertical", "horizontal"], default="vertical",
                    help="hướng vạch: dọc (hàng chạy ngang) | ngang (vật đi xuống/lên)")
    ap.add_argument("--line-pos", type=float, default=0.5, help="vị trí vạch 0..1")
    ap.add_argument("--anchor", default="CENTER", help="điểm neo xét cắt vạch")
    ap.add_argument("--save-dir", default=None, help="lưu frame annotate mỗi lượt (JPG)")
    # model
    ap.add_argument("--model", default="nvidia/LocateAnything-3B")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    proc_w = args.proc_width
    proc_h = max(2, round(proc_w * 9 / 16))  # giữ 16:9 (resize đồng đều, không lệch đếm)

    # ---- SELF-TEST (không GPU) ----
    if args.selftest:
        n = 26
        scn = build_scenario(proc_w, proc_h, "object", n, orient="vertical", line_pos=0.5)
        pipe = CountingPipeline(_FakeDet(n), scn, resize=False)
        res = pipe.run(_blank_frames(n, proc_w, proc_h), max_frames=n)
        print_header()
        print_row("selftest", "object", res, res.elapsed_s)
        ok = res.total_crossings >= 1
        print("-" * 92)
        print("✅ SELFTEST OK — đường ống đếm chạy đúng." if ok
              else "❌ SELFTEST FAIL — vật không cắt vạch (kiểm tra line-x).")
        return 0 if ok else 1

    # ---- Nguồn video ----
    videos = resolve_videos(args)
    if not videos:
        print("❌ Không có video nào. Cấp --video <path|URL> hoặc bỏ --no-milk.")
        return 2

    # ---- PREVIEW: chỉ vet nguồn ----
    if args.preview:
        print("🔎 PREVIEW nguồn video (mở JPG để xem có đúng băng chuyền + sản phẩm không):")
        for v in videos:
            preview(v, os.path.join(args.dest, "preview"))
        print("\n→ Nếu frame ĐÚNG là băng chuyền có sản phẩm: chạy lại BỎ --preview để đếm.")
        print("  Nếu SAI: gửi tôi URL/ảnh, tôi thay video khác.")
        return 0

    # ---- Query ----
    if args.queries:
        pairs = [("tuỳ chọn", q.strip()) for q in args.queries.split(",") if q.strip()]
    else:
        pairs = suite(lite=not args.suite, per_group=args.per_group)

    n_runs = len(videos) * len(pairs)
    print(f"▶️  {len(videos)} video × {len(pairs)} query = {n_runs} lượt "
          f"(proc={proc_w}x{proc_h}, max_frames={args.max_frames}, stride={args.stride}, "
          f"tokens={args.max_new_tokens}).")
    print("   LocateAnything-3B chậm (~vài giây/khung) → hãy kiên nhẫn.")

    detector = build_fast_detector(args.model, args.max_new_tokens, args.iou, args.max_boxes)

    print_header()
    t_all = time.time()
    for v in videos:
        for _group, q in pairs:
            save = None
            if args.save_dir:
                stem = os.path.splitext(os.path.basename(v))[0]
                qsafe = "".join(c if c.isalnum() else "_" for c in q)[:24]
                save = os.path.join(args.save_dir, f"{stem}__{qsafe}.jpg")
            t0 = time.time()
            res = run_video(detector, v, q, orient=args.orient, line_pos=args.line_pos,
                            anchor=args.anchor, proc_width=proc_w,
                            max_frames=args.max_frames, stride=args.stride,
                            save_annotated=save)
            print_row(v, q, res, time.time() - t0)
    print("-" * 92)
    print(f"Xong {n_runs} lượt trong {time.time() - t_all:.0f}s. "
          f"'qua vạch' = số sản phẩm cắt vạch dọc; 'det/fr' = số box TB/khung (đã NMS).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
