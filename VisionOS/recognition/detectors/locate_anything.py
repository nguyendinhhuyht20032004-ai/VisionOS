"""``LocateAnythingDetector`` — mô hình chế độ SMART (open-vocabulary).

Đây là **adapter** bọc lại ``la_counting.detector.LocateAnythingDetector`` đã có
sẵn trong repo (bọc ``nvidia/LocateAnything-3B``), quy đổi output sang kiểu
``recognition.base.Detection`` / ``DetectorResult`` để dùng chung interface
``detect(frame, prompt)`` với YOLO-NAS và detector giả.

Nhờ tái sử dụng, ta không viết lại phần nạp model + parse toạ độ (đã được unit
-test kỹ ở ``tests/test_parsing.py``); ``torch``/``transformers`` vẫn import lazy.
"""

from __future__ import annotations

import time
from typing import Optional

from ..base import BoundingBox, Detection, DetectorResult

__all__ = ["LocateAnythingDetector", "patch_modeling_source"]


def patch_modeling_source(code: str) -> str:
    """Vá nội dung file model — MẶC ĐỊNH đúng như NOTEBOOK KAGGLE ĐÃ CHẠY ĐƯỢC.

    Notebook chỉ vá **hai** thứ (và chạy tốt trên transformers 4.57.1):

    1. ``import decord`` / ``import lmdb`` → bọc ``try/except`` để
       ``transformers.check_imports`` không bắt cài (decord không có wheel Python
       3.12). Chỉ import-time, KHÔNG đổi phép tính của model.
    2. ``pixel_values = pixel_values.to(self.language_model.dtype)`` →
       ``.to(torch.float16)``: T4 (Turing) không có kernel bfloat16 → phải float16.
       **Đây là vá THEN CHỐT để model nhận diện được trên T4.**

    ⚠️ Các vá cũ (rope_theta / to_legacy_cache / from_legacy_cache / RoPE cache
    auto-extend / trả full cache) ĐÃ GỠ khỏi mặc định: trên 4.57.1 chúng ĐỔI phép
    tính RoPE/cache khiến model **trả rỗng** (det/frame=0). Chỉ bật lại khi chạy
    transformers KHÁC 4.57.x (cần shim tương thích): đặt env ``LA_COMPAT_PATCH=1``.
    """
    import os
    import re

    # (1) decord/lmdb: import-time guard (vô hại, cần cho Python 3.12). Idempotent:
    # sau khi bọc vào try (thụt lề) sẽ không khớp '^import' ở cột 0 nữa.
    code = re.sub(
        r"(?m)^(import (?:decord|lmdb)\b.*)$",
        "try:\n    \\1\nexcept Exception:\n    pass",
        code,
    )
    code = re.sub(
        r"(?m)^(from (?:decord|lmdb)\b.*)$",
        "try:\n    \\1\nexcept Exception:\n    pass",
        code,
    )
    # (2) VÁ THEN CHỐT (giống notebook): pixel_values → float16 cho T4. Idempotent
    # nhờ marker '# T4 fix' (vá lại không khớp chuỗi gốc nữa).
    code = code.replace(
        "pixel_values = pixel_values.to(self.language_model.dtype)",
        "pixel_values = pixel_values.to(torch.float16)  # T4 fix",
    )

    # ⚠️ Shim tương thích transformers ≠ 4.57.x — MẶC ĐỊNH TẮT (đổi RoPE/cache làm
    # model rỗng trên 4.57.1). Bật bằng LA_COMPAT_PATCH=1 khi buộc dùng bản khác.
    if os.environ.get("LA_COMPAT_PATCH"):
        code = code.replace(
            "self.rope_theta = config.rope_theta",
            "self.rope_theta = getattr(config, 'rope_theta', 1_000_000.0)",
        )
        code = re.sub(r"(\w+)\.to_legacy_cache\(\)", r"\1", code)
        _cache_fix = r"(\1 if hasattr(\1, 'get_seq_length') else DynamicCache())"
        code = re.sub(r"DynamicCache\.from_legacy_cache\((\w+)\)", _cache_fix, code)
        code = re.sub(r"\(DynamicCache\(\) if (\w+) is None else \1\)", _cache_fix, code)
        code = code.replace(
            "if seq_len > self.max_seq_len_cached",
            "if seq_len > self.max_seq_len_cached or not hasattr(self, 'cos_cached') or self.cos_cached is None",
        )
        code = code.replace(
            "return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)",
            "return self.cos_cached.to(dtype=x.dtype), self.sin_cached.to(dtype=x.dtype)",
        )
    return code


def _prepare_model_dir(model_id: str) -> str:
    """Tải snapshot model (hoặc dùng thư mục local) + vá modeling file + xoá cache.

    Trả về đường dẫn thư mục model đã vá để nạp bằng ``trust_remote_code``.
    """
    import os
    import shutil

    if os.path.isdir(model_id):
        model_dir = model_id
    else:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(model_id)

    # Xoá cache module động để bản vá có hiệu lực.
    mc = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    if os.path.isdir(mc):
        shutil.rmtree(mc, ignore_errors=True)

    # Vá TẤT CẢ file .py trong snapshot: decord/lmdb có thể nằm ở file processor /
    # image_processor (không chỉ modeling_locateanything.py) — chính là chỗ
    # AutoProcessor.from_pretrained gọi check_imports và crash.
    n = _patch_py_files(model_dir)
    if n:
        print(f"✅ Đã vá {n} file model (T4 float16 + gỡ ràng buộc decord/lmdb)")
    return model_dir


def _patch_py_files(model_dir: str) -> int:
    """Áp :func:`patch_modeling_source` cho MỌI file .py trong ``model_dir``.

    Trả về số file thực sự bị thay đổi. An toàn khi gọi lại (idempotent).
    """
    import glob
    import os

    changed = 0
    for path in glob.glob(os.path.join(model_dir, "*.py")):
        real = os.path.realpath(path)
        try:
            with open(real, encoding="utf-8") as fh:
                code = fh.read()
        except OSError:
            continue
        patched = patch_modeling_source(code)
        if patched != code:
            try:
                with open(real, "w", encoding="utf-8") as fh:
                    fh.write(patched)
                changed += 1
            except OSError:
                pass
    return changed


def _ensure_locate_deps() -> None:
    """Cài phụ thuộc runtime cho LocateAnything nếu THIẾU (Kaggle có Internet).

    ``decord`` không có wheel cho Python 3.12 → dùng ``eva-decord`` (drop-in, vẫn
    ``import decord``). ``lmdb`` có wheel sẵn. Đây là lớp bảo hiểm: kể cả không có
    2 gói này, bản vá try/except vẫn giúp chạy được khi suy luận ảnh.
    """
    import importlib

    need = []
    for module, pkg in (("decord", "eva-decord"), ("lmdb", "lmdb")):
        try:
            importlib.import_module(module)
        except Exception:
            need.append(pkg)
    if need:
        import subprocess
        import sys

        print(f"📦 Cài phụ thuộc còn thiếu cho LocateAnything: {need} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=False)

    # Cảnh báo phiên bản transformers: model được NVIDIA test với ĐÚNG 4.57.1.
    # Bản khác gây lỗi kiểu 'Qwen2Config has no attribute rope_theta'. Đổi phiên
    # bản cần RESTART kernel nên chỉ cảnh báo, không tự cài (tránh nửa vời).
    try:
        import transformers

        v = transformers.__version__
        if not v.startswith("4.57"):
            print("=" * 74)
            print(f"⚠️  transformers {v} KHÔNG khớp — LocateAnything-3B cần 4.57.1.")
            print("   Chạy 1 cell RỒI RESTART KERNEL, sau đó chạy lại run_eval:")
            print('     !pip install -q "transformers==4.57.1" accelerate')
            print("=" * 74)
    except Exception:
        pass


def _drop_full_frame(dets, w: int, h: int, max_frac: float = 0.85):
    """Bỏ box phủ ~CẢ khung ảnh — LocateAnything hay trả 1 box = TOÀN ảnh khi không định
    vị được từng vật (→ 'đếm cả khung' thay vì đếm vật). Loại trước khi NMS/track."""
    fa = float(w * h) or 1.0
    return [d for d in dets if d.bbox.area <= max_frac * fa]


def _dedup_dets(dets, iou_thr: float = 0.5, max_boxes: int = 60):
    """Khử box trùng (NMS class-agnostic) + chặn trần số box.

    LocateAnything hay sinh box lặp → gộp box chồng nhau (IoU≥``iou_thr``, giữ box
    lớn hơn khi điểm bằng nhau) rồi cắt còn tối đa ``max_boxes`` (theo diện tích).
    """
    if len(dets) <= 1:
        return dets
    from ..evaluation import nms_dedup

    boxes = [d.bbox for d in dets]
    scores = [d.confidence for d in dets]
    keep = nms_dedup(boxes, scores, iou_thr)
    kept = [dets[i] for i in keep]
    if len(kept) > max_boxes:
        kept = sorted(kept, key=lambda d: d.bbox.area, reverse=True)[:max_boxes]
    return kept


class LocateAnythingDetector:
    """Wrapper open-vocab: đếm bất kỳ vật gì mô tả bằng ngôn ngữ tự nhiên."""

    def __init__(self, model_dir: str = "nvidia/LocateAnything-3B", max_new_tokens: int = 1024):
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self._impl = None

    def load(self):
        # Import lazy: chỉ cần khi chạy model thật (kéo theo torch/transformers).
        from la_counting.detector import LocateAnythingDetector as _LA

        # (1) Cài decord(eva-decord)/lmdb nếu thiếu — lớp bảo hiểm chính.
        _ensure_locate_deps()
        # (2) Tải + vá MỌI file .py của model (T4 float16 + gỡ ràng buộc
        # decord/lmdb ở cả file processor) rồi load từ thư mục cục bộ đã vá.
        local_dir = _prepare_model_dir(self.model_dir)
        self._impl = _LA(local_dir, self.max_new_tokens)
        self._impl.load()
        return self

    def detect(self, frame, prompt: str, max_new_tokens: Optional[int] = None) -> DetectorResult:
        """Phát hiện mọi thực thể khớp mô tả ``prompt`` trong 1 frame BGR."""
        if self._impl is None:
            self.load()
        t0 = time.time()
        la_dets, raw = self._impl.detect_frame(frame, prompt, max_new_tokens)
        dets = [
            Detection(
                BoundingBox(*[float(v) for v in d.bbox]),
                d.class_name,
                float(d.confidence),
            )
            for d in la_dets
        ]
        # BỎ box 'CẢ KHUNG ẢNH' (LA hay trả 1 box = toàn ảnh khi không định vị được vật).
        h, w = frame.shape[:2]
        dets = _drop_full_frame(dets, w, h)
        # KHỬ BOX TRÙNG (NMS): LocateAnything greedy-decode dễ BÙNG NỔ box lặp
        # (100+/frame) → tracker loạn ID → KHÔNG đếm được (IN/OUT=0). Gộp box chồng
        # nhau (IoU≥0.5) rồi chặn trần để mỗi vật chỉ còn 1 box → track ổn định.
        dets = _dedup_dets(dets, iou_thr=0.5, max_boxes=60)
        return DetectorResult(
            dets, raw=raw, latency_ms=(time.time() - t0) * 1000,
            model_name="LocateAnything-3B",
        )

    # Tương thích ngược với code gọi kiểu la_counting (detect_frame → tuple).
    def detect_frame(self, frame, prompt, max_new_tokens=None):
        res = self.detect(frame, prompt, max_new_tokens)
        return res.detections, res.raw
