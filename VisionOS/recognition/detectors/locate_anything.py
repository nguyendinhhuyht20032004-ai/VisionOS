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
    """Vá nội dung ``modeling_locateanything.py`` (thuần chuỗi → test được).

    Hai vá, đều **idempotent** (vá lại lần nữa không đổi):

    1. **T4 (Turing) không có bfloat16 kernel** → ép ``pixel_values`` sang float16.
    2. ``import decord`` / ``import lmdb`` ở đầu file khiến
       ``transformers.check_imports`` **bắt buộc** cài 2 gói này (decord không có
       wheel cho Python 3.12 → lỗi). Chúng chỉ dùng cho video/dataset, KHÔNG cần
       khi suy luận ảnh → bọc vào ``try/except`` để check_imports bỏ qua.
    """
    import re

    code = code.replace(
        "pixel_values = pixel_values.to(self.language_model.dtype)",
        "pixel_values = pixel_values.to(torch.float16)  # T4 fix",
    )
    # chỉ khớp import ở CỘT 0 (top-level) → sau khi bọc vào try (thụt lề) sẽ không
    # khớp nữa ⇒ idempotent.
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
            print('     !pip install -q "transformers==4.57.1" "tokenizers>=0.20,<0.22" accelerate')
            print("=" * 74)
    except Exception:
        pass


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
        return DetectorResult(
            dets, raw=raw, latency_ms=(time.time() - t0) * 1000,
            model_name="LocateAnything-3B",
        )

    # Tương thích ngược với code gọi kiểu la_counting (detect_frame → tuple).
    def detect_frame(self, frame, prompt, max_new_tokens=None):
        res = self.detect(frame, prompt, max_new_tokens)
        return res.detections, res.raw
