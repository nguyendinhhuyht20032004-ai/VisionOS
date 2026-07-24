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

    f = os.path.join(model_dir, "modeling_locateanything.py")
    if os.path.exists(f):
        real = os.path.realpath(f)
        with open(real) as fh:
            code = fh.read()
        patched = patch_modeling_source(code)
        if patched != code:
            with open(real, "w") as fh:
                fh.write(patched)
            print("✅ Đã vá modeling file (T4 float16 + gỡ ràng buộc decord/lmdb)")
    return model_dir


class LocateAnythingDetector:
    """Wrapper open-vocab: đếm bất kỳ vật gì mô tả bằng ngôn ngữ tự nhiên."""

    def __init__(self, model_dir: str = "nvidia/LocateAnything-3B", max_new_tokens: int = 1024):
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self._impl = None

    def load(self):
        # Import lazy: chỉ cần khi chạy model thật (kéo theo torch/transformers).
        from la_counting.detector import LocateAnythingDetector as _LA

        # Tải + vá modeling file (T4 float16 + gỡ ràng buộc decord/lmdb) rồi load
        # từ thư mục cục bộ đã vá — nhờ vậy chạy được trên Kaggle T4 mà KHÔNG cần
        # cài decord/lmdb (hai gói này chỉ dùng cho video/dataset, không cần khi
        # suy luận ảnh).
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
