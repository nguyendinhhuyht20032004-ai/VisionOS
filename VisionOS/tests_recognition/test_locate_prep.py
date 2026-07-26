"""Test bản vá modeling file của LocateAnything (thuần chuỗi, không cần model)."""

import os

from recognition.detectors.locate_anything import _patch_py_files, patch_modeling_source

SAMPLE = """\
import torch
import decord
import lmdb
from decord import VideoReader
import numpy as np


class LocateAnythingModel:
    def forward(self, pixel_values):
        pixel_values = pixel_values.to(self.language_model.dtype)
        return pixel_values
"""


def test_pixel_values_dtype_left_untouched():
    # KHÔNG còn ép float16: dòng .to(self.language_model.dtype) giữ nguyên để tự
    # khớp dtype khi nạp (bf16). Nếu ép float16 sẽ tràn → CUBLAS_INTERNAL_ERROR.
    out = patch_modeling_source(SAMPLE)
    assert "pixel_values.to(self.language_model.dtype)" in out
    assert "torch.float16" not in out


def test_decord_lmdb_wrapped_in_try_except():
    out = patch_modeling_source(SAMPLE)
    # không còn import top-level (cột 0) cho decord/lmdb -> check_imports bỏ qua
    for line in out.splitlines():
        assert not (line == "import decord")
        assert not (line == "import lmdb")
        assert not line.startswith("from decord ")
    # đã bọc vào try/except
    assert "try:\n    import decord" in out
    assert "try:\n    import lmdb" in out
    assert "try:\n    from decord import VideoReader" in out


def test_unrelated_imports_untouched():
    out = patch_modeling_source(SAMPLE)
    assert out.startswith("import torch\n")    # torch giữ nguyên
    assert "\nimport numpy as np\n" in out     # numpy giữ nguyên
    assert "try:\n    import torch" not in out  # KHÔNG bọc nhầm torch/numpy


QWEN2_SNIPPET = """\
class Qwen2Attention:
    def __init__(self, config, layer_idx):
        self.rope_theta = config.rope_theta
        self.head_dim = config.hidden_size
"""


def test_rope_theta_patched_to_getattr():
    out = patch_modeling_source(QWEN2_SNIPPET)
    assert "getattr(config, 'rope_theta', 1_000_000.0)" in out
    assert "config.rope_theta" not in out


CACHE_SNIPPET = (
    "next_cache = next_decoder_cache.to_legacy_cache() "
    "if use_legacy_cache else next_decoder_cache\n"
)

FROM_CACHE_SNIPPET = (
    "        past_key_values = DynamicCache.from_legacy_cache(past_key_values)\n"
    "        past_key_values_length = past_key_values.get_seq_length()\n"
)

# Dạng ĐÃ bị bản vá TRƯỚC sửa (thiếu: để nguyên tuple → get_seq_length lỗi).
OLD_PATCHED_SNIPPET = (
    "        past_key_values = (DynamicCache() if past_key_values is None else past_key_values)\n"
    "        past_key_values_length = past_key_values.get_seq_length()\n"
)

ROBUST = "(past_key_values if hasattr(past_key_values, 'get_seq_length') else DynamicCache())"


def test_to_legacy_cache_call_removed():
    out = patch_modeling_source(CACHE_SNIPPET)
    assert ".to_legacy_cache()" not in out         # lời gọi đã bị gỡ
    # X.to_legacy_cache() -> X ⇒ hai nhánh ternary đều là next_decoder_cache
    assert "next_cache = next_decoder_cache if use_legacy_cache else next_decoder_cache" in out


def test_from_legacy_cache_replaced_with_cache_guard():
    out = patch_modeling_source(FROM_CACHE_SNIPPET)
    assert "from_legacy_cache" not in out          # hàm cũ đã bị thay
    assert ROBUST in out                           # cho ra Cache dù None/tuple/Cache


def test_old_buggy_patch_form_is_normalized():
    # Bản vá cũ có thêm _cos_cached (không tồn tại trong module thật) cũng phải được bình thường hóa
    out = patch_modeling_source(OLD_PATCHED_SNIPPET)
    assert "if past_key_values is None else past_key_values" not in out


ROPE_SNIPPET = """\
class Qwen2RotaryEmbedding:
    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)
"""


def test_rope_cache_guard_uses_real_attribute_name():
    # Bản vá cũ kiểm tra `hasattr(self, '_cos_cached')` — thuộc tính này KHÔNG
    # tồn tại trong modeling_qwen2.py gốc (buffer thật tên là `cos_cached`, không
    # có dấu gạch dưới). Vì hasattr() với tên sai luôn False, điều kiện "or" luôn
    # True → cache bị rebuild lại TỪ ĐẦU ở MỌI lần forward() (dead-code, tốn kém
    # thêm ở 36 layer x nhiều bước generate). Bản vá đúng phải dùng tên buffer
    # thật để guard chỉ thực sự hoạt động khi cache CHƯA từng được tạo.
    out = patch_modeling_source(ROPE_SNIPPET)
    assert "hasattr(self, 'cos_cached')" in out
    assert "self.cos_cached is None" in out
    assert "_cos_cached" not in out


def test_idempotent():
    combined = SAMPLE + QWEN2_SNIPPET + CACHE_SNIPPET + FROM_CACHE_SNIPPET + OLD_PATCHED_SNIPPET
    once = patch_modeling_source(combined)
    twice = patch_modeling_source(once)
    assert once == twice   # vá lại không đổi (an toàn khi load nhiều lần)


# --------------------------------------------------------------------------- #
# _patch_py_files: vá MỌI file .py (kể cả processor) — đây là lỗi đã gặp
# --------------------------------------------------------------------------- #
def test_patch_all_py_files_covers_processor(tmp_path):
    # file processor (chính chỗ AutoProcessor.from_pretrained crash) có import decord
    (tmp_path / "processing_locateanything.py").write_text("import decord\nimport lmdb\n")
    # modeling: dùng rope_theta để chắc chắn file này CÓ thứ để vá (không còn vá dtype)
    (tmp_path / "modeling_locateanything.py").write_text(
        "import torch\nself.rope_theta = config.rope_theta\n"
    )
    (tmp_path / "other.py").write_text("import torch\nimport numpy as np\n")

    changed = _patch_py_files(str(tmp_path))
    assert changed == 2   # processor + modeling đổi; other.py không

    proc = (tmp_path / "processing_locateanything.py").read_text()
    assert "try:\n    import decord" in proc and "try:\n    import lmdb" in proc
    assert not any(line == "import decord" for line in proc.splitlines())

    model = (tmp_path / "modeling_locateanything.py").read_text()
    assert "getattr(config, 'rope_theta', 1_000_000.0)" in model

    other = (tmp_path / "other.py").read_text()
    assert other == "import torch\nimport numpy as np\n"   # không đụng

    # gọi lại: không còn gì để đổi (idempotent)
    assert _patch_py_files(str(tmp_path)) == 0
