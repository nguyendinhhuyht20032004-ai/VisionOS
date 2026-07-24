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


def test_bfloat16_patched_to_float16():
    out = patch_modeling_source(SAMPLE)
    assert "pixel_values.to(torch.float16)" in out
    assert "self.language_model.dtype" not in out


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


def test_to_legacy_cache_call_removed():
    out = patch_modeling_source(CACHE_SNIPPET)
    assert ".to_legacy_cache()" not in out         # lời gọi đã bị gỡ
    # X.to_legacy_cache() -> X ⇒ hai nhánh ternary đều là next_decoder_cache
    assert "next_cache = next_decoder_cache if use_legacy_cache else next_decoder_cache" in out


def test_idempotent():
    combined = SAMPLE + QWEN2_SNIPPET + CACHE_SNIPPET
    once = patch_modeling_source(combined)
    twice = patch_modeling_source(once)
    assert once == twice   # vá lại không đổi (an toàn khi load nhiều lần)


# --------------------------------------------------------------------------- #
# _patch_py_files: vá MỌI file .py (kể cả processor) — đây là lỗi đã gặp
# --------------------------------------------------------------------------- #
def test_patch_all_py_files_covers_processor(tmp_path):
    # file processor (chính chỗ AutoProcessor.from_pretrained crash) có import decord
    (tmp_path / "processing_locateanything.py").write_text("import decord\nimport lmdb\n")
    (tmp_path / "modeling_locateanything.py").write_text(
        "import torch\npixel_values = pixel_values.to(self.language_model.dtype)\n"
    )
    (tmp_path / "other.py").write_text("import torch\nimport numpy as np\n")

    changed = _patch_py_files(str(tmp_path))
    assert changed == 2   # processor + modeling đổi; other.py không

    proc = (tmp_path / "processing_locateanything.py").read_text()
    assert "try:\n    import decord" in proc and "try:\n    import lmdb" in proc
    assert not any(line == "import decord" for line in proc.splitlines())

    model = (tmp_path / "modeling_locateanything.py").read_text()
    assert "pixel_values.to(torch.float16)" in model

    other = (tmp_path / "other.py").read_text()
    assert other == "import torch\nimport numpy as np\n"   # không đụng

    # gọi lại: không còn gì để đổi (idempotent)
    assert _patch_py_files(str(tmp_path)) == 0
