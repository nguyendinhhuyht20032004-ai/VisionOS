"""Test bản vá modeling file của LocateAnything (thuần chuỗi, không cần model)."""

from recognition.detectors.locate_anything import patch_modeling_source

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


def test_idempotent():
    once = patch_modeling_source(SAMPLE)
    twice = patch_modeling_source(once)
    assert once == twice   # vá lại không đổi (an toàn khi load nhiều lần)
