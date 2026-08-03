"""Test bản vá modeling file của LocateAnything (thuần chuỗi, không cần model).

MẶC ĐỊNH giờ khớp NOTEBOOK KAGGLE ĐÃ CHẠY ĐƯỢC: chỉ vá (1) decord/lmdb import-guard
và (2) pixel_values → float16 (T4). Các shim tương thích transformers ≠ 4.57.x
(rope_theta/cache/RoPE) chỉ bật khi env ``LA_COMPAT_PATCH=1`` — vì trên 4.57.1 chúng
làm model TRẢ RỖNG (det/frame=0).
"""

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


def test_pixel_values_patched_to_float16():
    # VÁ THEN CHỐT (như notebook): T4 không có kernel bfloat16 → ép float16, nếu không
    # model không chạy vision đúng → nhận diện rỗng.
    out = patch_modeling_source(SAMPLE)
    assert "pixel_values = pixel_values.to(torch.float16)  # T4 fix" in out
    assert "pixel_values.to(self.language_model.dtype)" not in out


def test_decord_lmdb_wrapped_in_try_except():
    out = patch_modeling_source(SAMPLE)
    for line in out.splitlines():
        assert not (line == "import decord")
        assert not (line == "import lmdb")
        assert not line.startswith("from decord ")
    assert "try:\n    import decord" in out
    assert "try:\n    import lmdb" in out
    assert "try:\n    from decord import VideoReader" in out


def test_unrelated_imports_untouched():
    out = patch_modeling_source(SAMPLE)
    assert out.startswith("import torch\n")
    assert "\nimport numpy as np\n" in out
    assert "try:\n    import torch" not in out


# --------------------------------------------------------------------------- #
# MẶC ĐỊNH: KHÔNG áp shim tương thích (rope/cache) — chúng làm model rỗng trên 4.57.1
# --------------------------------------------------------------------------- #
QWEN2_SNIPPET = """\
class Qwen2Attention:
    def __init__(self, config, layer_idx):
        self.rope_theta = config.rope_theta
        self.head_dim = config.hidden_size
"""
CACHE_SNIPPET = (
    "next_cache = next_decoder_cache.to_legacy_cache() "
    "if use_legacy_cache else next_decoder_cache\n"
)
FROM_CACHE_SNIPPET = (
    "        past_key_values = DynamicCache.from_legacy_cache(past_key_values)\n"
    "        past_key_values_length = past_key_values.get_seq_length()\n"
)
ROPE_SNIPPET = """\
class Qwen2RotaryEmbedding:
    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)
"""


def test_compat_patches_off_by_default():
    # Không có LA_COMPAT_PATCH → GIỮ NGUYÊN rope_theta/cache/RoPE (khớp notebook 4.57.1).
    assert patch_modeling_source(QWEN2_SNIPPET) == QWEN2_SNIPPET
    assert patch_modeling_source(CACHE_SNIPPET) == CACHE_SNIPPET
    assert patch_modeling_source(ROPE_SNIPPET) == ROPE_SNIPPET


def test_compat_patches_on_with_env(monkeypatch):
    # LA_COMPAT_PATCH=1 → áp shim cho transformers ≠ 4.57.x.
    monkeypatch.setenv("LA_COMPAT_PATCH", "1")
    q = patch_modeling_source(QWEN2_SNIPPET)
    assert "getattr(config, 'rope_theta', 1_000_000.0)" in q and "config.rope_theta" not in q
    c = patch_modeling_source(CACHE_SNIPPET)
    assert ".to_legacy_cache()" not in c
    f = patch_modeling_source(FROM_CACHE_SNIPPET)
    assert "from_legacy_cache" not in f
    r = patch_modeling_source(ROPE_SNIPPET)
    assert "self.cos_cached[:seq_len]" not in r and "return self.cos_cached.to" in r


def test_idempotent_default():
    combined = SAMPLE + QWEN2_SNIPPET
    once = patch_modeling_source(combined)
    twice = patch_modeling_source(once)
    assert once == twice


def test_idempotent_compat(monkeypatch):
    monkeypatch.setenv("LA_COMPAT_PATCH", "1")
    combined = SAMPLE + QWEN2_SNIPPET + CACHE_SNIPPET + FROM_CACHE_SNIPPET
    once = patch_modeling_source(combined)
    twice = patch_modeling_source(once)
    assert once == twice


# --------------------------------------------------------------------------- #
# _patch_py_files: vá MỌI file .py (kể cả processor)
# --------------------------------------------------------------------------- #
def test_patch_all_py_files_covers_processor(tmp_path):
    (tmp_path / "processing_locateanything.py").write_text("import decord\nimport lmdb\n")
    # modeling: dùng pixel_values (vá MẶC ĐỊNH) để chắc chắn file này CÓ thứ để vá
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
    assert "pixel_values.to(torch.float16)  # T4 fix" in model

    other = (tmp_path / "other.py").read_text()
    assert other == "import torch\nimport numpy as np\n"

    assert _patch_py_files(str(tmp_path)) == 0   # idempotent
