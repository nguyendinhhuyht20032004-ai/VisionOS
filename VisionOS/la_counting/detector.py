"""``LocateAnythingDetector`` — bọc mô hình nvidia/LocateAnything-3B.

Giữ đúng logic của notebook (patch T4 float16, ``_prep_input`` chuyển numpy ->
tensor, parse output), nhưng:

  * ``torch`` / ``transformers`` được import **lazy** trong ``load()`` để module
    này import được ở môi trường không có GPU (phần parse dùng chung
    ``la_counting.parsing``);
  * phần parse dùng lại ``parse_boxes`` -> đã được unit-test riêng.

Chỉ dùng module này khi thực sự chạy mô hình (Kaggle GPU / ``run_benchmark.py``).
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from .parsing import Detection, parse_boxes

__all__ = ["LocateAnythingDetector"]


class LocateAnythingDetector:
    def __init__(self, model_dir: str, max_new_tokens: int = 1024):
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self._loaded = False
        self.model = None
        self.tokenizer = None
        self.processor = None

    # ------------------------------------------------------------------ #
    # Nạp mô hình
    # ------------------------------------------------------------------ #
    def load(self):
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

        self._torch = torch
        self.dtype = torch.float16  # T4 (Turing) không có bfloat16 kernel

        # Compat: transformers mới kỳ vọng _tied_weights_keys là dict nhưng
        # modeling_qwen2.py bundled trả list (format cũ). Bọc an toàn: chỉ
        # can thiệp khi crash; tie_weights() vẫn chạy trước đó nên suy luận
        # không bị ảnh hưởng. Flag _locate_patched đảm bảo idempotent.
        try:
            import transformers.modeling_utils as _tmu
            if not getattr(_tmu.PreTrainedModel, "_locate_patched", False):
                _orig = _tmu.PreTrainedModel.get_expanded_tied_weights_keys

                def _safe(self, all_submodels=True):
                    try:
                        return _orig(self, all_submodels=all_submodels)
                    except (AttributeError, TypeError):
                        return {}

                _tmu.PreTrainedModel.get_expanded_tied_weights_keys = _safe
                _tmu.PreTrainedModel._locate_patched = True
        except Exception:
            pass

        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir, trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            config=config,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map="auto",
            attn_implementation="sdpa",  # ép SDPA cho T4
        )
        self.model.eval()
        self._loaded = True
        print(f"✅ Loaded in {time.time() - t0:.1f}s")
        if torch.cuda.is_available():
            print(f"   GPU Mem: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        return self

    # ------------------------------------------------------------------ #
    # Chuẩn hoá input: numpy.ndarray -> tensor, ép float16 cho số thực
    # ------------------------------------------------------------------ #
    def _prep_input(self, v):
        import numpy as np

        torch = self._torch
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        if torch.is_tensor(v):
            if v.is_floating_point():
                return v.to(device=self.model.device, dtype=torch.float16)
            return v.to(self.model.device)
        return v

    # ------------------------------------------------------------------ #
    # Suy luận
    # ------------------------------------------------------------------ #
    def detect_pil(
        self, pil_image, prompt: str, max_new_tokens: Optional[int] = None
    ) -> Tuple[List[Detection], str]:
        if not self._loaded:
            self.load()
        torch = self._torch
        w, h = pil_image.size
        max_tok = max_new_tokens or self.max_new_tokens

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Locate all instances of: {prompt}"},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text_prompt, images=[pil_image], return_tensors="pt"
        )
        inputs = {k: self._prep_input(v) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tok,
                do_sample=False,
                use_cache=True,
                tokenizer=self.tokenizer,
            )

        if isinstance(output, (list, tuple)) and hasattr(output[0], "shape"):
            raw = self.tokenizer.decode(output[0], skip_special_tokens=True)
        elif hasattr(output, "shape"):
            raw = self.tokenizer.decode(output, skip_special_tokens=True)
        else:
            raw = str(output)

        return parse_boxes(raw, prompt, w, h), raw

    def detect_frame(
        self, bgr_frame, prompt: str, max_new_tokens: Optional[int] = None
    ) -> Tuple[List[Detection], str]:
        import cv2
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        return self.detect_pil(pil, prompt, max_new_tokens)
