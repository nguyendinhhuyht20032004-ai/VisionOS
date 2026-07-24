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

        # Compat shim cho transformers MỚI hơn 4.57.1 (bản NVIDIA test model).
        # Model bundle sẵn modeling_qwen2.py / modeling_locateanything.py viết cho
        # 4.57.1; bản mới đổi cơ chế "tied weights" nên vỡ ở 2 chỗ:
        #   (a) get_expanded_tied_weights_keys: bundled đưa _tied_weights_keys dạng
        #       LIST (format cũ) nhưng bản mới coi là DICT → 'list' has no 'keys'.
        #   (b) all_tied_weights_keys: bản mới đặt trong post_init(), nhưng model
        #       composite (LocateAnythingForConditionalGeneration) bỏ qua post_init
        #       mới → thiếu attribute khi device_map suy luận thiết bị.
        # Cả hai vá đều VÔ HẠI cho suy luận: tie_weights() thật vẫn chạy khi load;
        # 2 thứ trên chỉ là metadata phục vụ device_map/checkpoint. Idempotent nhờ
        # cờ _locate_patched.
        try:
            import transformers.modeling_utils as _tmu
            _PTM = _tmu.PreTrainedModel
            if not getattr(_PTM, "_locate_patched", False):
                _orig = _PTM.get_expanded_tied_weights_keys

                def _safe(self, all_submodels=True):
                    try:
                        return _orig(self, all_submodels=all_submodels)
                    except (AttributeError, TypeError):
                        return {}

                _PTM.get_expanded_tied_weights_keys = _safe
                # default cấp lớp: instance nào không set (bỏ qua post_init mới) sẽ
                # đọc dict rỗng → len()==0 → "không có tied weight" → không lỗi.
                if not hasattr(_PTM, "all_tied_weights_keys"):
                    _PTM.all_tied_weights_keys = {}
                _PTM._locate_patched = True
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
        # ÉP TOÀN MODEL LÊN 1 GPU. Kaggle "GPU T4 x2" có cuda:0 + cuda:1; nếu để
        # device_map="auto" thì accelerate CHIA model ra 2 GPU, mà generate() bundled
        # của model tự .to()/cat tensor giả định 1 thiết bị → lỗi "tensors on cuda:1
        # different from cuda:0". Model 3B float16 (~6GB) thừa sức nằm gọn 1 T4 (16GB).
        # Nạp phẳng (không device_map, không hook accelerate) rồi .to() cho tương thích
        # tối đa với generate() quản lý thiết bị thủ công của model.
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            config=config,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",  # ép SDPA cho T4
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda:0")
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
