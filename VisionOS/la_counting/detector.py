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
        import os

        import torch
        import transformers
        from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

        self._torch = torch
        # LA_DEBUG_CPU=1 → nạp trên CPU (float32). CUDA "device-side assert" rất khó
        # đọc (báo bất đồng bộ, trỏ nhầm dòng); chạy CPU biến nó thành lỗi Python RÕ
        # RÀNG (vd IndexError: index 151655 out of bounds) → chỉ đúng thủ phạm.
        self._cpu_debug = bool(os.environ.get("LA_DEBUG_CPU"))
        # DTYPE: model được huấn luyện ở bfloat16 (Qwen2/NVIDIA).
        # float16 (dải hẹp, max ~65504) dễ TRÀN ở MLP → CUBLAS_STATUS_INTERNAL_ERROR.
        # float32 an toàn nhưng 3B×4B = ~12GB weights + activations → OOM trên T4 (15GB).
        # → Giữ bf16 (~7GB, vừa T4) + ép SDPA math backend trên T4 (tự upcast f32
        #   nội bộ khi tính attention, tránh CUBLAS crash). Cho phép LA_DTYPE ép tay.
        env_dtype = (os.environ.get("LA_DTYPE") or "").lower()
        if self._cpu_debug or not torch.cuda.is_available():
            self.dtype = torch.float32
        elif env_dtype in ("float16", "fp16", "half"):
            self.dtype = torch.float16
        elif env_dtype in ("float32", "fp32"):
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16  # mặc định: bf16 (dtype gốc của model)

        # T4 (Turing, cc 7.5): SDPA flash/mem_efficient backend CRASH với bf16 vì
        # T4 không có tensor-core bf16 gốc. Gọi math_sdp với float32 thì bị OOM/bug cuBLAS.
        # Giải pháp TỐT NHẤT: Dùng float16 + mem_efficient_attention.
        # LƯU Ý: Phải gọi .contiguous() trên q, k, v vì mem_efficient_attention trên
        # PyTorch 2.1 bị lỗi device-side assert nếu input không liền mạch (non-contiguous)
        # khi đi kèm với attention_mask.
        if torch.cuda.is_available() and not self._cpu_debug:
            cc = torch.cuda.get_device_capability()
            if cc[0] < 8:  # Turing (T4), Volta, Pascal...
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(True) # Bật xformers/cutlass fp16
                torch.backends.cuda.enable_math_sdp(True)
                
                if self.dtype == torch.bfloat16:
                    import torch.nn.functional as F
                    if not getattr(F, "_la_patched_sdpa", False):
                        _orig_sdpa = F.scaled_dot_product_attention
                        def _ultimate_sdpa(*args, **kwargs):
                            if len(args) >= 3 and torch.is_tensor(args[0]) and args[0].dtype == torch.bfloat16:
                                original_ndim = args[0].ndim
                                q = args[0]
                                k = args[1]
                                v = args[2]
                                
                                if original_ndim == 3:
                                    q = q.unsqueeze(1)
                                    k = k.unsqueeze(1)
                                    v = v.unsqueeze(1)
                                
                                # Cực kỳ quan trọng: phải .contiguous() TẤT CẢ tensor để tránh device-side assert
                                q = q.to(torch.float16).contiguous()
                                k = k.to(torch.float16).contiguous()
                                v = v.to(torch.float16).contiguous()
                                
                                new_args = [q, k, v] + list(args[3:])
                                
                                for i in range(3, len(new_args)):
                                    if torch.is_tensor(new_args[i]):
                                        if new_args[i].dtype == torch.bfloat16:
                                            new_args[i] = new_args[i].to(torch.float16).contiguous()
                                        else:
                                            new_args[i] = new_args[i].contiguous()
                                            
                                for key, val in kwargs.items():
                                    if torch.is_tensor(val):
                                        if val.dtype == torch.bfloat16:
                                            kwargs[key] = val.to(torch.float16).contiguous()
                                        else:
                                            kwargs[key] = val.contiguous()
                                
                                out = _orig_sdpa(*new_args, **kwargs)
                                
                                if original_ndim == 3:
                                    out = out.squeeze(1)
                                    
                                return out.to(torch.bfloat16)
                            return _orig_sdpa(*args, **kwargs)
                        F.scaled_dot_product_attention = _ultimate_sdpa
                        F._la_patched_sdpa = True
                print(f"⚠️  GPU cc={cc[0]}.{cc[1]} (<8.0) → Monkey-patch SDPA: "
                      f"Cast bf16 -> float16 + contiguous() để chạy mem_efficient_attention mượt mà.")

        # In RÕ phiên bản để hết đoán mò: model được NVIDIA test với transformers
        # 4.57.1. Bản khác vẫn chạy nhờ các bản vá độc lập phiên bản, nhưng biết
        # đúng phiên bản giúp chẩn đoán nhanh khi có sự cố.
        print(f"🔧 transformers=={transformers.__version__} · torch=={torch.__version__} "
              f"· CUDA {torch.cuda.is_available()} · dtype={self.dtype}"
              + ("  ⚙️ LA_DEBUG_CPU=1 (chạy CPU để lấy lỗi rõ ràng)" if self._cpu_debug else ""))

        # CHỐT PHIÊN BẢN: model chỉ tương thích transformers 4.57.x. Trên 5.x, dù
        # nạp được (nhờ các bản vá), attention/rotary vẫn sinh "device-side assert"
        # khó đọc lúc generate. Thà DỪNG NGAY với thông báo rõ còn hơn để user bơi
        # trong CUDA assert. (auto-pin trong run_eval lẽ ra đã đưa về 4.57.1.)
        if not transformers.__version__.startswith("4.57"):
            raise RuntimeError(
                "\n" + "=" * 72 + "\n"
                f"❌ Đang chạy transformers=={transformers.__version__} — model CẦN 4.57.x.\n"
                "   Auto-pin trong run_eval đã KHÔNG đưa được về 4.57.1 (pip bị chặn,\n"
                "   hoặc Kaggle giữ 2 bản transformers). Chạy 1 cell rồi chạy lại:\n"
                "     !pip install --force-reinstall --no-deps 'transformers==4.57.1'\n"
                "     !python -c \"import transformers; print(transformers.__version__)\"\n"
                "   (phải in ra 4.57.1). KHÔNG cần restart kernel — run_eval là tiến trình riêng.\n"
                + "=" * 72
            )

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
            # Bundled model CHỈ implement SDPA attention (không có eager/flash).
            # Crash CUBLAS trên T4 được fix bằng dtype=float32 (ở trên).
            attn_implementation="sdpa",
        )
        if torch.cuda.is_available() and not self._cpu_debug:
            self.model = self.model.to("cuda:0")
        self.model.eval()
        self._loaded = True
        print(f"✅ Loaded in {time.time() - t0:.1f}s (device={self.model.device})")
        if torch.cuda.is_available() and not self._cpu_debug:
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
                return v.to(device=self.model.device, dtype=self.dtype)
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

        # ------------------------------------------------------------------ #
        # Dựng input ĐÚNG cách của LocateAnything (theo NVlabs/Eagle worker).
        # LocateAnything có PROCESSOR RIÊNG (processing_locateanything.py) với:
        #   * ``py_apply_chat_template`` — chèn đúng token ảnh <IMG_CONTEXT> theo
        #     grid patch của ẢNH (KHÔNG phải ``apply_chat_template`` generic!),
        #   * ``process_vision_info`` — tách ảnh/video ra khỏi messages.
        # Dùng nhầm apply_chat_template generic → số token ảnh KHÔNG khớp số patch
        # → input_ids/position_ids lệch → "vectorized_gather_kernel index out of
        # bounds" (CUDA assert) đúng như lỗi đã gặp. Ảnh phải NẰM TRONG message.
        # ------------------------------------------------------------------ #
        instruction = (
            f"Locate all the instances that match the following description: {prompt}."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        proc = self.processor
        if hasattr(proc, "py_apply_chat_template"):
            text = proc.py_apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # text phải là LIST; ảnh lấy qua process_vision_info nếu có.
        if hasattr(proc, "process_vision_info"):
            images, videos = proc.process_vision_info(messages)
            inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt")
        else:
            inputs = proc(text=[text], images=[pil_image], return_tensors="pt")
        inputs = {k: self._prep_input(v) for k, v in inputs.items()}

        # KHÔNG truyền position_ids vào generate(): processor tạo position_ids dựa
        # trên toàn bộ input (ảnh + text) nhưng bundled model tính RoPE cos/sin cache
        # nội bộ với max_position_embeddings NHỎ hơn → index out of bounds. Bỏ để
        # model tự tính position_ids phù hợp với RoPE cache của nó.
        inputs.pop("position_ids", None)

        # generate() TÙY BIẾN của model: cần generation_mode="hybrid" (mặc định của
        # NVIDIA — Parallel Box Decoding). repetition_penalty chặn lặp box. Một số
        # kwargs có thể không được nhận ở bản generate này → thử rồi rút gọn dần.
        base = dict(**inputs, tokenizer=self.tokenizer, max_new_tokens=max_tok, use_cache=True)
        attempts = [
            dict(base, generation_mode="hybrid", do_sample=False, repetition_penalty=1.05),
            dict(base, generation_mode="hybrid", do_sample=False),
            dict(base, generation_mode="hybrid"),
            base,
        ]
        output, last_err = None, None
        with torch.no_grad():
            for kw in attempts:
                try:
                    output = self.model.generate(**kw)
                    break
                except TypeError as e:  # kwarg không được hỗ trợ → thử bộ gọn hơn
                    last_err = e
                    continue
        if output is None:
            raise last_err if last_err else RuntimeError("generate() thất bại")

        # generate có thể trả token ids (tensor / list) hoặc chuỗi đã decode.
        if isinstance(output, str):
            raw = output
        elif isinstance(output, (list, tuple)) and output and hasattr(output[0], "shape"):
            raw = self.tokenizer.decode(output[0], skip_special_tokens=True)
        elif hasattr(output, "shape"):
            seq = output[0] if output.dim() > 1 else output
            raw = self.tokenizer.decode(seq, skip_special_tokens=True)
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
