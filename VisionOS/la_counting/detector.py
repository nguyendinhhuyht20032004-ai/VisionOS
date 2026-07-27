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

        self.dtype = torch.float16

        # LỊCH SỬ (để không lặp lại đường đã thử):
        #   v1: bfloat16 toàn model  → device-side assert ngẫu nhiên ở Conv2d/Linear/RoPE.
        #   v2: float32 toàn model   → OOM (3B×4B ≈ 12GB weight, T4 chỉ ~15GB khả dụng).
        #   v3: float16 + monkey-patch nan_to_num từng Linear/DecoderLayer/MLP, upcast
        #       Q/K/V float32 NGAY TRƯỚC KHI GỌI torch SDPA gốc → VẪN crash đúng
        #       CUBLAS_STATUS_EXECUTION_FAILED tại chính lời gọi SDPA đó. Kết luận:
        #       đây KHÔNG PHẢI lỗi tràn số float16 (nếu vậy upcast float32 đã hết lỗi).
        #       Nhiều khả năng hơn: (a) kernel SDPA "hợp nhất" (fused) mà PyTorch chọn
        #       trên kiến trúc Turing có bug/không tương thích với 1 hình dạng cụ thể
        #       của attention_mask 4D tuỳ biến trong model này, hoặc (b) 1 lỗi index
        #       out-of-bounds xảy ra Ở KERNEL TRƯỚC ĐÓ làm hỏng context CUDA, khiến MỌI
        #       lệnh cuBLAS sau đó (kể cả không liên quan) đều báo lỗi chung chung —
        #       CUDA vốn chạy bất đồng bộ nên traceback hay trỏ nhầm thủ phạm.
        # v4 (bản này): THAY VÌ vá tiếp bên trong SDPA, NÉ HẲN con đường kernel SDPA
        #   hợp nhất bằng attn_implementation="eager" — cách chuẩn của HF khi phần
        #   cứng/kernel không tương thích SDPA (dùng matmul/softmax tường minh, dễ
        #   debug, không phụ thuộc heuristic chọn kernel của cuBLAS/cuDNN). Đồng thời
        #   thêm CUDA_LAUNCH_BLOCKING=1 (đặt ở run_eval.py, trước khi torch được nạp)
        #   để nếu VẪN lỗi thì traceback trỏ đúng dòng thật, không còn đoán mò.
        if torch.cuda.is_available() and not self._cpu_debug:
            cc = torch.cuda.get_device_capability()
            if cc[0] < 8:  # Turing (T4), Volta, Pascal...
                print(f"⚠️  GPU cc={cc[0]}.{cc[1]} (<8.0) → mặc định dùng attn_implementation=eager "
                      "(SDPA hợp nhất từng crash CUBLAS_STATUS_EXECUTION_FAILED trên kiến trúc này).")
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)

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
        # THỰC TẾ (đối chiếu trực tiếp modeling_qwen2.py trên HF): bundled model có
        # 4 lựa chọn attn_implementation — eager / flash_attention_2 / sdpa / magi
        # (mặc định gốc của model là "magi", cần gói magi_attention không có sẵn
        # trên Kaggle). "sdpa" là lựa chọn AN TOÀN nhất khi flash-attn không có
        # (T4/Turing cũng không được flash-attn 2 hỗ trợ tốt). Cho phép ép tay qua
        # LA_ATTN=eager để CHẨN ĐOÁN: "eager" có raise ValueError rõ ràng khi kích
        # model tự tính position_ids phù hợp với RoPE cache của nó.
        # MẶC ĐỊNH ĐỔI sang "eager" (xem lý do ở khối comment "v4" phía trên) — SDPA
        # đã crash lặp lại nhiều lần trên T4. Cho phép ép tay: LA_ATTN=sdpa để quay
        # lại đường cũ (đối chiếu), hoặc flash_attention_2 nếu gói có sẵn.
        attn_impl = os.environ.get("LA_ATTN", "eager")
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            config=config,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation=attn_impl,
        )
        if torch.cuda.is_available() and not self._cpu_debug:
            self.model = self.model.to("cuda:0")
            
        # =========================================================================
        # CHUỖI VÁ SÂU DƯỚI ĐÂY (global SDPA / Linear / DecoderLayer / MLP) chỉ còn
        # ý nghĩa khi ĐANG DÙNG attn_implementation="sdpa" (đường cũ, giữ lại để đối
        # chiếu qua LA_ATTN=sdpa). Với "eager" (mặc định mới), model không gọi
        # F.scaled_dot_product_attention nữa → patch vô nghĩa, CHỦ ĐỘNG BỎ QUA để
        # giảm bề mặt lỗi (ít code tự viết chen vào forward gốc của model hơn).
        if torch.cuda.is_available() and not self._cpu_debug and cc[0] < 8 and attn_impl == "sdpa":
            if not getattr(torch.nn.functional, "_la_patched_sdpa", False):
                orig_sdpa = torch.nn.functional.scaled_dot_product_attention
                def safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
                    # Turing (T4) cuBLAS SgemmStridedBatched crash khi xử lý float16
                    # với attention mask 4D tuỳ biến. Giải pháp: upcast sang float32
                    # để dùng cuBLAS float32 GEMM (ổn định 100% trên mọi GPU).
                    orig_dtype = query.dtype
                    if orig_dtype == torch.float16:
                        query = query.to(torch.float32)
                        key = key.to(torch.float32)
                        value = value.to(torch.float32)
                        if attn_mask is not None:
                            attn_mask = attn_mask.to(torch.float32)
                    out = orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
                    if out.dtype != orig_dtype:
                        out = out.to(orig_dtype)
                    return out
                torch.nn.functional.scaled_dot_product_attention = safe_sdpa
                torch.nn.functional._la_patched_sdpa = True
                print("🔧 Patched Global SDPA: float16→float32 upcast on Turing GPU (avoids cuBLAS crash)")

            # All Linear/MLP/lm_head nan_to_num patches removed because they corrupt the model's spatial reasoning.
        # Patch RotaryEmbedding để không bị index out of bounds khi model xài custom position_ids
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "model"):
            try:
                rotary_emb_cls = type(self.model.language_model.model.layers[0].self_attn.rotary_emb)
                if not getattr(rotary_emb_cls, "_la_patched", False):
                    _orig_rotary_forward = rotary_emb_cls.forward
                    def _safe_rotary_forward(self_emb, x, seq_len=None):
                        if seq_len is not None and seq_len > self_emb.max_seq_len_cached:
                            self_emb._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
                        # Trả về toàn bộ cache thay vì truncate tới seq_len, vì seq_len có thể nhỏ hơn max(position_ids)
                        return (
                            self_emb.cos_cached.to(dtype=x.dtype),
                            self_emb.sin_cached.to(dtype=x.dtype),
                        )
                    rotary_emb_cls.forward = _safe_rotary_forward
                    rotary_emb_cls._la_patched = True
                    print("🔧 Patched Qwen2RotaryEmbedding (disabled seq_len truncation)")
            except Exception as e:
                print(f"⚠️ Could not patch RotaryEmbedding: {e}")

        # =========================================================================

        self.model.eval()
        self._loaded = True
        print(f"✅ Loaded in {time.time() - t0:.1f}s (device={self.model.device}, attn={attn_impl})")
        if torch.cuda.is_available() and not self._cpu_debug:
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU Mem: allocated={alloc:.1f}GB reserved={reserved:.1f}GB / total={total:.1f}GB")
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

        
        # generate() TÙY BIẾN của model: cần generation_mode="hybrid" (mặc định của
        # NVIDIA — Parallel Box Decoding). repetition_penalty chặn lặp box. Một số
        # kwargs có thể không được nhận ở bản generate này → thử rồi rút gọn dần.
        base_full = dict(**inputs, tokenizer=self.tokenizer, max_new_tokens=max_tok, use_cache=True, generation_mode="hybrid", repetition_penalty=1.2)
        base_simple = dict(**inputs, tokenizer=self.tokenizer, max_new_tokens=max_tok, use_cache=True)
        attempts = [ base_full, base_simple ]
        output, last_err = None, None
        with torch.no_grad():
            for kw in attempts:
                try:
                    output = self.model.generate(**kw)
                    print(f"🔧 [DEBUG] generate() succeeded with keys: {list(kw.keys())}")
                    break
                except TypeError as e:  # kwarg không được hỗ trợ → thử rút gọn hơn
                    print(f"🔧 [DEBUG] TypeError with keys {list(kw.keys())}: {e}")
                    last_err = e
                    continue
                except RuntimeError as e:
                    # CUBLAS_STATUS_EXECUTION_FAILED thường bị nhầm là bug kernel,
                    # nhưng cũng CÓ THỂ chỉ là OOM đội lốt (cuBLAS không cấp phát
                    # được workspace khi VRAM gần đầy). In rõ số liệu để phân biệt
                    # NGAY LÚC CRASH thay vì đoán mò sau đó.
                    if torch.cuda.is_available():
                        try:
                            alloc = torch.cuda.memory_allocated() / 1024**3
                            reserved = torch.cuda.memory_reserved() / 1024**3
                            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            free_frac = 1 - reserved / total
                            print("=" * 74)
                            print(f"💥 RuntimeError trong generate(): {e}")
                            print(f"   GPU Mem lúc crash: allocated={alloc:.1f}GB reserved={reserved:.1f}GB "
                                  f"/ total={total:.1f}GB (còn trống ước ~{free_frac*100:.0f}%)")
                            if free_frac < 0.08:
                                print("   ⚠️  VRAM gần cạn lúc crash → NHIỀU KHẢ NĂNG là OOM đội lốt cuBLAS,"
                                      " không phải lỗi kernel. Thử giảm --n, giảm max_new_tokens, hoặc ảnh"
                                      " nhỏ hơn.")
                            else:
                                print("   ℹ️  VRAM còn nhiều lúc crash → khó là OOM thuần tuý. Xem traceback"
                                      " phía trên (CUDA_LAUNCH_BLOCKING=1 đã bật → dòng cuối là thủ phạm thật).")
                            print("=" * 74, flush=True)
                        except Exception:
                            pass
                    raise
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
