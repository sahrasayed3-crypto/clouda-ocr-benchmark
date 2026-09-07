from time import perf_counter

from .base import InferenceRequest, InferenceResult
from .chat_vl import ChatVLAdapter


def _ensure_default_rope_initializer():
    """Restore original RoPE initialization expected by the bundled model code."""
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def compute_default(config, device=None, seq_len=None, **kwargs):
        del seq_len, kwargs
        base = config.rope_theta
        partial = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS.setdefault("default", compute_default)
    return ROPE_INIT_FUNCTIONS["default"]


def _cache_position_for(input_ids, past_key_values=None):
    """Build the cache positions expected by the bundled generation hook."""
    import torch

    past_length = 0
    if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
        past_length = past_key_values.get_seq_length()
    return torch.arange(
        past_length,
        past_length + input_ids.shape[1],
        dtype=torch.long,
        device=input_ids.device,
    )


def _create_causal_mask_compat(mask_function, **kwargs):
    """Discard the deprecated cache_position kwarg when Transformers removed it."""
    import inspect

    if "cache_position" not in inspect.signature(mask_function).parameters:
        kwargs.pop("cache_position", None)
    return mask_function(**kwargs)


class PaddleOCRVLAdapter(ChatVLAdapter):
    name = "paddleocr_vl"
    model_class = "AutoModelForImageTextToText"
    trust_remote_code = True
    supports_batching = False

    def load(self, *, dtype: str = "bfloat16", attn_implementation: str = "sdpa") -> None:
        """Load the checkpoint's bundled implementation with its original RoPE semantics."""
        import torch
        import transformers
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        default_rope = _ensure_default_rope_initializer()
        model_path = str(self.model_path)
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
        model_class = get_class_from_dynamic_module(
            "modeling_paddleocr_vl.PaddleOCRVLForConditionalGeneration",
            model_path,
            local_files_only=True,
        )
        # Transformers v5's generic missing-key initializer calls this method
        # on repository RotaryEmbedding classes during final weight loading.
        import sys
        model_module = sys.modules[model_class.__module__]
        rotary_class = getattr(model_module, "RotaryEmbedding")
        if not hasattr(rotary_class, "compute_default_rope_parameters"):
            rotary_class.compute_default_rope_parameters = staticmethod(default_rope)
        original_prepare = model_class.prepare_inputs_for_generation
        if not getattr(original_prepare, "_ocr_cache_position_compat", False):
            def prepare_inputs_for_generation(model_self, input_ids, *args, cache_position=None, **kwargs):
                if cache_position is None:
                    cache_position = _cache_position_for(input_ids, kwargs.get("past_key_values"))
                return original_prepare(
                    model_self, input_ids, *args, cache_position=cache_position, **kwargs
                )

            prepare_inputs_for_generation._ocr_cache_position_compat = True
            model_class.prepare_inputs_for_generation = prepare_inputs_for_generation
        original_mask = model_module.create_causal_mask
        if not getattr(original_mask, "_ocr_mask_compat", False):
            def create_causal_mask(**kwargs):
                return _create_causal_mask_compat(original_mask, **kwargs)

            create_causal_mask._ocr_mask_compat = True
            model_module.create_causal_mask = create_causal_mask
        self.model = model_class.from_pretrained(
            model_path,
            torch_dtype=self._torch_dtype(torch, dtype),
            device_map="auto",
            attn_implementation=attn_implementation,
            local_files_only=True,
            use_safetensors=True,
        ).eval()

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Use the local repository's official Transformers OCR preprocessing."""
        import torch
        from PIL import Image

        if self.model is None or self.processor is None:
            raise RuntimeError("adapter is not loaded")
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            image = Image.open(request.image_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": request.prompt},
            ]}]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(self.model.device)
            preprocess = perf_counter() - started
            generation = {"max_new_tokens": 512, "do_sample": False, **request.generation}
            inference_started = perf_counter()
            with torch.inference_mode():
                output = self.model.generate(**inputs, **generation)
            inference = perf_counter() - inference_started
            post_started = perf_counter()
            # The repository example removes both the prompt and terminal token.
            generated = output[0][inputs["input_ids"].shape[-1]:-1]
            text = self.processor.decode(generated, skip_special_tokens=True)
            post = perf_counter() - post_started
            return InferenceResult(text, text, inference, preprocess, post, self._peak_vram(torch))
        except Exception as exc:
            return InferenceResult(error=f"{type(exc).__name__}: {exc}")
