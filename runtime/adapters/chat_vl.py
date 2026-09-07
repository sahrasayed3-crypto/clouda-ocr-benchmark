"""Official Transformers chat-template adapter shared by Qwen-like VLMs."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter

from .base import BaseAdapter, InferenceRequest, InferenceResult


class ChatVLAdapter(BaseAdapter):
    model_class = "AutoModelForImageTextToText"
    trust_remote_code = False
    supports_batching = True
    processor_kwargs = {}
    processor_call_kwargs = {}

    def load(self, *, dtype: str = "bfloat16", attn_implementation: str = "sdpa") -> None:
        import torch
        import transformers

        cls = getattr(transformers, self.model_class)
        self.processor = transformers.AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code, local_files_only=True,
            **self.processor_kwargs,
        )
        self.model = cls.from_pretrained(
            self.model_path, torch_dtype=self._torch_dtype(torch, dtype), device_map="auto",
            attn_implementation=attn_implementation, trust_remote_code=self.trust_remote_code,
            local_files_only=True, use_safetensors=True,
        ).eval()

    def _prepare_chat_inputs(self, messages):
        return self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            processor_kwargs=deepcopy(self.processor_call_kwargs),
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        import torch
        if self.model is None or self.processor is None:
            raise RuntimeError("adapter is not loaded")
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(request.image_path)},
                {"type": "text", "text": request.prompt},
            ]}]
            inputs = self._prepare_chat_inputs(messages)
            inputs = inputs.to(self.model.device)
            preprocess = perf_counter() - started
            generation = {"max_new_tokens": 2048, "do_sample": False, **request.generation}
            inference_started = perf_counter()
            with torch.inference_mode():
                ids = self.model.generate(**inputs, **generation)
            inference = perf_counter() - inference_started
            post_started = perf_counter()
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, ids)]
            text = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0]
            post = perf_counter() - post_started
            return InferenceResult(text, text, inference, preprocess, post, self._peak_vram(torch))
        except Exception as exc:
            return InferenceResult(error=f"{type(exc).__name__}: {exc}")
