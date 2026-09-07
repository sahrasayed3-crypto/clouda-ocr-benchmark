from __future__ import annotations

from time import perf_counter

from .base import BaseAdapter, InferenceRequest, InferenceResult


class ArabicNougatAdapter(BaseAdapter):
    name = "arabic_nougat"
    supports_batching = True

    def load(self, *, dtype: str = "bfloat16", attn_implementation: str = "eager") -> None:
        import torch
        from transformers import NougatProcessor, VisionEncoderDecoderModel
        self.processor = NougatProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.model_path, torch_dtype=self._torch_dtype(torch, dtype),
            attn_implementation={"decoder": attn_implementation, "encoder": "eager"},
            local_files_only=True, use_safetensors=True,
        ).to("cuda").eval()

    def infer(self, request: InferenceRequest) -> InferenceResult:
        import torch
        from PIL import Image
        if self.model is None: raise RuntimeError("adapter is not loaded")
        try:
            torch.cuda.reset_peak_memory_stats(); t0 = perf_counter()
            pixels = self.processor(Image.open(request.image_path).convert("RGB"), return_tensors="pt").pixel_values
            pixels = pixels.to(self.model.dtype).to(self.model.device); prep = perf_counter() - t0
            gen = {"repetition_penalty": 1.5, "min_length": 1,
                   "max_new_tokens": self.model.decoder.config.max_position_embeddings,
                   "bad_words_ids": [[self.processor.tokenizer.unk_token_id]], **request.generation}
            t1 = perf_counter()
            with torch.inference_mode(): output = self.model.generate(pixels, **gen)
            infer = perf_counter() - t1; t2 = perf_counter()
            text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            return InferenceResult(text, text, infer, prep, perf_counter()-t2, self._peak_vram(torch))
        except Exception as exc: return InferenceResult(error=f"{type(exc).__name__}: {exc}")

