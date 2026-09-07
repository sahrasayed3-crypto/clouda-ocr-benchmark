from __future__ import annotations

from tempfile import TemporaryDirectory
from time import perf_counter

from .base import BaseAdapter, InferenceRequest, InferenceResult


class DeepSeekOCR2Adapter(BaseAdapter):
    name = "deepseek_ocr_2"

    def load(self, *, dtype: str = "bfloat16", attn_implementation: str = "flash_attention_2") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.processor = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            self.model_path, _attn_implementation=attn_implementation, trust_remote_code=True,
            use_safetensors=True, local_files_only=True,
        ).eval().cuda().to(self._torch_dtype(torch, dtype))

    def infer(self, request: InferenceRequest) -> InferenceResult:
        import torch
        if self.model is None: raise RuntimeError("adapter is not loaded")
        try:
            torch.cuda.reset_peak_memory_stats(); t0 = perf_counter()
            prompt = request.prompt if request.prompt.startswith("<image>") else "<image>\n" + request.prompt
            prep = perf_counter() - t0; t1 = perf_counter()
            kwargs = {"base_size": 1024, "image_size": 768, "crop_mode": True,
                      "save_results": False, "eval_mode": True, **request.generation}
            # The bundled implementation creates output_path unconditionally,
            # even when save_results=False. Use an isolated disposable directory.
            with TemporaryDirectory(prefix="deepseek_ocr2_") as output_path:
                raw = self.model.infer(
                    self.processor,
                    prompt=prompt,
                    image_file=str(request.image_path),
                    output_path=output_path,
                    **kwargs,
                )
            infer = perf_counter() - t1; t2 = perf_counter(); text = str(raw).strip()
            return InferenceResult(text, raw, infer, prep, perf_counter()-t2, self._peak_vram(torch))
        except Exception as exc: return InferenceResult(error=f"{type(exc).__name__}: {exc}")
