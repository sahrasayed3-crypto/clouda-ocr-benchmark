"""Qari always composes the local LoRA adapter with the local Qwen base."""
from __future__ import annotations

from pathlib import Path

from .chat_vl import ChatVLAdapter


class QariAdapter(ChatVLAdapter):
    name = "qari"
    model_class = "AutoModelForImageTextToText"
    processor_call_kwargs = {"text_kwargs": {"truncation": False}}

    def __init__(self, model_path: Path, base_model_path: Path, **kwargs):
        super().__init__(model_path, **kwargs)
        self.base_model_path = base_model_path.resolve()

    def load(self, *, dtype: str = "float16", attn_implementation: str = "sdpa") -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        base = AutoModelForImageTextToText.from_pretrained(
            self.base_model_path, torch_dtype=self._torch_dtype(torch, dtype), device_map="auto",
            attn_implementation=attn_implementation, local_files_only=True, use_safetensors=True,
        )
        self.model = PeftModel.from_pretrained(base, self.model_path, local_files_only=True).eval()
        if not getattr(self.model, "peft_config", None):
            raise RuntimeError("Qari PEFT adapter was not attached to the local Qwen base")
