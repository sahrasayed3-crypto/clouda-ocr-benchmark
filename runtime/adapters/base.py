"""Common lazy adapter interface; importing this module never imports ML frameworks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass
class InferenceRequest:
    image_path: Path
    prompt: str
    generation: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    extracted_text: str = ""
    raw_response: Any = None
    elapsed_inference_time: float = 0.0
    preprocessing_time: float = 0.0
    postprocessing_time: float = 0.0
    peak_gpu_vram_bytes: int | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseAdapter:
    name = "base"
    supports_batching = False

    def __init__(self, model_path: Path, **_: Any) -> None:
        self.model_path = model_path.resolve()
        self.model = None
        self.processor = None

    def load(self, *, dtype: str = "bfloat16", attn_implementation: str = "sdpa") -> None:
        raise NotImplementedError

    def infer(self, request: InferenceRequest) -> InferenceResult:
        raise NotImplementedError

    @staticmethod
    def _torch_dtype(torch: Any, name: str) -> Any:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]

    @staticmethod
    def _peak_vram(torch: Any) -> int | None:
        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

