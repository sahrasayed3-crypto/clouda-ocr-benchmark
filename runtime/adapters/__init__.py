"""Registry for all lazy local-model adapters."""
from pathlib import Path

from .ain import AINAdapter
from .arabic_nougat import ArabicNougatAdapter
from .deepseek_ocr_2 import DeepSeekOCR2Adapter
from .dots_mocr import DotsMOCRAdapter
from .hunyuanocr import HunyuanOCRAdapter
from .paddleocr_vl import PaddleOCRVLAdapter
from .qari import QariAdapter
from .qwen3_vl_4b import Qwen3VLAdapter

ROOT = Path.home() / "ocr_models"
ADAPTERS = {
    "qari": (QariAdapter, {"model_path": ROOT/"Qari-OCR-0.4.0", "base_model_path": ROOT/"Qwen3-VL-4B-Instruct"}),
    "qwen3_vl_4b": (Qwen3VLAdapter, {"model_path": ROOT/"Qwen3-VL-4B-Instruct"}),
    "arabic_nougat": (ArabicNougatAdapter, {"model_path": ROOT/"Arabic-Nougat-Large"}),
    "ain": (AINAdapter, {"model_path": ROOT/"AIN-7B"}),
    "paddleocr_vl": (PaddleOCRVLAdapter, {"model_path": ROOT/"PaddleOCR-VL-1.6"}),
    "hunyuanocr": (HunyuanOCRAdapter, {"model_path": ROOT/"HunyuanOCR-1.5"}),
    "deepseek_ocr_2": (DeepSeekOCR2Adapter, {"model_path": ROOT/"DeepSeek-OCR-2"}),
    "dots_mocr": (DotsMOCRAdapter, {"model_path": ROOT/"DotsMOCR"}),
}


def create_adapter(name: str):
    if name not in ADAPTERS: raise KeyError(f"unknown model {name!r}; choose from {sorted(ADAPTERS)}")
    cls, kwargs = ADAPTERS[name]
    return cls(**kwargs)

