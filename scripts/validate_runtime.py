#!/usr/bin/env python3
"""CPU-only runtime validation; never instantiates a model or reads tensor payloads."""
from __future__ import annotations

import argparse
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def check(name: str, fn, results: list[dict]) -> None:
    try:
        detail = fn()
        results.append({"check": name, "status": "pass", "detail": detail})
    except Exception as exc:
        results.append({"check": name, "status": "fail", "detail": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=ROOT/"reports/cpu_validation.json")
    args = parser.parse_args()
    results: list[dict] = []
    models = yaml.safe_load((ROOT/"configs/models.yaml").read_text(encoding="utf-8"))["models"]
    prompts = yaml.safe_load((ROOT/"configs/prompts.yaml").read_text(encoding="utf-8"))

    check("eight_model_configs", lambda: len(models) if len(models) == 8 else (_ for _ in ()).throw(ValueError(len(models))), results)
    check("prompt_config", lambda: sorted(prompts), results)

    def paths():
        missing = []
        for name, spec in models.items():
            for role, raw in spec["paths"].items():
                path = Path(os.path.expanduser(raw))
                if not path.exists(): missing.append(f"{name}.{role}: {path}")
        if missing: raise FileNotFoundError("; ".join(missing))
        return "all local model paths resolve"
    check("model_paths", paths, results)

    manifest = Path.home()/"ocr_benchmark/manifests/benchmark_manifest.jsonl"
    def manifest_check():
        rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(rows) != 177: raise ValueError(f"expected 177 benchmark rows, found {len(rows)}")
        if len({r["distorted_id"] for r in rows}) != len(rows): raise ValueError("duplicate distorted_id")
        return {"rows": len(rows), "unique_ids": len(rows)}
    check("benchmark_manifest", manifest_check, results)

    def configs_present():
        required_any = ("processor_config.json", "preprocessor_config.json", "image_processor_config.json")
        detail = {}
        for name, spec in models.items():
            model_path = Path(os.path.expanduser(spec["paths"].get("model", spec["paths"].get("adapter"))))
            has_tokenizer = any(model_path.glob("tokenizer*")) or (model_path/"sentencepiece.bpe.model").is_file()
            has_processor = any((model_path/item).is_file() for item in required_any)
            if not has_tokenizer or not has_processor:
                raise FileNotFoundError(f"{name}: tokenizer={has_tokenizer}, processor={has_processor}")
            detail[name] = {"tokenizer": True, "processor": True}
        return detail
    check("tokenizer_processor_metadata", configs_present, results)

    def adapter_imports():
        sys.path.insert(0, str(ROOT))
        from adapters import ADAPTERS, create_adapter
        for name in ADAPTERS:
            adapter = create_adapter(name)
            if adapter.model is not None or adapter.processor is not None: raise RuntimeError(name)
        return sorted(ADAPTERS)
    check("lazy_adapter_imports", adapter_imports, results)

    def compile_code():
        count = 0
        for base in (ROOT/"adapters", ROOT/"runtime", ROOT/"scripts"):
            for path in base.glob("*.py"):
                py_compile.compile(str(path), doraise=True); count += 1
        for model_dir in (Path.home()/"ocr_models/DeepSeek-OCR-2", Path.home()/"ocr_models/DotsMOCR", Path.home()/"ocr_models/PaddleOCR-VL-1.6"):
            for path in model_dir.glob("*.py"):
                py_compile.compile(str(path), doraise=True); count += 1
        return {"compiled_files": count}
    check("python_syntax", compile_code, results)

    env_imports = {
        "modern_vlm": "import torch,transformers,accelerate,peft,yaml,PIL; assert torch.cuda.is_available() is False",
        "legacy_ocr": "import torch,transformers,accelerate,qwen_vl_utils,flash_attn,yaml,PIL; assert flash_attn.__version__ == '2.7.3'; assert torch.cuda.is_available() is False",
        "dots_mocr": "import torch,transformers,accelerate,qwen_vl_utils,flash_attn,yaml,PIL; assert flash_attn.__version__ == '2.8.0.post2'; assert torch.cuda.is_available() is False",
        "paddle_native": "import paddle,paddleocr; assert paddle.device.is_compiled_with_cuda() is False",
    }
    for env_name, code in env_imports.items():
        def env_check(env_name=env_name, code=code):
            proc = subprocess.run([str(ROOT/f"envs/{env_name}/bin/python"), "-c", code], capture_output=True, text=True, timeout=120)
            if proc.returncode: raise RuntimeError((proc.stderr or proc.stdout).strip())
            return "imports pass; no CUDA device/workload"
        check(f"environment_imports.{env_name}", env_check, results)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"cpu_only": True, "model_instantiation": False,
              "status": "pass" if all(x["status"] == "pass" for x in results) else "fail",
              "checks": results}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
