#!/usr/bin/env python3
"""Validate local model artifacts without importing ML frameworks or tensors."""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any


MODELS = {
    "qari": {"dir": "Qari-OCR-0.4.0", "config": "adapter_config.json", "base": "Qwen3-VL-4B-Instruct"},
    "qwen3_vl_4b": {"dir": "Qwen3-VL-4B-Instruct", "config": "config.json"},
    "arabic_nougat": {"dir": "Arabic-Nougat-Large", "config": "config.json"},
    "ain": {"dir": "AIN-7B", "config": "config.json"},
    "paddleocr_vl": {"dir": "PaddleOCR-VL-1.6", "config": "config.json"},
    "hunyuanocr": {"dir": "HunyuanOCR-1.5", "config": "config.json"},
    "deepseek_ocr_2": {"dir": "DeepSeek-OCR-2", "config": "config.json"},
    "dots_mocr": {"dir": "DotsMOCR", "config": "config.json"},
}


def inspect_safetensors(path: Path) -> dict[str, Any]:
    result = {"valid": False, "tensor_count": 0, "data_bytes": 0, "error": None}
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            prefix = fh.read(8)
            if len(prefix) != 8:
                raise ValueError("missing 8-byte header length")
            header_len = struct.unpack("<Q", prefix)[0]
            if header_len < 2 or header_len > size - 8:
                raise ValueError(f"invalid header length {header_len} for {size}-byte file")
            header = json.loads(fh.read(header_len))
        entries = [(k, v) for k, v in header.items() if k != "__metadata__"]
        data_capacity = size - 8 - header_len
        occupied: list[tuple[int, int, str]] = []
        for name, item in entries:
            if not isinstance(item, dict) or "data_offsets" not in item:
                raise ValueError(f"tensor {name!r} lacks data_offsets")
            start, end = item["data_offsets"]
            if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= data_capacity):
                raise ValueError(f"tensor {name!r} has out-of-range offsets {start}:{end}")
            occupied.append((start, end, name))
        occupied.sort()
        for left, right in zip(occupied, occupied[1:]):
            if left[1] > right[0]:
                raise ValueError(f"overlapping tensor ranges: {left[2]} and {right[2]}")
        result.update(valid=True, tensor_count=len(entries),
                      data_bytes=sum(end - start for start, end, _ in occupied))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def validate_index(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"valid": False, "referenced_shards": [], "missing_shards": [], "error": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("missing non-empty weight_map")
        shards = sorted(set(weight_map.values()))
        missing = [name for name in shards if not (path.parent / name).is_file()]
        result.update(valid=not missing, referenced_shards=shards, missing_shards=missing)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _presence(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    present = [name for name in names if (path / name).is_file()]
    return {"present": bool(present), "files": present}


def inspect_model(name: str, spec: dict[str, str], models_root: Path) -> dict[str, Any]:
    path = models_root / spec["dir"]
    metadata_files = sorted((path / ".cache/huggingface/download").rglob("*.metadata"))
    revisions = sorted({p.read_text(encoding="utf-8").splitlines()[0]
                        for p in metadata_files if p.read_text(encoding="utf-8").splitlines()})
    weights = sorted(path.rglob("*.safetensors"))
    integrity = {str(p.relative_to(path)): inspect_safetensors(p) for p in weights}
    indexes = sorted(path.rglob("*.safetensors.index.json"))
    index_results = {str(p.relative_to(path)): validate_index(p) for p in indexes}
    incomplete = [p for p in path.rglob("*") if p.is_file() and
                  p.name.endswith((".incomplete", ".part", ".download", ".tmp"))]
    zero_byte = [p for p in path.rglob("*") if p.is_file() and p.stat().st_size == 0
                 and ".cache/huggingface/download" not in str(p)]
    config = _presence(path, (spec["config"],))
    tokenizer = _presence(path, ("tokenizer.json", "tokenizer.model", "vocab.json"))
    processor = _presence(path, ("processor_config.json", "preprocessor_config.json"))
    generation = _presence(path, ("generation_config.json",))
    notes: list[str] = []
    if name == "qari":
        adapter_cfg = json.loads((path / "adapter_config.json").read_text()) if config["present"] else {}
        expected = models_root / spec["base"]
        relation_ok = expected.is_dir() and adapter_cfg.get("base_model_name_or_path", "").endswith("Qwen3-VL-4B-Instruct")
        notes.append(f"Qari PEFT base relationship: {'valid' if relation_ok else 'invalid'}; local base={expected}")
    else:
        relation_ok = True
    status = "ok" if (path.is_dir() and weights and config["present"] and tokenizer["present"] and
                      processor["present"] and not incomplete and relation_ok and
                      all(v["valid"] for v in integrity.values()) and all(v["valid"] for v in index_results.values())) else "failed"
    if not generation["present"]:
        notes.append("generation_config.json absent; optional unless model-specific generation defaults are required")
    return {
        "model_name": name, "path": str(path),
        "repository_revision": revisions[0] if len(revisions) == 1 else None,
        "total_disk_size_bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
        "weight_files": [str(p.relative_to(path)) for p in weights],
        "weight_size_bytes": sum(p.stat().st_size for p in weights),
        "shard_count": len(weights), "index_files": index_results,
        "required_config": config, "tokenizer": tokenizer, "processor": processor,
        "generation_config": generation, "incomplete_file_count": len(incomplete),
        "incomplete_files": [str(p.relative_to(path)) for p in incomplete],
        "informational_zero_byte_files": [str(p.relative_to(path)) for p in zero_byte],
        "safetensors_integrity": integrity, "integrity_status": status, "notes": notes,
    }


def build_report(models_root: Path) -> dict[str, Any]:
    records = [inspect_model(name, spec, models_root) for name, spec in MODELS.items()]
    return {"format_version": 1, "validation_method": "metadata-only safetensors header/range/index validation",
            "models_root": str(models_root), "models": records,
            "all_ok": all(r["integrity_status"] == "ok" for r in records)}


if __name__ == "__main__":
    root = Path(os.environ.get("OCR_MODELS_ROOT", Path.home() / "ocr_models")).resolve()
    report = build_report(root)
    output = root / "model_integrity_report.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "all_ok": report["all_ok"],
                      "statuses": {r["model_name"]: r["integrity_status"] for r in report["models"]}}, indent=2))
