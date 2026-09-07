#!/usr/bin/env python3
"""Unified benchmark CLI. Use --dry-run on CPU; inference is an explicit action."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import yaml

from adapters import ADAPTERS, create_adapter
from adapters.base import InferenceRequest
from runtime.runner import completed_ids, load_manifest, result_record, select_samples


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    p.add_argument("--manifest", type=Path, default=Path.home()/"ocr_benchmark/manifests/benchmark_manifest.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int); p.add_argument("--sample-id")
    p.add_argument("--resume", action="store_true"); p.add_argument("--overwrite", action="store_true")
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260825); p.add_argument("--prompt", choices=("ocr", "layout"), default="ocr")
    p.add_argument("--dtype", default=None); p.add_argument("--attn-implementation", default=None)
    p.add_argument("--gpu-model", default="unknown"); p.add_argument("--gpu-vram-gb", type=float, default=0)
    p.add_argument("--dry-run", action="store_true", help="validate paths/config/selection without loading a model")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.batch_size < 1: raise ValueError("--batch-size must be at least 1")
    if args.warmup < 0: raise ValueError("--warmup cannot be negative")
    root = Path(__file__).resolve().parent
    prompts = yaml.safe_load((root/"configs/prompts.yaml").read_text(encoding="utf-8"))
    models = yaml.safe_load((root/"configs/models.yaml").read_text(encoding="utf-8"))["models"]
    rows = select_samples(load_manifest(args.manifest), sample_id=args.sample_id, limit=args.limit)
    spec = models[args.model]
    for path in spec["paths"].values():
        if not Path(os.path.expanduser(path)).exists(): raise FileNotFoundError(path)
    summary = {"model": args.model, "samples": len(rows), "batch_size": args.batch_size,
               "model_paths": spec["paths"], "prompt": args.prompt, "dry_run": args.dry_run}
    if args.dry_run:
        print(json.dumps(summary, indent=2)); return 0
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output/"results.jsonl"
    if results_path.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"{results_path} exists; use --resume or --overwrite")
    if args.overwrite: results_path.write_text("")
    done = completed_ids(results_path) if args.resume else set()
    random.seed(args.seed)
    adapter = create_adapter(args.model)
    if args.batch_size > 1:
        raise ValueError(
            "native batched generation is not yet standardized for this adapter; use --batch-size 1 "
            "to preserve native preprocessing and accurate per-sample timing"
        )
    dtype = args.dtype or spec["dtype"]
    attention = args.attn_implementation or spec["attention"]
    adapter.load(dtype=dtype, attn_implementation=attention)
    generation = spec["generation"]
    run_meta = {"gpu_model": args.gpu_model, "gpu_vram_gb": args.gpu_vram_gb,
                "dtype": dtype, "batch_size": args.batch_size,
                "image_preprocessing": spec["preprocessing"], "generation": generation,
                "peak_gpu_vram_bytes": None, "seed": args.seed}
    manifest_root = args.manifest.parents[1]
    pending = [row for row in rows if row["distorted_id"] not in done]
    if args.warmup and pending:
        warm_row = pending[0]
        warm_image = manifest_root/warm_row["distorted_image_path"]
        warm_prompt = spec.get("required_native_prompt") or prompts[args.prompt]["text"]
        for _ in range(args.warmup):
            adapter.infer(InferenceRequest(warm_image, warm_prompt, generation))
    for row in rows:
        sid = row["distorted_id"]
        if sid in done: continue
        image = manifest_root/row["distorted_image_path"]
        gt = (manifest_root/row["source"]["ground_truth_path"]).read_text(encoding="utf-8")
        prompt = spec.get("required_native_prompt") or prompts[args.prompt]["text"]
        result = adapter.infer(InferenceRequest(image, prompt, generation))
        timing = result.as_dict(); run_meta["peak_gpu_vram_bytes"] = timing.get("peak_gpu_vram_bytes")
        record = result_record(args.model, sid, result.extracted_text, gt, timing, run_meta)
        with results_path.open("a", encoding="utf-8") as fh: fh.write(json.dumps(record, ensure_ascii=False)+"\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
