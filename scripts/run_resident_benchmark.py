#!/usr/bin/env python3
"""Calibrate and run one OCR adapter while retaining its model on the GPU.

This runner deliberately delegates result records and text metrics to the original
runtime.  It changes scheduling only: token-aware native batches, an incremental
checkpoint, and hardware telemetry.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import yaml

from adapters import ADAPTERS, create_adapter
from adapters.base import InferenceRequest
from runtime.runner import completed_ids, load_manifest, result_record


class GpuMonitor:
    def __init__(self, interval: float = 0.5) -> None:
        self.interval, self.samples, self._stop = interval, [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set(); self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True,
                )
                util, memory, power, temperature = [v.strip() for v in result.stdout.splitlines()[0].split(",")]
                self.samples.append({"timestamp": now, "gpu_utilization_pct": float(util),
                                     "memory_used_mib": float(memory), "power_draw_w": float(power),
                                     "temperature_c": float(temperature)})
            except (subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval)

    def between(self, start: float, end: float) -> dict:
        values = [row for row in self.samples if start <= row["timestamp"] <= end]
        if not values:
            return {"samples": 0}
        return {"samples": len(values),
                "avg_gpu_utilization_pct": statistics.fmean(v["gpu_utilization_pct"] for v in values),
                "max_gpu_utilization_pct": max(v["gpu_utilization_pct"] for v in values),
                "peak_memory_mib": max(v["memory_used_mib"] for v in values),
                "peak_power_w": max(v["power_draw_w"] for v in values),
                "peak_temperature_c": max(v["temperature_c"] for v in values)}


def batches(rows: list[dict], size: int):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def prefetch(paths: list[Path]) -> None:
    """Warm the OS page cache without transforming benchmark images."""
    for path in paths:
        with path.open("rb") as handle:
            handle.read(1024 * 1024)


def run_batch(adapter, rows, root: Path, prompt: str, generation: dict):
    requests = [InferenceRequest(root / row["distorted_image_path"], prompt, generation) for row in rows]
    started = time.perf_counter()
    results = adapter.infer_batch(requests)
    return results, time.perf_counter() - started


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True,
                   help="JSON list with id and visual_tokens computed by the native processor")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-implementation", required=True)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    configs = yaml.safe_load((root / "configs/models.yaml").read_text())["models"]
    prompts = yaml.safe_load((root / "configs/prompts.yaml").read_text())
    spec = configs[args.model]
    manifest = load_manifest(args.manifest)
    benchmark_root = args.manifest.parents[1]
    token_map = {item["id"]: item["visual_tokens"] for item in json.loads(args.profile.read_text())}
    if set(token_map) != {row["distorted_id"] for row in manifest}:
        raise ValueError("token profile does not exactly match the benchmark manifest")
    prompt = spec.get("required_native_prompt") or prompts["ocr"]["text"]
    generation = spec["generation"]
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if batch_sizes != sorted(set(batch_sizes)) or batch_sizes[0] != 1:
        raise ValueError("--batch-sizes must be increasing and start with 1")
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.jsonl"
    if results_path.exists() and not args.resume:
        raise FileExistsError(f"{results_path} exists; use --resume")
    done = completed_ids(results_path) if args.resume else set()
    monitor = GpuMonitor(); adapter = create_adapter(args.model)
    adapter.load(dtype=args.dtype, attn_implementation=args.attn_implementation)
    monitor.start()
    run_started = time.perf_counter()
    try:
        ordered = sorted(manifest, key=lambda row: token_map[row["distorted_id"]], reverse=True)
        # Functional smoke is retained as diagnostics, not mixed into final outputs.
        smoke_rows = [ordered[-1]]
        smoke_start = time.perf_counter(); smoke, smoke_wall = run_batch(adapter, smoke_rows, benchmark_root, prompt, generation)
        smoke_end = time.perf_counter()
        if len(smoke) != 1 or smoke[0].error or not smoke[0].extracted_text.strip():
            raise RuntimeError(f"functional smoke failed: {smoke[0].error if smoke else 'no result'}")
        (args.output / "smoke.json").write_text(json.dumps({"sample_id": smoke_rows[0]["distorted_id"],
            "prediction_raw": smoke[0].extracted_text, "wall_seconds": smoke_wall,
            "telemetry": monitor.between(smoke_start, smoke_end)}, ensure_ascii=False, indent=2))

        # The calibration pool is the largest native visual-token pages.  This
        # exercises the highest-memory real batch for each candidate size.
        calibration = []
        reference = None
        records = []
        for size in batch_sizes:
            rows = ordered[:size]
            started = time.perf_counter(); results, wall = run_batch(adapter, rows, benchmark_root, prompt, generation); ended = time.perf_counter()
            stable = len(results) == size and all(not r.error and r.extracted_text.strip() for r in results)
            if size == 1 and stable:
                reference = results[0].extracted_text
            identical_reference = bool(stable and reference is not None and results[0].extracted_text == reference)
            entry = {"batch_size": size, "pages": size, "wall_seconds": wall,
                     "pages_per_minute": size / wall * 60 if wall else 0,
                     "stable": stable, "reference_output_identical": identical_reference,
                     "errors": [r.error for r in results if r.error],
                     "peak_model_vram_bytes": max((r.peak_gpu_vram_bytes or 0) for r in results),
                     "telemetry": monitor.between(started, ended)}
            records.append(entry)
            if not stable or not identical_reference:
                break
        stable_records = [r for r in records if r["stable"] and r["reference_output_identical"]]
        if not stable_records:
            raise RuntimeError("no stable calibrated batch size")
        selected = max(stable_records, key=lambda r: r["pages_per_minute"])["batch_size"]
        (args.output / "calibration.json").write_text(json.dumps({"candidates": records, "selected_batch_size": selected}, indent=2))

        pending = [row for row in ordered if row["distorted_id"] not in done]
        run_meta = {"gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "gpu_vram_gb": 95.6,
                    "dtype": args.dtype, "batch_size": selected, "image_preprocessing": spec["preprocessing"],
                    "generation": generation, "peak_gpu_vram_bytes": None, "seed": args.seed}
        batch_log = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor, results_path.open("a", encoding="utf-8") as handle:
            next_prefetch = None
            for index, group in enumerate(batches(pending, selected)):
                following = pending[(index + 1) * selected:(index + 2) * selected]
                if following:
                    next_prefetch = executor.submit(prefetch, [benchmark_root / r["distorted_image_path"] for r in following])
                started = time.perf_counter(); results, wall = run_batch(adapter, group, benchmark_root, prompt, generation); ended = time.perf_counter()
                if len(results) != len(group):
                    raise RuntimeError(f"adapter returned {len(results)} results for batch of {len(group)}")
                telemetry = monitor.between(started, ended)
                batch_log.append({"batch": index, "pages": len(group), "wall_seconds": wall,
                                  "pages_per_minute": len(group) / wall * 60 if wall else 0, "telemetry": telemetry})
                for row, result in zip(group, results):
                    gt = (benchmark_root / row["source"]["ground_truth_path"]).read_text(encoding="utf-8")
                    timing = result.as_dict(); run_meta["peak_gpu_vram_bytes"] = timing.get("peak_gpu_vram_bytes")
                    handle.write(json.dumps(result_record(args.model, row["distorted_id"], result.extracted_text, gt, timing, run_meta), ensure_ascii=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                if next_prefetch is not None:
                    next_prefetch.result()
        ended = time.perf_counter()
        (args.output / "run_telemetry.json").write_text(json.dumps({"selected_batch_size": selected,
            "wall_seconds": ended - run_started, "batches": batch_log,
            "overall_telemetry": monitor.between(run_started, ended)}, indent=2))
    finally:
        monitor.stop()
        with (args.output / "gpu_telemetry.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["timestamp", "gpu_utilization_pct", "memory_used_mib", "power_draw_w", "temperature_c"])
            writer.writeheader(); writer.writerows(monitor.samples)
        try:
            import torch
            del adapter.model; adapter.model = None
            torch.cuda.empty_cache()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
