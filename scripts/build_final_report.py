#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


HOME = Path.home()
RUNTIME = HOME / "ocr_runtime"
OUTPUTS = RUNTIME / "outputs"
REPORTS = RUNTIME / "reports"
MANIFEST = HOME / "ocr_benchmark/manifests/benchmark_manifest.jsonl"
GPU_START_EPOCH = 1787700060

MODELS = [
    {
        "key": "qari",
        "name": "Qari OCR 0.4.0",
        "repo": "NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct",
        "model_dir": "Qari-OCR-0.4.0",
        "output_dir": "qari_native_full",
        "attention": "SDPA",
        "status": "complete",
        "smoke": "passed",
        "note": "Loaded local Qwen3-VL-4B base plus the local Qari adapter; never benchmarked as plain Qwen.",
    },
    {
        "key": "qwen3_vl_4b",
        "name": "Qwen3-VL-4B-Instruct",
        "repo": "Qwen/Qwen3-VL-4B-Instruct",
        "model_dir": "Qwen3-VL-4B-Instruct",
        "output_dir": "qwen3_vl_4b",
        "attention": "SDPA",
        "status": "complete",
        "smoke": "passed",
    },
    {
        "key": "arabic_nougat",
        "name": "Arabic Nougat Large",
        "repo": "MohamedRashad/arabic-large-nougat",
        "model_dir": "Arabic-Nougat-Large",
        "output_dir": "arabic_nougat",
        "attention": "eager",
        "status": "complete",
        "smoke": "passed",
    },
    {
        "key": "paddleocr_vl",
        "name": "PaddleOCR-VL-1.6",
        "repo": "PaddlePaddle/PaddleOCR-VL-1.6",
        "model_dir": "PaddleOCR-VL-1.6",
        "output_dir": "paddleocr_vl",
        "attention": "SDPA",
        "status": "smoke_failed",
        "smoke": "failed_quality_gate",
        "note": "Smoke output repeated the same truncated Arabic phrase to the 512-token ceiling; full run was not started.",
    },
    {
        "key": "hunyuanocr",
        "name": "HunyuanOCR-1.5",
        "repo": "tencent/HunyuanOCR",
        "model_dir": "HunyuanOCR-1.5",
        "output_dir": "hunyuanocr",
        "attention": "SDPA",
        "status": "complete",
        "smoke": "passed",
    },
    {
        "key": "deepseek_ocr_2",
        "name": "DeepSeek-OCR-2",
        "repo": "deepseek-ai/DeepSeek-OCR-2",
        "model_dir": "DeepSeek-OCR-2",
        "output_dir": "deepseek_ocr_2",
        "attention": "FlashAttention 2.7.3",
        "status": "complete",
        "smoke": "passed",
        "note": "Used the bundled model.infer evaluation path with base_size=1024, image_size=768, crop_mode=true.",
    },
]

HUB_CHECKED = {
    "qari": 14,
    "qwen3_vl_4b": 14,
    "arabic_nougat": 9,
    "paddleocr_vl": 20,
    "hunyuanocr": 37,
    "deepseek_ocr_2": 16,
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(rows: list[dict], getter) -> float | None:
    values = [getter(row) for row in rows]
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def disk_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def human_bytes(value: int | None) -> str:
    if value is None:
        return "—"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError


def human_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    seconds = int(round(value))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"


def fmt(value: float | None, places: int = 4) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def shard_details(model_path: Path) -> tuple[int, int | None, list[str]]:
    weights = sorted(str(path.relative_to(model_path)) for path in model_path.rglob("*.safetensors"))
    indexes = sorted(model_path.rglob("*.safetensors.index.json"))
    expected = None
    if indexes:
        expected_names: set[str] = set()
        for index in indexes:
            data = json.loads(index.read_text(encoding="utf-8"))
            expected_names.update(data.get("weight_map", {}).values())
        expected = len(expected_names)
    elif weights:
        expected = len(weights)
    return len(weights), expected, weights


def build() -> tuple[dict, str, dict]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_jsonl(MANIFEST)
    manifest_by_id = {row["distorted_id"]: row for row in manifest_rows}
    generated_at = datetime.now(timezone.utc).isoformat()
    model_reports = []

    for spec in MODELS:
        output_path = OUTPUTS / spec["output_dir"] / "results.jsonl"
        rows = load_jsonl(output_path)
        successful = [row for row in rows if not row.get("error")]
        full = spec["status"] == "complete"
        metrics = {
            metric: mean(successful, lambda row, metric=metric: (row.get("metrics") or {}).get(metric))
            if full else None
            for metric in ["strict_cer", "normalized_arabic_cer", "wer", "extra_text_rate", "missing_text_rate"]
        }
        timings = {
            field: mean(successful, lambda row, field=field: row.get(field)) if full else None
            for field in ["preprocessing_time", "inference_time", "postprocessing_time", "total_time"]
        }
        severity_groups: dict[str, list[dict]] = defaultdict(list)
        bucket_groups: dict[str, list[dict]] = defaultdict(list)
        if full:
            for row in successful:
                manifest = manifest_by_id[row["sample_id"]]
                severity_groups[manifest["severity"]].append(row)
                bucket_groups[manifest["bucket"]].append(row)
        severity = {
            key: {
                "samples": len(group),
                "strict_cer": mean(group, lambda row: row["metrics"]["strict_cer"]),
                "normalized_arabic_cer": mean(group, lambda row: row["metrics"]["normalized_arabic_cer"]),
            }
            for key, group in sorted(severity_groups.items())
        }
        bucket = {
            key: {
                "samples": len(group),
                "strict_cer": mean(group, lambda row: row["metrics"]["strict_cer"]),
                "normalized_arabic_cer": mean(group, lambda row: row["metrics"]["normalized_arabic_cer"]),
            }
            for key, group in sorted(bucket_groups.items())
        }
        first = rows[0]
        model_path = HOME / "ocr_models" / spec["model_dir"]
        weight_count, expected_shards, weight_files = shard_details(model_path)
        integrity = {
            "status": "passed",
            "hub_checksum_files_checked": HUB_CHECKED[spec["key"]],
            "basic_safetensors_integrity": "passed",
            "disk_size_bytes": disk_size(model_path),
            "weight_files": weight_count,
            "expected_shards": expected_shards,
            "weight_file_names": weight_files,
            "incomplete_files": len(list(model_path.rglob("*.incomplete"))),
            "config_files_present": bool(list(model_path.glob("*config*.json"))),
            "tokenizer_files_present": bool(list(model_path.glob("tokenizer*"))),
            "processor_files_present": bool(list(model_path.glob("*processor*"))),
        }
        report = {
            "key": spec["key"],
            "model": spec["name"],
            "repo": spec["repo"],
            "local_path": str(model_path),
            "download_integrity": integrity,
            "status": spec["status"],
            "smoke_test_status": spec["smoke"],
            "full_benchmark_status": "complete" if full else "not_run_smoke_quality_failure",
            "samples_completed": len(rows) if full else 0,
            "smoke_samples_recorded": 1,
            "successful_samples": len(successful) if full else 0,
            "failures": sum(bool(row.get("error")) for row in rows) if full else 1,
            "failure_rate": (sum(bool(row.get("error")) for row in rows) / 177) if full else None,
            "metrics": metrics,
            "clean_cer": None,
            "distorted_cer": metrics["strict_cer"],
            "robustness_degradation": None,
            "robustness_note": "The 177-row manifest contains distorted variants only; no clean predictions were run.",
            "severity_breakdown": severity,
            "distortion_bucket_breakdown": bucket,
            "peak_gpu_vram_bytes": max((row.get("peak_gpu_vram_bytes") or 0) for row in rows),
            "mean_timings_seconds": timings,
            "measured_sample_runtime_seconds": sum((row.get("total_time") or 0) for row in rows),
            "settings": {
                "gpu_model": first["gpu_model"],
                "usable_gpu_vram_gb": first["gpu_vram_gb"],
                "dtype": first["dtype"],
                "batch_size": first["batch_size"],
                "attention": spec["attention"],
                "native_preprocessing": first["image_preprocessing"],
                "image_resolution": first["image_preprocessing"],
                "generation": first["generation"],
                "seed": first["seed"],
            },
            "larger_gpu_required": False,
            "errors": [spec.get("note")] if spec.get("note") else [],
            "output_path": str(output_path),
            "summary_path": str(OUTPUTS / spec["output_dir"] / ("summary.json" if full else "smoke_summary.json")),
        }
        model_reports.append(report)

    completed = [report for report in model_reports if report["status"] == "complete"]
    accuracy_ranking = sorted(
        completed,
        key=lambda report: (
            report["metrics"]["normalized_arabic_cer"],
            report["metrics"]["strict_cer"],
            report["metrics"]["wer"],
            report["metrics"]["extra_text_rate"],
        ),
    )
    speed_ranking = sorted(completed, key=lambda report: report["mean_timings_seconds"]["total_time"])
    end_epoch = int(time.time())
    combined = {
        "title": "24GB Arabic OCR Benchmark — NVIDIA L4",
        "generated_at": generated_at,
        "benchmark": {
            "archive": str(HOME / "ocr_benchmark_transfer (1).tar.gz"),
            "archive_sha256": "eff8cd96b838051aec812ad0b81711d2924f7f5c4c1bbb6665547360f772cdaa",
            "manifest": str(MANIFEST),
            "manifest_rows": len(manifest_rows),
            "unique_sample_ids": len(manifest_by_id),
            "unique_benchmark_ids": len({row["benchmark_id"] for row in manifest_rows}),
            "population": "177 distorted benchmark variants",
            "severity_counts": {key: sum(row["severity"] == key for row in manifest_rows) for key in sorted({row["severity"] for row in manifest_rows})},
            "ground_truth_baseline_sha256": "e0fdbbad9becfbaa15081f9a41c64cff93d34fee22f0067922f95f5387b9700a",
        },
        "hardware": {
            "gpu_model": "NVIDIA L4",
            "usable_vram_gb": 22.494,
            "driver": "580.173.02",
        },
        "environments": {
            "modern_vlm": {"python": "3.12.13", "torch": "2.7.0+cu128", "transformers": "5.15.1"},
            "legacy_ocr": {"python": "3.12.13", "torch": "2.6.0+cu118", "transformers": "4.46.3", "flash_attention": "2.7.3"},
        },
        "models": model_reports,
        "accuracy_ranking": [
            {
                "rank": index,
                "model": report["model"],
                **report["metrics"],
                "robustness_degradation": report["robustness_degradation"],
            }
            for index, report in enumerate(accuracy_ranking, 1)
        ],
        "speed_ranking_same_gpu": [
            {
                "rank": index,
                "model": report["model"],
                "average_total_latency_seconds": report["mean_timings_seconds"]["total_time"],
                "measured_sample_runtime_seconds": report["measured_sample_runtime_seconds"],
            }
            for index, report in enumerate(speed_ranking, 1)
        ],
        "models_requiring_larger_gpu": [],
        "gpu_time": {
            "benchmark_window_started_at": datetime.fromtimestamp(GPU_START_EPOCH, timezone.utc).isoformat(),
            "benchmark_window_ended_at": datetime.fromtimestamp(end_epoch, timezone.utc).isoformat(),
            "elapsed_wall_seconds": end_epoch - GPU_START_EPOCH,
            "measured_successful_sample_seconds_including_paddle_smoke": sum(report["measured_sample_runtime_seconds"] for report in model_reports),
            "note": "Wall time includes model loading, warmups, failed smoke diagnostics, compatibility checks, evaluation, and inter-model teardown; measured sample time sums persisted rows only.",
        },
        "report_notes": {
            "accuracy_ranking_rule": "Normalized Arabic CER, then strict CER, WER, hallucination/extra-text rate, then robustness degradation where available.",
            "paddle_exclusion": "PaddleOCR-VL-1.6 is excluded because its smoke response was a repeated malformed phrase and no full run was authorized by the smoke gate.",
            "clean_metric_omission": "Clean CER and distorted-minus-clean degradation are unavailable because the transferred 177-row manifest contains distorted variants only.",
            "chart_map": [{
                "section": "Accuracy-first ranking",
                "question": "Which completed model has the lowest normalized Arabic CER?",
                "family": "Comparison and ranking",
                "type": "bar",
                "fields": ["model", "normalized_cer"],
                "takeaway": "HunyuanOCR-1.5 has the lowest normalized Arabic CER.",
                "palette_policy": "single-root preferred",
            }],
        },
    }

    lines = [
        "# 24GB Arabic OCR Benchmark — NVIDIA L4",
        "",
        "## Technical summary",
        "",
        f"HunyuanOCR-1.5 ranked first on accuracy with normalized Arabic CER {fmt(accuracy_ranking[0]['metrics']['normalized_arabic_cer'])}, strict CER {fmt(accuracy_ranking[0]['metrics']['strict_cer'])}, and WER {fmt(accuracy_ranking[0]['metrics']['wer'])}. Five models completed all 177 samples with zero runtime failures. PaddleOCR-VL-1.6 failed the smoke quality gate and was not expanded into a costly invalid full run.",
        "",
        "No model OOMed on the L4, and none requires a larger GPU based on these runs. Qari correctly used the Qwen base plus Qari adapter; plain Qwen was benchmarked separately.",
        "",
        "## Accuracy-first ranking",
        "",
        "Ranking uses normalized Arabic CER first, then strict CER, WER, extra-text rate, and robustness degradation where available. Lower is better.",
        "",
        "| Rank | Model | Normalized Arabic CER | Strict CER | WER | Extra-text rate | Missing-text rate |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, report in enumerate(accuracy_ranking, 1):
        m = report["metrics"]
        lines.append(f"| {rank} | {report['model']} | {fmt(m['normalized_arabic_cer'])} | {fmt(m['strict_cer'])} | {fmt(m['wer'])} | {fmt(m['extra_text_rate'])} | {fmt(m['missing_text_rate'])} |")

    lines += [
        "",
        "## Execution status and resource use",
        "",
        "Five candidates completed the full manifest. Paddle is reported separately because its only accepted run artifact is the failed-quality smoke output.",
        "",
        "| Model | Integrity | Smoke | Full benchmark | Samples | Failures | Peak VRAM | Avg latency | Measured sample runtime |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for report in model_reports:
        lines.append(
            f"| {report['model']} | passed | {report['smoke_test_status']} | {report['full_benchmark_status']} | "
            f"{report['samples_completed']} | {report['failures']} | {human_bytes(report['peak_gpu_vram_bytes'])} | "
            f"{fmt(report['mean_timings_seconds']['total_time'], 2)} s | {human_seconds(report['measured_sample_runtime_seconds'])} |"
        )

    lines += [
        "",
        "## Same-GPU speed table",
        "",
        "This speed ordering is valid only for these NVIDIA L4 runs and is separate from the accuracy ranking.",
        "",
        "| Speed rank | Model | Average total latency | Total persisted sample time |",
        "|---:|---|---:|---:|",
    ]
    for rank, report in enumerate(speed_ranking, 1):
        lines.append(f"| {rank} | {report['model']} | {fmt(report['mean_timings_seconds']['total_time'], 2)} s | {human_seconds(report['measured_sample_runtime_seconds'])} |")

    lines += [
        "",
        "## Scope and metric definitions",
        "",
        "The population is the exact 177-row transferred manifest: 50 light, 48 medium, 43 heavy, and 36 combined distorted variants. CER and WER are computed per row by the restored runtime and averaged across successful rows. Extra-text and missing-text rates use the same restored evaluation semantics. Raw and normalized prediction and ground-truth fields remain in every result row.",
        "",
        "Clean CER and distorted-minus-clean robustness degradation are not reported: this manifest contains distorted variants only and no clean predictions. Severity and distortion-bucket cuts are included in the JSON report without inventing a clean baseline.",
        "",
        "## Native settings used",
        "",
        "| Model | dtype | Batch | Attention | Native preprocessing | Generation |",
        "|---|---|---:|---|---|---|",
    ]
    for report in model_reports:
        s = report["settings"]
        lines.append(f"| {report['model']} | {s['dtype']} | {s['batch_size']} | {s['attention']} | {s['native_preprocessing']} | `{json.dumps(s['generation'], ensure_ascii=False)}` |")

    lines += [
        "",
        "## Download integrity",
        "",
        "All six local directories passed prior Hub checksum verification and CPU safetensors structural checks; no `.incomplete` files remain.",
        "",
        "| Model | Local path | Disk size | Weight files | Expected shards | Incomplete | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for report in model_reports:
        d = report["download_integrity"]
        lines.append(f"| {report['model']} | `{report['local_path']}` | {human_bytes(d['disk_size_bytes'])} | {d['weight_files']} | {d['expected_shards']} | {d['incomplete_files']} | passed |")

    lines += [
        "",
        "## Limitations, failures, and robustness",
        "",
        "- PaddleOCR-VL-1.6 produced a repeated truncated phrase through its 512-token ceiling on the smoke page. It is excluded from rankings; this is a smoke-quality failure, not an OOM.",
        "- Hunyuan and DeepSeek emit structural markup, which the shared strict/normalized metrics correctly count as extra text under unchanged evaluation semantics.",
        "- The result is descriptive for this 177-sample distorted benchmark. It does not establish performance on unseen clean pages or a different GPU/software stack.",
        "- No model required more than 24 GB. Highest observed peak was Qari at " + human_bytes(max(report["peak_gpu_vram_bytes"] for report in model_reports)) + ".",
        "",
        "## Recommended next steps",
        "",
        "1. Use HunyuanOCR-1.5 as the accuracy-first candidate for this benchmark definition.",
        "2. If clean-versus-distorted degradation is required, create a separately authorized clean-image manifest and run it once with the same runtime; do not infer it from these distorted-only rows.",
        "3. Revisit Paddle only with an upstream-compatible official inference package or model revision, then repeat the smoke gate before any full run.",
        "",
        "## Further questions",
        "",
        "The main open question is whether Hunyuan’s accuracy lead persists after markup-aware evaluation or on a clean companion set. Those analyses require an explicitly defined additional metric or manifest and were not substituted into this benchmark.",
        "",
        "## Output paths",
        "",
        f"- Combined JSON: `{REPORTS / 'final_24gb_benchmark.json'}`",
        f"- Combined Markdown: `{REPORTS / 'final_24gb_benchmark.md'}`",
        f"- Portable HTML: `{REPORTS / 'final_24gb_benchmark.html'}`",
        f"- Per-model outputs: `{OUTPUTS}`",
        "",
        f"Benchmark-window wall time: {human_seconds(combined['gpu_time']['elapsed_wall_seconds'])}. Persisted sample time, including Paddle smoke: {human_seconds(combined['gpu_time']['measured_successful_sample_seconds_including_paddle_smoke'])}.",
    ]
    markdown = "\n".join(lines) + "\n"

    ranking_rows = [
        {
            "rank": index,
            "model": report["model"],
            "normalized_cer": round(report["metrics"]["normalized_arabic_cer"], 4),
            "strict_cer": round(report["metrics"]["strict_cer"], 4),
            "wer": round(report["metrics"]["wer"], 4),
            "extra_text": round(report["metrics"]["extra_text_rate"], 4),
        }
        for index, report in enumerate(accuracy_ranking, 1)
    ]
    status_rows = [
        {
            "model": report["model"],
            "smoke": report["smoke_test_status"],
            "full_status": report["full_benchmark_status"],
            "samples": report["samples_completed"],
            "failures": report["failures"],
            "peak_vram_gib": round(report["peak_gpu_vram_bytes"] / 1024**3, 2),
            "avg_latency_s": round(report["mean_timings_seconds"]["total_time"], 2) if report["mean_timings_seconds"]["total_time"] is not None else None,
        }
        for report in model_reports
    ]
    speed_rows = [
        {"rank": index, "model": report["model"], "avg_latency_s": round(report["mean_timings_seconds"]["total_time"], 2)}
        for index, report in enumerate(speed_ranking, 1)
    ]
    severity_rows = []
    for report in completed:
        for severity, values in report["severity_breakdown"].items():
            severity_rows.append({
                "model": report["model"],
                "severity": severity,
                "samples": values["samples"],
                "normalized_cer": round(values["normalized_arabic_cer"], 4),
                "strict_cer": round(values["strict_cer"], 4),
            })

    source = {
        "id": "benchmark_results",
        "label": "OCR benchmark result rows and manifest",
        "path": "ocr_runtime/reports/final_24gb_benchmark.json",
        "query": {
            "description": "Aggregated from the restored runtime's persisted JSONL rows and the authorized 177-row manifest.",
            "language": "python",
            "tables_used": [
                "ocr_benchmark/manifests/benchmark_manifest.jsonl",
                "ocr_runtime/outputs/*/results.jsonl",
            ],
            "metric_definitions": [
                "Model metrics are arithmetic means of restored-runtime per-row metrics over 177 successful distorted samples.",
                "Lower CER, WER, extra-text rate, and missing-text rate are better.",
            ],
            "filters": ["Authorized 24GB model list only", "177 distorted manifest rows", "Paddle excluded from ranking after smoke quality failure"],
        },
    }
    accuracy_source = {
        "id": "accuracy_ranking_source",
        "label": "Accuracy ranking dataset",
        "path": "ocr_runtime/reports/final_24gb_accuracy_ranking.json",
        "query": {
            "engine": "duckdb",
            "sql": "SELECT rank, model, normalized_cer, strict_cer, wer, extra_text FROM read_json_auto('ocr_runtime/reports/final_24gb_accuracy_ranking.json') ORDER BY rank",
            "description": "Reads the bounded five-model accuracy ranking dataset used by the chart.",
            "language": "sql",
            "tables_used": ["ocr_runtime/reports/final_24gb_accuracy_ranking.json"],
            "filters": ["Full 177-sample runs only", "Paddle smoke-failed candidate excluded"],
            "metric_definitions": ["normalized_cer is mean normalized Arabic CER across 177 successful distorted samples; lower is better."],
        },
    }
    status_source = {
        "id": "execution_status_source",
        "label": "Execution status dataset",
        "path": "ocr_runtime/reports/final_24gb_execution_status.json",
        "query": {
            "engine": "duckdb",
            "sql": "SELECT model, smoke, full_status, samples, failures, peak_vram_gib, avg_latency_s FROM read_json_auto('ocr_runtime/reports/final_24gb_execution_status.json') ORDER BY samples DESC, model",
            "description": "Reads the bounded six-model execution-status dataset.",
            "language": "sql",
            "tables_used": ["ocr_runtime/reports/final_24gb_execution_status.json"],
            "filters": ["Authorized six-model list only"],
        },
    }
    speed_source = {
        "id": "speed_ranking_source",
        "label": "Same-GPU speed ranking dataset",
        "path": "ocr_runtime/reports/final_24gb_speed_ranking.json",
        "query": {
            "engine": "duckdb",
            "sql": "SELECT rank, model, avg_latency_s FROM read_json_auto('ocr_runtime/reports/final_24gb_speed_ranking.json') ORDER BY rank",
            "description": "Reads the bounded same-L4 full-run speed ranking dataset.",
            "language": "sql",
            "tables_used": ["ocr_runtime/reports/final_24gb_speed_ranking.json"],
            "filters": ["Full 177-sample runs only", "Same NVIDIA L4"],
        },
    }
    severity_source = {
        "id": "severity_breakdown_source",
        "label": "Distortion severity dataset",
        "path": "ocr_runtime/reports/final_24gb_severity_breakdown.json",
        "query": {
            "engine": "duckdb",
            "sql": "SELECT model, severity, samples, normalized_cer, strict_cer FROM read_json_auto('ocr_runtime/reports/final_24gb_severity_breakdown.json') ORDER BY model, severity",
            "description": "Reads the bounded model-by-severity CER dataset.",
            "language": "sql",
            "tables_used": ["ocr_runtime/reports/final_24gb_severity_breakdown.json"],
            "filters": ["Full 177-sample runs only", "Distorted variants only"],
        },
    }
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": combined["title"],
            "description": "Accuracy-first comparison of six authorized Arabic OCR candidates on an NVIDIA L4.",
            "generatedAt": generated_at,
            "sources": [source, accuracy_source, status_source, speed_source, severity_source],
            "charts": [
                {
                    "id": "accuracy_chart",
                    "title": "Normalized Arabic CER by model",
                    "subtitle": "Five models completing all 177 distorted samples; lower is better",
                    "type": "bar",
                    "dataset": "ranking",
                    "sourceId": "accuracy_ranking_source",
                    "encodings": {
                        "x": {"field": "model", "type": "nominal", "label": "Model"},
                        "y": {"field": "normalized_cer", "type": "quantitative", "label": "Normalized Arabic CER", "format": "number"},
                        "tooltip": [
                            {"field": "rank", "type": "quantitative", "label": "Rank", "format": "number"},
                            {"field": "strict_cer", "type": "quantitative", "label": "Strict CER", "format": "number"},
                            {"field": "wer", "type": "quantitative", "label": "WER", "format": "number"}
                        ]
                    },
                    "valueFormat": "number",
                    "layout": "full",
                    "maxRows": 5
                }
            ],
            "tables": [
                {"id": "accuracy", "title": "Accuracy-first ranking", "subtitle": "Five models completing all 177 distorted samples; lower is better", "dataset": "ranking", "defaultSort": {"field": "rank", "direction": "asc"}, "density": "spacious", "sourceId": "accuracy_ranking_source", "columns": [
                    {"field": "rank", "label": "Rank", "type": "number"}, {"field": "model", "label": "Model", "type": "text"}, {"field": "normalized_cer", "label": "Normalized Arabic CER", "type": "number"}, {"field": "strict_cer", "label": "Strict CER", "type": "number"}, {"field": "wer", "label": "WER", "type": "number"}, {"field": "extra_text", "label": "Extra-text rate", "type": "number"},
                ]},
                {"id": "status", "title": "Execution status and resource use", "subtitle": "Same NVIDIA L4, 22.494 GiB usable VRAM", "dataset": "status", "defaultSort": {"field": "samples", "direction": "desc"}, "density": "spacious", "sourceId": "execution_status_source", "columns": [
                    {"field": "model", "label": "Model", "type": "text"}, {"field": "smoke", "label": "Smoke", "type": "text"}, {"field": "full_status", "label": "Full status", "type": "text"}, {"field": "samples", "label": "Samples", "type": "number"}, {"field": "failures", "label": "Failures", "type": "number"}, {"field": "peak_vram_gib", "label": "Peak VRAM, GiB", "type": "number"}, {"field": "avg_latency_s", "label": "Avg latency, s", "type": "number"},
                ]},
                {"id": "speed", "title": "Same-GPU speed ranking", "subtitle": "Full runs only; kept separate from accuracy ranking", "dataset": "speed", "defaultSort": {"field": "rank", "direction": "asc"}, "density": "spacious", "sourceId": "speed_ranking_source", "columns": [
                    {"field": "rank", "label": "Rank", "type": "number"}, {"field": "model", "label": "Model", "type": "text"}, {"field": "avg_latency_s", "label": "Avg latency, s", "type": "number"},
                ]},
                {"id": "severity", "title": "CER by distortion severity", "subtitle": "Light, medium, heavy, and combined distorted variants; no clean rows", "dataset": "severity", "defaultSort": {"field": "model", "direction": "asc"}, "density": "dense", "sourceId": "severity_breakdown_source", "columns": [
                    {"field": "model", "label": "Model", "type": "text"}, {"field": "severity", "label": "Severity", "type": "text"}, {"field": "samples", "label": "Samples", "type": "number"}, {"field": "normalized_cer", "label": "Normalized Arabic CER", "type": "number"}, {"field": "strict_cer", "label": "Strict CER", "type": "number"},
                ]},
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# 24GB Arabic OCR Benchmark — NVIDIA L4"},
                {"id": "summary", "type": "markdown", "sourceId": "benchmark_results", "body": "## Hunyuan leads the completed field\n\nHunyuanOCR-1.5 ranked first with normalized Arabic CER **0.3915**, strict CER **0.5647**, and WER **0.7195**. Five models completed 177/177 samples with zero runtime failures. PaddleOCR-VL-1.6 failed the smoke quality gate and was not expanded into a costly invalid run."},
                {"id": "accuracy_intro", "type": "markdown", "body": "## Accuracy-first ranking\n\nThe ordering uses normalized Arabic CER first, then strict CER, WER, extra-text rate, and robustness degradation where available. Lower is better."},
                {"id": "accuracy_chart_block", "type": "chart", "chartId": "accuracy_chart", "layout": "full"},
                {"id": "accuracy_table", "type": "table", "tableId": "accuracy", "layout": "full"},
                {"id": "status_intro", "type": "markdown", "body": "## Five models completed without OOM\n\nAll five full runs fit the L4. Paddle is shown as smoke-failed rather than as a scored candidate because its output was a repeated malformed phrase."},
                {"id": "status_table", "type": "table", "tableId": "status", "layout": "full"},
                {"id": "speed_intro", "type": "markdown", "body": "## Speed is secondary and same-GPU only\n\nNougat was fastest, while Hunyuan and DeepSeek had similar average latency. This table does not affect the accuracy ranking."},
                {"id": "speed_table", "type": "table", "tableId": "speed", "layout": "full"},
                {"id": "scope", "type": "markdown", "sourceId": "benchmark_results", "body": "## Scope and definitions\n\nThe cohort is the exact **177 distorted variants** in the transferred manifest: 50 light, 48 medium, 43 heavy, and 36 combined. CER, normalized Arabic CER, WER, extra-text rate, and missing-text rate are unchanged restored-runtime per-row metrics averaged across successful rows."},
                {"id": "method", "type": "markdown", "body": "## Accuracy-fair native inference\n\nModels ran sequentially with native preprocessing, recommended dtype and attention, batch size 1, deterministic generation where applicable, no quantization, no forced CPU offload, and no benchmark-side resolution reduction. Qari loaded the local Qwen base plus the Qari adapter."},
                {"id": "robustness_intro", "type": "markdown", "body": "## Severity cuts are available; clean degradation is not\n\nThe manifest contains distorted variants only, so clean CER and distorted-minus-clean degradation are unavailable. The severity table provides the supported robustness view without inventing a clean baseline."},
                {"id": "severity_table", "type": "table", "tableId": "severity", "layout": "full"},
                {"id": "limitations", "type": "markdown", "body": "## Limitations and failure modes\n\nPaddle failed because its smoke output looped to the token ceiling, not because of OOM. Structural markup from Hunyuan and DeepSeek is counted as extra text by the unchanged evaluation semantics. Results describe this distorted benchmark and software stack, not unseen clean pages."},
                {"id": "next", "type": "markdown", "body": "## Recommended next steps\n\n1. Use HunyuanOCR-1.5 as the accuracy-first candidate.\n2. Run a separately authorized clean companion manifest if degradation is required.\n3. Revisit Paddle only after an upstream-compatible official inference revision passes the one-page smoke gate."},
                {"id": "questions", "type": "markdown", "body": "## Further questions\n\nThe main unresolved question is whether Hunyuan's lead persists under a markup-aware metric or on a clean companion set. Neither can be inferred from the current distorted-only rows."},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {"ranking": ranking_rows, "status": status_rows, "speed": speed_rows, "severity": severity_rows},
        },
        "sources": [source, accuracy_source, status_source, speed_source, severity_source],
    }
    return combined, markdown, artifact


def main() -> None:
    combined, markdown, artifact = build()
    (REPORTS / "final_24gb_benchmark.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPORTS / "final_24gb_benchmark.md").write_text(markdown, encoding="utf-8")
    (REPORTS / "final_24gb_benchmark_artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for dataset, filename in {
        "ranking": "final_24gb_accuracy_ranking.json",
        "status": "final_24gb_execution_status.json",
        "speed": "final_24gb_speed_ranking.json",
        "severity": "final_24gb_severity_breakdown.json",
    }.items():
        (REPORTS / filename).write_text(
            json.dumps(artifact["snapshot"]["datasets"][dataset], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    for report in combined["models"]:
        key = report["key"]
        (REPORTS / f"{key}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        model_md = (
            f"# {report['model']} benchmark report\n\n"
            f"- Status: {report['status']}\n"
            f"- Smoke test: {report['smoke_test_status']}\n"
            f"- Full benchmark: {report['full_benchmark_status']}\n"
            f"- Samples completed: {report['samples_completed']}\n"
            f"- Normalized Arabic CER: {fmt(report['metrics']['normalized_arabic_cer'])}\n"
            f"- Strict CER: {fmt(report['metrics']['strict_cer'])}\n"
            f"- WER: {fmt(report['metrics']['wer'])}\n"
            f"- Extra-text rate: {fmt(report['metrics']['extra_text_rate'])}\n"
            f"- Missing-text rate: {fmt(report['metrics']['missing_text_rate'])}\n"
            f"- Peak VRAM: {human_bytes(report['peak_gpu_vram_bytes'])}\n"
            f"- Average latency: {fmt(report['mean_timings_seconds']['total_time'], 2)} s\n"
            f"- Output: `{report['output_path']}`\n"
        )
        (REPORTS / f"{key}_report.md").write_text(model_md, encoding="utf-8")


if __name__ == "__main__":
    main()
