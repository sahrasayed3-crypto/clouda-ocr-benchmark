# Reproducibility

## 1. Canonical manifest

`benchmark/MANIFEST.csv` is the canonical 177-sample registry; `benchmark/manifests/benchmark_manifest.jsonl` is the machine-readable equivalent with per-sample source metadata, seeds, QC fields and SHA-256 hashes for the distorted image, the clean source image and the ground truth. The two are cross-checked (same IDs, same asset hashes) in `audit_public/CONSISTENCY_AUDIT.json`.

## 2. Model revisions and run identity

| Model | Version / revision | Canonical run ID | Environment lock |
|---|---|---|---|
| HunyuanOCR-1.5 | HunyuanOCR-1.5 | `hunyuanocr` | `configs/locks/modern_vlm.freeze.txt` |
| MBZUAI/AIN-7B | AIN-7B | `ain_native_fa2_96gb_20260826` | `configs/locks/legacy_ocr.freeze.txt` |
| Qari OCR 0.4.0 | Qari OCR 0.4.0 (Qwen3-VL-4B base) | `qari_native_full` | `configs/locks/modern_vlm.freeze.txt` |
| Qwen3-VL-4B-Instruct | Qwen3-VL-4B-Instruct | `qwen3_vl_4b` | `configs/locks/modern_vlm.freeze.txt` |
| DeepSeek-OCR-2 | DeepSeek-OCR-2 | `deepseek_ocr_2` | `configs/locks/legacy_ocr.freeze.txt` |
| Arabic Nougat Large | Arabic Nougat Large | `arabic_nougat` | `configs/locks/legacy_ocr.freeze.txt` |
| dots.mocr | `e539fbb52280393adc081b289ec597430a0f9031` | `dots_mocr_lightning_final_20260831` | `results/dots.mocr/provenance/lightning_export/metadata/inference_environment.freeze.txt` |
| PaddleOCR-VL-1.6 | 1.6 (failed smoke) | `paddleocr_vl` | `configs/locks/paddle_native.freeze.txt` |

dots.mocr model-configuration file hashes are recorded in `results/dots.mocr/provenance/lightning_export/metadata/model_revision.json`. Only one canonical run per model is published; no runs with different checkpoints, revisions, prompts or benchmark subsets were merged.

## 3. Configs and prompts

- `configs/models.yaml` — per-model dtype, attention, preprocessing and generation settings (model weights are referenced via `<LOCAL_MODEL_CACHE>/`; users supply their own weights).
- `configs/prompts.yaml` — the benchmark OCR prompt ("Transcribe all visible text exactly in reading order…") and the optional layout prompt (excluded from primary pure-OCR scores). Native model-specific prompts are recorded in `configs/models.yaml` (e.g. PaddleOCR `"OCR:"`, DeepSeek `"<image>\nFree OCR."`, Qari `"Free OCR."`).
- `configs/environments.yaml` — environment composition per model group.
- `runtime/RUNBOOK.md` — exact per-model inference commands (smoke test then full resume run).

## 4. Runtime environment

- Two main GPU environments: `modern_vlm` (torch 2.7.0+cu128, transformers 5.15.1) and `legacy_ocr` (torch 2.6.0+cu118, transformers 4.46.3, flash-attn 2.7.3), Python 3.12 — full package pins in `configs/locks/*.freeze.txt`.
- dots.mocr ran in a dedicated Lightning environment (Python 3.12.13, torch 2.8.0+cu128, transformers 4.55.4, flash-attention 2.8.3, qwen-vl-utils 0.0.14; full freeze in its provenance metadata).
- All inference: deterministic (`do_sample=false`), batch size 1, seed 20260825, per-model dtype/attention as recorded in every result row.

## 5. Hardware metadata

GPU model, VRAM, dtype, peak VRAM and per-phase timings are recorded **per sample** in every `results/<model>/results.jsonl` row. Aggregate hardware blocks are in each `summary.json`. AIN-7B GPU telemetry is included (`results/MBZUAI_AIN-7B/gpu_telemetry.csv`, `run_telemetry.json`, `calibration.json`); dots.mocr hardware capture is in its Lightning provenance.

## 6. Per-model result files and integrity

`results/<model>/results.jsonl` are the canonical stored outputs (ground-truth text fields removed for licensing; predictions, metrics, timing and metadata are unmodified). Source-file SHA-256 hashes of the original files, plus the transformation applied, are recorded in `SANITIZATION_REPORT.md`. Every published file's SHA-256 is in `CHECKSUMS.txt`.

## 7. Normalization and metric implementation

The exact scoring implementation is shipped: `runtime/runtime/normalization.py` (`normalize_arabic`) and `runtime/runtime/metrics.py` (`edit_counts`, `evaluate_text`). No external metric library is required. During the public-release audit, all per-sample stored metrics and all summary aggregates were recomputed with this implementation and matched exactly (`audit_public/CONSISTENCY_AUDIT.json`).

## 8. How to rerun scoring (no inference needed)

```bash
# recompute a model's summary from its stored outputs
python scripts/evaluate_results.py results/HunyuanOCR-1.5/results.jsonl --output /tmp/hunyuan_summary.json

# compare with the published summary
python - <<'PY'
import json
a = json.load(open('/tmp/hunyuan_summary.json'))
b = json.load(open('results/HunyuanOCR-1.5/summary.json'))
for k in ('samples','successful','failures'):
    assert a[k] == b[k], k
for k, v in b['mean_metrics'].items():
    assert abs(a['mean_metrics'][k] - v) < 1e-12, k
print('metrics verified')
PY
```

To rescore from scratch, join each result row with its ground truth (from `benchmark/ground_truth_public/` for released samples, or from the upstream datasets for the remainder) and call `runtime/runtime/metrics.py:evaluate_text`.

## 9. How to rerun inference (full replication)

1. Provision the GPU environment per `configs/environments.yaml` and the matching lock file.
2. Obtain model weights from their official repositories into `<LOCAL_MODEL_CACHE>/` and point `configs/models.yaml` at them.
3. Obtain the benchmark inputs: 55 of 177 distorted pages derive from datasets whose licenses do not permit redistribution here (`metadata_public/GROUND_TRUTH_INDEX.csv`); obtain them from the pinned upstream sources under those sources' own access and license terms using `metadata_public/SAMPLE_SOURCE_INDEX.csv`, or supply the hash-verified distorted images independently.
4. Follow `runtime/RUNBOOK.md` per model (smoke sample `dist_b1aa183ffd20fcdf`, then `--resume` full run).
5. Score with `scripts/evaluate_results.py` and compare against `results/<model>/summary.json`.

## 10. How to verify hashes

```bash
sha256sum -c CHECKSUMS.txt
```

`CHECKSUMS.txt` lists every file in this package (sorted by path) except itself. The canonical benchmark asset hashes (distorted images, clean images, ground truth) are additionally embedded in `benchmark/MANIFEST.csv` and `benchmark/manifests/benchmark_manifest.jsonl` for independent verification against upstream data.

## 11. Benchmark generation reproducibility

The benchmark can be reconstructed from the pinned upstream sources and the published deterministic generation pipeline, subject to the original datasets' access and license terms. No assurance of byte-identical regeneration of third-party upstream data is implied, and access to every upstream source is not assumed.

The full distortion/generation pipeline (`benchmark_generation/`) is deterministic given the recorded seeds: sample allocation (`ocrbench/allocation.py`), rendering (`render.py`, `paginate.py`, `layouts.py`), distortions (`distortions.py`), QC gating (`config.py`, `pipeline.py`) and hashing (`hashing.py`). Generation configs are in `benchmark_generation/configs/`; the QC gate report is `benchmark_generation/qc_report.json`.
