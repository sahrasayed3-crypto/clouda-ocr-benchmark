# Arabic OCR benchmark GPU runbook

All setup and validation before this point is CPU-only. Do not run a full command until its one-sample smoke test succeeds. Commands below preserve native precision and preprocessing and record the fairness metadata in every output row.

Benchmark: `<BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl` (177 distorted samples)  
Smoke sample: `dist_b1aa183ffd20fcdf`  
Runner: `<RUNTIME_ROOT>/benchmark.py`

The smoke output is reused by the full command with `--resume`. If a smoke run fails, inspect its `results.jsonl`; do not use `--overwrite` unless intentionally restarting that model's output.

## Qari OCR 0.4.0

Recommended GPU: L4 24 GB; FP16; batch 1. The adapter always attaches `<LOCAL_MODEL_CACHE>/Qari-OCR-0.4.0` to the local `Qwen3-VL-4B-Instruct` base.

```bash
source <RUNTIME_ROOT>/envs/modern_vlm/bin/activate
python <RUNTIME_ROOT>/benchmark.py --model qari --sample-id dist_b1aa183ffd20fcdf --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/qari --gpu-model L4 --gpu-vram-gb 24 --batch-size 1
python <RUNTIME_ROOT>/benchmark.py --model qari --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/qari --gpu-model L4 --gpu-vram-gb 24 --batch-size 1 --warmup 1 --resume
```

## Qwen3-VL-4B-Instruct

Recommended GPU: L4 24 GB; BF16; batch 1.

```bash
source <RUNTIME_ROOT>/envs/modern_vlm/bin/activate
python <RUNTIME_ROOT>/benchmark.py --model qwen3_vl_4b --sample-id dist_b1aa183ffd20fcdf --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/qwen3_vl_4b --gpu-model L4 --gpu-vram-gb 24 --batch-size 1
python <RUNTIME_ROOT>/benchmark.py --model qwen3_vl_4b --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/qwen3_vl_4b --gpu-model L4 --gpu-vram-gb 24 --batch-size 1 --warmup 1 --resume
```

## Arabic Nougat Large

Recommended GPU: L4 24 GB; BF16; batch 1. The architecture is promptless and uses its native Nougat processor.

```bash
source <RUNTIME_ROOT>/envs/legacy_ocr/bin/activate
python <RUNTIME_ROOT>/benchmark.py --model arabic_nougat --sample-id dist_b1aa183ffd20fcdf --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/arabic_nougat --gpu-model L4 --gpu-vram-gb 24 --batch-size 1
python <RUNTIME_ROOT>/benchmark.py --model arabic_nougat --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/arabic_nougat --gpu-model L4 --gpu-vram-gb 24 --batch-size 1 --warmup 1 --resume
```

## AIN-7B

Recommended GPU: L40S 48 GB; BF16; batch 1; native 4–16384 visual-token budget.

```bash
source <RUNTIME_ROOT>/envs/legacy_ocr/bin/activate
python <RUNTIME_ROOT>/benchmark.py --model ain --sample-id dist_b1aa183ffd20fcdf --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/ain --gpu-model L40S --gpu-vram-gb 48 --batch-size 1
python <RUNTIME_ROOT>/benchmark.py --model ain --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/ain --gpu-model L40S --gpu-vram-gb 48 --batch-size 1 --warmup 1 --resume
```

## PaddleOCR-VL-1.6

Recommended GPU: L4 24 GB; BF16; batch 1; official Transformers OCR path and pixel budget. The optional Paddle document-parser environment is not the primary run because its auxiliary pipeline weights are not present locally.

```bash
source <RUNTIME_ROOT>/envs/modern_vlm/bin/activate
python <RUNTIME_ROOT>/benchmark.py --model paddleocr_vl --sample-id dist_b1aa183ffd20fcdf --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/paddleocr_vl --gpu-model L4 --gpu-vram-gb 24 --batch-size 1
python <RUNTIME_ROOT>/benchmark.py --model paddleocr_vl --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/paddleocr_vl --gpu-model L4 --gpu-vram-gb 24 --batch-size 1 --warmup 1 --resume
```

## HunyuanOCR-1.5

Recommended GPU: L4 24 GB; BF16; batch 1; official Transformers target model. DFlash speculative decoding is not required for accuracy.

```bash
source <RUNTIME_ROOT>/envs/modern_vlm/bin/activate
python <RUNTIME_ROOT>/benchmark.py --model hunyuanocr --sample-id dist_b1aa183ffd20fcdf --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/hunyuanocr --gpu-model L4 --gpu-vram-gb 24 --batch-size 1
python <RUNTIME_ROOT>/benchmark.py --model hunyuanocr --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/hunyuanocr --gpu-model L4 --gpu-vram-gb 24 --batch-size 1 --warmup 1 --resume
```

## DeepSeek-OCR-2

Recommended GPU: L4 24 GB; BF16; batch 1. Its official pinned FlashAttention wheel is already installed; the helper below only verifies it.

```bash
source <RUNTIME_ROOT>/envs/legacy_ocr/bin/activate
<RUNTIME_ROOT>/scripts/install_gpu_extras.sh deepseek_ocr_2
python <RUNTIME_ROOT>/benchmark.py --model deepseek_ocr_2 --sample-id dist_b1aa183ffd20fcdf --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/deepseek_ocr_2 --gpu-model L4 --gpu-vram-gb 24 --batch-size 1
python <RUNTIME_ROOT>/benchmark.py --model deepseek_ocr_2 --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/deepseek_ocr_2 --gpu-model L4 --gpu-vram-gb 24 --batch-size 1 --warmup 1 --resume
```

## dots.mocr

Recommended GPU: L40S 48 GB; BF16; batch 1, retaining the native 24000-token output ceiling. Its official pinned FlashAttention wheel is already installed; the helper below only verifies it.

```bash
source <RUNTIME_ROOT>/envs/dots_mocr/bin/activate
<RUNTIME_ROOT>/scripts/install_gpu_extras.sh dots_mocr
python <RUNTIME_ROOT>/benchmark.py --model dots_mocr --sample-id dist_b1aa183ffd20fcdf --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/dots_mocr --gpu-model L40S --gpu-vram-gb 48 --batch-size 1
python <RUNTIME_ROOT>/benchmark.py --model dots_mocr --manifest <BENCHMARK_ROOT>/manifests/benchmark_manifest.jsonl --output <RUNTIME_ROOT>/outputs/dots_mocr --gpu-model L40S --gpu-vram-gb 48 --batch-size 1 --warmup 1 --resume
```

## Evaluation after a run

```bash
python <RUNTIME_ROOT>/scripts/evaluate_results.py <RUNTIME_ROOT>/outputs/qari/results.jsonl --output <RUNTIME_ROOT>/outputs/qari/summary.json
```

The summary reports accuracy and timing, but timing is only comparable when the candidate runs used the same GPU model.
