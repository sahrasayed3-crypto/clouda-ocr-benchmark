# Results

All values below were recomputed offline from the stored per-sample outputs in `results/<model>/results.jsonl` using the canonical evaluator (`runtime/runtime/metrics.py`, `runtime/runtime/normalization.py`). No OCR inference was rerun. Aggregates are means over each model's valid evaluated pairs.

Required metric note: error rates are not capped at 1.0; insertion errors can make edit distance exceed the reference length.

## A. Primary leaderboard — models with complete 177/177 valid evaluated pairs

Primary metric: **Normalized Arabic CER** (lower is better).

| Rank | Model | Version / revision | Valid pairs | Normalized Arabic CER | CER | WER | Hardware | Runtime sec/page |
|---:|---|---|---:|---:|---:|---:|---|---:|
| 1 | HunyuanOCR-1.5 | HunyuanOCR-1.5 | 177/177 | 0.391497102343 | 0.564666303999 | 0.719510772462 | NVIDIA L4 | 22.0037 |
| 2 | MBZUAI/AIN-7B | AIN-7B | 177/177 | 0.837028332500 | 0.752904696351 | 0.643977368106 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 7.5903 |
| 3 | Qari OCR 0.4.0 | Qari OCR 0.4.0 | 177/177 | 1.076259820088 | 1.419240556704 | 1.638313456431 | NVIDIA L4 | 48.6011 |
| 4 | Qwen3-VL-4B-Instruct | Qwen3-VL-4B-Instruct | 177/177 | 1.286567026274 | 2.007940166793 | 1.529543234329 | NVIDIA L4 | 48.0877 |
| 5 | DeepSeek-OCR-2 | DeepSeek-OCR-2 | 177/177 | 1.486473469125 | 1.727660839069 | 1.549825480979 | NVIDIA L4 | 22.3366 |
| 6 | Arabic Nougat Large | Arabic Nougat Large | 177/177 | 2.226340721331 | 2.391472153897 | 1.876030865759 | NVIDIA L4 | 3.1724 |

Runtime is the mean wall time per page from each stored run (preprocessing + inference + postprocessing). All runs used deterministic decoding (`do_sample=false`), batch size 1, and seed 20260825.

Canonical statement: HunyuanOCR-1.5 ranked first on this historical 177-page distorted Arabic benchmark v0.1.0 among models with complete 177/177 valid coverage, using normalized Arabic CER as the primary metric.

## B. Additional evaluated run — dots.mocr

- Model revision: `e539fbb52280393adc081b289ec597430a0f9031`
- Statuses: **177** — Successful outputs: **169** — Failures: **8** (deterministic empty outputs after repeated identical-setting retries)
- Because 8 of 177 pages failed, dots.mocr is **not** listed in the primary 177/177 leaderboard.
- Coverage-qualified aggregate over its 169 valid pairs: normalized Arabic CER **20.377487907976**, WER 38.884466129364, strict CER 20.676199710648.
- The 8 failure sample IDs are recorded in `results/dots.mocr/progress.json` and `audit_public/CONSISTENCY_AUDIT.json`.
- Provenance: verified Lightning export (`results/dots.mocr/provenance/lightning_export/`), run ID `dots_mocr_lightning_final_20260831`.
- Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition; 25.5337 s/page mean.

## C. Failed / unranked models

| Model | Status | Status rows | Valid pairs | Failures | Score |
|---|---|---:|---:|---:|---|
| PaddleOCR-VL-1.6 | FAILED | 1 (smoke only) | 0 | 0 | none — unranked |

PaddleOCR-VL-1.6's single smoke output repeated a truncated phrase through its 512-token ceiling; the full benchmark run was never started. No diagnostic/smoke metrics are ranked. Evidence: `results/PaddleOCR-VL-1.6/`.

## D. Common successful 169-sample subset

All seven scored models (the six full-coverage models plus dots.mocr) share 169 sample IDs with valid outputs. Ranking on this subset:

| Rank | Model | Samples | Normalized Arabic CER | WER | CER |
|---:|---|---:|---:|---:|---:|
| 1 | HunyuanOCR-1.5 | 169 | 0.372044885032 | 0.706726272539 | 0.547388622015 |
| 2 | MBZUAI/AIN-7B | 169 | 0.843021836060 | 0.630505459918 | 0.749066183178 |
| 3 | Qwen3-VL-4B-Instruct | 169 | 1.026305411235 | 1.473953846717 | 1.691950485542 |
| 4 | Qari OCR 0.4.0 | 169 | 1.086816720886 | 1.667768700353 | 1.441300871514 |
| 5 | Arabic Nougat Large | 169 | 1.201064817859 | 1.418598683580 | 1.294888245771 |
| 6 | DeepSeek-OCR-2 | 169 | 1.414038617023 | 1.507213669428 | 1.650864047871 |
| 7 | dots.mocr | 169 | 20.377487907976 | 38.884466129364 | 20.676199710648 |

The common-subset ordering of the six full-coverage models differs from the primary leaderboard in one position: Qwen3-VL-4B-Instruct and Qari OCR 0.4.0 swap ranks because the 8 excluded pages (all dots.mocr failures) affect the models unevenly. HunyuanOCR-1.5 remains first on both.

## E. Interpretation notes

- Aggregates are means over valid evaluated pairs; models with failures (only dots.mocr) have coverage-qualified aggregates that are not directly comparable to 177/177 aggregates. Use the common subset for strict comparison.
- Structural markup (HTML tables produced by HunyuanOCR-1.5 and DeepSeek-OCR-2 on table-style pages) is counted as extra text by the unchanged evaluation semantics; this is a deliberate, uniformly applied scoring choice and not a model-specific adjustment.
- dots.mocr's very high aggregate is driven by extreme insertion loops on a minority of distorted pages (median normalized CER 0.15625 over its successful pages); see `results/dots.mocr/dots_mocr_report.json` for per-bucket medians and breakdowns.
- Results are specific to Clouda OCR's historical distorted Arabic benchmark v0.1.0 and should not be interpreted as universal OCR performance.

## F. Hardware comparability caveat

Six runs used a single NVIDIA L4 (24 GB). MBZUAI/AIN-7B and dots.mocr ran on a single NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB). Do not rank runtime across different GPU families. Runtime figures are contextual and should only be compared within the same hardware environment. Among completed NVIDIA L4 runs, Arabic Nougat Large was the fastest (≈3.17 s/page mean wall time).
