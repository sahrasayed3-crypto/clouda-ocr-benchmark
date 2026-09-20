# Clouda OCR Benchmark — Public Release

Arabic-first OCR and Document AI benchmark infrastructure from Clouda OCR.

This repository publishes the canonical evaluation state of Clouda OCR's current distorted Arabic OCR benchmark: a 177-page evaluation set, per-model stored OCR outputs, verified metric aggregates, and full provenance for reproducibility.

- Project page: https://cloudaocr.xyz
- GitHub: https://github.com/sahrasayed3-crypto/clouda-ocr
- Contact: contact@cloudaocr.xyz

## 1. What is being measured

Models transcribe **distorted Arabic document page images** (single-image, page-level OCR). Each of the 177 benchmark samples is a distorted page image plus a reference transcription. Inputs include photographic and rendering-style degradations:

| Distortion bucket | Samples |
|---|---:|
| combined (multi-step realistic profiles) | 36 |
| blur | 22 |
| jpeg compression | 22 |
| noise | 22 |
| skew | 12 |
| perspective | 12 |
| low resolution | 12 |
| contrast | 12 |
| shadow | 9 |
| faded / old paper | 9 |
| bleedthrough | 9 |
| **Total** | **177** |

## 2. Canonical benchmark size

**177 unique distorted samples** (`benchmark/MANIFEST.csv`, 177 unique sample IDs, 0 duplicates). The 177 distorted images are derived from 100 clean source pages drawn from 14 Hugging Face-hosted source datasets (see `metadata_public/DATASET_SOURCES.csv`).

## 3. Primary metric

**Normalized Arabic CER** (lower is better): character error rate computed after a deterministic Arabic normalization (NFC, alef/ya unification, tatweel and diacritic removal, whitespace collapse), averaged over each model's valid evaluated pairs. The exact implementation is `runtime/runtime/metrics.py` and `runtime/runtime/normalization.py`.

**Note:** error rates are not capped at 1.0; insertion errors can make edit distance exceed the reference length.

## 4. Primary leaderboard

Models with complete **177/177 valid evaluated pairs** only. Results are specific to Clouda OCR's current distorted Arabic benchmark and should not be interpreted as universal OCR performance.

| Rank | Model | Valid pairs | Normalized Arabic CER | CER | WER |
|---:|---|---:|---:|---:|---:|
| 1 | HunyuanOCR-1.5 | 177/177 | 0.391497 | 0.564666 | 0.719511 |
| 2 | MBZUAI/AIN-7B | 177/177 | 0.837028 | 0.752905 | 0.643977 |
| 3 | Qari OCR 0.4.0 | 177/177 | 1.076260 | 1.419241 | 1.638313 |
| 4 | Qwen3-VL-4B-Instruct | 177/177 | 1.286567 | 2.007940 | 1.529543 |
| 5 | DeepSeek-OCR-2 | 177/177 | 1.486473 | 1.727661 | 1.549825 |
| 6 | Arabic Nougat Large | 177/177 | 2.226341 | 2.391472 | 1.876031 |

HunyuanOCR-1.5 ranked first on Clouda OCR's current 177-page distorted Arabic benchmark among models with complete 177/177 valid coverage, using normalized Arabic CER as the primary metric.

Full details (hardware, runtime, run identifiers): [`RESULTS.md`](RESULTS.md), [`summaries/FINAL_RESULTS.csv`](summaries/FINAL_RESULTS.csv).

## 5. Coverage explanation

A "valid evaluated pair" is one benchmark sample with a stored, non-failed model output. All values above are recomputed offline from the stored per-sample outputs using the canonical evaluator; no OCR inference was rerun during reconciliation. Models that did not complete all 177 samples are **not** placed in the primary leaderboard, regardless of their partial scores.

## 6. Additional evaluated run: dots.mocr

dots.mocr (revision `e539fbb52280393adc081b289ec597430a0f9031`) produced **177 statuses: 169 successful outputs and 8 deterministic-empty failures** (empty OCR text after repeated identical-setting retries; failure IDs recorded in `results/dots.mocr/progress.json`). It is therefore **not** listed as having 177 valid evaluated pairs and is excluded from the primary leaderboard. Its coverage-qualified aggregate over its 169 valid pairs is reported in [`RESULTS.md`](RESULTS.md) alongside the other models on the shared 169-sample common subset.

## 7. Failed run: PaddleOCR-VL-1.6

PaddleOCR-VL-1.6 **failed** (a one-sample smoke run exhibited a repeated truncated output loop through the 512-token ceiling; the full run was never started). It is **unranked** and has no valid benchmark score. Evidence: `results/PaddleOCR-VL-1.6/`.

## 8. Common successful 169-sample subset

All scored models share 169 sample IDs with valid outputs. On this fully comparable subset:

| Rank | Model | Samples | Normalized Arabic CER | WER | CER |
|---:|---|---:|---:|---:|---:|
| 1 | HunyuanOCR-1.5 | 169 | 0.372045 | 0.706726 | 0.547389 |
| 2 | MBZUAI/AIN-7B | 169 | 0.843022 | 0.630505 | 0.749066 |
| 3 | Qwen3-VL-4B-Instruct | 169 | 1.026305 | 1.473954 | 1.691950 |
| 4 | Qari OCR 0.4.0 | 169 | 1.086817 | 1.667769 | 1.441301 |
| 5 | Arabic Nougat Large | 169 | 1.201065 | 1.418599 | 1.294888 |
| 6 | DeepSeek-OCR-2 | 169 | 1.414039 | 1.507214 | 1.650864 |
| 7 | dots.mocr | 169 | 20.377488 | 38.884466 | 20.676200 |

dots.mocr's very high error is dominated by extreme insertion loops on a subset of distorted pages; see `results/dots.mocr/dots_mocr_report.md` for the per-bucket breakdown.

## 9. Runtime / hardware caveat

Six runs used a single NVIDIA L4 (24 GB); MBZUAI/AIN-7B and dots.mocr ran on a single NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB). Runtime figures are contextual and should only be compared within the same hardware environment. Among completed NVIDIA L4 runs, Arabic Nougat Large was the fastest (≈3.17 s/page mean wall time).

## 10. Methodology and reproducibility

- [METHODOLOGY.md](METHODOLOGY.md) — sample registry, distortion pipeline, GT mapping, metric definitions, ranking logic.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — manifests, model revisions, configs, prompts, environments, how to rerun scoring and verify hashes.

## 11. Data redistribution note

Benchmark inputs derive from third-party public datasets with differing redistribution rights. **Raw page images are not redistributed in this repository**, and ground-truth text is released only for sources with a declared redistribution-permissive license (with attribution); see [DATA_NOTICE.md](DATA_NOTICE.md) and [LICENSE_REVIEW.md](LICENSE_REVIEW.md). Per-sample hashes and source references preserve full auditability.

## 12. Integrity

`CHECKSUMS.txt` contains SHA-256 for every file in this repository (sorted by path; the file does not list its own hash). Verify with:

```bash
sha256sum -c CHECKSUMS.txt
```

## 13. Repository structure

```
CLOUDA_OCR_BENCHMARK_PUBLIC/
├── README.md, RESULTS.md, METHODOLOGY.md, REPRODUCIBILITY.md,
│   DATA_NOTICE.md, LICENSE_REVIEW.md, SANITIZATION_REPORT.md, CHECKSUMS.txt
├── benchmark/            canonical 177-sample manifest + manifests (metadata only)
│   └── ground_truth_public/   license-cleared ground-truth texts (122 of 177)
├── results/              per-model stored outputs (GT fields removed), summaries, provenance
├── summaries/            FINAL_RESULTS.{csv,md}, COMMON_SUBSET_RESULTS.csv, MODEL_STATUS.csv
├── configs/              model/runtime/prompt configs and environment freezes
├── runtime/              benchmark runner, adapters, canonical metrics implementation
├── scripts/              evaluation and validation scripts
├── benchmark_generation/ distortion-pipeline sources, configs, QC report
├── metadata_public/      sanitized source linkage, dataset licenses, aliases
└── audit_public/         FINAL_PUBLIC_AUDIT.{json,md} and consistency audit
```

## 14. Citation

Version 0.1.0 is archived on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22859930.svg)](https://doi.org/10.5281/zenodo.22859930)

Wahbah, S. K. N. (2026). *Clouda OCR Arabic OCR Benchmark v0.1.0 — 177-Page Reproducible Evaluation* (Version 0.1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.22859930

For this specific release, cite the version DOI above. The all-versions DOI is `10.5281/zenodo.22859929`.

## 15. Disclaimer

Results are specific to Clouda OCR's current distorted Arabic benchmark and should not be interpreted as universal OCR performance. Error rates are not capped at 1.0; insertion errors can make edit distance exceed the reference length. No claim of state-of-the-art or universal superiority is made.
