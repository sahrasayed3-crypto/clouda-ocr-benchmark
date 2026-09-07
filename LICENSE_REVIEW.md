# License Review — Benchmark Source Datasets

All 177 benchmark samples derive from 14 Hugging Face-hosted source datasets (train split; repository revisions pinned in `benchmark/MANIFEST.csv`). Public accessibility of a dataset does not by itself establish redistribution rights. License evidence below is the per-sample `license_or_permission_note` recorded at ingestion time into `benchmark/manifests/benchmark_manifest.jsonl`. **No LICENSE file was present locally for any source**; classification is therefore evidence-conservative and must be re-verified against each repository's current license before any redistribution.

Classification legend:

- `ATTRIBUTION_REQUIRED` — a redistribution-permissive open license is declared in the source metadata; reuse requires attribution (and compliance with the declared license terms).
- `UNKNOWN` — no license information available locally; redistribution rights not established.

## Source dataset table

| Source dataset | Samples | Declared license note | Classification | Raw images in public package | Ground truth in public package |
|---|---:|---|---|---|---|
| mohres/The_Arabic_E-Book_Corpus | 32 | cc-by-4.0 | ATTRIBUTION_REQUIRED | NO | YES (32) |
| MohamedRashad/arabic-img2md | 26 | gpl-3.0 | ATTRIBUTION_REQUIRED (copyleft) | NO | YES (26) |
| ahmedheakl/arocrbench_hindawi | 15 | unspecified | UNKNOWN | NO | NO |
| calfa-ai/tarima | 15 | apache-2.0 | ATTRIBUTION_REQUIRED | NO | YES (15) |
| calfa-ai/RASAM-1 | 11 | apache-2.0 | ATTRIBUTION_REQUIRED | NO | YES (11) |
| craneset/arabic-ocr | 11 | mit | ATTRIBUTION_REQUIRED | NO | YES (11) |
| ahmedheakl/arocrbench_historicalbooks | 11 | unspecified | UNKNOWN | NO | NO |
| calfa-ai/RASAM-2 | 11 | apache-2.0 | ATTRIBUTION_REQUIRED | NO | YES (11) |
| ahmedheakl/arocrbench_tables | 10 | unspecified | UNKNOWN | NO | NO |
| calfa-ai/iskandar | 9 | etalab-2.0 | ATTRIBUTION_REQUIRED | NO | YES (9) |
| ahmedheakl/arocrbench_khattparagraph | 8 | unspecified | UNKNOWN | NO | NO |
| calfa-ai/baybars | 7 | etalab-2.0 | ATTRIBUTION_REQUIRED | NO | YES (7) |
| ahmedheakl/arocrbench_historyar | 7 | unspecified | UNKNOWN | NO | NO |
| ahmedheakl/arocrbench_doclaynet | 4 | unspecified | UNKNOWN | NO | NO |
| **Total** | **177** | | | | 122 released / 55 withheld |

Per-sample mapping with SHA-256 hashes: `metadata_public/GROUND_TRUTH_INDEX.csv` and `metadata_public/SAMPLE_SOURCE_INDEX.csv`. Per-source summary: `metadata_public/DATASET_SOURCES.csv`.

## Decisions applied in this package

1. **Raw images (all 177 samples): withheld.** Even for license-declared sources, image rights can differ from metadata rights and underlying documents may carry their own terms. Raw image redistribution is deferred until rights are confirmed directly with each source.
2. **Ground truth: released only for declared-license sources (122 samples).** Each released text retains its upstream identification (dataset, split, revision, source sample ID, SHA-256) in `GROUND_TRUTH_INDEX.csv`.
3. **Ground truth for UNKNOWN sources (55 samples): withheld.** The `ahmedheakl/arocrbench_*` aggregates provide no local license evidence; no license is assumed or invented.
4. **GPL-3.0 (MohamedRashad/arabic-img2md, 26 samples):** released as declared, flagged `ATTRIBUTION_REQUIRED_COPYLEFT`. Users redistributing these texts onward should review GPL-3.0 copyleft obligations for the derived work.
5. **Rendered benchmark PDFs and contact sheets: withheld** (derived from the same source pages).
6. **OCR model outputs in `results/` are published** as Clouda OCR's own evaluation artifacts; embedded ground-truth text fields were removed from the released result streams for consistency with the policy above (see SANITIZATION_REPORT.md).

## Model licenses (evaluated checkpoints — obtained separately)

| Model | Source | License evidence locally |
|---|---|---|
| HunyuanOCR-1.5 | official repository | not included |
| MBZUAI/AIN-7B | official repository | not included |
| Qari OCR 0.4.0 | official repository | not included |
| Qwen3-VL-4B-Instruct | official repository | not included |
| DeepSeek-OCR-2 | official repository | not included |
| Arabic Nougat Large | official repository | not included |
| dots.mocr | `e539fbb52280393adc081b289ec597430a0f9031` | model license agreement referenced in the Lightning export manifest (`results/dots.mocr/provenance/lightning_export/SHA256SUMS.txt` lists `dots.mocr LICENSE AGREEMENT`); file itself not redistributed |
| PaddleOCR-VL-1.6 | official repository | not included |

## Conservative-verification checklist before any future redistribution

- [ ] Fetch the current LICENSE / dataset card of each source repository.
- [ ] Confirm the pinned revision's terms have not changed.
- [ ] Obtain written permission where terms are unclear.
- [ ] Re-run the per-sample classification and update `metadata_public/DATASET_SOURCES.csv`.
