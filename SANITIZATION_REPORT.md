# Sanitization Report — Clouda OCR Public Release

- Generated (UTC): 2026-09-07
- Source workspace: `<WORKSPACE_ROOT>` (local review workspace; absolute local path intentionally withheld from this public document)
- Output path: this repository root (`CLOUDA_OCR_BENCHMARK_PUBLIC/`)
- Source workspace status: **read-only input; not modified, renamed, or deleted** during this release build
- Publication status: package prepared for human review; nothing was uploaded or published automatically

## Volume summary

| Item | Value |
|---|---:|
| Source files inspected | 707 |
| Source total size | 378,385,660 bytes (~360.9 MB) |
| Public files (incl. this report and CHECKSUMS.txt) | 171 |
| Public package size | ~13.2 MB |
| SHA-256 checksum entries in CHECKSUMS.txt | 170 (every file except CHECKSUMS.txt itself) |
| Checksum verification failures | 0 |

## Exact duplicates

- **0 exact duplicates** (all 707 source files have unique SHA-256).
- Same-filename-different-content collisions found (`MANIFEST.csv`, `results.jsonl`, `summary.json`, `metrics.py`, `__init__.py`, five `arabic_ocr_0000{96..100}.png`) are per-model/per-directory artifacts, not duplicate copies. No deduplication exclusions were necessary.

## Files excluded from the public package

| Excluded group | Files | Reason |
|---|---:|---|
| Distorted benchmark images | 177 | raw asset redistribution withheld (license review) |
| Clean original images | 100 | raw asset redistribution withheld |
| Source image extracts (`input/images/`) | 100 | raw asset redistribution withheld |
| Input metadata JSONs (`input/metadata/`) | 100 | contained temporary signed Hugging Face dataset-server URLs; per-sample metadata preserved instead in benchmark manifests and `metadata_public/SAMPLE_SOURCE_INDEX.csv` |
| Source ground-truth files | 100 (122 of 177 sample references released from these) | license-gated release; 55 sample references withheld (unknown license) |
| Rendered benchmark PDFs | 2 | derived from source pages; redistribution withheld; ~115 MB |
| QC contact sheets | 9 | image-derived previews of source pages |
| Fonts | 6 | render-time assets; obtain from official sources (licenses not verified locally) |
| Sample corpus texts | 3 | no local license evidence |
| `source_index/` | 2 | contained personal user-profile download paths; hashes preserved in `metadata_public/TRANSFER_ARCHIVE_HASHES.csv` |
| `audit/CLEANUP_REPORT.md` | 1 | **stale/superseded**: described dots.mocr as still a 30-row partial run (superseded by the 2026-08-31 Lightning merge audit) |
| `audit/DATA_INTEGRITY_REPORT.md`, `audit/UNRESOLVED_ISSUES.md` | 2 | local working-state reports; substance regenerated in `audit_public/` |
| `metadata/FILE_INVENTORY.csv`, `metadata/DUPLICATES.csv` | 2 | local inventory / historical cleanup evidence; superseded by this report and `audit_public/` |
| `BENCHMARK_MASTER_INVENTORY.{json,md}`, `FINAL_LIGHTNING_MERGE_AUDIT.json`, top-level `README.md` | 4 | contained local machine drive paths (consolidated-root and export-folder locations); substance regenerated as public documentation and `audit_public/` |
| `results/dots.mocr/provenance/.../README_EXPORT.md` | 1 | studio-path mapping document; replaced by sanitized `PROVENANCE.md` |
| `opencode.json` | 1 | local tool configuration, not benchmark content |

## Stale / contradictory files excluded

- `audit/CLEANUP_REPORT.md` — asserted dots.mocr was still a 30-row partial run (30 of 177); superseded by the verified Lightning final merge (177 statuses / 169 valid / 8 failures). Excluded from active documentation. The older 30-row partial itself is not present in this workspace; its supersession is recorded in `results/dots.mocr/provenance/lightning_export/ocr_runtime/reports/final_integrity_audit.json` (original partial SHA-256 preserved).

## Local paths sanitized

| File | Transformation |
|---|---|
| `configs/models.yaml` | `~/ocr_models/` → `<LOCAL_MODEL_CACHE>/` |
| `runtime/RUNBOOK.md` | `~/ocr_models/`, `~/ocr_runtime/`, `~/ocr_benchmark/` → `<LOCAL_MODEL_CACHE>/`, `<RUNTIME_ROOT>/`, `<BENCHMARK_ROOT>/` |
| `results/dots.mocr/inference.log` | cloud-studio container paths → `<RUNTIME_ROOT>`; ephemeral build-home paths → `<BUILD_ROOT>` |
| `results/dots.mocr/provenance/lightning_export/metadata/model_revision.json` | studio model path → `<LOCAL_MODEL_CACHE>/DotsMOCR` |
| `results/dots.mocr/provenance/lightning_export/metadata/inference_environment.freeze.txt` cloud-studio and build-home `file://` wheel URLs → `local-wheel#sha256=...` (hashes preserved) |
| `metadata_public/TRANSFER_ARCHIVE_HASHES.csv` (generated) | local download paths stripped; archive names + SHA-256 retained |

Total: **6 files with path sanitization**; 4 top-level reports and 1 source-index file excluded wholesale for path content; 100 input-metadata files excluded (each contained one signed URL + ground-truth text).

## Temporary URLs removed

- **100 temporary signed Hugging Face dataset-server URLs** (cache-asset endpoints carrying expiry, signature and key-pair query parameters) removed by excluding the raw `input/metadata/*.json` files. Stable source identity (dataset, config, split, revision, row index, dimensions, hashes) is preserved in `benchmark/manifests/benchmark_manifest.jsonl` and `metadata_public/SAMPLE_SOURCE_INDEX.csv`.
- Second-pass scan found **0** remaining signed-URL parameters (expiry, signature, key-pair, or AWS X-Amz query parameters). The only remaining dataset-server host mention is a non-sensitive API base-URL constant in `benchmark_generation/ocrbench/ingest.py` (no query parameters, no credentials).

## Secrets and credentials

- **0 secrets or credentials found or copied.** Keyword-pattern screening (token, secret, cookie, session, authorization, bearer, API-key patterns) produced only false positives: ML tokenizer package names in freeze files and the literal audit field name `secret_pattern_hits` in historical evidence. No API keys, bearer tokens, HF tokens, cookies, passwords, or auth headers exist in the source workspace text files, and none appear in the public package.
- No credential appears to have been committed historically in the inspected workspace.

## Ground-truth field stripping (result transformation)

- 9 files transformed: 8 × `results/<model>/results.jsonl` + `results/dots.mocr/attempts.jsonl`.
- Fields removed: `raw_ground_truth`, `normalized_ground_truth`. All other content (predictions, model responses, per-sample metrics, timings, hardware metadata, generation settings, errors) is **unmodified**.
- Per-sample ground-truth linkage is preserved via `ground_truth_sha256` in `benchmark/MANIFEST.csv` and `metadata_public/GROUND_TRUTH_INDEX.csv`; original source-file SHA-256 hashes and per-file transformation records are listed in `audit_public/CONSISTENCY_AUDIT.json` provenance notes.

## Source datasets detected and redistribution status

14 source datasets (train split; revisions pinned in manifests). Summary (full table in `LICENSE_REVIEW.md` and `metadata_public/DATASET_SOURCES.csv`):

| Classification | Sources | Samples |
|---|---:|---:|
| ATTRIBUTION_REQUIRED (declared open license) | 8 | 122 |
| ATTRIBUTION_REQUIRED_COPYLEFT (gpl-3.0) | 1 (also counted above) | 26 (subset of 122) |
| UNKNOWN (no local license evidence) | 6 | 55 |

- Raw assets included in the public package: **0** (no images, no PDFs, no contact sheets).
- Ground truth released: 122 sample references (`benchmark/ground_truth_public/`); withheld: 55 (`UNKNOWN`).

## Benchmark consistency result

- Manifest: 177 rows / 177 unique IDs / 0 duplicates; CSV ↔ JSONL consistent (IDs and asset hashes); 177/177 ground-truth references present with matching SHA-256; 0 broken distorted-image references; 0 asset hash mismatches.
- All 6 full-coverage models: 177/177 valid pairs, 0 duplicate/unknown/missing result IDs.
- dots.mocr: 177 statuses / 169 valid / 8 deterministic-empty failures; revision `e539fbb52280393adc081b289ec597430a0f9031`; Lightning export subset hash-verified (7/7 retained entries match).
- PaddleOCR-VL-1.6: FAILED, 1 smoke row, unranked.
- Full detail: `audit_public/CONSISTENCY_AUDIT.json`.

## Metric validation result

- Per-sample stored metrics recomputed with the repository's own `runtime/runtime/metrics.py` + `normalization.py`: **0 mismatches across all 8 result streams** (1,415 evaluated rows).
- `summaries/FINAL_RESULTS.csv`, `summaries/COMMON_SUBSET_RESULTS.csv`, and all `summary.json` files match the recomputation exactly (0 differences).
- Leaderboard verified: HunyuanOCR-1.5 first (normalized Arabic CER 0.391497) among the six 177/177 models.

## Checksum result

- CHECKSUMS.txt: 171 entries (every file except itself), regenerated after final content, **verified: 0 failures**.

## Second-pass scan result (output-only)

- 172 output files screened (text formats): **0** personal local paths of any kind (Windows user profiles, drive letters, home/download/desktop/appdata directories), **0** signed-URL parameters, **0** secret or credential material, **0** loopback or local-hostname addresses, **0** local usernames.
- 30 `ipv4_like` regex hits are package **version strings** in environment freeze files (e.g. `nvidia-cublas-cu12==12.8.3.14`) — not IP addresses.
- Dataset-owner names (e.g. `ahmedheakl`, `calfa-ai`, `MohamedRashad`, `mohres`, `craneset`) are public Hugging Face repository identifiers, intentionally retained; they are not personal data.

## Unresolved issues

1. Redistribution rights for the 6 `UNKNOWN` sources (55 samples, all `ahmedheakl/arocrbench_*`) remain unestablished — their raw images and ground truth are withheld until licenses are confirmed.
2. Raw image redistribution for even the license-declared sources is deferred pending direct confirmation (see LICENSE_REVIEW.md checklist).
3. Dots.mocr Lightning export: 38 of 45 listed export entries were not carried into the public release (model configs, runtime duplicates, superseded partial); their hashes are preserved for provenance.

## Publication polish pass

A final documentation-only polish pass was applied before publication: project links finalized (project page `https://cloudaocr.xyz`, GitHub `https://github.com/sahrasayed3-crypto/clouda-ocr`, contact `contact@cloudaocr.xyz`), unresolved link labels removed, citation set to forthcoming, dataset wording corrected to "Hugging Face-hosted source datasets", and conservative reconstruction wording added. No benchmark data, results, metrics, or manifests changed; CHECKSUMS.txt was regenerated and re-verified afterwards.

## Publication polish pass

A final documentation-only polish pass was applied before publication: project links finalized (project page `https://cloudaocr.xyz`, GitHub `https://github.com/sahrasayed3-crypto/clouda-ocr`, contact `contact@cloudaocr.xyz`), unresolved link labels removed, citation set to forthcoming, dataset wording corrected to "Hugging Face-hosted source datasets", and conservative reconstruction wording added. No benchmark data, results, metrics, or manifests changed; CHECKSUMS.txt was regenerated and re-verified afterwards.

## Publication readiness

**READY FOR HUMAN REVIEW BEFORE PUBLICATION** — all critical safety-gate items pass; nothing has been uploaded or published.

