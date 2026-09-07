# Methodology

## 1. Canonical sample registry

The canonical registry is `benchmark/MANIFEST.csv` — **177 rows, 177 unique `sample_id` values, 0 duplicates** — mirrored by `benchmark/manifests/benchmark_manifest.jsonl` (same 177 IDs and asset hashes; cross-checked). Each row records:

- `sample_id` (e.g. `dist_b1aa183ffd20fcdf`) — the distorted-sample identity used by every result stream;
- `distortion_type`, `kind` (`atomic` 141 / profile 36), `severity`, `steps`, `seed`;
- relative paths and **SHA-256 of the distorted image, the ground-truth text, and the clean source image**;
- `source_dataset`, `source_split`, `source_repository_revision` and `source_sample_id` linking back to the upstream dataset.

Sample IDs are content-derived (`dist_` + 16 hex chars of a SHA-256 over source id, bucket, distortion, severity, index and seed — see `benchmark_generation/ocrbench/allocation.py`), so the registry is deterministic and reproducible.

## 2. Benchmark subset selection

100 clean source pages were ingested from 14 Hugging Face-hosted source datasets (counts in `benchmark_generation/qc_report.json` and `metadata_public/DATASET_SOURCES.csv`). Each clean page can contribute multiple distorted variants; 177 distorted pages were produced. Five replacements were made during ingestion (recorded in `final_run.json`) when a selected viewer row lacked ground truth.

Source datasets (train split; revisions pinned in the manifest):

| Source dataset | Samples | Declared license note |
|---|---:|---|
| mohres/The_Arabic_E-Book_Corpus | 32 | cc-by-4.0 |
| MohamedRashad/arabic-img2md | 26 | gpl-3.0 |
| ahmedheakl/arocrbench_hindawi | 15 | unspecified |
| calfa-ai/tarima | 15 | apache-2.0 |
| calfa-ai/RASAM-1 | 11 | apache-2.0 |
| craneset/arabic-ocr | 11 | mit |
| ahmedheakl/arocrbench_historicalbooks | 11 | unspecified |
| calfa-ai/RASAM-2 | 11 | apache-2.0 |
| ahmedheakl/arocrbench_tables | 10 | unspecified |
| calfa-ai/iskandar | 9 | etalab-2.0 |
| ahmedheakl/arocrbench_khattparagraph | 8 | unspecified |
| calfa-ai/baybars | 7 | etalab-2.0 |
| ahmedheakl/arocrbench_historyar | 7 | unspecified |
| ahmedheakl/arocrbench_doclaynet | 4 | unspecified |

## 3. Distorted input

Distortions are applied by the deterministic pipeline in `benchmark_generation/ocrbench/` (`distortions.py`, `render.py`, `pipeline.py`) using per-sample seeds recorded in the manifest. Atomic distortions (blur, jpeg, noise, skew, perspective, low resolution, contrast, shadow, faded, bleedthrough) and combined multi-step realistic profiles are QC-gated: each distorted page must pass quality checks (contrast, sharpness, ink-ratio bounds — `qc` field in the manifest). The rendered benchmark and contact-sheet previews are **not redistributed** (see DATA_NOTICE.md); the distorted page images are fully identified by their SHA-256 in the manifest.

## 4. Ground truth mapping

Ground truth is the transcription field of the upstream source sample, extracted at ingestion into `ground_truth/<benchmark_id>.txt` and recorded with SHA-256 in the manifest. All 177 samples reference an existing ground-truth file (177/177 present, hash-verified; 82 unique files because distorted variants share a source page). In this public package, ground-truth text is released only for license-cleared sources (122 of 177 samples — `benchmark/ground_truth_public/` and `metadata_public/GROUND_TRUTH_INDEX.csv`); the remaining 55 references are preserved as hashes and source links.

## 5. Result schema

Each `results/<model>/results.jsonl` row is one stored sample evaluation:

- identity: `model`, `sample_id`;
- outputs: `raw_prediction`, `normalized_prediction`, `raw_model_response` (ground-truth fields are removed in this public package; see SANITIZATION_REPORT.md);
- metrics: `strict_cer`, `normalized_arabic_cer`, `wer`, `extra_text_rate`, `missing_text_rate`, substitution/insertion/deletion counts;
- runtime: preprocessing/inference/postprocessing/total times, `gpu_model`, `gpu_vram_gb`, `dtype`, `batch_size`, generation settings, `seed`;
- `error`: non-null marks a failed status (no valid pair).

A **valid evaluated pair** = a result row with `error == null` for a sample present in the canonical manifest. A **failure** = a row with a non-null `error` (or, for dots.mocr, a recorded deterministic-empty final status).

## 6. Failure handling

Failures remain in the result stream as explicit statuses with failure reasons (see `results/dots.mocr/progress.json`). Failed samples are excluded from metric aggregation; the model's aggregate is then **coverage-qualified** and the model is excluded from the primary 177/177 leaderboard. No missing predictions are invented and no outputs are re-rolled.

## 7. Normalization

`runtime/runtime/normalization.py` implements the full normalization: NFC composition; alef variants (أ إ آ ٱ → ا); final ya (ى → ي); tatweel removal; Arabic diacritic removal (U+0610–061A, U+064B–065F, U+0670, U+06D6–06ED); whitespace collapse. Raw strings are never mutated; normalization is applied to local copies for scoring only.

## 8. Metrics

`runtime/runtime/metrics.py` computes, per sample:

- **strict CER** = character-level Levenshtein distance / reference length (raw strings);
- **normalized Arabic CER** = the same after Arabic normalization, over the normalized reference length;
- **WER** = word-level edit distance / reference word count (raw strings);
- plus insertion/deletion breakdown rates.

Rates are means over valid pairs. Lengths of 0 score 0.0 only if the hypothesis is also empty.

## 9. Why errors can exceed 1.0

Error rates are not capped at 1.0: the edit distance is normalized by the reference length, so insertion errors (model outputs longer than the reference) can push distance beyond the reference length and the rate above 1.0. This is intentional and applied identically to all models.

## 10. Primary ranking logic

Primary leaderboard: models with **exactly 177/177 valid evaluated pairs**, ranked by mean normalized Arabic CER (lower is better). Models with any failure or incomplete coverage are excluded from the primary ranking regardless of score.

## 11. Common-subset ranking logic

The common successful subset is the intersection of valid sample IDs across all scored models (169 samples — exactly the complement of the 8 dots.mocr failures). All scored models are ranked on this identical sample set; it is the strict like-for-like comparison that includes coverage-limited models such as dots.mocr.

## 12. Hardware context

Runs were executed on a single NVIDIA L4 (24 GB) or a single NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) as recorded per row in the result streams. Identical decoding settings (deterministic, batch size 1, seed 20260825) were used across models. Timing is contextual only and must not be compared across GPU families.

## 13. Metric implementation reference

The authoritative implementation shipped in this repository:

- `runtime/runtime/normalization.py` — `normalize_arabic(text)`;
- `runtime/runtime/metrics.py` — `edit_counts(reference, hypothesis)` (character/word list Levenshtein with explicit substitution/insertion/deletion accounting) and `evaluate_text(raw_ground_truth, raw_prediction)`;
- scoring entry point: `scripts/evaluate_results.py`.

Per-sample stored metrics in every `results.jsonl` row were verified against this implementation during the public-release audit with zero mismatches (see `audit_public/CONSISTENCY_AUDIT.json`).
