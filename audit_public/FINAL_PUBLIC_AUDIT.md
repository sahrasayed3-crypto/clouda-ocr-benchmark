# Final Public Audit

Generated (UTC): `2026-09-07T14:40:24.843306+00:00`

**Status: PASS**

## Checks

| Check | Result |
|---|---|
| canonical manifest rows | 177 |
| canonical unique sample ids | 177 |
| canonical duplicate sample ids | 0 |
| broken ground truth references | 0 |
| broken distorted image references | 0 |
| asset hash mismatches | 0 |
| manifest csv jsonl id mismatch | 0 |
| unknown result ids | 0 |
| duplicate result ids | 0 |
| per sample metric mismatches vs recompute | 0 |
| summary csv vs recompute mismatches | 0 |
| leaderboard leader | HunyuanOCR-1.5 |
| full coverage models ranked | 6 |
| dots statuses | 177 |
| dots successful | 169 |
| dots failures | 8 |
| dots listed in primary leaderboard | False |
| common subset samples | 169 |
| paddle status | FAILED |
| paddle ranked | False |
| stale 30 of 177 claims in active docs | 0 |
| local path hits | 0 |
| signed url hits | 0 |
| secret hits | 0 |
| archive dirs included | 0 |
| backup dirs included | 0 |
| raw image assets included | 0 |
| ground truth released | 122 |
| ground truth withheld | 55 |
| license sources reviewed | 14 |
| license unknown sources | 6 |
| second pass files scanned | 171 |

## Metric verification

- All per-sample stored metrics in every published `results.jsonl` were recomputed with `runtime/runtime/metrics.py` / `normalization.py`: **0 mismatches**.
- All published aggregates (`summaries/FINAL_RESULTS.csv`, `COMMON_SUBSET_RESULTS.csv`, per-model `summary.json`) match the recomputation exactly.

- Full consistency audit: `audit_public/CONSISTENCY_AUDIT.json`.
- dots.mocr deterministic failure IDs: `dist_d81844ae32aab663, dist_8bcaa349fa14f63a, dist_34b97e5a4f2bf1b0, dist_48dc8a47e830a050, dist_230e8b6d9ab74c61, dist_208318d9d5970cf9, dist_7ffbb7c9291660a0, dist_48ca82ad05989502`.
- Stale 30-of-177 dots.mocr claims: none present in active documentation (historical `audit/CLEANUP_REPORT.md` was excluded during sanitization).

## Publication polish pass

Generated (UTC): see `FINAL_PUBLIC_AUDIT.json` (`publication_polish_pass.timestamp_utc`).

Documentation-only publication-readiness corrections; no benchmark data, results, metrics, or manifests were changed:

- Project links finalized (`https://cloudaocr.xyz`, `https://github.com/sahrasayed3-crypto/clouda-ocr`, `contact@cloudaocr.xyz`); all unresolved link labels removed.
- Citation entry set to: "Citation information will be added in a future release."
- Dataset wording corrected to "Hugging Face-hosted source datasets" — public accessibility does not imply redistribution rights; license classifications unchanged (14 reviewed / 6 UNKNOWN / 0 RESTRICTED).
- Conservative reconstruction wording added to REPRODUCIBILITY.md and DATA_NOTICE.md; no assured-reconstruction claim is made.
- Stale-wording scan: 0 remaining hits for any removed label, the former repository URL, the previous dataset phrasing, or assured-reconstruction claims.
- CHECKSUMS.txt regenerated and verified after the edits.
