# dots.mocr — Lightning export provenance (sanitized)

The canonical dots.mocr run was finalized from a verified Lightning export on 2026-08-31 (run ID `dots_mocr_lightning_final_20260831`). The export's own file list and hashes are preserved here.

## Contents

- `MANIFEST.csv` — original export file list (source→archive mapping; studio paths replaced by logical labels in the original export layout).
- `SHA256SUMS.txt` — SHA-256 manifest of the **full** export tree (45 entries). The full tree included the benchmark manifest, runtime code, model configuration files and the original 30-row partial run. Only the entries below were retained in this repository; the remaining listed artifacts were not carried into the public release.
- `metadata/export_verification.json` — verification pass over the export: 177 status rows, 177 unique IDs, 0 duplicates, 169 successful / 8 failed.
- `metadata/hardware.json` — GPU capture: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.6 GB recorded.
- `metadata/model_revision.json` — model revision `e539fbb52280393adc081b289ec597430a0f9031`, model configuration file hashes, runtime versions and inference configuration (source path sanitized to `<LOCAL_MODEL_CACHE>/DotsMOCR`).
- `metadata/inference_environment.freeze.txt` — full pip freeze of the inference environment (local wheel URLs sanitized).
- `ocr_runtime/reports/final_integrity_audit.json` — the export's final integrity audit (status: pass).

## Entry verification status

Of the 45 entries in `SHA256SUMS.txt`, 7 retained entries are present in this repository and all 7 match their recorded SHA-256. The other 38 entries describe files not carried into the public release (model configuration files, the original 30-row partial run, and duplicated runtime code); their hashes are preserved for provenance only. The original 30-row partial run was superseded by this final merged run (original partial SHA-256 `fc5bf552bec558f5f6c996886be4abf5283f181f461304828f1a4ccf422457f9`, recorded in `final_integrity_audit.json`).
