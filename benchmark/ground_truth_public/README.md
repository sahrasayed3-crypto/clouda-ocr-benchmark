# Ground-truth release subset

This directory contains the ground-truth transcription texts for the **122 of 177** benchmark samples whose source datasets carry a declared redistribution-permissive license (see `../../LICENSE_REVIEW.md`).

- Per-sample license status, source dataset, revision and SHA-256: `../../metadata_public/GROUND_TRUTH_INDEX.csv`.
- The remaining 55 samples (sources with unspecified license terms) are **withheld**; their hashes and upstream references are preserved in the same index for independent reconstruction.
- Files are named `<benchmark_id>.txt` and match the `ground_truth_sha256` recorded in `../../benchmark/MANIFEST.csv`.
- File names do not encode the distortion variant; multiple distorted samples of one source page share one ground-truth file (the 177 samples reference 82 unique ground-truth files overall; the 57 unique ground-truth files referenced by the 122 released samples are all present here).
