# Data Notice

## What is published

This repository publishes **benchmark evaluation artifacts**: the canonical sample registry, per-model stored OCR outputs (predictions and metrics), verified metric aggregates, source linkage metadata, and the code used for scoring. It does **not** publish the full benchmark image corpus.

## What is withheld

- **Raw page images** (clean originals, distorted pages, source image extracts, rendered benchmark PDFs, QC contact sheets) are **not redistributed** in any form. Every image is identified by its SHA-256 and upstream source reference in `benchmark/MANIFEST.csv` and `metadata_public/SAMPLE_SOURCE_INDEX.csv`.
- **Ground-truth text** is published only for the 122 of 177 samples whose source datasets carry a declared redistribution-permissive license (`benchmark/ground_truth_public/`). Ground truth for the remaining 55 samples (sources with unspecified license terms) is withheld; its SHA-256 and upstream reference are preserved.
- Model configuration files and weights for the evaluated models are not included; users must obtain them from their official repositories.

## Why

Benchmark inputs derive from third-party public datasets (14 sources) with differing redistribution terms. A dataset being publicly viewable on Hugging Face does not by itself establish the right to redistribute it here. Where license evidence is absent locally, Clouda OCR classifies the source as UNKNOWN and withholds the corresponding raw content. See `LICENSE_REVIEW.md` for the per-source assessment.

## Auditability without redistribution

The combination of per-sample SHA-256 hashes, upstream dataset names, pinned repository revisions, split and row indices in the manifests means any auditor with lawful access to the upstream datasets can verify, byte-for-byte, that the published results were computed on the exact benchmark samples claimed.

## Reproduction

The benchmark can be reconstructed from the pinned upstream sources and the published deterministic generation pipeline, subject to the original datasets' access and license terms; the published results can then be recomputed from the included scoring implementation. Because the withheld assets are identified by hash and source reference, anyone who lawfully obtains the upstream datasets can verify that the published results correspond to the exact benchmark samples claimed. The scoring implementation and per-sample outputs needed to reproduce every published number are included in this repository.

## Ownership

Nothing in this notice implies that Clouda OCR owns the source datasets or the underlying documents. Clouda OCR claims ownership only of its own benchmark construction, evaluation outputs and tooling. All upstream material remains governed by its respective licenses and terms of access.
