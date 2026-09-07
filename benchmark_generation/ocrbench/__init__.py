"""Arabic OCR benchmark dataset preparation toolkit (CPU-only).

This package renders Arabic / mixed Arabic-English source text into realistic
document page images, derives byte-exact ground truth directly from the source
text, applies a configurable distortion engine, and exports a manifest, PDFs
and QC reports.

It never runs, downloads or depends on any OCR model, vision-language model or
other machine-learning inference engine. Ground truth is produced by slicing
the supplied source text, not by recognising anything.
"""

__all__ = ["BENCHMARK_VERSION"]

BENCHMARK_VERSION = "benchmark-v1.0"
