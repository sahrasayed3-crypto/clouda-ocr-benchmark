#!/usr/bin/env python3
"""Aggregate runner results without altering raw or normalized text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("strict_cer", "normalized_arabic_cer", "wer", "extra_text_rate", "missing_text_rate")
TIMINGS = ("preprocessing_time", "inference_time", "postprocessing_time", "total_time")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.results.read_text(encoding="utf-8").splitlines() if x.strip()]
    ok = [x for x in rows if not x.get("error")]
    def mean(key, nested=False):
        values = [(x["metrics"][key] if nested else x[key]) for x in ok]
        return sum(values)/len(values) if values else None
    hardware = sorted({(x.get("gpu_model"), x.get("gpu_vram_gb")) for x in rows})
    report = {
        "samples": len(rows), "successful": len(ok), "failures": len(rows)-len(ok),
        "mean_metrics": {key: mean(key, True) for key in METRICS},
        "mean_timings_seconds": {key: mean(key) for key in TIMINGS},
        "hardware": [{"gpu_model": x[0], "gpu_vram_gb": x[1]} for x in hardware],
        "speed_comparison_warning": "Do not compare timing across different GPU models.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
