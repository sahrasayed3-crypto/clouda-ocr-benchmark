from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import evaluate_text
from .normalization import normalize_arabic


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if "distorted_id" not in row or "distorted_image_path" not in row or "source" not in row:
            raise ValueError("benchmark manifest row lacks distorted_id, distorted_image_path, or source")
    return rows


def select_samples(rows: list[dict[str, Any]], *, sample_id: str | None, limit: int | None) -> list[dict[str, Any]]:
    selected = [r for r in rows if sample_id is None or r["distorted_id"] == sample_id]
    if sample_id is not None and not selected:
        raise ValueError(f"sample id not found: {sample_id}")
    return selected[:limit] if limit is not None else selected


def completed_ids(path: Path) -> set[str]:
    if not path.is_file(): return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if not row.get("error"): done.add(row["sample_id"])
    return done


def result_record(model: str, sample_id: str, prediction: str, ground_truth: str,
                  timing: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluate_text(ground_truth, prediction) if not timing.get("error") else None
    return {
        "model": model, "sample_id": sample_id,
        "raw_prediction": prediction, "normalized_prediction": normalize_arabic(prediction),
        "raw_ground_truth": ground_truth, "normalized_ground_truth": normalize_arabic(ground_truth),
        "metrics": metrics, "raw_model_response": timing.get("raw_response"),
        "preprocessing_time": timing.get("preprocessing_time", 0.0),
        "inference_time": timing.get("elapsed_inference_time", timing.get("inference_time", 0.0)),
        "postprocessing_time": timing.get("postprocessing_time", 0.0),
        "total_time": (float(timing.get("preprocessing_time", 0.0)) +
                       float(timing.get("elapsed_inference_time", timing.get("inference_time", 0.0))) +
                       float(timing.get("postprocessing_time", 0.0))),
        "error": timing.get("error"), **run,
    }
