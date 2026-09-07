"""Framework-independent OCR metrics with explicit edit accounting."""
from __future__ import annotations

from typing import Sequence

from .normalization import normalize_arabic


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> dict[str, int]:
    # Cell: (cost, substitutions, insertions, deletions); tuple ordering makes ties deterministic.
    prev = [(j, 0, j, 0) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        cur = [(i, 0, 0, i)]
        for j, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                cur.append(prev[j - 1])
                continue
            sub = (prev[j-1][0]+1, prev[j-1][1]+1, prev[j-1][2], prev[j-1][3])
            ins = (cur[j-1][0]+1, cur[j-1][1], cur[j-1][2]+1, cur[j-1][3])
            delete = (prev[j][0]+1, prev[j][1], prev[j][2], prev[j][3]+1)
            cur.append(min((sub, ins, delete), key=lambda x: (x[0], x[1], x[2], x[3])))
        prev = cur
    distance, substitutions, insertions, deletions = prev[-1]
    return {"distance": distance, "substitutions": substitutions,
            "insertions": insertions, "deletions": deletions}


def _rate(distance: int, length: int) -> float:
    return float(distance / length) if length else (0.0 if distance == 0 else 1.0)


def evaluate_text(raw_ground_truth: str, raw_prediction: str) -> dict[str, float | int]:
    strict = edit_counts(list(raw_ground_truth), list(raw_prediction))
    ngt, npred = normalize_arabic(raw_ground_truth), normalize_arabic(raw_prediction)
    normalized = edit_counts(list(ngt), list(npred))
    words = edit_counts(raw_ground_truth.split(), raw_prediction.split())
    denom = len(raw_ground_truth)
    return {
        "strict_cer": _rate(strict["distance"], denom),
        "normalized_arabic_cer": _rate(normalized["distance"], len(ngt)),
        "wer": _rate(words["distance"], len(raw_ground_truth.split())),
        "extra_text_rate": _rate(strict["insertions"], denom),
        "missing_text_rate": _rate(strict["deletions"], denom),
        "substitutions": strict["substitutions"], "insertions": strict["insertions"],
        "deletions": strict["deletions"],
    }

