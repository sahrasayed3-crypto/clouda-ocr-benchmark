"""Balanced deterministic distortion allocation."""
from __future__ import annotations

import hashlib
from typing import Any


def allocate(source_ids: list[str], config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    if not source_ids:
        raise ValueError("cannot allocate without source samples")
    records: list[dict[str, Any]] = []
    cursor = seed % len(source_ids)
    severities = list(config.get("default_severity_mix", {"light": 1, "medium": 1, "heavy": 1}))
    for bucket, body in config.get("buckets", {}).items():
        distortions = list(body["distortions"])
        for i in range(int(body["count"])):
            source_id = source_ids[(cursor + i * 37) % len(source_ids)]
            distortion = distortions[i % len(distortions)]
            severity = severities[(i // len(distortions)) % len(severities)]
            token = f"{source_id}|{bucket}|{distortion}|{severity}|{i}|{seed}"
            records.append({
                "distorted_id": "dist_" + hashlib.sha256(token.encode()).hexdigest()[:16],
                "benchmark_id": source_id, "kind": "atomic", "bucket": bucket,
                "distortion": distortion, "severity": severity,
            })
        cursor = (cursor + int(body["count"])) % len(source_ids)
    profiles = config.get("combined", {}).get("profiles", {})
    for profile, count in profiles.items():
        for i in range(int(count)):
            source_id = source_ids[(cursor + i * 41) % len(source_ids)]
            token = f"{source_id}|combined|{profile}|{i}|{seed}"
            records.append({
                "distorted_id": "dist_" + hashlib.sha256(token.encode()).hexdigest()[:16],
                "benchmark_id": source_id, "kind": "profile", "bucket": "combined",
                "profile": profile, "severity": "combined",
            })
        cursor = (cursor + int(count)) % len(source_ids)
    return records

