#!/usr/bin/env python3
"""Record exact environment packages and disk consumption."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def disk_bytes(path: Path) -> int:
    result = subprocess.run(["du", "-sb", str(path)], check=True, capture_output=True, text=True)
    return int(result.stdout.split()[0])


def main() -> int:
    envs = {}
    for name in ("modern_vlm", "legacy_ocr", "dots_mocr", "paddle_native"):
        freeze = ROOT/f"configs/locks/{name}.freeze.txt"
        envs[name] = {
            "path": str(ROOT/f"envs/{name}"),
            "disk_size_bytes": disk_bytes(ROOT/f"envs/{name}"),
            "freeze_file": str(freeze),
            "packages": freeze.read_text(encoding="utf-8").splitlines(),
        }
    report = {
        "runtime_root": str(ROOT),
        "runtime_total_disk_size_bytes": disk_bytes(ROOT),
        "shared_gpu_libraries": {
            "cu128": {"path": str(ROOT/"shared/cu128"), "disk_size_bytes": disk_bytes(ROOT/"shared/cu128")},
            "cu118": {"path": str(ROOT/"shared/cu118"), "disk_size_bytes": disk_bytes(ROOT/"shared/cu118")},
        },
        "environments": envs,
        "notes": [
            "modern_vlm and dots_mocr share the cu128 Torch/torchvision files via .pth files.",
            "legacy_ocr shares the cu118 Torch/torchvision files via a .pth file.",
            "No CUDA workload or model inference was executed during preparation.",
        ],
    }
    output = ROOT/"reports/environment_report.json"
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runtime_total_disk_size_bytes": report["runtime_total_disk_size_bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
