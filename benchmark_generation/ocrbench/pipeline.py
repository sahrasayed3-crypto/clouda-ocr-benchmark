"""End-to-end CPU benchmark generation and artifact reporting."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .allocation import allocate
from .config import BenchmarkConfig, load_config
from .distortions import apply_distortion, apply_profile
from .hashing import derive_seed, sha256_file
from .ingest import ingest
from .metrics import assess

SEED = 20260825


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records), encoding="utf-8")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    keys = sorted({k for row in records for k in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG", optimize=True)


def generate_distortions(root: Path, cfg: BenchmarkConfig, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = allocate([s["benchmark_id"] for s in sources], cfg.allocation, SEED)
    by_id = {s["benchmark_id"]: s for s in sources}
    clean_cache: dict[str, np.ndarray] = {}
    readability = cfg.readability
    records: list[dict[str, Any]] = []
    for job in jobs:
        source = by_id[job["benchmark_id"]]
        clean = clean_cache.setdefault(job["benchmark_id"], _read_rgb(root / source["clean_image_path"]))
        # A different real page supplies verso structure; resize is handled by distortion.
        verso_id = sources[(int(source["benchmark_id"].split("_")[-1]) % len(sources))]["benchmark_id"]
        verso = clean_cache.setdefault(verso_id, _read_rgb(root / by_id[verso_id]["clean_image_path"]))
        seed = derive_seed(SEED, job["distorted_id"], distortion=job.get("distortion", job.get("profile", "")), severity=job["severity"])
        rng = np.random.default_rng(seed)
        output = root / "distorted" / job["bucket"] / f"{job['distorted_id']}.png"
        if job["kind"] == "atomic":
            spec = cfg.distortion(job["distortion"])
            steps = [{"distortion": job["distortion"], "severity": job["severity"]}]
            out = (_read_rgb(output) if output.is_file() else
                   apply_distortion(job["distortion"], clean, spec.params(job["severity"]), rng, {"verso": verso}))
        else:
            profile = cfg.profile(job["profile"])
            steps = [{"distortion": s.distortion, "severity": s.severity} for s in profile.steps]
            out = (_read_rgb(output) if output.is_file() else
                   apply_profile(profile.steps, clean, cfg, rng, {"verso": verso}))
        qc = assess(out, clean, min_contrast_ratio=readability["min_contrast_ratio"],
                    min_sharpness_ratio=readability["min_sharpness_ratio"],
                    min_ink_ratio=readability["min_ink_ratio"], max_ink_ratio=readability["max_ink_ratio"])
        if not output.is_file():
            _save_rgb(output, out)
        records.append({**job, "seed": seed, "steps": steps,
                        "distorted_image_path": str(output.relative_to(root)),
                        "distorted_sha256": sha256_file(output), "qc": qc.as_dict()})
    return records


def _thumb(path: Path, size: tuple[int, int], label: str) -> Image.Image:
    with Image.open(path) as source:
        im = ImageOps.contain(source.convert("RGB"), (size[0] - 20, size[1] - 55))
    cell = Image.new("RGB", size, "white")
    cell.paste(im, ((size[0] - im.width) // 2, 42 + (size[1] - 55 - im.height) // 2))
    ImageDraw.Draw(cell).text((10, 10), label, fill="black", font=ImageFont.load_default())
    return cell


def _sheet(path: Path, cells: list[tuple[Path, str]], columns: int = 4) -> None:
    size = (330, 430)
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * size[0], rows * size[1]), (225, 225, 225))
    for i, (image_path, label) in enumerate(cells):
        sheet.paste(_thumb(image_path, size, label), ((i % columns) * size[0], (i // columns) * size[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, optimize=True)


def generate_previews(root: Path, cfg: BenchmarkConfig, sources: list[dict[str, Any]]) -> list[str]:
    preview_dir = root / "qc/contact_sheets"
    preview_dir.mkdir(parents=True, exist_ok=True)
    base_source = sources[0]
    clean_path = root / base_source["clean_image_path"]
    clean = _read_rgb(clean_path)
    verso = _read_rgb(root / sources[1]["clean_image_path"])
    groups = {
        "blur": "gaussian_blur", "noise": "gaussian_noise", "jpeg_compression": "jpeg_compression",
        "skew_perspective": "perspective", "shadow_lighting": "shadow",
        "faded_old_paper": "paper_degradation", "bleedthrough": "bleedthrough",
    }
    outputs: list[str] = []
    for group, distortion in groups.items():
        cells = [(clean_path, "Original")]
        for severity in ("light", "medium", "heavy"):
            out = apply_distortion(distortion, clean, cfg.distortion(distortion).params(severity),
                                   np.random.default_rng(derive_seed(SEED, group, distortion=distortion, severity=severity)),
                                   {"verso": verso})
            temp = preview_dir / f".{group}_{severity}.png"
            _save_rgb(temp, out)
            cells.append((temp, severity.title()))
        path = preview_dir / f"preview_{group}.png"
        _sheet(path, cells)
        outputs.append(str(path.relative_to(root)))
    cells = [(clean_path, "Original")]
    for profile_name in ("old_book_light", "phone_photo_medium", "bad_scan_heavy"):
        profile = cfg.profile(profile_name)
        out = apply_profile(profile.steps, clean, cfg, np.random.default_rng(derive_seed(SEED, profile_name)), {"verso": verso})
        temp = preview_dir / f".combined_{profile_name}.png"
        _save_rgb(temp, out)
        cells.append((temp, profile_name))
    combined_path = preview_dir / "preview_combined_realistic.png"
    _sheet(combined_path, cells)
    outputs.append(str(combined_path.relative_to(root)))
    # One source from every repository, preserving repository-level diversity.
    seen: set[str] = set()
    mixed: list[tuple[Path, str]] = []
    for source in sources:
        if source["source_dataset"] not in seen:
            seen.add(source["source_dataset"])
            mixed.append((root / source["clean_image_path"], source["source_dataset"].split("/")[-1]))
    mixed_path = preview_dir / "preview_mixed_sources.png"
    _sheet(mixed_path, mixed, columns=4)
    outputs.append(str(mixed_path.relative_to(root)))
    return outputs


def _pdf(path: Path, root: Path, records: list[dict[str, Any]], image_key: str, caption_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=A4, pageCompression=1)
    page_w, page_h = A4
    for record in records:
        image_path = root / record[image_key]
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            scale = min((page_w - 40) / rgb.width, (page_h - 70) / rgb.height)
            buf = io.BytesIO(); rgb.save(buf, "JPEG", quality=88, optimize=True); buf.seek(0)
            w, h = rgb.width * scale, rgb.height * scale
            pdf.drawImage(ImageReader(buf), (page_w - w) / 2, (page_h - h) / 2 - 10, w, h)
        pdf.setFont("Helvetica", 8); pdf.drawString(20, page_h - 20, str(record[caption_key]))
        pdf.showPage()
    pdf.save()


def run(root: Path) -> dict[str, Any]:
    cfg = load_config(root / "configs/distortions.yaml", root / "configs/layout.yaml", root / "configs/allocation.yaml")
    sources = ingest(root / "configs/source_selectors.csv", root)
    _write_jsonl(root / "manifests/source_manifest.jsonl", sources)
    _write_csv(root / "manifests/source_manifest.csv", sources)
    distortions = generate_distortions(root, cfg, sources)
    manifest = [{**d, "source": next(s for s in sources if s["benchmark_id"] == d["benchmark_id"])} for d in distortions]
    _write_jsonl(root / "manifests/benchmark_manifest.jsonl", manifest)
    _write_csv(root / "manifests/benchmark_manifest.csv", distortions)
    failures = [{"distorted_id": d["distorted_id"], "reasons": d["qc"]["reasons"]} for d in distortions if not d["qc"]["passed"]]
    report = {
        "requested_sources": 100, "resolved_sources": len(sources),
        "clean_images": len(sources), "distorted_images": len(distortions),
        "source_counts": dict(Counter(s["source_dataset"] for s in sources)),
        "replacement_count": sum(bool(s["replacement_status"]) for s in sources),
        "replacements": [s for s in sources if s["replacement_status"]],
        "qc_failures": failures, "config_hash": cfg.hash,
        "source_manifest_sha256": sha256_file(root / "manifests/source_manifest.jsonl"),
        "benchmark_manifest_sha256": sha256_file(root / "manifests/benchmark_manifest.jsonl"),
    }
    (root / "reports").mkdir(exist_ok=True); (root / "qc").mkdir(exist_ok=True)
    (root / "reports/summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "qc/qc_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    previews = generate_previews(root, cfg, sources)
    clean_pdf = root / "pdfs/clean_benchmark.pdf"
    distorted_pdf = root / "pdfs/distorted_benchmark.pdf"
    if not clean_pdf.is_file():
        _pdf(clean_pdf, root, sources, "clean_image_path", "benchmark_id")
    if not distorted_pdf.is_file():
        _pdf(distorted_pdf, root, distortions, "distorted_image_path", "distorted_id")
    report["previews"] = previews
    return report
