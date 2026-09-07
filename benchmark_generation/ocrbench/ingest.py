"""Revision-aware ingestion of approved Hugging Face dataset selectors."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

from huggingface_hub import HfApi
from PIL import Image, ImageDraw

from .render import FontBook, TextMeasurer

VIEWER = "https://datasets-server.huggingface.co"
_LAST_VIEWER_CALL = 0.0
CRANE_ARCHIVE = (
    "https://drive.usercontent.google.com/download?"
    "id=1PZ2VmHQBOPTrMpBf8ZFKSBjQFmcqPu1f&export=download&confirm=t"
)


@dataclass(frozen=True)
class Selector:
    dataset: str
    split: str
    row: int

    @classmethod
    def parse(cls, value: str) -> "Selector":
        dataset, split, row = (part.strip() for part in value.split(","))
        return cls(dataset, split, int(row))


def choose_nearest(requested: int, candidates: Iterable[int], used: set[int]) -> int:
    available = [n for n in candidates if n not in used]
    if not available:
        raise ValueError("no unused replacement row remains")
    return min(available, key=lambda n: (abs(n - requested), n))


def read_selectors(path: Path) -> list[Selector]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [Selector(r["dataset"], r["split"], int(r["row"])) for r in csv.DictReader(fh)]


def _get_json(endpoint: str, **params: Any) -> dict[str, Any]:
    global _LAST_VIEWER_CALL
    url = f"{VIEWER}/{endpoint}?{urlencode(params)}"
    error: Exception | None = None
    # Anonymous Dataset Viewer traffic is rate limited. A small global spacing
    # is faster than repeatedly tripping the limiter and makes long runs stable.
    spacing = 2.2 - (time.monotonic() - _LAST_VIEWER_CALL)
    if spacing > 0:
        time.sleep(spacing)
    for delay in (0, 15, 30, 60):
        if delay:
            time.sleep(delay)
        try:
            with urlopen(url, timeout=90) as response:
                _LAST_VIEWER_CALL = time.monotonic()
                return json.load(response)
        except Exception as exc:  # transient Viewer/CDN errors are common
            error = exc
    raise RuntimeError(f"Dataset Viewer request failed: {url}: {error}")


def _row(dataset: str, split: str, index: int) -> dict[str, Any] | None:
    result = _get_json(
        "rows", dataset=dataset, config="default", split=split,
        offset=index, length=1,
    )
    rows = result.get("rows", [])
    if not rows or int(rows[0].get("row_idx", -1)) != index:
        return None
    return rows[0]["row"]


def _download(url: str) -> bytes:
    error: Exception | None = None
    for delay in (0, 1, 2, 4):
        if delay:
            time.sleep(delay)
        try:
            with urlopen(url, timeout=120) as response:
                return response.read()
        except Exception as exc:
            error = exc
    raise RuntimeError(f"download failed: {url}: {error}")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ground_truth(row: dict[str, Any]) -> tuple[str, str]:
    for key in ("answer", "transcription", "markdown"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value, key
    texts = row.get("texts")
    if isinstance(texts, list) and any(str(v).strip() for v in texts):
        return "\n".join(str(v) for v in texts if str(v).strip()), "texts"
    # The tables dataset's JSON data is the authoritative semantic content.
    data = row.get("data")
    if isinstance(data, str) and data.strip():
        return data, "data"
    return "", ""


def _render_ebook(text: str, fontbook: FontBook, measurer: TextMeasurer) -> tuple[bytes, str]:
    # Preserve a page-sized exact prefix, ending at whitespace, without rewriting.
    segment = text[:5200]
    if len(text) > len(segment):
        cut = max(segment.rfind("\n"), segment.rfind(" "))
        if cut > 2500:
            segment = segment[:cut]
    font = fontbook.get("amiri", 34)
    title_font = fontbook.get("amiri", 44, bold=True)
    lines = measurer.wrap(font, segment, 1080)
    max_lines = 42
    if len(lines) > max_lines:
        segment = segment[: lines[max_lines - 1].end]
        lines = lines[:max_lines]
    page = Image.new("L", (1240, 1754), 255)
    draw = ImageDraw.Draw(page)
    draw.text((1140, 75), "نص من مجموعة الكتب العربية", font=title_font,
              fill=25, anchor="ra", direction="rtl", language="ar")
    y = 155
    for line in lines:
        draw.text((1140, y), line.text, font=font, fill=25,
                  anchor="ra", direction="rtl", language="ar")
        y += 37
    out = io.BytesIO()
    page.save(out, format="PNG", optimize=True)
    return out.getvalue(), segment


def _crane_pairs(count: int) -> list[tuple[str, bytes, str]]:
    """Read only selected members of the public 4.39GB ZIP using HTTP ranges."""
    from remotezip import RemoteZip

    pairs: list[tuple[str, bytes, str]] = []
    with RemoteZip(CRANE_ARCHIVE) as archive:
        names = set(archive.namelist())
        for image_name in sorted(n for n in names if n.startswith("ar_pub/images/") and n.endswith(".png")):
            stem = Path(image_name).stem
            text_name = f"ar_pub/texts/{stem}.txt"
            if text_name not in names:
                continue
            text = archive.read(text_name).decode("utf-8-sig", errors="strict").strip()
            if not text:
                continue
            pairs.append((stem, archive.read(image_name), text))
            if len(pairs) == count:
                return pairs
    raise RuntimeError(f"Craneset public archive exposed only {len(pairs)} valid pairs")


def ingest(selectors_path: Path, root: Path) -> list[dict[str, Any]]:
    selectors = read_selectors(selectors_path)
    if len(selectors) != 100:
        raise ValueError(f"expected exactly 100 selectors, got {len(selectors)}")
    for directory in ("input/images", "input/metadata", "originals", "ground_truth", "manifests"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    api = HfApi()
    revisions: dict[str, str] = {}
    licenses: dict[str, str] = {}
    for dataset in sorted({s.dataset for s in selectors}):
        info = api.dataset_info(dataset)
        revisions[dataset] = info.sha
        card = info.cardData or {}
        license_value = card.get("license", "unspecified")
        licenses[dataset] = json.dumps(license_value, ensure_ascii=False) if not isinstance(license_value, str) else license_value

    fontbook = FontBook(json.load(open(root / "_layout.json")) if False else __import__("yaml").safe_load((root / "configs/layout.yaml").read_text()), root)
    measurer = TextMeasurer()
    used: dict[tuple[str, str], set[int]] = {}
    crane_selectors = [s for s in selectors if s.dataset == "craneset/arabic-ocr"]
    crane_pairs = _crane_pairs(len(crane_selectors)) if crane_selectors else []
    crane_pos = 0
    partial_path = root / "manifests/source_manifest.partial.jsonl"
    records: list[dict[str, Any]] = []
    if partial_path.is_file():
        records = [json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    timestamp = datetime.now(timezone.utc).isoformat()

    for ordinal, selector in enumerate(selectors, 1):
        benchmark_id = f"arabic_ocr_{ordinal:06d}"
        if ordinal <= len(records) and records[ordinal - 1].get("benchmark_id") == benchmark_id:
            continue
        key = (selector.dataset, selector.split)
        used.setdefault(key, set())
        resolved = selector.row
        replacement_reason = ""
        row: dict[str, Any] = {}

        if selector.dataset == "craneset/arabic-ocr":
            stem, image_bytes, gt = crane_pairs[crane_pos]
            crane_pos += 1
            original_id = stem
            original_filename = f"{stem}.png"
            source_type = "public_archive_page"
            replacement_reason = "Viewer row lacks ground truth; replaced by valid unused page/transcription pair from the same dataset public archive"
            gt_field = "texts archive"
            original_meta = {"archive": "ocr-data.zip", "member": original_filename}
        else:
            candidate = _row(selector.dataset, selector.split, selector.row)
            reason = ""
            if candidate is None:
                reason = "requested row index is outside the resolved split"
            elif "text" not in candidate and not isinstance(candidate.get("image"), dict):
                reason = "requested row has no source image or source text"
            elif "text" not in candidate and not _ground_truth(candidate)[0].strip():
                reason = "requested row has missing or empty authoritative ground truth"
            if reason:
                # Nearest-index search expands monotonically and never crosses dataset/split.
                found = None
                for distance in range(1, 10000):
                    for index in (selector.row - distance, selector.row + distance):
                        if index < 0 or index in used[key]:
                            continue
                        probe = _row(selector.dataset, selector.split, index)
                        if probe is None:
                            continue
                        if "text" in probe or (isinstance(probe.get("image"), dict) and _ground_truth(probe)[0].strip()):
                            found = (index, probe)
                            break
                    if found:
                        break
                if not found:
                    raise RuntimeError(f"no valid same-dataset replacement for {selector}")
                resolved, candidate = found
                replacement_reason = reason
            row = candidate
            if "text" in row:
                image_bytes, gt = _render_ebook(str(row["text"]), fontbook, measurer)
                gt_field = "text"
                source_type = "rendered_text_page"
            else:
                image_info = row["image"]
                image_bytes = _download(image_info["src"])
                gt, gt_field = _ground_truth(row)
                source_type = "original_image"
            original_id = str(row.get("id") or row.get("uid") or row.get("image_name") or resolved)
            original_filename = str(row.get("original_filename") or row.get("image_name") or f"row_{resolved}")
            original_meta = row

        # Decode once as corruption check; preserve original bytes separately.
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            width, height = image.size
            clean = image.convert("RGB")
            clean_path = root / "originals" / f"{benchmark_id}.png"
            clean.save(clean_path, format="PNG", optimize=True)
        source_suffix = Path(original_filename).suffix.lower()
        if source_suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}:
            source_suffix = ".bin"
        source_path = root / "input/images" / f"{benchmark_id}{source_suffix}"
        source_path.write_bytes(image_bytes)
        gt_bytes = gt.encode("utf-8")
        gt_path = root / "ground_truth" / f"{benchmark_id}.txt"
        gt_path.write_bytes(gt_bytes)
        metadata_path = root / "input/metadata" / f"{benchmark_id}.json"
        metadata_path.write_text(json.dumps(original_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        used[key].add(resolved)
        records.append({
            "benchmark_id": benchmark_id,
            "source_dataset": selector.dataset,
            "source_split": selector.split,
            "requested_row_index": selector.row,
            "resolved_row_index": resolved,
            "original_sample_id": original_id,
            "original_filename": original_filename,
            "repository_revision": revisions[selector.dataset],
            "source_type": source_type,
            "source_image_path": str(source_path.relative_to(root)),
            "clean_image_path": str(clean_path.relative_to(root)),
            "ground_truth_path": str(gt_path.relative_to(root)),
            "ground_truth_field": gt_field,
            "ground_truth_sha256": _sha(gt_bytes),
            "source_sha256": _sha(image_bytes),
            "clean_sha256": _sha(clean_path.read_bytes()),
            "license_or_permission_note": licenses[selector.dataset],
            "replacement_status": bool(replacement_reason),
            "replacement_reason": replacement_reason,
            "ingestion_timestamp": timestamp,
            "width": width,
            "height": height,
            "metadata_path": str(metadata_path.relative_to(root)),
        })
        _write = json.dumps(records[-1], ensure_ascii=False, sort_keys=True) + "\n"
        with partial_path.open("a", encoding="utf-8") as partial:
            partial.write(_write)
    return records
