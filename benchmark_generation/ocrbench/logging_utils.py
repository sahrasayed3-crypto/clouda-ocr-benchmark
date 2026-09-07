"""Structured logging for the benchmark pipeline.

Emits human-readable lines to the console and, optionally, newline-delimited
JSON to a log file so that runs can be diffed and audited.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Mapping

_CONSOLE_FMT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_DATE_FMT = "%H:%M:%S"


class JsonLinesHandler(logging.Handler):
    """Write each record as one JSON object per line."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload: dict[str, Any] = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            extra = getattr(record, "context", None)
            if isinstance(extra, Mapping):
                payload["context"] = dict(extra)
            if record.exc_info:
                payload["exception"] = self.format(record)
            self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:  # logging must never crash the pipeline
            self.handleError(record)

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure root logging once. Safe to call repeatedly."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    root.addHandler(console)

    if log_file is not None:
        jsonl = JsonLinesHandler(log_file)
        jsonl.setLevel(logging.DEBUG)
        root.addHandler(jsonl)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_context(logger: logging.Logger, level: int, message: str, **context: Any) -> None:
    """Log `message` with a structured `context` payload attached."""
    logger.log(level, message, extra={"context": context})
