"""Deterministic Arabic normalization; raw strings are never mutated or overwritten."""
from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TRANSLATION = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ـ": ""})


def normalize_arabic(text: str) -> str:
    value = unicodedata.normalize("NFC", text).translate(_TRANSLATION)
    value = _DIACRITICS.sub("", value)
    return " ".join(value.split())

