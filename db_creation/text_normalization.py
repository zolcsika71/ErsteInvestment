"""Shared text-normalization helpers."""

from __future__ import annotations

import unicodedata
from typing import Any


def normalized_key(value: Any) -> str:
    """Return a trimmed, Unicode-normalized, case-insensitive lookup key."""
    return unicodedata.normalize("NFC", str(value).strip()).casefold()
