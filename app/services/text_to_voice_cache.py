"""
In-memory cache of pending "Convert to Voice" text per user.

Same pattern and trade-offs as `transcript_cache.py` / `tts_cache.py`:
plain in-process dict, resets on restart, single-process only. Overwritten
each time the user sends a new eligible plain-text message; entries expire
after `CACHE_TTL_SECONDS`.
"""

import time
from dataclasses import dataclass
from threading import Lock

CACHE_TTL_SECONDS = 15 * 60


@dataclass
class _CacheEntry:
    text: str
    stored_at: float


_cache: dict[str, _CacheEntry] = {}
_lock = Lock()


def set_pending_text(wa_phone: str, text: str) -> None:
    with _lock:
        _cache[wa_phone] = _CacheEntry(text=text, stored_at=time.monotonic())


def get_pending_text(wa_phone: str) -> str | None:
    """Return cached plain text, or None if missing/expired."""
    with _lock:
        entry = _cache.get(wa_phone)
        if entry is None:
            return None
        if time.monotonic() - entry.stored_at > CACHE_TTL_SECONDS:
            del _cache[wa_phone]
            return None
        return entry.text
