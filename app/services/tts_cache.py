"""
In-memory cache of the most recent summary/translation text per user,
keyed by phone number — used by the "Listen" TTS button.

Same pattern and trade-offs as `transcript_cache.py`: plain in-process
dict, resets on restart, single-process only. Overwritten each time a
new summary or translation is generated for that phone; entries expire
after `CACHE_TTL_SECONDS`.
"""

import time
from dataclasses import dataclass
from threading import Lock

CACHE_TTL_SECONDS = 15 * 60


@dataclass
class _CacheEntry:
    text: str
    language: str
    stored_at: float


_cache: dict[str, _CacheEntry] = {}
_lock = Lock()


def set_tts_cache(phone: str, text: str, language: str) -> None:
    with _lock:
        _cache[phone] = _CacheEntry(text=text, language=language, stored_at=time.monotonic())


def get_tts_cache(phone: str) -> tuple[str, str] | None:
    """
    Return (text, language) for the most recent summary/translation, or
    None if missing or expired.
    """
    with _lock:
        entry = _cache.get(phone)
        if entry is None:
            return None
        if time.monotonic() - entry.stored_at > CACHE_TTL_SECONDS:
            del _cache[phone]
            return None
        return entry.text, entry.language
