"""
In-memory "last transcript" cache, keyed by phone number.

Used by Translate / Summarize (interactive buttons and slash-command
fallbacks) to re-run the LLM on a user's most recent cleaned transcript
without re-running Whisper (cheaper, faster, and Whisper output doesn't
change between calls anyway). Plain in-process dict, same trade-offs as
`rate_limiter.py` — resets on restart, single-process only, which is fine
at this app's scale.

Each new voice note simply overwrites the previous entry for that phone
number (there's only ever "the most recent transcript" per user, not a
history) and entries expire after `CACHE_TTL_SECONDS` so Translate on a
stale transcript correctly reports "no recent transcript" rather than
resurfacing old content.
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


def set_last_transcript(wa_phone: str, text: str, language: str) -> None:
    with _lock:
        _cache[wa_phone] = _CacheEntry(text=text, language=language, stored_at=time.monotonic())


def get_last_transcript(wa_phone: str) -> tuple[str, str] | None:
    """
    Return (text, language) for the most recent transcript, or None if
    there isn't one (never sent a voice note, or it expired).
    """
    with _lock:
        entry = _cache.get(wa_phone)
        if entry is None:
            return None
        if time.monotonic() - entry.stored_at > CACHE_TTL_SECONDS:
            del _cache[wa_phone]
            return None
        return entry.text, entry.language
