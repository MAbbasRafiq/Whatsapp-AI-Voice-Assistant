"""
Simple in-memory rate limiter for voice note processing and TTS.

Limits each phone number to 1 voice note every `RATE_LIMIT_WINDOW_SECONDS`
seconds, and separately gates TTS (/voice + Listen) with a cooldown plus
an in-flight busy lock so double-taps cannot run two synthesizers at once.

This is intentionally a plain in-process dict, not backed by Redis/DB —
it resets on every restart and doesn't work across multiple app
instances, but for this app's current scale (single Railway web process)
that's an acceptable trade-off for the simplicity it buys us.
If this ever needs to run with >1 process, swap this for a shared store
(Redis, or a DB-backed check) without changing the call site.
"""

import time
from threading import Lock

RATE_LIMIT_WINDOW_SECONDS = 10.0
TTS_RATE_LIMIT_WINDOW_SECONDS = 10.0

# wa_phone -> monotonic timestamp of the last *accepted* voice note.
_last_voice_note_at: dict[str, float] = {}
# wa_phone -> monotonic timestamp of the last *finished* TTS attempt.
_last_tts_at: dict[str, float] = {}
# Phones currently inside the TTS pipeline (synthesize → upload → send).
_tts_in_progress: set[str] = set()
_lock = Lock()


def check_and_record(wa_phone: str) -> bool:
    """
    Check whether `wa_phone` is allowed to send a voice note right now,
    and if so, record this attempt as the new "last accepted" timestamp.

    Returns:
        True if allowed (and the timestamp was recorded).
        False if the phone is still within the rate-limit window (the
        existing timestamp is left untouched, so the window doesn't keep
        sliding forward on repeated rejected attempts).
    """
    now = time.monotonic()
    with _lock:
        last_at = _last_voice_note_at.get(wa_phone)
        if last_at is not None and (now - last_at) < RATE_LIMIT_WINDOW_SECONDS:
            return False
        _last_voice_note_at[wa_phone] = now
        return True


def try_acquire_tts(wa_phone: str) -> str:
    """
    Gate a TTS request for `wa_phone`.

    Returns:
        "ok" — caller must call `release_tts` when finished.
        "busy" — another TTS job is already in flight for this phone.
        "rate_limited" — still inside the post-TTS cooldown window.
    """
    now = time.monotonic()
    with _lock:
        if wa_phone in _tts_in_progress:
            return "busy"
        last_at = _last_tts_at.get(wa_phone)
        if last_at is not None and (now - last_at) < TTS_RATE_LIMIT_WINDOW_SECONDS:
            return "rate_limited"
        _tts_in_progress.add(wa_phone)
        return "ok"


def release_tts(wa_phone: str) -> None:
    """Clear the in-flight flag and start the TTS cooldown window."""
    with _lock:
        _tts_in_progress.discard(wa_phone)
        _last_tts_at[wa_phone] = time.monotonic()
