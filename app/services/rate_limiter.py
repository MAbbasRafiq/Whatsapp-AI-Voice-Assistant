"""
Simple in-memory rate limiter for voice note processing.

Limits each phone number to 1 voice note every `RATE_LIMIT_WINDOW_SECONDS`
seconds. This is intentionally a plain in-process dict, not backed by
Redis/DB — it resets on every restart and doesn't work across multiple
app instances, but for this app's current scale (single Railway web
process) that's an acceptable trade-off for the simplicity it buys us.
If this ever needs to run with >1 process, swap this for a shared store
(Redis, or a DB-backed check) without changing the call site.
"""

import time
from threading import Lock

RATE_LIMIT_WINDOW_SECONDS = 10.0

# wa_phone -> monotonic timestamp of the last *accepted* voice note.
_last_voice_note_at: dict[str, float] = {}
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
