"""
Short-lived per-user conversation state.

Currently used only for the "Other language..." translate flow: after the
user picks that list option we set `waiting_for_language` so the next
plain-text message is treated as the target language name rather than a
generic nudge.

Same in-process dict trade-offs as `transcript_cache.py` / `rate_limiter.py`
(resets on restart, single-process only). Entries expire after
`STATE_TTL_SECONDS` so a forgotten pending prompt can't linger forever.
"""

import time
from dataclasses import dataclass
from threading import Lock

STATE_TTL_SECONDS = 15 * 60


@dataclass
class _UserState:
    waiting_for_language: bool = False
    updated_at: float = 0.0


_states: dict[str, _UserState] = {}
_lock = Lock()


def set_waiting_for_language(wa_phone: str, waiting: bool = True) -> None:
    """Mark (or clear) that the next text from this user is a language name."""
    with _lock:
        if not waiting:
            _states.pop(wa_phone, None)
            return
        _states[wa_phone] = _UserState(
            waiting_for_language=True,
            updated_at=time.monotonic(),
        )


def clear_waiting_for_language(wa_phone: str) -> None:
    set_waiting_for_language(wa_phone, waiting=False)


def is_waiting_for_language(wa_phone: str) -> bool:
    """
    True if this user has an unexpired pending "type a language" prompt.
    Expired entries are removed as a side effect.
    """
    with _lock:
        entry = _states.get(wa_phone)
        if entry is None or not entry.waiting_for_language:
            return False
        if time.monotonic() - entry.updated_at > STATE_TTL_SECONDS:
            del _states[wa_phone]
            return False
        return True
