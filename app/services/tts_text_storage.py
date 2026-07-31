"""
Background archival of Convert-to-Voice text to Supabase Postgres.

Inserts one row into the `tts_texts` table when a user taps Convert to
Voice. Designed to be strictly fire-and-forget:

- Callers schedule an insert and continue immediately (never await it).
- Failures are logged only — they never raise into the TTS / WhatsApp
  reply path.

Uses the same SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY as voice Storage
archival. Does not touch Storage buckets.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any

from app.config import settings
from app.utils import mask_phone

logger = logging.getLogger(__name__)

# Keep strong refs to in-flight tasks so the event loop doesn't GC them
# before they finish (asyncio only holds weak refs to create_task results).
_background_tasks: set[asyncio.Task[None]] = set()


def _supabase_configured() -> bool:
    return bool(settings.supabase_url and settings.supabase_service_role_key)


@lru_cache
def _get_supabase_client() -> Any:
    """Lazily construct a sync Supabase client (service role → bypasses RLS)."""
    from supabase import create_client

    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def _insert_sync(phone_number: str, text: str) -> None:
    """Blocking Supabase table insert — run via asyncio.to_thread."""
    client = _get_supabase_client()
    client.table("tts_texts").insert(
        {
            "phone_number": phone_number,
            "text": text,
        }
    ).execute()


async def _archive_tts_text(wa_phone: str, text: str) -> None:
    """Perform the insert; swallow and log all errors."""
    if not _supabase_configured():
        logger.warning(
            "Skipping TTS text archival — SUPABASE_URL / "
            "SUPABASE_SERVICE_ROLE_KEY not configured | sender=%s",
            mask_phone(wa_phone),
        )
        return

    try:
        await asyncio.to_thread(_insert_sync, wa_phone, text)
        logger.info(
            "TTS text archived to Supabase | table=tts_texts | sender=%s | chars=%d",
            mask_phone(wa_phone),
            len(text),
        )
    except Exception:
        logger.exception(
            "TTS text archival failed (ignored — does not affect user reply) | "
            "table=tts_texts | sender=%s",
            mask_phone(wa_phone),
        )


def schedule_tts_text_archive(wa_phone: str, text: str) -> None:
    """
    Fire-and-forget: schedule a background insert and return immediately.

    Never awaits the insert. Safe to call from the Convert-to-Voice path —
    a failure here cannot delay or alter TTS / WhatsApp replies.
    """
    try:
        task = asyncio.create_task(
            _archive_tts_text(wa_phone, text),
            name=f"tts-text-archive-{wa_phone}",
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except Exception:
        # Even scheduling must never break the pipeline.
        logger.exception(
            "Failed to schedule TTS text archival | sender=%s",
            mask_phone(wa_phone),
        )
