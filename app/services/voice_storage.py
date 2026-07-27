"""
Background voice-note archival to Supabase Storage.

Uploads raw voice/audio bytes to a private bucket so recordings are retained
for later use. Designed to be strictly fire-and-forget:

- Callers schedule an upload and continue immediately (never await it).
- Failures are logged only — they never raise into the voice pipeline and
  never affect transcription, LLM cleanup, or WhatsApp replies.

Object key layout (bucket = SUPABASE_STORAGE_BUCKET, default voice-recordings):

    <phone_number>/<whatsapp_message_id>.<ext>
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

_MIME_TO_EXT: dict[str, str] = {
    "audio/ogg": "ogg",
    "audio/opus": "ogg",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/aac": "aac",
    "audio/amr": "amr",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
    "audio/flac": "flac",
}


def extension_from_mime(mime_type: str | None) -> str:
    """Map a WhatsApp/Graph API mime_type to a file extension (default ogg)."""
    if not mime_type:
        return "ogg"
    base = mime_type.split(";", 1)[0].strip().lower()
    return _MIME_TO_EXT.get(base, "ogg")


def _storage_configured() -> bool:
    return bool(settings.supabase_url and settings.supabase_service_role_key)


@lru_cache
def _get_supabase_client() -> Any:
    """Lazily construct a sync Supabase client (service role → bypasses RLS)."""
    from supabase import create_client

    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def _upload_sync(
    audio_bytes: bytes,
    object_path: str,
    content_type: str,
) -> None:
    """Blocking Supabase Storage upload — run via asyncio.to_thread."""
    client = _get_supabase_client()
    client.storage.from_(settings.supabase_storage_bucket).upload(
        path=object_path,
        file=audio_bytes,
        file_options={
            "content-type": content_type,
            "upsert": "false",
        },
    )


async def _upload_voice_recording(
    audio_bytes: bytes,
    wa_phone: str,
    wa_message_id: str,
    mime_type: str | None,
) -> None:
    """Perform the upload; swallow and log all errors."""
    if not _storage_configured():
        logger.warning(
            "Skipping voice storage upload — SUPABASE_URL / "
            "SUPABASE_SERVICE_ROLE_KEY not configured | sender=%s | id=%s",
            mask_phone(wa_phone),
            wa_message_id,
        )
        return

    ext = extension_from_mime(mime_type)
    object_path = f"{wa_phone}/{wa_message_id}.{ext}"
    content_type = (mime_type or "audio/ogg").split(";", 1)[0].strip() or "audio/ogg"

    try:
        await asyncio.to_thread(_upload_sync, audio_bytes, object_path, content_type)
        logger.info(
            "Voice recording archived to Supabase Storage | bucket=%s | path=%s | sender=%s",
            settings.supabase_storage_bucket,
            object_path,
            mask_phone(wa_phone),
        )
    except Exception:
        logger.exception(
            "Voice storage upload failed (ignored — does not affect user reply) | "
            "bucket=%s | path=%s | sender=%s | id=%s",
            settings.supabase_storage_bucket,
            object_path,
            mask_phone(wa_phone),
            wa_message_id,
        )


def schedule_voice_upload(
    audio_bytes: bytes,
    wa_phone: str,
    wa_message_id: str,
    mime_type: str | None = None,
) -> None:
    """
    Fire-and-forget: schedule a background upload and return immediately.

    Never awaits the upload. Safe to call from the voice pipeline — a
    failure here cannot delay or alter the user-facing response path.
    """
    try:
        task = asyncio.create_task(
            _upload_voice_recording(audio_bytes, wa_phone, wa_message_id, mime_type),
            name=f"voice-storage-{wa_message_id}",
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except Exception:
        # Even scheduling must never break the pipeline.
        logger.exception(
            "Failed to schedule voice storage upload | sender=%s | id=%s",
            mask_phone(wa_phone),
            wa_message_id,
        )
